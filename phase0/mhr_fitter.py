import logging

import roma
import torch
import torch.nn as nn

from sam_3d_body.models.modules.geometry_utils import rot6d_to_rotmat
from sam_3d_body.models.modules.mhr_utils import (
    compact_cont_to_model_params_body,
    compact_model_params_to_cont_body,
    mhr_param_hand_mask,
)

from .coord_utils import compute_pelvis_alignment

logger = logging.getLogger(__name__)

# Stage definitions: (params_to_optimize, num_iters, learning_rate)
STAGES = [
    {
        "name": "stage1_global_body",
        "params": ["global_rot_6d", "body_pose_cont", "global_trans"],
        "num_iters": 150,
        "lr": 0.02,
    },
    {
        "name": "stage2_shape_scale",
        "params": ["global_rot_6d", "body_pose_cont", "global_trans",
                    "shape_params", "scale_params"],
        "num_iters": 200,
        "lr": 0.01,
    },
    {
        "name": "stage3_hands",
        "params": ["global_rot_6d", "body_pose_cont", "global_trans",
                    "shape_params", "scale_params", "hand_params"],
        "num_iters": 400,
        "lr": 0.01,
    },
]


class MHRFitter:
    """Differentiable optimization to fit MHR parameters to SMPL-X targets."""

    def __init__(self, mhr_head, device="cuda"):
        """
        Args:
            mhr_head: MHRHead instance with populated buffers (from SAM 3D Body checkpoint)
            device: torch device
        """
        self.mhr_head = mhr_head
        self.device = device
        self.mhr_head.eval()

        # Cache zero-pose in continuous space
        with torch.no_grad():
            self._zero_pose_cont = compact_model_params_to_cont_body(
                torch.zeros(1, 133)
            ).squeeze(0).to(device)

    def _init_params(self, batch_size):
        """Initialize optimization parameters.

        Returns:
            dict of parameter tensors, all with requires_grad=True
        """
        params = {}

        # Global rotation as 6D (identity rotation)
        params["global_rot_6d"] = torch.tensor(
            [[1, 0, 0, 0, 1, 0]], dtype=torch.float32, device=self.device
        ).expand(batch_size, -1).clone().requires_grad_(True)

        # Body pose in continuous space — zero-pose is NOT all zeros
        params["body_pose_cont"] = self._zero_pose_cont.unsqueeze(0).expand(
            batch_size, -1
        ).clone().requires_grad_(True)

        # Shape params (45D)
        params["shape_params"] = torch.zeros(
            batch_size, 45, device=self.device, dtype=torch.float32
        ).requires_grad_(True)

        # Scale params (28D)
        params["scale_params"] = torch.zeros(
            batch_size, 28, device=self.device, dtype=torch.float32
        ).requires_grad_(True)

        # Hand params (108D = 54 left + 54 right)
        params["hand_params"] = torch.zeros(
            batch_size, 108, device=self.device, dtype=torch.float32
        ).requires_grad_(True)

        # Global translation (3D)
        params["global_trans"] = torch.zeros(
            batch_size, 3, device=self.device, dtype=torch.float32
        ).requires_grad_(True)

        return params

    def _mhr_forward_from_params(self, params):
        """Run MHR forward pass from optimization parameters.

        Returns:
            verts: (B, V, 3) mesh vertices
            j3d_70: (B, 70, 3) keypoints
            model_params: (B, 204) raw MHR model params
        """
        # Convert 6D rotation to euler
        global_rot_rotmat = rot6d_to_rotmat(params["global_rot_6d"])  # (B, 3, 3)
        global_rot_euler = roma.rotmat_to_euler("ZYX", global_rot_rotmat)  # (B, 3)

        # Convert continuous body pose to euler model params
        body_pose_euler = compact_cont_to_model_params_body(params["body_pose_cont"])
        # Zero out hand params in body pose (hands handled separately)
        body_pose_euler[:, mhr_param_hand_mask] = 0
        # Zero out jaw
        body_pose_euler[:, -3:] = 0

        # Run MHR forward
        B = params["global_trans"].shape[0]
        expr_params = torch.zeros(
            B, self.mhr_head.num_face_comps, device=self.device, dtype=torch.float32
        )
        output = self.mhr_head.mhr_forward(
            global_trans=params["global_trans"],
            global_rot=global_rot_euler,
            body_pose_params=body_pose_euler,
            hand_pose_params=params["hand_params"],
            scale_params=params["scale_params"],
            shape_params=params["shape_params"],
            expr_params=expr_params,
            return_keypoints=True,
            return_joint_coords=True,
            return_model_params=True,
        )

        verts, j3d_308, jcoords, model_params = output
        j3d_70 = j3d_308[:, :70]  # 308 -> 70 keypoints

        # NOTE: We do NOT apply the Y,Z flip here. The flip is applied
        # post-forward in the inference pipeline. Our fitting targets are
        # in MHR's pre-flip coordinate space.

        return verts, j3d_70, model_params

    def _find_target_pelvis(self, targets, mhr_indices):
        """Find pelvis position in target correspondences (hip midpoint)."""
        lhip_pos = rhip_pos = None
        for i, idx in enumerate(mhr_indices):
            if idx == 9:
                lhip_pos = i
            elif idx == 10:
                rhip_pos = i
        if lhip_pos is not None and rhip_pos is not None:
            return (targets[:, lhip_pos] + targets[:, rhip_pos]) / 2.0
        return targets.mean(dim=1)

    def _compute_loss(self, params, j3d_70, targets, mhr_indices, weights,
                      stage_idx, use_pelvis_relative=True,
                      reg_shape=0.001, reg_scale=0.001, reg_pose=0.0005,
                      reg_hand=0.0001, reg_trans=1.0):
        """Compute fitting loss.

        Args:
            params: dict of optimization parameters
            j3d_70: (B, 70, 3) MHR keypoint predictions
            targets: (B, N, 3) target positions (in MHR pre-flip coords)
            mhr_indices: list of N MHR70 indices
            weights: (N,) per-correspondence weights
            stage_idx: 0, 1, or 2 for current fitting stage
            use_pelvis_relative: if True, compute loss in pelvis-relative coords
            reg_trans: weight for absolute pelvis translation loss

        Returns:
            total_loss: scalar
            loss_dict: dict of individual loss terms
        """
        loss_dict = {}

        if use_pelvis_relative:
            mhr_rel, target_rel, _ = compute_pelvis_alignment(
                j3d_70, targets, mhr_indices
            )
        else:
            mhr_idx_tensor = torch.tensor(
                mhr_indices, device=j3d_70.device, dtype=torch.long
            )
            mhr_rel = j3d_70[:, mhr_idx_tensor]
            target_rel = targets

        # L1 weighted joint position loss
        diff = (mhr_rel - target_rel).abs()  # (B, N, 3)
        weighted_diff = diff * weights.unsqueeze(0).unsqueeze(-1)  # broadcast
        joint_loss = weighted_diff.mean()
        loss_dict["joint"] = joint_loss

        total_loss = joint_loss

        # Absolute pelvis translation loss — gives gradient signal for global_trans
        # (pelvis-relative loss cancels out global_trans entirely)
        mhr_pelvis = (j3d_70[:, 9] + j3d_70[:, 10]) / 2.0  # (B, 3)
        target_pelvis = self._find_target_pelvis(targets, mhr_indices)
        trans_loss = (mhr_pelvis - target_pelvis).abs().mean() * reg_trans
        loss_dict["trans"] = trans_loss
        total_loss = total_loss + trans_loss

        # Pose regularization (deviation from zero-pose in continuous space)
        pose_deviation = params["body_pose_cont"] - self._zero_pose_cont.unsqueeze(0)
        pose_reg = (pose_deviation ** 2).mean() * reg_pose
        loss_dict["pose_reg"] = pose_reg
        total_loss = total_loss + pose_reg

        # Shape/scale regularization (stage 2+)
        if stage_idx >= 1:
            shape_reg = (params["shape_params"] ** 2).mean() * reg_shape
            scale_reg = (params["scale_params"] ** 2).mean() * reg_scale
            loss_dict["shape_reg"] = shape_reg
            loss_dict["scale_reg"] = scale_reg
            total_loss = total_loss + shape_reg + scale_reg

        # Hand regularization (stage 3)
        if stage_idx >= 2:
            hand_reg = (params["hand_params"] ** 2).mean() * reg_hand
            loss_dict["hand_reg"] = hand_reg
            total_loss = total_loss + hand_reg

        loss_dict["total"] = total_loss
        return total_loss, loss_dict

    def _run_stage(self, params, targets, mhr_indices, weights, stage_cfg, stage_idx):
        """Run a single optimization stage.

        Args:
            params: dict of all parameters (only subset will be optimized)
            targets: (B, N, 3) target positions
            mhr_indices: list of MHR70 indices
            weights: (N,) per-correspondence weights
            stage_cfg: dict with 'params', 'num_iters', 'lr'
            stage_idx: 0, 1, or 2

        Returns:
            params: updated parameters dict
        """
        # Freeze all, then unfreeze stage params
        for key in params:
            params[key].requires_grad_(False)

        opt_params = []
        for key in stage_cfg["params"]:
            params[key].requires_grad_(True)
            opt_params.append(params[key])

        optimizer = torch.optim.Adam(opt_params, lr=stage_cfg["lr"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=stage_cfg["num_iters"], eta_min=stage_cfg["lr"] * 0.01
        )

        for it in range(stage_cfg["num_iters"]):
            optimizer.zero_grad()

            verts, j3d_70, model_params = self._mhr_forward_from_params(params)
            total_loss, loss_dict = self._compute_loss(
                params, j3d_70, targets, mhr_indices, weights, stage_idx
            )

            if torch.isnan(total_loss):
                logger.warning(f"NaN loss at {stage_cfg['name']} iter {it}, skipping")
                break

            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(opt_params, max_norm=1.0)

            optimizer.step()
            scheduler.step()

        return params

    def _compute_final_metrics(self, j3d_70, targets, mhr_indices):
        """Compute MPJPE metrics after fitting.

        Args:
            j3d_70: (B, 70, 3) MHR predictions
            targets: (B, N, 3) target positions
            mhr_indices: list of N MHR70 indices

        Returns:
            dict with 'mpjpe' (B,) in meters and 'max_joint_error' (B,)
        """
        mhr_idx_tensor = torch.tensor(
            mhr_indices, device=j3d_70.device, dtype=torch.long
        )
        mhr_at_corr = j3d_70[:, mhr_idx_tensor]  # (B, N, 3)

        # Pelvis-relative
        mhr_pelvis = (j3d_70[:, 9] + j3d_70[:, 10]) / 2.0
        mhr_rel = mhr_at_corr - mhr_pelvis.unsqueeze(1)

        # Find SMPLX pelvis from targets
        lhip_pos = rhip_pos = None
        for i, idx in enumerate(mhr_indices):
            if idx == 9:
                lhip_pos = i
            elif idx == 10:
                rhip_pos = i
        if lhip_pos is not None and rhip_pos is not None:
            target_pelvis = (targets[:, lhip_pos] + targets[:, rhip_pos]) / 2.0
        else:
            target_pelvis = targets.mean(dim=1)
        target_rel = targets - target_pelvis.unsqueeze(1)

        # Per-joint error
        per_joint_error = (mhr_rel - target_rel).norm(dim=-1)  # (B, N)
        mpjpe = per_joint_error.mean(dim=1)  # (B,)
        max_error = per_joint_error.max(dim=1).values  # (B,)

        return {"mpjpe": mpjpe, "max_joint_error": max_error}

    @torch.no_grad()
    def _get_final_outputs(self, params):
        """Run final forward pass and collect outputs."""
        verts, j3d_70, model_params = self._mhr_forward_from_params(params)
        return verts, j3d_70, model_params

    def fit_batch(self, target_positions, mhr_indices, weights):
        """Fit MHR parameters to target positions via 3-stage optimization.

        Args:
            target_positions: (B, N, 3) target 3D positions in MHR pre-flip coords
            mhr_indices: list of N MHR70 indices for correspondence
            weights: (N,) per-correspondence weights

        Returns:
            dict with:
                'model_params': (B, 204) raw MHR model params
                'shape_params': (B, 45) shape params
                'scale_params': (B, 28) scale params
                'keypoints_3d': (B, 70, 3) fitted keypoints
                'vertices': (B, V, 3) fitted vertices
                'metrics': dict with 'mpjpe', 'max_joint_error'
                'global_rot_6d': (B, 6)
                'body_pose_cont': (B, 260)
                'hand_params': (B, 108)
                'global_trans': (B, 3)
        """
        B = target_positions.shape[0]
        params = self._init_params(B)

        # Run 3 stages
        for stage_idx, stage_cfg in enumerate(STAGES):
            params = self._run_stage(
                params, target_positions, mhr_indices, weights,
                stage_cfg, stage_idx
            )

        # Get final outputs
        with torch.no_grad():
            verts, j3d_70, model_params = self._get_final_outputs(params)
            metrics = self._compute_final_metrics(j3d_70, target_positions, mhr_indices)

        return {
            "model_params": model_params.detach(),
            "shape_params": params["shape_params"].detach(),
            "scale_params": params["scale_params"].detach(),
            "keypoints_3d": j3d_70.detach(),
            "vertices": verts.detach(),
            "metrics": {k: v.detach() for k, v in metrics.items()},
            "global_rot_6d": params["global_rot_6d"].detach(),
            "body_pose_cont": params["body_pose_cont"].detach(),
            "hand_params": params["hand_params"].detach(),
            "global_trans": params["global_trans"].detach(),
        }
