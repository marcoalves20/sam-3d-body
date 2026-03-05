import torch


def smplx_to_mhr_pre_flip(positions):
    """Convert positions from SMPL-X coord system (OpenCV: Y-down, Z-forward)
    to MHR's pre-flip space (Y-up, Z-backward).

    The Y,Z flip `[..., [1,2]] *= -1` happens AFTER mhr_forward() in mhr_head.py:339-343.
    So fitting targets must be in MHR's internal space (before that flip).
    This means we need to apply the inverse flip to SMPL-X positions.

    Args:
        positions: (..., 3) tensor in SMPL-X/OpenCV coordinates

    Returns:
        (..., 3) tensor in MHR pre-flip coordinates
    """
    out = positions.clone()
    out[..., 1] *= -1  # flip Y
    out[..., 2] *= -1  # flip Z
    return out


def compute_pelvis_alignment(mhr_j3d_70, smplx_targets, mhr_indices):
    """Compute pelvis-relative coordinates for both MHR and SMPLX targets.

    MHR pelvis is approximated as the mean of left_hip (idx 9) and right_hip (idx 10).
    SMPLX pelvis is joint 0 but we use the same hip-midpoint approach for consistency.

    Args:
        mhr_j3d_70: (B, 70, 3) MHR 70-keypoint predictions
        smplx_targets: (B, N, 3) SMPLX target positions at correspondence points
        mhr_indices: list of N MHR70 indices corresponding to smplx_targets

    Returns:
        mhr_pelvis_rel: (B, N, 3) MHR positions relative to pelvis
        smplx_pelvis_rel: (B, N, 3) SMPLX positions relative to pelvis
        mhr_pelvis: (B, 3) MHR pelvis position
    """
    # MHR pelvis = midpoint of left hip (9) and right hip (10)
    mhr_pelvis = (mhr_j3d_70[:, 9] + mhr_j3d_70[:, 10]) / 2.0  # (B, 3)

    # Extract MHR positions at correspondence indices
    mhr_idx_tensor = torch.tensor(mhr_indices, device=mhr_j3d_70.device, dtype=torch.long)
    mhr_at_corr = mhr_j3d_70[:, mhr_idx_tensor]  # (B, N, 3)

    # Make pelvis-relative
    mhr_pelvis_rel = mhr_at_corr - mhr_pelvis.unsqueeze(1)

    # For SMPLX targets, find the hip midpoint within the correspondences
    # The targets already have the SMPLX positions at the matching joints.
    # We need to find the SMPLX hip midpoint. The L_hip and R_hip should be
    # in the targets if body joints are included (MHR indices 9 and 10).
    lhip_pos = None
    rhip_pos = None
    for i, mhr_idx in enumerate(mhr_indices):
        if mhr_idx == 9:
            lhip_pos = i
        elif mhr_idx == 10:
            rhip_pos = i

    if lhip_pos is not None and rhip_pos is not None:
        smplx_pelvis = (smplx_targets[:, lhip_pos] + smplx_targets[:, rhip_pos]) / 2.0
    else:
        # Fallback: use mean of all target points
        smplx_pelvis = smplx_targets.mean(dim=1)

    smplx_pelvis_rel = smplx_targets - smplx_pelvis.unsqueeze(1)

    return mhr_pelvis_rel, smplx_pelvis_rel, mhr_pelvis


def project_3d_to_2d(joints_3d, cam_int, cam_t=None):
    """Project 3D joints to 2D pixel coordinates via perspective projection.

    Args:
        joints_3d: (B, N, 3) 3D joint positions
        cam_int: (B, 3, 3) camera intrinsic matrices
        cam_t: (B, 3) optional camera translation to apply first

    Returns:
        joints_2d: (B, N, 2) projected 2D pixel coordinates
    """
    pts = joints_3d
    if cam_t is not None:
        pts = pts + cam_t.unsqueeze(1)

    # Perspective divide
    z = pts[..., 2:3].clamp(min=1e-6)
    pts_norm = pts[..., :2] / z  # (B, N, 2)

    # Apply intrinsics: u = fx * X/Z + cx, v = fy * Y/Z + cy
    fx = cam_int[:, 0, 0].unsqueeze(1)  # (B, 1)
    fy = cam_int[:, 1, 1].unsqueeze(1)
    cx = cam_int[:, 0, 2].unsqueeze(1)
    cy = cam_int[:, 1, 2].unsqueeze(1)

    u = fx * pts_norm[..., 0] + cx  # (B, N)
    v = fy * pts_norm[..., 1] + cy

    return torch.stack([u, v], dim=-1)  # (B, N, 2)
