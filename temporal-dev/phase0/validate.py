import logging
import os
import pickle

import cv2
import numpy as np
import torch
import webdataset as wds

logger = logging.getLogger(__name__)


def compute_validation_metrics(fitted_results_list):
    """Compute aggregate validation metrics from a list of fitted results.

    Args:
        fitted_results_list: list of dicts from MHRFitter.fit_batch(),
            each containing 'metrics' with 'mpjpe' (B,) in meters

    Returns:
        dict with aggregate metrics
    """
    all_mpjpe = []
    for result in fitted_results_list:
        mpjpe = result["metrics"]["mpjpe"]
        if isinstance(mpjpe, torch.Tensor):
            mpjpe = mpjpe.cpu().numpy()
        all_mpjpe.append(mpjpe)

    all_mpjpe = np.concatenate(all_mpjpe)
    all_mpjpe_mm = all_mpjpe * 1000  # convert to mm

    metrics = {
        "mean_mpjpe_mm": float(np.mean(all_mpjpe_mm)),
        "median_mpjpe_mm": float(np.median(all_mpjpe_mm)),
        "mpjpe_95th_mm": float(np.percentile(all_mpjpe_mm, 95)),
        "success_rate_10mm": float(np.mean(all_mpjpe_mm < 10)),
        "success_rate_20mm": float(np.mean(all_mpjpe_mm < 20)),
        "success_rate_50mm": float(np.mean(all_mpjpe_mm < 50)),
        "total_samples": len(all_mpjpe_mm),
    }
    return metrics


def round_trip_test(mhr_head, model_params, shape_params, expected_j3d, tolerance=1e-4):
    """Verify that stored model_params reproduce the expected keypoints.

    Loads model_params, re-runs mhr_forward, and checks output matches.

    Args:
        mhr_head: MHRHead instance
        model_params: (B, 204) stored model params
        shape_params: (B, 45) stored shape params
        expected_j3d: (B, 70, 3) expected keypoints
        tolerance: max allowed difference

    Returns:
        bool: True if round-trip matches within tolerance
    """
    with torch.no_grad():
        device = model_params.device

        # Decompose model_params (204D)
        # model_params layout: [global_trans*10 (3), global_rot (3), body_pose (127-6=124?? no...]
        # From mhr_head.mhr_forward():
        #   full_pose_params = cat([global_trans * 10, global_rot, body_pose], dim=1)  # B x 136
        #   model_params = cat([full_pose_params, scales], dim=1)  # B x 204
        # So: model_params[:, :3] = global_trans * 10
        #     model_params[:, 3:136] = global_rot (3) + body_pose (130)
        #     model_params[:, 136:204] = scales (68)

        global_trans = model_params[:, :3] / 10.0
        global_rot = model_params[:, 3:6]
        body_pose = model_params[:, 6:136]
        scales = model_params[:, 136:204]

        # Reconstruct via mhr model call
        full_pose_params = model_params[:, :136]
        curr_skinned_verts, curr_skel_state = mhr_head.mhr(
            shape_params, model_params, None
        )
        curr_joint_coords, curr_joint_quats, _ = torch.split(
            curr_skel_state, [3, 4, 1], dim=2
        )
        curr_skinned_verts = curr_skinned_verts / 100
        curr_joint_coords = curr_joint_coords / 100

        # Get 70 keypoints via keypoint_mapping
        model_vert_joints = torch.cat(
            [curr_skinned_verts, curr_joint_coords], dim=1
        )
        model_keypoints = (
            (
                mhr_head.keypoint_mapping
                @ model_vert_joints.permute(1, 0, 2).flatten(1, 2)
            )
            .reshape(-1, model_vert_joints.shape[0], 3)
            .permute(1, 0, 2)
        )
        j3d_70 = model_keypoints[:, :70]

        # Compare (without Y,Z flip since expected_j3d is in pre-flip space)
        diff = (j3d_70 - expected_j3d).abs().max().item()
        passed = diff < tolerance

        if not passed:
            logger.warning(
                f"Round-trip test failed: max diff = {diff:.6f} (tolerance = {tolerance})"
            )
        else:
            logger.info(f"Round-trip test passed: max diff = {diff:.6f}")

        return passed


def visualize_fitting_result(image, fitted_vertices, cam_int, global_trans, faces,
                             output_path=None):
    """Render fitted MHR mesh overlaid on image.

    Args:
        image: (H, W, 3) BGR image as numpy array
        fitted_vertices: (V, 3) mesh vertices in camera coords
        cam_int: (3, 3) camera intrinsics
        global_trans: (3,) camera translation
        faces: (F, 3) mesh face indices
        output_path: if provided, save result here

    Returns:
        overlay: (H, W, 3) overlay image as float32 [0, 1]
    """
    try:
        from sam_3d_body.visualization.renderer import Renderer
    except ImportError:
        logger.warning("Cannot import Renderer, skipping visualization")
        return None

    focal_length = float(cam_int[0, 0])
    renderer = Renderer(focal_length=focal_length, faces=faces)

    verts_np = fitted_vertices.cpu().numpy() if isinstance(fitted_vertices, torch.Tensor) else fitted_vertices
    cam_t_np = global_trans.cpu().numpy() if isinstance(global_trans, torch.Tensor) else global_trans

    # Apply Y,Z flip for camera-space rendering
    verts_render = verts_np.copy()
    verts_render[:, [1, 2]] *= -1

    camera_center = [float(cam_int[0, 2]), float(cam_int[1, 2])]

    overlay = renderer(
        vertices=verts_render,
        cam_t=cam_t_np,
        image=image.astype(np.float32),
        camera_center=camera_center,
    )

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, (overlay * 255).astype(np.uint8))

    return overlay


def run_validation(webdataset_dir, mhr_head, num_samples=100, visualize=False,
                   output_dir=None):
    """Run validation on produced WebDataset shards.

    Args:
        webdataset_dir: directory containing tar shards
        mhr_head: MHRHead instance
        num_samples: number of samples to validate
        visualize: whether to render overlays
        output_dir: directory for visualization output

    Returns:
        validation report dict
    """
    import glob

    tar_files = sorted(glob.glob(os.path.join(webdataset_dir, "*.tar")))
    if not tar_files:
        logger.warning(f"No tar files found in {webdataset_dir}")
        return {"error": "no tar files found"}

    url = tar_files[0] if len(tar_files) == 1 else f"{webdataset_dir}/{{000000..{len(tar_files)-1:06d}}}.tar"

    dataset = wds.WebDataset(tar_files).decode("pil").to_tuple(
        "jpg", "annotation.pyd", "metadata.json", "__key__"
    )

    all_mpjpe = []
    vis_count = 0

    for sample_idx, sample in enumerate(dataset):
        if sample_idx >= num_samples:
            break

        jpg_data, annotations, metadata, key = sample

        for anno in annotations:
            if not anno.get("mhr_valid", False):
                continue

            model_params = torch.tensor(
                anno["mhr_params"]["model_params"], dtype=torch.float32
            ).unsqueeze(0).to(next(mhr_head.parameters()).device)
            shape_params = torch.tensor(
                anno["mhr_params"]["shape_params"], dtype=torch.float32
            ).unsqueeze(0).to(model_params.device)

            loss = anno["metadata"].get("loss", 0.0)
            all_mpjpe.append(loss)

    report = {}
    if all_mpjpe:
        all_mpjpe_mm = np.array(all_mpjpe) * 1000
        report["mean_mpjpe_mm"] = float(np.mean(all_mpjpe_mm))
        report["median_mpjpe_mm"] = float(np.median(all_mpjpe_mm))
        report["num_valid"] = len(all_mpjpe)
        report["success_rate_50mm"] = float(np.mean(all_mpjpe_mm < 50))
    else:
        report["error"] = "no valid annotations found"

    return report
