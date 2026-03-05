#!/usr/bin/env python3
"""3D scatter plot comparing SMPL-X vs MHR fitted vertices."""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Config ──────────────────────────────────────────────────────────────
NUM_SAMPLES = 3

BEDLAM_NPZ = "/home/marco/Desktop/BD_data/training_labels/all_npz_12_training/20221010_3-10_500_batch01hand_zoom_suburb_d_6fps.npz"
IMAGE_ROOT = "/home/marco/Desktop/BD_data/training_images/20221010_3-10_500_batch01hand_zoom_suburb_d_6fps/png"
SMPLX_MODEL = "/home/marco/Desktop/SportsMotion/dev/sam-3d-body/temporal-dev/phase0"
SAM3D_CKPT = "/home/marco/Desktop/SportsMotion/dev/models/sam3d_body/model.ckpt"
MHR_MODEL = "/home/marco/Desktop/SportsMotion/dev/models/sam3d_body/mhr_model.pt"
OUTPUT_DIR = "/home/marco/Desktop/BD_data/test_output/vis"
DEVICE = "cuda:0"


def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading models...")
    from sam_3d_body.build_models import load_sam_3d_body
    model, cfg = load_sam_3d_body(SAM3D_CKPT, device=device, mhr_path=MHR_MODEL)
    mhr_head = model.head_pose; mhr_head.eval()
    del model; torch.cuda.empty_cache()

    from phase0.bedlam_loader import (
        BEDLAMDataset, collate_bedlam_batch, create_smplx_model, run_smplx_forward_batch,
    )
    from phase0.coord_utils import smplx_to_mhr_pre_flip
    from phase0.joint_mapping import extract_smplx_targets
    from phase0.mhr_fitter import MHRFitter
    from torch.utils.data import DataLoader

    smplx_model = create_smplx_model(SMPLX_MODEL, device=device)

    print("Loading data...")
    data = np.load(BEDLAM_NPZ, allow_pickle=True)
    subset_dir = os.path.join(OUTPUT_DIR, "subset_npz")
    os.makedirs(subset_dir, exist_ok=True)
    subset = {k: data[k][:NUM_SAMPLES] for k in data.keys()}
    np.savez(os.path.join(subset_dir, "scatter_subset.npz"), **subset)

    dataset = BEDLAMDataset(subset_dir, IMAGE_ROOT, num_betas=10)
    loader = DataLoader(dataset, batch_size=NUM_SAMPLES, shuffle=False,
                        num_workers=0, collate_fn=collate_bedlam_batch)
    batch = next(iter(loader))

    print("Running SMPL-X forward + MHR fitting...")
    pose_cam = batch["pose_cam"].to(device)
    shape = batch["shape"].to(device)
    trans_cam = batch["trans_cam"].to(device)

    smplx_out = run_smplx_forward_batch(smplx_model, pose_cam, shape, trans_cam)
    smplx_verts = smplx_out["vertices"]

    targets, mhr_idx, weights = extract_smplx_targets(
        smplx_out["joints"], smplx_verts,
        include_hands=True, include_feet=True, include_fingertips=True,
    )
    targets_flip = smplx_to_mhr_pre_flip(targets)

    fitter = MHRFitter(mhr_head, device=device)
    fitted = fitter.fit_batch(targets_flip, mhr_idx, weights)

    # MHR verts: flip to camera space to match SMPL-X
    mhr_verts = fitted["vertices"].clone()
    mhr_verts[..., [1, 2]] *= -1

    # MHR keypoints: flip to camera space for pelvis alignment
    mhr_kps = fitted["keypoints_3d"].clone()
    mhr_kps[..., [1, 2]] *= -1

    # SMPL-X joints for pelvis
    smplx_joints = smplx_out["joints"]

    mpjpe = fitted["metrics"]["mpjpe"]

    print("Plotting 3D scatter...")
    for i in range(NUM_SAMPLES):
        sv = smplx_verts[i].cpu().numpy()
        mv = mhr_verts[i].cpu().numpy()

        # Pelvis-align using actual joint positions (not vertex centroid)
        # SMPL-X pelvis = midpoint of L_hip (joint 1) and R_hip (joint 2)
        smplx_pelvis = ((smplx_joints[i, 1] + smplx_joints[i, 2]) / 2.0).cpu().numpy()
        # MHR pelvis = midpoint of L_hip (kp 9) and R_hip (kp 10)
        mhr_pelvis = ((mhr_kps[i, 9] + mhr_kps[i, 10]) / 2.0).cpu().numpy()

        sv = sv - smplx_pelvis
        mv = mv - mhr_pelvis

        # Subsample for visibility
        rng = np.random.default_rng(42)
        sv_sub = sv[rng.choice(len(sv), size=2000, replace=False)]
        mv_sub = mv[rng.choice(len(mv), size=min(2000, len(mv)), replace=False)]

        fig = plt.figure(figsize=(16, 12))

        # Two views: front-ish and side
        for pidx, (elev, azim, title) in enumerate([
            (0, 0, "Front view"),
            (0, 90, "Side view"),
        ]):
            ax = fig.add_subplot(1, 2, pidx + 1, projection='3d')
            ax.scatter(sv_sub[:, 0], sv_sub[:, 2], sv_sub[:, 1],
                       s=1, alpha=0.3, c='green', label='SMPL-X GT')
            ax.scatter(mv_sub[:, 0], mv_sub[:, 2], mv_sub[:, 1],
                       s=1, alpha=0.3, c='blue', label='MHR Fitted')
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_zlabel('Y')
            ax.set_title(title)
            ax.view_init(elev=elev, azim=azim)
            ax.legend(markerscale=10, loc='upper right')

            # Equal aspect ratio
            max_range = max(
                np.ptp(sv_sub[:, 0]), np.ptp(sv_sub[:, 1]), np.ptp(sv_sub[:, 2]),
                np.ptp(mv_sub[:, 0]), np.ptp(mv_sub[:, 1]), np.ptp(mv_sub[:, 2]),
            ) / 2
            mid_x = (sv_sub[:, 0].mean() + mv_sub[:, 0].mean()) / 2
            mid_y = (sv_sub[:, 1].mean() + mv_sub[:, 1].mean()) / 2
            mid_z = (sv_sub[:, 2].mean() + mv_sub[:, 2].mean()) / 2
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_z - max_range, mid_z + max_range)
            ax.set_zlim(mid_y - max_range, mid_y + max_range)

        fig.suptitle(
            f'Sample {i} — SMPL-X (green) vs MHR Fitted (blue) — MPJPE: {mpjpe[i].item()*1000:.1f}mm',
            fontsize=14, fontweight='bold',
        )
        plt.tight_layout()

        out_path = os.path.join(OUTPUT_DIR, f"scatter3d_{i:03d}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {out_path}")

    print(f"\nDone! Plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
