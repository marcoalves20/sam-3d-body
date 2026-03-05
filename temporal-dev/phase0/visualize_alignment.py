#!/usr/bin/env python3
"""Visualize SMPL-X vs fitted MHR meshes side by side on BEDLAM images."""

import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Config ──────────────────────────────────────────────────────────────
NUM_SAMPLES = 5

BEDLAM_NPZ = "/home/marco/Desktop/BD_data/training_labels/all_npz_12_training/20221010_3-10_500_batch01hand_zoom_suburb_d_6fps.npz"
IMAGE_ROOT = "/home/marco/Desktop/BD_data/training_images/20221010_3-10_500_batch01hand_zoom_suburb_d_6fps/png"
SMPLX_MODEL = "/home/marco/Desktop/SportsMotion/dev/sam-3d-body/temporal-dev/phase0"
SAM3D_CKPT = "/home/marco/Desktop/SportsMotion/dev/models/sam3d_body/model.ckpt"
MHR_MODEL = "/home/marco/Desktop/SportsMotion/dev/models/sam3d_body/mhr_model.pt"
OUTPUT_DIR = "/home/marco/Desktop/BD_data/test_output/vis"
DEVICE = "cuda:0"


def render_mesh_on_image(renderer, verts, cam_t, image, color, focal_length):
    """Render a mesh overlay on an image."""
    rendered = renderer(
        vertices=verts,
        cam_t=cam_t,
        image=image,
        mesh_base_color=color,
    )
    return (rendered * 255).astype(np.uint8)


def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load models ─────────────────────────────────────────────────────
    print("Loading models...")
    from sam_3d_body.build_models import load_sam_3d_body

    model, cfg = load_sam_3d_body(SAM3D_CKPT, device=device, mhr_path=MHR_MODEL)
    mhr_head = model.head_pose
    mhr_head.eval()
    mhr_faces = mhr_head.faces.cpu().numpy()
    del model
    torch.cuda.empty_cache()

    from phase0.bedlam_loader import (
        BEDLAMDataset,
        collate_bedlam_batch,
        create_smplx_model,
        run_smplx_forward_batch,
    )
    from phase0.coord_utils import smplx_to_mhr_pre_flip
    from phase0.joint_mapping import extract_smplx_targets
    from phase0.mhr_fitter import MHRFitter

    smplx_model = create_smplx_model(SMPLX_MODEL, device=device)
    smplx_faces = smplx_model.faces_tensor.cpu().numpy()

    # ── Load dataset ────────────────────────────────────────────────────
    print("Loading data...")
    # Create a small subset
    data = np.load(BEDLAM_NPZ, allow_pickle=True)
    subset_dir = os.path.join(OUTPUT_DIR, "subset_npz")
    os.makedirs(subset_dir, exist_ok=True)
    subset = {k: data[k][:NUM_SAMPLES] for k in data.keys()}
    np.savez(os.path.join(subset_dir, "vis_subset.npz"), **subset)

    dataset = BEDLAMDataset(subset_dir, IMAGE_ROOT, num_betas=10)
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=NUM_SAMPLES, shuffle=False,
                        num_workers=0, collate_fn=collate_bedlam_batch)
    batch = next(iter(loader))

    # ── SMPL-X forward ──────────────────────────────────────────────────
    print("Running SMPL-X forward...")
    pose_cam = batch["pose_cam"].to(device)
    shape = batch["shape"].to(device)
    trans_cam = batch["trans_cam"].to(device)
    cam_int = batch["cam_int"].to(device)

    smplx_out = run_smplx_forward_batch(smplx_model, pose_cam, shape, trans_cam)
    smplx_joints = smplx_out["joints"]
    smplx_verts = smplx_out["vertices"]

    # ── MHR fitting ─────────────────────────────────────────────────────
    print("Fitting MHR parameters...")
    targets, mhr_idx, weights = extract_smplx_targets(
        smplx_joints, smplx_verts, include_hands=True,
        include_feet=True, include_fingertips=True,
    )
    targets_flip = smplx_to_mhr_pre_flip(targets)

    fitter = MHRFitter(mhr_head, device=device)
    fitted = fitter.fit_batch(targets_flip, mhr_idx, weights)

    mhr_verts = fitted["vertices"]      # (B, V, 3) pre-flip
    mhr_j3d = fitted["keypoints_3d"]    # (B, 70, 3) pre-flip

    # Apply Y,Z flip to bring MHR into camera/OpenCV space (same as SMPL-X)
    mhr_verts_cam = mhr_verts.clone()
    mhr_verts_cam[..., [1, 2]] *= -1
    mhr_j3d_cam = mhr_j3d.clone()
    mhr_j3d_cam[..., [1, 2]] *= -1

    mpjpe = fitted["metrics"]["mpjpe"]

    # ── Render ──────────────────────────────────────────────────────────
    print("Rendering visualizations...")
    from sam_3d_body.visualization.renderer import Renderer

    for i in range(NUM_SAMPLES):
        imgname = batch["imgname"][i]
        img_path = os.path.join(IMAGE_ROOT, imgname)
        if not os.path.exists(img_path):
            print(f"  Skipping {imgname} (image not found)")
            continue

        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Camera params
        fx = cam_int[i, 0, 0].item()
        fy = cam_int[i, 1, 1].item()
        focal = (fx + fy) / 2.0
        cx = cam_int[i, 0, 2].item()
        cy = cam_int[i, 1, 2].item()

        # SMPL-X vertices are already in camera space (OpenCV)
        sv = smplx_verts[i].cpu().numpy()
        # MHR vertices flipped to camera space
        mv = mhr_verts_cam[i].cpu().numpy()

        # Camera translation = zeros (vertices already include translation)
        cam_t = np.zeros(3)

        # --- SMPL-X render (green) ---
        renderer_smplx = Renderer(focal_length=focal, faces=smplx_faces)
        img_smplx = render_mesh_on_image(
            renderer_smplx, sv, cam_t, img_rgb,
            color=(0.4, 0.8, 0.4),  # green
            focal_length=focal,
        )

        # --- MHR render (blue) ---
        renderer_mhr = Renderer(focal_length=focal, faces=mhr_faces)
        img_mhr = render_mesh_on_image(
            renderer_mhr, mv, cam_t, img_rgb,
            color=(0.4, 0.6, 0.9),  # blue
            focal_length=focal,
        )

        # Compose: [original | SMPL-X (green) | MHR (blue)]
        h, w = img_rgb.shape[:2]
        canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)
        canvas[:, :w] = img_rgb
        canvas[:, w:2*w] = img_smplx
        canvas[:, 2*w:] = img_mhr

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, "Original", (10, 30), font, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, "SMPL-X GT", (w + 10, 30), font, 0.8, (100, 255, 100), 2)
        label = f"MHR Fitted ({mpjpe[i].item()*1000:.1f}mm)"
        cv2.putText(canvas, label, (2*w + 10, 30), font, 0.8, (100, 150, 255), 2)

        out_path = os.path.join(OUTPUT_DIR, f"align_{i:03d}.png")
        cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        print(f"  Saved {out_path} (MPJPE={mpjpe[i].item()*1000:.1f}mm)")

    print(f"\nDone! {NUM_SAMPLES} visualizations saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
