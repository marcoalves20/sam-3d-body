#!/usr/bin/env python3
"""Quick end-to-end test of the phase0 pipeline on a small sample."""

import os
import sys
import time

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Config ──────────────────────────────────────────────────────────────
NUM_SAMPLES = 10
BATCH_SIZE = 10

BEDLAM_NPZ = "/home/marco/Desktop/BD_data/training_labels/all_npz_12_training/20221010_3-10_500_batch01hand_zoom_suburb_d_6fps.npz"
IMAGE_ROOT = "/home/marco/Desktop/BD_data/training_images/20221010_3-10_500_batch01hand_zoom_suburb_d_6fps/png"
SMPLX_MODEL = "/home/marco/Desktop/SportsMotion/dev/sam-3d-body/temporal-dev/phase0"
SAM3D_CKPT = "/home/marco/Desktop/SportsMotion/dev/models/sam3d_body/model.ckpt"
MHR_MODEL = "/home/marco/Desktop/SportsMotion/dev/models/sam3d_body/mhr_model.pt"
OUTPUT_DIR = "/home/marco/Desktop/BD_data/test_output"
DEVICE = "cuda:0"


def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Step 1: Create a tiny NPZ subset ────────────────────────────────
    print("\n[1/6] Creating tiny NPZ subset...")
    data = np.load(BEDLAM_NPZ, allow_pickle=True)
    print(f"  Full NPZ has {len(data['imgname'])} records")

    # Pick first NUM_SAMPLES with unique images so we can verify visually
    subset_dir = os.path.join(OUTPUT_DIR, "subset_npz")
    os.makedirs(subset_dir, exist_ok=True)

    subset = {}
    for key in data.keys():
        subset[key] = data[key][:NUM_SAMPLES]

    subset_path = os.path.join(subset_dir, "test_subset.npz")
    np.savez(subset_path, **subset)
    print(f"  Saved {NUM_SAMPLES} records to {subset_path}")

    # Verify images exist
    missing = 0
    for imgname in subset["imgname"]:
        img_path = os.path.join(IMAGE_ROOT, str(imgname))
        if not os.path.exists(img_path):
            print(f"  WARNING: missing image {img_path}")
            missing += 1
    print(f"  Image check: {NUM_SAMPLES - missing}/{NUM_SAMPLES} found")

    # ── Step 2: Load models ─────────────────────────────────────────────
    print("\n[2/6] Loading models...")
    t0 = time.time()

    from phase0.bedlam_loader import (
        BEDLAMDataset,
        collate_bedlam_batch,
        create_smplx_model,
        run_smplx_forward_batch,
    )
    from phase0.coord_utils import smplx_to_mhr_pre_flip
    from phase0.joint_mapping import extract_smplx_targets
    from phase0.mhr_fitter import MHRFitter

    # Load MHR head
    from sam_3d_body.build_models import load_sam_3d_body

    print("  Loading SAM 3D Body checkpoint...")
    model, cfg = load_sam_3d_body(
        checkpoint_path=SAM3D_CKPT,
        device=device,
        mhr_path=MHR_MODEL,
    )
    mhr_head = model.head_pose
    mhr_head.eval()
    for p in mhr_head.parameters():
        p.requires_grad = False
    del model  # free memory
    torch.cuda.empty_cache()

    print("  Loading SMPL-X model...")
    smplx_model = create_smplx_model(SMPLX_MODEL, device=device)

    print(f"  Models loaded in {time.time() - t0:.1f}s")

    # ── Step 3: Load data ───────────────────────────────────────────────
    print("\n[3/6] Loading dataset...")
    dataset = BEDLAMDataset(
        npz_dir=subset_dir,
        image_root=IMAGE_ROOT,
        num_betas=10,
    )
    print(f"  Dataset: {len(dataset)} records")

    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_bedlam_batch,
    )

    # ── Step 4: SMPL-X forward ──────────────────────────────────────────
    print("\n[4/6] Running SMPL-X forward pass...")
    batch = next(iter(loader))
    pose_cam = batch["pose_cam"].to(device)
    shape = batch["shape"].to(device)
    trans_cam = batch["trans_cam"].to(device)
    cam_int = batch["cam_int"].to(device)

    smplx_out = run_smplx_forward_batch(smplx_model, pose_cam, shape, trans_cam)
    print(f"  joints: {smplx_out['joints'].shape}")
    print(f"  vertices: {smplx_out['vertices'].shape}")

    # ── Step 5: Extract targets & fit ───────────────────────────────────
    print("\n[5/6] Extracting targets and fitting MHR parameters...")
    target_positions, mhr_indices, weights = extract_smplx_targets(
        smplx_out["joints"],
        smplx_out["vertices"],
        include_hands=True,
        include_feet=True,
        include_fingertips=True,
    )
    print(f"  Target positions: {target_positions.shape}")
    print(f"  MHR indices: {len(mhr_indices)} correspondences, weights: {len(weights)}")

    # Convert to MHR pre-flip coords
    target_positions = smplx_to_mhr_pre_flip(target_positions)

    # Fit
    t0 = time.time()
    fitter = MHRFitter(mhr_head, device=device)
    fitted = fitter.fit_batch(target_positions, mhr_indices, weights)
    fit_time = time.time() - t0

    print(f"  Fitting done in {fit_time:.1f}s ({BATCH_SIZE / fit_time:.1f} samples/sec)")
    print(f"  Output model_params: {fitted['model_params'].shape}")
    print(f"  Output keypoints_3d: {fitted['keypoints_3d'].shape}")

    mpjpe = fitted["metrics"]["mpjpe"]
    max_err = fitted["metrics"]["max_joint_error"]
    print(f"\n  Per-sample MPJPE (mm):")
    for i in range(len(mpjpe)):
        print(f"    sample {i}: MPJPE={mpjpe[i].item()*1000:.1f}mm, max_err={max_err[i].item()*1000:.1f}mm")
    print(f"  Mean MPJPE: {mpjpe.mean().item()*1000:.1f}mm")

    # ── Step 6: Assemble annotations ────────────────────────────────────
    print("\n[6/6] Assembling annotations...")
    from phase0.export_webdataset import assemble_annotation

    annotations = []
    for i in range(BATCH_SIZE):
        rec = {
            "imgname": batch["imgname"][i],
            "person_idx": batch["person_idx"][i],
            "center": batch["center"][i].numpy(),
            "scale": batch["scale"][i].item(),
        }
        anno = assemble_annotation(fitted, rec, cam_int[i], i, quality_threshold=0.05)
        annotations.append(anno)

    valid = sum(1 for a in annotations if a["mhr_valid"])
    print(f"  Annotations: {len(annotations)} total, {valid} valid (mhr_valid=True)")

    # Save a sample annotation for inspection
    sample_anno_path = os.path.join(OUTPUT_DIR, "sample_annotation.npz")
    np.savez(sample_anno_path, **{k: v for k, v in annotations[0].items()
                                   if isinstance(v, (np.ndarray, int, float, str, bool))})
    print(f"  Sample annotation saved to {sample_anno_path}")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST PASSED — Pipeline ran end-to-end successfully!")
    print(f"  Samples: {BATCH_SIZE}")
    print(f"  Mean MPJPE: {mpjpe.mean().item()*1000:.1f}mm")
    print(f"  Valid: {valid}/{len(annotations)}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
