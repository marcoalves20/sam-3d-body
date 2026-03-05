#!/usr/bin/env python3
"""Master CLI script for BEDLAM -> MHR data conversion pipeline.

Usage:
    python phase0/run_conversion.py \
        --bedlam_npz_dir /data/bedlam/processed_npz \
        --bedlam_image_dir /data/bedlam/images \
        --smplx_model_path /data/body_models/smplx \
        --sam3d_checkpoint /models/sam3d_body/model.ckpt \
        --mhr_model_path /models/mhr_model.pt \
        --output_dir data/webdataset/bedlam \
        --batch_size 64 --num_workers 4 --gpu 0 \
        --quality_threshold 0.05 --validate --vis_samples 50

Multi-GPU: Split by NPZ files via --npz_start N --npz_end M.
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase0.bedlam_loader import (
    BEDLAMDataset,
    collate_bedlam_batch,
    create_smplx_model,
    run_smplx_forward_batch,
)
from phase0.coord_utils import smplx_to_mhr_pre_flip
from phase0.export_webdataset import assemble_annotation, write_all_shards
from phase0.joint_mapping import extract_smplx_targets
from phase0.mhr_fitter import MHRFitter
from phase0.validate import compute_validation_metrics, run_validation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_mhr_head(sam3d_checkpoint, mhr_model_path, device):
    """Load SAM 3D Body checkpoint and extract the MHRHead.

    Args:
        sam3d_checkpoint: path to SAM 3D Body model checkpoint
        mhr_model_path: path to MHR model (.pt)
        device: torch device

    Returns:
        mhr_head: MHRHead with populated buffers
    """
    from sam_3d_body.build_models import load_sam_3d_body

    model, cfg = load_sam_3d_body(
        checkpoint_path=sam3d_checkpoint,
        device=device,
        mhr_path=mhr_model_path,
    )

    mhr_head = model.head_pose
    mhr_head.eval()
    for param in mhr_head.parameters():
        param.requires_grad = False

    return mhr_head


def parse_args():
    parser = argparse.ArgumentParser(
        description="BEDLAM -> MHR Data Conversion Pipeline"
    )

    # Input paths
    parser.add_argument("--bedlam_npz_dir", type=str, required=True,
                        help="Directory containing BEDLAM NPZ files")
    parser.add_argument("--bedlam_image_dir", type=str, required=True,
                        help="Root directory for BEDLAM images")
    parser.add_argument("--smplx_model_path", type=str, required=True,
                        help="Path to SMPL-X model directory")
    parser.add_argument("--sam3d_checkpoint", type=str, required=True,
                        help="Path to SAM 3D Body checkpoint (.ckpt)")
    parser.add_argument("--mhr_model_path", type=str, required=True,
                        help="Path to MHR model (.pt)")

    # Output
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for WebDataset shards")

    # Processing
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_betas", type=int, default=10)
    parser.add_argument("--quality_threshold", type=float, default=0.05,
                        help="MPJPE threshold (meters) for mhr_valid")
    parser.add_argument("--samples_per_shard", type=int, default=1000)

    # Multi-GPU splitting
    parser.add_argument("--npz_start", type=int, default=None,
                        help="Start index for NPZ file range (inclusive)")
    parser.add_argument("--npz_end", type=int, default=None,
                        help="End index for NPZ file range (exclusive)")

    # Validation
    parser.add_argument("--validate", action="store_true",
                        help="Run validation after conversion")
    parser.add_argument("--vis_samples", type=int, default=0,
                        help="Number of samples to visualize (0=disabled)")

    # Feature flags
    parser.add_argument("--include_hands", action="store_true", default=True)
    parser.add_argument("--no_hands", action="store_true")
    parser.add_argument("--include_feet", action="store_true", default=True)
    parser.add_argument("--no_feet", action="store_true")
    parser.add_argument("--include_fingertips", action="store_true", default=True)
    parser.add_argument("--no_fingertips", action="store_true")

    args = parser.parse_args()

    if args.no_hands:
        args.include_hands = False
    if args.no_feet:
        args.include_feet = False
    if args.no_fingertips:
        args.include_fingertips = False

    return args


def main():
    args = parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Step 1: Load models
    logger.info("Loading MHR head from SAM 3D Body checkpoint...")
    mhr_head = load_mhr_head(args.sam3d_checkpoint, args.mhr_model_path, device)
    logger.info("MHR head loaded successfully")

    logger.info("Loading SMPL-X model...")
    smplx_model = create_smplx_model(
        args.smplx_model_path, gender="neutral",
        num_betas=args.num_betas, device=device,
    )
    logger.info("SMPL-X model loaded successfully")

    # Step 2: Create dataset and dataloader
    logger.info("Creating BEDLAM dataset...")
    npz_pattern = "*.npz"
    dataset = BEDLAMDataset(
        npz_dir=args.bedlam_npz_dir,
        image_root=args.bedlam_image_dir,
        num_betas=args.num_betas,
        npz_pattern=npz_pattern,
    )

    # Filter by NPZ index range if specified
    if args.npz_start is not None or args.npz_end is not None:
        import glob
        all_npz = sorted(glob.glob(os.path.join(args.bedlam_npz_dir, npz_pattern)))
        start = args.npz_start or 0
        end = args.npz_end or len(all_npz)
        selected_npz = set(os.path.basename(f) for f in all_npz[start:end])

        original_len = len(dataset.records)
        dataset.records = [
            r for r in dataset.records if r["npz_file"] in selected_npz
        ]
        logger.info(
            f"Filtered to NPZ files [{start}:{end}]: "
            f"{original_len} -> {len(dataset.records)} records"
        )

    logger.info(f"Dataset has {len(dataset)} person instances")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_bedlam_batch,
        drop_last=False,
        pin_memory=True,
    )

    # Step 3: Create fitter
    fitter = MHRFitter(mhr_head, device=device)

    # Step 4: Process batches
    all_annotations = []
    all_records = []
    all_fitted_results = []
    stats = {"total": 0, "valid": 0, "failed": 0}
    start_time = time.time()

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Fitting")):
        B = batch["pose_cam"].shape[0]

        try:
            # Move to device
            pose_cam = batch["pose_cam"].to(device)
            shape = batch["shape"].to(device)
            trans_cam = batch["trans_cam"].to(device)
            cam_int = batch["cam_int"].to(device)

            # Run SMPL-X forward pass
            smplx_output = run_smplx_forward_batch(smplx_model, pose_cam, shape, trans_cam)

            # Extract targets
            target_positions, mhr_indices, weights = extract_smplx_targets(
                smplx_output["joints"],
                smplx_output["vertices"],
                include_hands=args.include_hands,
                include_feet=args.include_feet,
                include_fingertips=args.include_fingertips,
            )

            # Convert to MHR pre-flip coordinate space
            target_positions = smplx_to_mhr_pre_flip(target_positions)

            # Fit MHR parameters
            fitted = fitter.fit_batch(target_positions, mhr_indices, weights)

            # Assemble annotations
            for i in range(B):
                rec = {
                    "imgname": batch["imgname"][i],
                    "person_idx": batch["person_idx"][i],
                    "center": batch["center"][i].numpy(),
                    "scale": batch["scale"][i].item(),
                }
                anno = assemble_annotation(
                    fitted, rec, cam_int[i], i,
                    quality_threshold=args.quality_threshold,
                )
                all_annotations.append(anno)
                all_records.append(rec)

                mpjpe = fitted["metrics"]["mpjpe"][i].item()
                stats["total"] += 1
                if mpjpe < args.quality_threshold:
                    stats["valid"] += 1

            all_fitted_results.append(fitted)

        except torch.cuda.OutOfMemoryError:
            logger.warning(f"OOM at batch {batch_idx}, skipping")
            torch.cuda.empty_cache()
            stats["failed"] += B
            continue
        except Exception as e:
            logger.error(f"Error at batch {batch_idx}: {e}")
            stats["failed"] += B
            continue

        # Log progress
        if (batch_idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = stats["total"] / elapsed
            logger.info(
                f"Batch {batch_idx + 1}: {stats['total']} samples processed, "
                f"{stats['valid']} valid ({stats['valid']/max(stats['total'],1)*100:.1f}%), "
                f"{rate:.1f} samples/sec"
            )

    elapsed = time.time() - start_time
    logger.info(
        f"Fitting complete: {stats['total']} samples in {elapsed:.1f}s "
        f"({stats['total']/elapsed:.1f} samples/sec)"
    )
    logger.info(
        f"Valid: {stats['valid']}/{stats['total']} "
        f"({stats['valid']/max(stats['total'],1)*100:.1f}%), "
        f"Failed: {stats['failed']}"
    )

    # Step 5: Compute validation metrics
    if all_fitted_results:
        metrics = compute_validation_metrics(all_fitted_results)
        logger.info(f"Validation metrics: {json.dumps(metrics, indent=2)}")

        # Save metrics
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")

    # Step 6: Write WebDataset shards
    logger.info("Writing WebDataset shards...")
    shard_paths = write_all_shards(
        args.output_dir,
        all_annotations,
        all_records,
        args.bedlam_image_dir,
        samples_per_shard=args.samples_per_shard,
    )
    logger.info(f"Written {len(shard_paths)} shards to {args.output_dir}")

    # Step 7: Optional post-conversion validation
    if args.validate:
        logger.info("Running post-conversion validation...")
        val_report = run_validation(
            args.output_dir,
            mhr_head,
            num_samples=min(100, stats["total"]),
            visualize=args.vis_samples > 0,
            output_dir=os.path.join(args.output_dir, "vis") if args.vis_samples > 0 else None,
        )
        logger.info(f"Validation report: {json.dumps(val_report, indent=2)}")

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
