import io
import json
import os
import pickle
from collections import defaultdict

import cv2
import numpy as np
import webdataset as wds

from .coord_utils import project_3d_to_2d


def assemble_annotation(fitted_result, bedlam_record, cam_int, sample_idx,
                        quality_threshold=0.05):
    """Assemble a single annotation dict from fitted results.

    Args:
        fitted_result: dict from MHRFitter.fit_batch() for a single sample
        bedlam_record: dict with bedlam metadata (imgname, center, scale, etc.)
        cam_int: (3,3) camera intrinsic matrix
        sample_idx: index within the batch
        quality_threshold: MPJPE threshold in meters for mhr_valid

    Returns:
        annotation dict or None if fitting failed
    """
    mpjpe = fitted_result["metrics"]["mpjpe"][sample_idx].item()

    # Get 3D keypoints and flip to camera coords for storage
    j3d = fitted_result["keypoints_3d"][sample_idx].clone()  # (70, 3)
    # Apply the Y,Z flip that happens post-forward to get camera-space keypoints
    j3d[..., [1, 2]] *= -1

    # Project to 2D
    cam_int_tensor = cam_int.unsqueeze(0)  # (1, 3, 3)
    j3d_for_proj = j3d.unsqueeze(0)  # (1, 70, 3)
    j2d = project_3d_to_2d(j3d_for_proj, cam_int_tensor).squeeze(0)  # (70, 2)

    # Build keypoints arrays
    # keypoints_2d: (70, 3) with [x, y, confidence]
    keypoints_2d = np.zeros((70, 3), dtype=np.float32)
    keypoints_2d[:, :2] = j2d.cpu().numpy()
    keypoints_2d[:, 2] = 1.0  # all visible

    # keypoints_3d: (70, 3)
    keypoints_3d = j3d.cpu().numpy().astype(np.float32)

    # Model params
    model_params = fitted_result["model_params"][sample_idx].cpu().numpy().astype(np.float32)
    shape_params = fitted_result["shape_params"][sample_idx].cpu().numpy().astype(np.float32)

    # Center and scale from BEDLAM record
    center = bedlam_record["center"]
    if isinstance(center, np.ndarray):
        center = center.tolist()
    scale = bedlam_record["scale"]
    if isinstance(scale, (np.ndarray, np.floating)):
        scale = float(scale)

    anno = {
        "person_id": bedlam_record.get("person_idx", sample_idx),
        "keypoints_2d": keypoints_2d,
        "keypoints_3d": keypoints_3d,
        "mhr_params": {
            "model_params": model_params,
            "shape_params": shape_params,
        },
        "mhr_valid": mpjpe < quality_threshold,
        "bbox": _compute_bbox_from_center_scale(center, scale),
        "bbox_format": "xywh",
        "bbox_score": 1.0,
        "center": np.array(center, dtype=np.float32),
        "scale": np.array(scale, dtype=np.float32),
        "metadata": {
            "cam_int": cam_int.cpu().numpy().astype(np.float32),
            "loss": mpjpe,
            "imgname": bedlam_record["imgname"],
        },
    }
    return anno


def _compute_bbox_from_center_scale(center, scale, pixel_std=200.0):
    """Compute xywh bbox from center and scale (BEDLAM convention)."""
    if isinstance(center, (list, tuple)):
        cx, cy = center[0], center[1]
    else:
        cx, cy = float(center[0]), float(center[1])
    w = h = float(scale) * pixel_std
    x = cx - w / 2
    y = cy - h / 2
    return np.array([x, y, w, h], dtype=np.float32)


def group_annotations_by_image(annotations, records):
    """Group annotations by their source image path.

    Args:
        annotations: list of annotation dicts
        records: list of bedlam record dicts (parallel to annotations)

    Returns:
        dict mapping image_path -> list of annotations
    """
    groups = defaultdict(list)
    for anno, rec in zip(annotations, records):
        if anno is not None:
            groups[rec["imgname"]].append(anno)
    return dict(groups)


def write_webdataset_shard(shard_path, image_groups, image_root, max_per_shard=1000):
    """Write a single WebDataset tar shard.

    Args:
        shard_path: output path for the tar file
        image_groups: dict mapping image_path -> list of annotations
        image_root: root directory for BEDLAM images
        max_per_shard: max samples per shard (each sample = one image with all its people)
    """
    os.makedirs(os.path.dirname(shard_path), exist_ok=True)
    count = 0

    with wds.TarWriter(shard_path) as sink:
        for img_path, annos in image_groups.items():
            if count >= max_per_shard:
                break

            full_img_path = os.path.join(image_root, img_path)
            if not os.path.exists(full_img_path):
                continue

            img = cv2.imread(full_img_path)
            if img is None:
                continue

            # Create key from image path
            key = img_path.replace("/", "-").replace("\\", "-")
            if key.endswith(".png") or key.endswith(".jpg"):
                key = key[:-4]
            key = f"bedlam-{key}"

            sample = {
                "__key__": key,
                "jpg": cv2.imencode(".jpg", img)[1].tobytes(),
                "metadata.json": {"width": img.shape[1], "height": img.shape[0]},
                "annotation.pyd": annos,
            }

            sink.write(sample)
            count += 1

    return count


def write_all_shards(output_dir, annotations, records, image_root, samples_per_shard=1000):
    """Write all WebDataset shards from annotations.

    Args:
        output_dir: directory to write tar files
        annotations: list of annotation dicts
        records: list of bedlam record dicts
        image_root: root directory for images
        samples_per_shard: max images per tar shard

    Returns:
        list of shard paths written
    """
    os.makedirs(output_dir, exist_ok=True)

    # Group by image
    image_groups = group_annotations_by_image(annotations, records)
    img_paths = list(image_groups.keys())

    shard_paths = []
    shard_idx = 0

    for start in range(0, len(img_paths), samples_per_shard):
        end = min(start + samples_per_shard, len(img_paths))
        shard_groups = {p: image_groups[p] for p in img_paths[start:end]}

        shard_path = os.path.join(output_dir, f"{shard_idx:06d}.tar")
        count = write_webdataset_shard(shard_path, shard_groups, image_root, samples_per_shard)
        shard_paths.append(shard_path)
        shard_idx += 1

    return shard_paths
