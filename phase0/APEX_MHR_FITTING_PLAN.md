# Plan: Apex Multi-Camera → MHR Fitting Pipeline

## Context

We need to generate SAM 3D Body training data from Marco's multi-camera mocap system (Apex). Apex produces high-fidelity 3D joint positions + local rotation matrices from a 27-joint kinematic skeleton (no hands). We'll fit the MHR body model to these results and export WebDataset shards — one training sample per camera view per frame.

**Approach: Post-solve fitting** — Apex mocap is already high-fidelity, so we fit MHR to the final 3D skeleton rather than integrating into the multi-camera solve. This is simpler, decoupled, and reuses the proven phase0 fitter.

## Key Decisions

- **Rotations**: Initialize MHR global rotation from Apex pelvis rotation (high-impact, low-effort). Full rotation loss on internal joints deferred as optional future enhancement.
- **Hands**: Apex has no finger data → skip hand fitting (2-stage instead of 3-stage), hands stay at MHR zero-pose.
- **Temporal**: Warm-start each frame from previous frame's result. Share shape/scale params across a sequence (fit once on first N frames, then freeze).
- **Coordinate transform**: Apex Z-up `[x,y,z]` → MHR pre-flip Y-up `[x, z, -y]`.

## Files to Create

### 1. `phase0/apex_loader.py` — Apex data ingestion
- `load_apex_kinematics(json_path)` → list of per-frame dicts with joint positions + local 3x3 rotations
- `load_apex_calibration(calib_dir)` → dict of camera_id → {R, t, K, dist, img_size}
- `load_apex_images(image_dir, camera_names)` → dict of camera_id → list of image paths
- Reference: `/home/marco/Desktop/SportsMotion/dev/apex/apex/export/exporters/kinematics_to_json.py` (JSON format)
- Reference: `/home/marco/Desktop/SportsMotion/dev/apex/apex/tracking/core/data_structures/calibration.py` (camera format)

### 2. `phase0/apex_joint_mapping.py` — Apex→MHR70 correspondences
- ~15 position correspondences (hips, knees, ankles, shoulders, elbows, wrists, neck, feet)
- No head landmarks (nose/eyes/ears) — Apex has no face joints
- No hand joints (MHR 21-62 skipped)
- Foot joint mapped to heel position with low weight
- Verified: compound joints collapse to last body name in chain (see `_detect_compound_and_skips` in `kinematics_to_json.py`)

```
Apex JSON Name → MHR70 Index  (Weight)
l_thigh        → 9  L_hip     (1.0)
r_thigh        → 10 R_hip     (1.0)
l_shank        → 11 L_knee    (1.0)
r_shank        → 12 R_knee    (1.0)
l_ankle_2      → 13 L_ankle   (0.8)   [compound: ankle_0/1/2 → ankle_2]
r_ankle_2      → 14 R_ankle   (0.8)
l_shoulder     → 5  L_shoulder(1.0)   [ball joint, not compound]
r_shoulder     → 6  R_shoulder(1.0)
l_forearm      → 7  L_elbow   (1.0)   [single hinge]
r_forearm      → 8  R_elbow   (1.0)
l_wrist_2      → 62 L_wrist   (0.8)   [compound: wrist_0/1/2 → wrist_2]
r_wrist_2      → 41 R_wrist   (0.8)
neck_2         → 69 neck      (0.6)   [compound: neck_0/1/2 → neck_2]
l_foot         → 17 L_heel    (0.3)   [single hinge]
r_foot         → 20 R_heel    (0.3)
```

### 3. `phase0/coord_utils.py` — Add apex coordinate transforms (modify existing)
- `apex_zup_to_mhr_preflip(positions)`: `[x,y,z] → [x, z, -y]`
- `apex_rotation_to_mhr_preflip(R)`: change-of-basis for 3x3 rotation matrices
- `mhr_preflip_to_camera(positions, R_w2c, t_w2c)`: chain MHR→Apex world→camera for 2D projection

### 4. `phase0/mhr_fitter.py` — Extend with Apex initialization (modify existing)
- `fit_batch()` gets new optional args: `global_rot_init`, `global_trans_init`, `num_stages=3`
- When `num_stages=2`, skip hand stage entirely
- Init `global_rot_6d` from Apex pelvis rotation (converted to MHR pre-flip frame)
- Init `global_trans` from Apex pelvis position (accounting for MHR ×10 scaling)
- Increase shape/scale regularization (fewer correspondences → more underconstrained)

### 5. `phase0/apex_export.py` — Per-camera annotation assembly + WebDataset export
- `assemble_apex_annotation(fitted, camera, img_size)` → annotation dict matching SAM 3D format
- Transforms fitted MHR keypoints: pre-flip → Apex world → camera space
- 2D projection via camera K matrix
- Visibility filtering: behind-camera + out-of-bounds → confidence=0
- Skip camera views with <8 visible keypoints
- Bbox from visible 2D keypoints with 20% padding
- Reuse `write_webdataset_shard()` from existing `phase0/export_webdataset.py`

### 6. `phase0/run_apex_conversion.py` — Master CLI script
- Args: `--apex_kinematics`, `--apex_calibration`, `--apex_image_dir`, `--sam3d_checkpoint`, `--mhr_model`, `--output_dir`, `--batch_size`, `--gpu`
- Flow: load session → batch frames → fit MHR → per-camera export → write shards
- Temporal warm-start: use previous frame's params as initialization
- Shape/scale sharing: fit shape on first batch, freeze for rest of sequence

## Data Flow

```
Apex JSON + Calibration + Images
        │
        ▼
   apex_loader.py
   (frames, cameras, image_paths)
        │
        ▼
   apex_joint_mapping.py
   (target_positions in Apex Z-up)
        │
        ▼
   coord_utils.py
   apex_zup_to_mhr_preflip()
   (targets in MHR pre-flip space)
        │
        ▼
   mhr_fitter.py
   fit_batch() with apex init
   (model_params, shape_params, keypoints_3d)
        │
        ├──► For each camera:
        │    apex_export.py
        │    (annotation per camera view)
        │
        ▼
   WebDataset tar shards
   (jpg + metadata.json + annotation.pyd)
```

## Training Data Output Format (per sample)

```python
{
    "keypoints_2d": (70, 3),        # [x, y, confidence] — projected MHR70
    "keypoints_3d": (70, 3),        # MHR70 in camera space
    "mhr_params": {
        "model_params": (204,),     # fitted MHR model params
        "shape_params": (45,),      # fitted shape params
    },
    "mhr_valid": bool,              # MPJPE < 50mm threshold
    "bbox": (4,),                   # [x, y, w, h]
    "center": (2,), "scale": float,
    "metadata": {"cam_int": (3,3), "loss": float},
}
```

## Implementation Order

1. `apex_loader.py` — parse real Apex output, verify joint names
2. `apex_joint_mapping.py` — build correspondence table
3. `coord_utils.py` — add apex transforms
4. `mhr_fitter.py` — add apex init + 2-stage mode
5. Small test: fit one frame, scatter-plot to verify alignment
6. `apex_export.py` — per-camera annotation + WebDataset
7. `run_apex_conversion.py` — end-to-end CLI
8. Validate: load produced WebDataset, overlay on camera images

## Verification

1. **Scatter plot test**: Fit one Apex frame, plot Apex joints vs MHR keypoints (like phase0 scatter plots)
2. **2D reprojection overlay**: Project fitted MHR onto camera images, visually verify alignment
3. **Round-trip test**: Load produced WebDataset through SAM 3D dataloader, verify all fields parse correctly
4. **MPJPE metrics**: Report per-joint and mean errors
