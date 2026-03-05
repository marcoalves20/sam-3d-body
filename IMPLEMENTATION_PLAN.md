# SAM 3D Body: Finetuning & Temporal Video Inference — Implementation Plan

This plan is organized into **7 phases** (0–6), each self-contained and delegatable to a dedicated subagent. Phase 0 must be completed first (it produces the training data). Phases 1–3 are independent of each other and can be parallelized. Phases 4–6 build on Phase 1 and should be done sequentially.

---

## Phase 0: BEDLAM → MHR Data Conversion Pipeline

**Goal:** Convert the BEDLAM dataset (SMPL-X annotations) into SAM 3D Body's MHR format via differentiable optimization, producing training-ready WebDataset tar files.

**Why this is necessary:** SAM 3D Body uses the Momentum Human Rig (MHR) — a fundamentally different parametric body model from SMPL-X. MHR has 127 joints (vs 55), a 260D continuous pose latent space (vs per-joint axis-angle), 45 shape PCA components, and 28 scale components that don't exist in SMPL-X. There is no analytical conversion. We must *fit* MHR parameters to match SMPL-X output meshes/joints.

### Step 0.1 — BEDLAM Data Loader

**File to create:** `data/scripts/bedlam_loader.py`

Load BEDLAM's `.npz` annotation files and extract per-person SMPL-X parameters.

**BEDLAM NPZ fields we need:**

| Field | Shape | Description |
|---|---|---|
| `imgname` | `(N,)` str | Image filename |
| `pose_cam` | `(N, 165)` | SMPL-X pose in camera coords (axis-angle) |
| `shape` | `(N, 10)` or `(N, 16)` | Beta shape coefficients |
| `trans_cam` | `(N, 3)` | Translation in camera coords |
| `cam_int` | `(N, 3, 3)` | Camera intrinsic matrix |
| `cam_ext` | `(N, 4, 4)` | Camera extrinsic matrix |
| `center` | `(N, 2)` | Bounding box center |
| `scale` | `(N,)` | Bounding box scale |
| `gtkps` | `(N, K, 2-3)` | 2D ground truth keypoints |
| `gender` | `(N,)` str | `'m'`, `'f'`, or `'n'` |

**SMPL-X pose_cam decomposition (165D, all axis-angle):**
```
[0:3]     global_orient    (1 joint × 3)
[3:66]    body_pose        (21 joints × 3)
[66:69]   jaw_pose         (1 joint × 3)
[69:72]   left_eye_pose    (1 joint × 3)
[72:75]   right_eye_pose   (1 joint × 3)
[75:120]  left_hand_pose   (15 joints × 3)
[120:165] right_hand_pose  (15 joints × 3)
```

**The loader should:**
1. Load NPZ files from BEDLAM's `data_processing/` output.
2. For each person annotation, extract all fields above.
3. Run the SMPL-X forward pass to produce 3D joint positions and mesh vertices as fitting targets.
4. Group annotations by image (multiple people per image).
5. Support iteration over samples in batches for GPU-parallel fitting.

**SMPL-X model initialization** (matching BEDLAM's setup):
```python
import smplx

smplx_model = smplx.create(
    model_path="data/body_models/smplx",
    model_type="smplx",
    gender="neutral",         # BEDLAM uses gendered models but neutral works for fitting
    flat_hand_mean=True,
    num_betas=10,             # or 16 for BEDLAM v2
    num_expression_coeffs=10,
    use_pca=False,
    batch_size=1,
)
```

**SMPL-X forward pass to get targets:**
```python
smplx_output = smplx_model(
    global_orient=pose_cam[:, :3],      # (B, 3)
    body_pose=pose_cam[:, 3:66],        # (B, 63)
    jaw_pose=pose_cam[:, 66:69],        # (B, 3)
    leye_pose=pose_cam[:, 69:72],       # (B, 3)
    reye_pose=pose_cam[:, 72:75],       # (B, 3)
    left_hand_pose=pose_cam[:, 75:120], # (B, 45)
    right_hand_pose=pose_cam[:, 120:],  # (B, 45)
    betas=shape,                         # (B, 10)
    transl=trans_cam,                    # (B, 3)
)

# Targets for fitting:
target_joints = smplx_output.joints       # (B, 127, 3) — all SMPL-X joints
target_vertices = smplx_output.vertices   # (B, 10475, 3) — mesh
```

### Step 0.2 — Joint Correspondence Map (SMPL-X → MHR70)

**File to create:** `data/scripts/joint_mapping.py`

Define the mapping between SMPL-X joints and MHR70 keypoints. This is critical for the fitting loss.

**SMPL-X joint ordering** (first 22 are body, then hands):
```
SMPL-X Body (indices 0-21):
 0: pelvis, 1: left_hip, 2: right_hip, 3: spine1,
 4: left_knee, 5: right_knee, 6: spine2, 7: left_ankle,
 8: right_ankle, 9: spine3, 10: left_foot, 11: right_foot,
 12: neck, 13: left_collar, 14: right_collar, 15: head,
 16: left_shoulder, 17: right_shoulder, 18: left_elbow,
 19: right_elbow, 20: left_wrist, 21: right_wrist

SMPL-X Hands (indices 25-45 left, 40-55 right):
 25-39: left hand (15 joints: root, index1-3, middle1-3, pinky1-3, ring1-3, thumb1-3)
 40-54: right hand (same structure)

SMPL-X Extra (regressed from vertices):
 22: jaw, 23: left_eye, 24: right_eye
 55-66: various face landmarks
```

**MHR70 keypoint ordering** (from `sam_3d_body/metadata/mhr70.py`):
```
 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
 9: left_hip, 10: right_hip, 11: left_knee, 12: right_knee,
13: left_ankle, 14: right_ankle,
15-20: feet (big_toe, small_toe, heel × 2),
21-41: right hand (thumb tip→pinky base, then wrist),
42-62: left hand (thumb tip→pinky base, then wrist),
63-64: olecranon (back of elbow) L/R,
65-66: cubital fossa (inner elbow) L/R,
67-68: acromion (top of shoulder) L/R,
69: neck
```

**The mapping should define:**

```python
# Body joints that have clear 1-to-1 correspondence:
SMPLX_TO_MHR70_BODY = {
    # MHR70_idx: SMPLX_joint_idx
    5: 16,   # left_shoulder
    6: 17,   # right_shoulder
    7: 18,   # left_elbow
    8: 19,   # right_elbow
    9: 1,    # left_hip
    10: 2,   # right_hip
    11: 4,   # left_knee
    12: 5,   # right_knee
    13: 7,   # left_ankle
    14: 8,   # right_ankle
    62: 20,  # left_wrist
    41: 21,  # right_wrist
    69: 12,  # neck
}

# Hand joints — SMPL-X has 15/hand, MHR has 20/hand (4 per finger: tip, 1st, 2nd, 3rd joint)
# The 3rd joint (MCP/base) indices map to SMPL-X's finger roots.
# Tips are NOT directly in SMPL-X joints — they must be regressed from vertices.
SMPLX_TO_MHR70_RIGHT_HAND = {
    # MHR70_idx: SMPLX_joint_idx (where direct mapping exists)
    # thumb: SMPL-X indices 52-54 (thumb1, thumb2, thumb3)
    24: 52,  # right_thumb_third_joint (MCP) → smplx thumb1
    23: 53,  # right_thumb_second_joint → smplx thumb2
    22: 54,  # right_thumb_first_joint → smplx thumb3
    # 21: tip — regress from vertices
    # index: SMPL-X indices 40-42
    28: 40,  # right_index_third_joint → smplx index1
    27: 41,  # right_index_second_joint → smplx index2
    26: 42,  # right_index_first_joint → smplx index3
    # middle: SMPL-X indices 43-45
    32: 43,  # right_middle_third_joint → smplx middle1
    31: 44,  # right_middle_second → smplx middle2
    30: 45,  # right_middle_first → smplx middle3
    # ring: SMPL-X indices 49-51
    36: 49,  # right_ring_third → smplx ring1
    35: 50,  # right_ring_second → smplx ring2
    34: 51,  # right_ring_first → smplx ring3
    # pinky: SMPL-X indices 46-48
    40: 46,  # right_pinky_third → smplx pinky1
    39: 47,  # right_pinky_second → smplx pinky2
    38: 48,  # right_pinky_first → smplx pinky3
}
# (Mirror for left hand: SMPL-X 25-39 → MHR70 42-62)
```

**Important notes:**
- Some MHR70 keypoints (nose, eyes, ears, feet, fingertips, olecranon, cubital fossa, acromion) are NOT direct SMPL-X joints — they are regressed from mesh vertices via `keypoint_mapping` (a 308 × (18439+127) matrix). For the fitting loss, we should only use joints that have clear correspondence.
- Fingertip positions can be approximated from SMPL-X vertex positions using known vertex indices (the `smplx` package provides these).
- We'll supervise on ~30-40 matched joints/keypoints. The remaining MHR70 keypoints will be indirectly constrained via vertex alignment.

### Step 0.3 — MHR Fitting Optimizer

**File to create:** `data/scripts/fit_mhr_to_smplx.py`

This is the core fitting script. For each BEDLAM annotation, optimize MHR parameters to match SMPL-X output.

**Parameters to optimize:**

| Parameter | Dim | Init | Notes |
|---|---|---|---|
| `global_rot_6d` | 6 | Identity [1,0,0,0,1,0] | Global rotation in 6D |
| `body_pose_cont` | 260 | Zero-pose continuous repr | `compact_model_params_to_cont_body(zeros(133))` |
| `shape_params` | 45 | Zeros | Shape PCA coefficients |
| `scale_params` | 28 | Zeros | Scale PCA coefficients |
| `hand_params` | 108 | Zeros | 54 per hand (PCA space) |

**Fitting procedure:**

```python
def fit_mhr_to_smplx(
    mhr_head: MHRHead,
    target_joints_3d: torch.Tensor,   # (B, N_matched, 3) from SMPL-X
    target_vertices: torch.Tensor,     # (B, 10475, 3) from SMPL-X (optional)
    joint_mapping: dict,               # MHR70_idx → SMPLX_joint_idx
    num_iterations: int = 500,
    lr: float = 0.01,
):
    B = target_joints_3d.shape[0]
    device = target_joints_3d.device

    # Initialize optimizable parameters
    global_rot_6d = torch.tensor([[1,0,0,0,1,0]], dtype=torch.float32, device=device)
    global_rot_6d = global_rot_6d.expand(B, -1).clone().requires_grad_(True)

    body_cont = compact_model_params_to_cont_body(
        torch.zeros(B, 133, device=device)
    ).requires_grad_(True)

    shape = torch.zeros(B, 45, device=device, requires_grad=True)
    scale = torch.zeros(B, 28, device=device, requires_grad=True)
    hand = torch.zeros(B, 108, device=device, requires_grad=True)

    optimizer = torch.optim.Adam(
        [global_rot_6d, body_cont, shape, scale, hand],
        lr=lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iterations)

    # MHR70 indices and corresponding SMPL-X joint indices
    mhr_indices = sorted(joint_mapping.keys())
    smplx_indices = [joint_mapping[k] for k in mhr_indices]

    for step in range(num_iterations):
        optimizer.zero_grad()

        # Convert 6D → Euler for MHR forward
        global_rot_rotmat = rot6d_to_rotmat(global_rot_6d)
        global_rot_euler = roma.rotmat_to_euler("ZYX", global_rot_rotmat)

        # Convert continuous → model params
        body_euler = compact_cont_to_model_params_body(body_cont)
        body_euler[:, mhr_param_hand_mask] = 0   # hands handled separately
        body_euler[:, -3:] = 0                    # jaw zeroed

        # MHR forward pass
        verts, j3d_308, jcoords, _, _ = mhr_head.mhr_forward(
            global_trans=global_rot_euler * 0,     # translation handled separately
            global_rot=global_rot_euler,
            body_pose_params=body_euler,
            hand_pose_params=hand,
            scale_params=scale,
            shape_params=shape,
            expr_params=torch.zeros(B, 72, device=device),
            return_keypoints=True,
            return_joint_coords=True,
            return_model_params=True,
            return_joint_rotations=True,
        )

        j3d_70 = j3d_308[:, :70]
        # Apply camera system flip (MHR convention)
        j3d_70[..., [1, 2]] *= -1
        verts[..., [1, 2]] *= -1

        # === LOSS COMPUTATION ===

        # 1. Joint alignment loss (pelvis-relative)
        pred_joints = j3d_70[:, mhr_indices]          # (B, N_matched, 3)
        target_matched = target_joints_3d[:, smplx_indices]

        # Pelvis-relative (align roots)
        # MHR pelvis = mean of indices 9,10 (left_hip, right_hip)
        pred_pelvis = j3d_70[:, [9, 10]].mean(dim=1, keepdim=True)
        # SMPL-X pelvis = index 0
        target_pelvis = target_joints_3d[:, [0]]  # or mean of hips

        loss_joints = F.l1_loss(
            pred_joints - pred_pelvis,
            target_matched - target_pelvis
        )

        # 2. Shape/scale regularization (prevent degenerate solutions)
        loss_reg = 0.01 * (shape.pow(2).mean() + scale.pow(2).mean())

        # 3. Pose regularization (keep close to natural poses)
        loss_pose_reg = 0.001 * body_cont.pow(2).mean()

        # 4. (Optional) Vertex chamfer distance — only if SMPL-X verts available
        # This is expensive but dramatically improves surface alignment.
        # Use nearest-neighbor matching on a subsampled set (~2000 vertices).
        # loss_verts = chamfer_distance_l1(verts_subsample, target_verts_subsample)

        loss = loss_joints + loss_reg + loss_pose_reg
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Return fitted parameters
    return {
        "global_rot_6d": global_rot_6d.detach(),
        "body_pose_cont": body_cont.detach(),
        "shape_params": shape.detach(),
        "scale_params": scale.detach(),
        "hand_params": hand.detach(),
        # Also save the converted euler params for the annotation:
        "body_pose_euler": compact_cont_to_model_params_body(body_cont).detach(),
        "global_rot_euler": roma.rotmat_to_euler("ZYX", rot6d_to_rotmat(global_rot_6d)).detach(),
    }
```

**Multi-stage fitting (recommended for quality):**

1. **Stage 1 (100 iters, lr=0.02):** Fit global rotation + body pose only (freeze shape, scale, hands). Coarse body alignment.
2. **Stage 2 (200 iters, lr=0.01):** Unfreeze shape and scale. Refine body + shape.
3. **Stage 3 (200 iters, lr=0.005):** Unfreeze hands. Full refinement.

This prevents degenerate early solutions where shape/scale compensate for wrong poses.

**Batch processing:** The fitting is fully batched (B samples in parallel on GPU). Recommended batch size: 64–256 per GPU. For ~1M BEDLAM instances at 500 iters each, with B=128 on an A100: ~2-3 GPU-days.

### Step 0.4 — Coordinate System Alignment

**File to add to:** `data/scripts/fit_mhr_to_smplx.py` (or separate utility)

BEDLAM and MHR use different coordinate systems. This must be handled before fitting.

**BEDLAM (SMPL-X in camera coords):**
- `pose_cam` is in camera space (OpenCV convention: X-right, Y-down, Z-forward)
- `trans_cam` is camera-space translation
- `cam_int` is the camera intrinsic matrix (3×3)
- `cam_ext` is world-to-camera (4×4)

**MHR (SAM 3D Body convention):**
- Y and Z axes are flipped after MHR forward pass: `j3d[..., [1, 2]] *= -1` (line 341 of mhr_head.py)
- The MHR model works in its own coordinate system internally
- Camera translation is predicted separately by the camera head
- Global translation in MHR forward pass is multiplied by 10 (line 212: `global_trans * 10`)

**What we need to do:**
1. The SMPL-X `pose_cam` already has `global_orient` in camera space. We use this to set MHR's `global_rot`.
2. The `trans_cam` becomes part of the camera translation (handled by the camera head during training, NOT by the MHR model). We store it separately.
3. Body/hand poses are relative (local joint rotations), so they transfer between coordinate systems — the fitting optimizer handles the mapping.
4. After fitting, when saving the annotation, the `global_rot` should be stored in MHR's Euler convention (ZYX), and body pose in the 133-dim Euler format (not the 260D continuous — the continuous form is the network's output representation, the annotation stores the "model_params" form).

**Alignment function:**
```python
def align_smplx_to_mhr_coords(smplx_joints, smplx_trans):
    """
    SMPL-X joints are in camera coordinates (OpenCV).
    MHR internally flips Y,Z after forward pass.
    We need targets in MHR's pre-flip space.

    Since MHR does: output[..., [1,2]] *= -1 (i.e., negate Y and Z),
    our targets should be in the space BEFORE this flip,
    meaning we need to apply the INVERSE flip to SMPL-X joints.
    """
    aligned = smplx_joints.clone()
    aligned[..., [1, 2]] *= -1  # undo the flip MHR will apply
    return aligned
```

### Step 0.5 — Annotation Assembly & WebDataset Export

**File to create:** `data/scripts/bedlam_to_webdataset.py`

After fitting, assemble the fitted MHR params + BEDLAM images + camera intrinsics into SAM 3D Body's WebDataset format.

**Output annotation format** (must match `create_webdataset.py` exactly):

```python
anno = {
    "person_id": int,                        # from BEDLAM 'sub' field
    "keypoints_2d": np.ndarray,              # (70, 3) — x, y, confidence
    "keypoints_3d": np.ndarray,              # (70, 3) — from fitted MHR forward pass

    "mhr_params": {
        "model_params": np.ndarray,          # (133+68,) = pose(133) + scale(68)
                                             # This is the full_pose_params(136) flattened
                                             # with scales appended, matching mhr_forward's
                                             # internal model_params format
        "shape_params": np.ndarray,          # (45,) shape PCA coefficients
    },
    "mhr_valid": True,                       # fitted successfully
    "mhr_version": "bedlam_fitted",          # provenance tag

    "bbox": np.ndarray,                      # (4,) XYXY from BEDLAM center+scale
    "bbox_format": "xyxy",
    "bbox_score": 1.0,
    "center": np.ndarray,                    # (2,) from BEDLAM
    "scale": np.ndarray,                     # (2,) from BEDLAM

    "metadata": {
        "cam_int": np.ndarray,               # (3, 3) from BEDLAM cam_int
        "loss": float,                        # fitting loss for quality filtering
    },
    # mask: omitted (BEDLAM doesn't provide instance masks in NPZ,
    #        but depth maps could be used to generate them)
}
```

**To derive 2D keypoints:**
```python
# Run fitted MHR forward → get j3d_70 (3D keypoints)
# Project using BEDLAM camera intrinsics:
j3d_cam = j3d_70 + pred_cam_t[:, None, :]   # add camera translation
j2d = j3d_cam[:, :, :2] / j3d_cam[:, :, 2:3]  # perspective divide
j2d[:, :, 0] = j2d[:, :, 0] * fx + cx          # apply intrinsics
j2d[:, :, 1] = j2d[:, :, 1] * fy + cy
keypoints_2d = np.concatenate([j2d, np.ones((*j2d.shape[:-1], 1))], axis=-1)  # add confidence=1
```

**To derive bbox from BEDLAM center+scale:**
```python
# BEDLAM stores center (cx, cy) and scale (s) where the bbox is
# 200px × s in each dimension (BEDLAM convention)
bbox_size = scale * 200
x1 = center[0] - bbox_size / 2
y1 = center[1] - bbox_size / 2
x2 = center[0] + bbox_size / 2
y2 = center[1] + bbox_size / 2
bbox = np.array([x1, y1, x2, y2])
```

**Image handling:** BEDLAM images are PNGs in `bedlam_download_dir/[split]/[sequence]/png/[camera]/[frame].png`. The WebDataset writer needs to read these and encode as JPEG.

**Quality filtering:** Discard fitted samples with `loss > threshold` (e.g., fitting loss > 0.05m MPJPE). This removes cases where the fitting diverged.

### Step 0.6 — Validation & Sanity Checks

**File to create:** `data/scripts/validate_bedlam_conversion.py`

Before using the converted data for training, validate the pipeline:

1. **Visual validation:** For a random subset (~100 samples), render the fitted MHR mesh overlaid on the BEDLAM image. Compare against rendering the SMPL-X mesh on the same image. They should align closely.

2. **Quantitative validation:** For all samples, compute:
   - MPJPE between fitted MHR joints and SMPL-X target joints (should be < 10mm for well-fitted samples)
   - Per-vertex error between MHR mesh and nearest SMPL-X vertex (expect ~5-15mm for body, ~20mm for hands)
   - Fitting success rate (% of samples with MPJPE < 20mm)

3. **Annotation format check:** Load a sample from the produced WebDataset and verify it can be consumed by `prepare_batch()` → `forward_pose_branch()` without errors.

4. **Round-trip test:** Take the stored `model_params` and `shape_params`, run `mhr_forward()`, verify the output matches the fitted joints/vertices within numerical precision.

### Step 0.7 — Execution Script

**File to create:** `data/scripts/run_bedlam_conversion.py`

Master script that orchestrates the full pipeline:

```python
# CLI:
# python data/scripts/run_bedlam_conversion.py \
#     --bedlam_dir /path/to/bedlam \
#     --bedlam_npz /path/to/bedlam/processed_npz \
#     --smplx_model_path /path/to/smplx/models \
#     --sam3d_checkpoint /path/to/sam3d_body_checkpoint \
#     --output_dir data/webdataset/bedlam \
#     --batch_size 128 \
#     --num_workers 8 \
#     --gpus 0,1,2,3 \
#     --fit_iterations 500 \
#     --quality_threshold 0.05

# Steps:
# 1. Load BEDLAM NPZ files
# 2. Initialize SMPL-X model
# 3. Initialize MHR model (from SAM3D checkpoint)
# 4. For each batch of annotations:
#    a. Run SMPL-X forward pass → target joints/vertices
#    b. Align coordinates (Step 0.4)
#    c. Run MHR fitting (Step 0.3)
#    d. Quality filter
#    e. Assemble annotations (Step 0.5)
# 5. Write WebDataset tar shards
# 6. Run validation (Step 0.6)
# 7. Print statistics (total samples, success rate, mean fitting error)
```

**Multi-GPU parallelization:** Split BEDLAM NPZ files across GPUs. Each GPU processes its own subset independently and writes separate tar shards. This is embarrassingly parallel.

**Expected output:**
- `data/webdataset/bedlam/000000.tar` through `data/webdataset/bedlam/NNNNNN.tar`
- Each tar file contains ~1000 image+annotation pairs
- Total: ~380k images → ~1M person instances (after multi-person splitting)

### Files Created in Phase 0

| File | Purpose |
|---|---|
| `data/scripts/bedlam_loader.py` | Load BEDLAM NPZ + run SMPL-X forward |
| `data/scripts/joint_mapping.py` | SMPL-X ↔ MHR70 joint correspondence |
| `data/scripts/fit_mhr_to_smplx.py` | Differentiable MHR fitting optimizer |
| `data/scripts/bedlam_to_webdataset.py` | Assemble & export to WebDataset format |
| `data/scripts/validate_bedlam_conversion.py` | Quality validation |
| `data/scripts/run_bedlam_conversion.py` | Master orchestration script |

### Dependencies to Install

```bash
pip install smplx     # SMPL-X body model
# Also need SMPL-X model files downloaded from https://smpl-x.is.tue.mpg.de/
# Place at: data/body_models/smplx/
```

---

## Phase 1: Training Infrastructure

**Goal:** Build the missing training loop so the pretrained model can be finetuned on new data.

The released codebase is inference-only. The model already inherits from `pl.LightningModule` (via `BaseLightningModule` → `BaseModel` → `SAM3DBody`), but has no `training_step`, loss functions, optimizer config, or data module. We need to add all of these.

### Step 1.1 — Create Loss Module

**File to create:** `sam_3d_body/losses.py`

Implement the following losses as a single `SAM3DBodyLoss` module:

| Loss | Formula | Weight | When to Apply |
|---|---|---|---|
| 3D Keypoint (MPJPE) | `L1(J_pred - J_pred[:,pelvis], J_gt - J_gt[:,pelvis])` | `w_j3d=5.0` | Always (pelvis = mean of indices 9,10) |
| 2D Reprojection | `L1(proj_2d_pred, kpts_2d_gt) * visibility` | `w_j2d=5.0` | When 2D GT available |
| Vertex | `L1(V_pred, V_gt)` | `w_verts=2.0` | When GT mesh available |
| Body Pose Params | `MSE(body_pose_pred, body_pose_gt)` | `w_pose=1.0` | When GT MHR params available |
| Shape Params | `MSE(shape_pred, shape_gt)` | `w_shape=0.5` | When GT MHR params available |
| Scale Params | `MSE(scale_pred, scale_gt)` | `w_scale=0.5` | When GT MHR params available |
| Hand Pose Params | `MSE(hand_pred, hand_gt)` | `w_hand=1.0` | When GT MHR hand params available |
| Acceleration Penalty | `L2(J[t] - 2*J[t-1] + J[t-2])` | `w_accel=1.0` | Phase 6 only (temporal training) |

**Key details:**
- The MHR head outputs `pred_keypoints_3d` as shape `(B, 70, 3)` — pelvis indices are `[9, 10]` (see `SAM3DBody.pelvis_idx`).
- `pred_keypoints_2d` is shape `(B, 70, 2)` in full image space.
- `pred_vertices` is shape `(B, 18439, 3)` when returned.
- Body pose is `(B, 260)` continuous, shape `(B, 45)`, scale `(B, 28)`, hand `(B, 108)`.
- The loss `forward()` should accept `(predictions: Dict, targets: Dict)` where predictions come from `pose_output["mhr"]` and targets come from the data loader batch.
- Each loss term should be gated by a `has_*` flag in the target dict so it gracefully handles partial GT annotations (some datasets have 2D only, some have full MHR params).

### Step 1.2 — Add `training_step` and `validation_step` to SAM3DBody

**File to edit:** `sam_3d_body/models/meta_arch/sam3d_body.py`

Add these methods to the `SAM3DBody` class:

```python
def training_step(self, batch, batch_idx):
    # batch comes from the DataModule (Step 1.4)
    # It should already be in the format expected by forward_step():
    #   batch["img"]: (B, N, 3, 256, 256)
    #   batch["keypoints_3d"]: (B, N, 70, 3)  — GT
    #   batch["keypoints_2d"]: (B, N, 70, 2)  — GT
    #   batch["has_3d"]: (B, N)  — mask for which persons have 3D GT
    #   etc.

    self.hand_batch_idx = []
    self.body_batch_idx = list(range(batch["img"].shape[0] * batch["img"].shape[1]))
    pose_output = self.forward_pose_branch(batch)

    # pose_output["mhr"] is the Dict from MHRHead with all predictions.
    # The decoder does intermediate predictions; pose_output["mhr"] is the last layer.
    loss, loss_dict = self.loss_fn(pose_output["mhr"], batch)

    # Log losses
    for k, v in loss_dict.items():
        self.log(f"train/{k}", v, prog_bar=(k == "loss"))

    return loss

def validation_step(self, batch, batch_idx):
    # Same forward pass but no grad
    self.hand_batch_idx = []
    self.body_batch_idx = list(range(batch["img"].shape[0] * batch["img"].shape[1]))
    pose_output = self.forward_pose_branch(batch)
    loss, loss_dict = self.loss_fn(pose_output["mhr"], batch)

    for k, v in loss_dict.items():
        self.log(f"val/{k}", v, sync_dist=True)

    return loss
```

**Important:** `forward_pose_branch` internally calls `self.backbone(x)` → `self.forward_decoder(...)` which returns intermediate predictions via `do_interm_preds`. For training, we should also supervise intermediate decoder layer outputs (the `pose_output` list returned by the decoder). This means the loss should be applied to all layers with a decay weight (e.g., layer 0: 0.1, layer 1: 0.3, layer 2: 0.5, layer 3: 1.0). To enable this, `forward_pose_branch` needs a minor modification to return the full intermediate predictions list, not just the last one (currently line 1132 does `pose_output = pose_output[-1]`).

### Step 1.3 — Add `configure_optimizers` to SAM3DBody

**File to edit:** `sam_3d_body/models/meta_arch/sam3d_body.py`

```python
def configure_optimizers(self):
    # Differential learning rates
    backbone_params = list(self.backbone.parameters())
    decoder_params = (
        list(self.decoder.parameters())
        + list(self.decoder_hand.parameters())
    )
    head_params = (
        list(self.head_pose.parameters())
        + list(self.head_pose_hand.parameters())
        + list(self.head_camera.parameters())
        + list(self.head_camera_hand.parameters())
    )
    other_params = [
        p for p in self.parameters()
        if not any(p is pp for pp in backbone_params + decoder_params + head_params)
    ]

    param_groups = [
        {"params": backbone_params, "lr": self.cfg.TRAIN.LR_BACKBONE},     # e.g. 1e-5
        {"params": decoder_params, "lr": self.cfg.TRAIN.LR_DECODER},       # e.g. 1e-4
        {"params": head_params, "lr": self.cfg.TRAIN.LR_HEAD},             # e.g. 1e-4
        {"params": other_params, "lr": self.cfg.TRAIN.LR_HEAD},
    ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=self.cfg.TRAIN.MAX_EPOCHS
    )
    return [optimizer], [scheduler]
```

Also add new config keys under `TRAIN`:
```yaml
TRAIN:
  LR_BACKBONE: 1.0e-5
  LR_DECODER: 1.0e-4
  LR_HEAD: 1.0e-4
  WEIGHT_DECAY: 0.01
  MAX_EPOCHS: 50
  FREEZE_BACKBONE: false  # when true, set backbone LR to 0 and requires_grad=False
```

### Step 1.4 — Create Training DataModule

**File to create:** `sam_3d_body/data/datamodule.py`

Build a `pl.LightningDataModule` that wraps the existing WebDataset pipeline.

**Data source:** WebDataset tar files created by `data/scripts/create_webdataset.py`. Each sample contains:
- `jpg`: image bytes
- `metadata.json`: `{"width": int, "height": int}`
- `annotation.pyd`: list of per-person annotation dicts with keys: `person_id`, `keypoints_2d`, `keypoints_3d`, `mhr_params` (dict with `model_params`, `shape_params`), `mhr_valid`, `bbox`, `center`, `scale`, `metadata.cam_int`, `mask`

**The DataModule should:**
1. Use `webdataset.WebDataset` to read tar files.
2. For each sample, decode the image, parse annotations, apply transforms (reuse `GetBBoxCenterScale`, `TopdownAffine`, `VisionTransformWrapper` from `sam_3d_body/data/transforms/common.py`).
3. Construct batch dicts matching the format `forward_pose_branch` expects (see `prepare_batch.py` for the exact structure), plus GT fields for loss computation.
4. Handle multi-person samples: each image may have multiple annotated people. During training, randomly sample 1 person per image (or use all with padding).
5. Support train/val splits via separate tar file lists.
6. Integrate with `DistributedSampler` for multi-GPU training (WebDataset handles this natively via `nodesplitter`).

**Output batch format:**
```python
{
    # Model inputs (same as inference batch)
    "img": (B, 1, 3, 256, 256),       # 1 person per image for training
    "bbox": (B, 1, 4),
    "bbox_center": (B, 1, 2),
    "bbox_scale": (B, 1, 2),
    "ori_img_size": (B, 1, 2),
    "img_size": (B, 1, 2),
    "affine_trans": (B, 1, 2, 3),
    "mask": (B, 1, 1, 256, 256),
    "mask_score": (B, 1),
    "person_valid": (B, 1),
    "cam_int": (B, 3, 3),

    # GT targets
    "keypoints_3d": (B, 1, 70, 3),    # MHR70 3D keypoints
    "keypoints_2d": (B, 1, 70, 2),    # MHR70 2D keypoints
    "has_3d": (B, 1),                  # whether 3D GT is available
    "has_mhr": (B, 1),                 # whether MHR params are available
    "body_pose_gt": (B, 1, 260),       # continuous body pose
    "shape_gt": (B, 1, 45),
    "scale_gt": (B, 1, 28),
    "hand_gt": (B, 1, 108),
}
```

### Step 1.5 — Create Training Script

**File to create:** `train.py`

A minimal PyTorch Lightning trainer script:

```python
# Pseudocode structure:
# 1. Parse args: checkpoint_path, data_dir, config overrides
# 2. Load pretrained model via load_sam_3d_body()
# 3. Attach loss_fn to model
# 4. Optionally freeze backbone (set requires_grad=False)
# 5. Create DataModule
# 6. Configure pl.Trainer with:
#    - precision="bf16-mixed"
#    - callbacks: ModelCheckpoint, LearningRateMonitor
#    - loggers: WandbLogger and/or TensorBoardLogger
#    - gradient_clip_val=1.0
#    - accumulate_grad_batches (for effective larger batch sizes)
# 7. trainer.fit(model, datamodule)
```

**CLI arguments:**
- `--checkpoint_path`: path to pretrained checkpoint
- `--data_dir`: path to WebDataset tar files
- `--freeze_backbone`: flag to freeze ViT backbone
- `--max_epochs`: training epochs
- `--batch_size`: per-GPU batch size
- `--gpus`: number of GPUs
- `--lr`: base learning rate (heads/decoder)
- `--lr_backbone`: backbone learning rate
- `--resume_from`: resume training from a checkpoint

### Step 1.6 — Create Training Config

**File to create:** `configs/train_finetune.yaml`

```yaml
TRAIN:
  USE_FP16: true
  FP16_TYPE: "bfloat16"
  LR_BACKBONE: 1.0e-5
  LR_DECODER: 1.0e-4
  LR_HEAD: 1.0e-4
  WEIGHT_DECAY: 0.01
  MAX_EPOCHS: 50
  FREEZE_BACKBONE: false
  BATCH_SIZE: 32
  NUM_WORKERS: 8
  GRADIENT_CLIP_VAL: 1.0
  ACCUMULATE_GRAD_BATCHES: 1

LOSS:
  W_J3D: 5.0
  W_J2D: 5.0
  W_VERTS: 2.0
  W_POSE: 1.0
  W_SHAPE: 0.5
  W_SCALE: 0.5
  W_HAND: 1.0
  W_ACCEL: 0.0          # 0 for single-frame, >0 for temporal
  INTERMEDIATE_DECAY: [0.1, 0.3, 0.5, 1.0]  # per decoder layer

DATA:
  TRAIN_SHARDS: "data/webdataset/train/*.tar"
  VAL_SHARDS: "data/webdataset/val/*.tar"
  NUM_WORKERS: 8
  PERSONS_PER_IMAGE: 1
```

---

## Phase 2: Temporal — Token Warm-Starting (Level 2)

**Goal:** Use previous frame's pose prediction to initialize the next frame's decoder, so the model only predicts the *delta* between frames. Zero additional parameters, zero retraining.

**Prerequisite:** None (works with pretrained model as-is).

### Step 2.1 — Create Video Inference Wrapper

**File to create:** `sam_3d_body/video_estimator.py`

This wraps `SAM3DBodyEstimator` to process videos frame-by-frame with temporal state:

```python
class VideoEstimator:
    def __init__(self, estimator: SAM3DBodyEstimator):
        self.estimator = estimator
        self.prev_predictions = {}  # track_id → prev output dict

    def process_frame(self, img, bboxes=None, track_ids=None, ...):
        """Process a single video frame with temporal warm-starting."""
        # 1. Run detection (or use provided bboxes + track_ids)
        # 2. Match detections to tracked persons (IoU matching)
        # 3. For matched persons, retrieve prev_predictions[track_id]
        # 4. Call model with warm-started init_estimate
        # 5. Store predictions for next frame
        # 6. Return per-person results

    def process_video(self, video_path, ...):
        """Process a full video file."""
        # Frame-by-frame loop with process_frame()
```

### Step 2.2 — Modify `forward_decoder` to Accept External Init

**File to edit:** `sam_3d_body/models/meta_arch/sam3d_body.py`

The method `forward_decoder()` (line 289) already accepts `init_estimate` and `prev_estimate` parameters. Currently, when `init_estimate=None` (which is always the case in `forward_pose_branch`, line 1126), it uses the learned `self.init_pose` embedding.

**Change `forward_pose_branch`** to optionally accept an external `init_estimate`:

```python
def forward_pose_branch(self, batch: Dict, init_estimates: Optional[Dict] = None) -> Dict:
    # ... existing backbone code ...

    # Instead of always passing init_estimate=None:
    init_est = None
    prev_est = None
    if init_estimates is not None:
        # init_estimates contains previous frame's raw MHR + camera params
        # Shape: (B*N, 1, npose + ncam) where npose=465, ncam=3
        init_est = init_estimates.get("init_estimate", None)
        prev_est = init_estimates.get("prev_estimate", None)

    tokens_output, pose_output = self.forward_decoder(
        image_embeddings[self.body_batch_idx],
        init_estimate=init_est,           # was None
        keypoints=keypoints_prompt[self.body_batch_idx],
        prev_estimate=prev_est,           # was None
        condition_info=condition_info[self.body_batch_idx],
        batch=batch,
    )
```

**Construct `init_estimate` from previous frame output** in the `VideoEstimator`:

```python
# From previous frame's pose_output["mhr"]:
prev_pose = torch.cat([
    pose_output["mhr"]["pred_pose_raw"],  # (B, 6) — global rot 6D
    pose_output["mhr"]["body_pose_continuous"],  # (B, 260) — continuous body pose
    pose_output["mhr"]["shape"],           # (B, 45)
    pose_output["mhr"]["scale"],           # (B, 28)
    pose_output["mhr"]["hand"],            # (B, 108)
    pose_output["mhr"]["face"],            # (B, 72)  — if applicable
], dim=-1).unsqueeze(1)  # (B, 1, 465)

prev_cam = pose_output["mhr"]["pred_cam"].unsqueeze(1)  # (B, 1, 3)
init_estimate = torch.cat([prev_pose, prev_cam], dim=-1)  # (B, 1, 468)
```

Note: Check that this matches `self.head_pose.npose + self.head_camera.ncam` exactly. The current `init_pose` embedding has dim `self.head_pose.npose` = 465, and `init_camera` has dim 3, for a total of 468. Verify these dimensions match the concatenation above.

### Step 2.3 — Add IoU-Based Person Tracking

**File to edit:** `sam_3d_body/video_estimator.py` (same file as Step 2.1)

Simple IoU matching between frames (no need for a full tracker):

```python
def match_detections(prev_boxes, curr_boxes, iou_threshold=0.3):
    """Match current detections to previous frame using IoU."""
    # Compute IoU matrix (N_prev x N_curr)
    # Hungarian matching or greedy assignment
    # Return: list of (prev_idx, curr_idx) matches
    # Unmatched curr detections get fresh initialization
```

For more robust tracking, optionally integrate ByteTrack or similar — but IoU matching is sufficient as a starting point since consecutive frames have high bbox overlap.

---

## Phase 3: Temporal — Post-Processing Smoothing (Level 1)

**Goal:** Apply real-time temporal filters to per-frame outputs to reduce jitter. No model changes, no retraining.

**Prerequisite:** Phase 2 (needs the `VideoEstimator` wrapper).

### Step 3.1 — Implement OneEuro Filter

**File to create:** `sam_3d_body/temporal/filters.py`

Implement the [OneEuro filter](https://cristal.univ-lille.fr/~casiez/1euro/) for real-time smoothing:

```python
class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """
        min_cutoff: minimum cutoff frequency (lower = more smoothing)
        beta: speed coefficient (higher = less smoothing when fast motion)
        d_cutoff: cutoff frequency for derivative
        """

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        """Filter a single frame's values. x can be any shape."""
```

Also implement a simple exponential moving average as a fallback:

```python
class ExponentialMovingAverage:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.prev = None

    def __call__(self, x):
        if self.prev is None:
            self.prev = x
            return x
        self.prev = self.alpha * x + (1 - self.alpha) * self.prev
        return self.prev
```

### Step 3.2 — Integrate Smoothing into VideoEstimator

**File to edit:** `sam_3d_body/video_estimator.py`

Apply smoothing to the appropriate output channels per tracked person:

```python
# What to smooth (and what NOT to smooth):
SMOOTH_TARGETS = {
    "pred_cam_t": OneEuroFilter(min_cutoff=0.5, beta=0.5),   # camera translation
    "body_pose": OneEuroFilter(min_cutoff=1.0, beta=0.3),    # body pose euler angles
    "hand": OneEuroFilter(min_cutoff=1.5, beta=0.2),         # hand pose (faster motion)
    "shape": ExponentialMovingAverage(alpha=0.1),             # shape should be near-constant
    "scale": ExponentialMovingAverage(alpha=0.1),             # scale should be near-constant
    "global_rot": OneEuroFilter(min_cutoff=0.8, beta=0.4),   # global rotation
}
# Do NOT smooth: pred_keypoints_2d (derived from smoothed params),
#                pred_vertices (recompute from smoothed params)
```

After smoothing MHR params, **re-run `head_pose.mhr_forward()`** with the smoothed params to get consistent vertices and keypoints.

### Step 3.3 — Add Interpolation for Dropped Detections

**File to edit:** `sam_3d_body/video_estimator.py`

When a tracked person is not detected for 1–2 frames (occlusion), interpolate:

```python
def handle_missing_detection(self, track_id, current_frame_idx):
    """Interpolate prediction for a temporarily lost track."""
    history = self.track_history[track_id]  # list of (frame_idx, prediction)
    if len(history) < 2:
        return None
    # Linear interpolation of MHR params from last 2 observations
    # Mark confidence as low
    # Kill track if missing for > N frames (e.g., 10)
```

---

## Phase 4: Temporal — Backbone Feature Caching (Level 3)

**Goal:** Skip the expensive ViT backbone on intermediate frames by caching and warping features. This is the biggest speed optimization — the backbone is ~70-80% of per-frame compute.

**Prerequisite:** Phase 2 (needs `VideoEstimator`).

### Step 4.1 — Implement Keyframe Scheduling

**File to create:** `sam_3d_body/temporal/keyframe.py`

```python
class KeyframeScheduler:
    def __init__(self, interval=5, adaptive=True):
        self.interval = interval         # run full backbone every N frames
        self.adaptive = adaptive         # if True, adapt based on motion
        self.frame_count = 0

    def should_run_backbone(self, prev_prediction=None, curr_bbox=None):
        """Decide whether this frame needs a full backbone pass."""
        self.frame_count += 1

        # Always run on first frame
        if self.frame_count == 1:
            return True

        # Fixed interval
        if not self.adaptive:
            return self.frame_count % self.interval == 0

        # Adaptive: run backbone if motion is large
        if prev_prediction is not None:
            bbox_shift = compute_bbox_iou(prev_prediction["bbox"], curr_bbox)
            if bbox_shift < 0.7:  # significant motion
                return True

        return self.frame_count % self.interval == 0
```

### Step 4.2 — Add Feature Caching to forward_pose_branch

**File to edit:** `sam_3d_body/models/meta_arch/sam3d_body.py`

Add a `cached_forward_pose_branch` method that optionally reuses cached backbone features:

```python
def cached_forward_pose_branch(
    self,
    batch: Dict,
    cached_features: Optional[torch.Tensor] = None,
    init_estimates: Optional[Dict] = None,
) -> Dict:
    batch_size, num_person = batch["img"].shape[:2]

    if cached_features is None:
        # FULL PASS: run backbone (expensive)
        x = self.data_preprocess(self._flatten_person(batch["img"]), ...)
        ray_cond = self.get_ray_condition(batch)
        ray_cond = self._flatten_person(ray_cond)
        # ... (existing ray_cond cropping logic) ...
        batch["ray_cond"] = ray_cond[self.body_batch_idx].clone()

        image_embeddings = self.backbone(x.type(self.backbone_dtype))
        if isinstance(image_embeddings, tuple):
            image_embeddings = image_embeddings[-1]
        image_embeddings = image_embeddings.type(x.dtype)
    else:
        # CACHED PASS: skip backbone entirely
        image_embeddings = cached_features
        # Still need ray_cond for the decoder
        ray_cond = self.get_ray_condition(batch)
        ray_cond = self._flatten_person(ray_cond)
        batch["ray_cond"] = ray_cond[self.body_batch_idx].clone()

    # Mask conditioning, decoder, heads — all run every frame (they're cheap)
    # ... (rest of forward_pose_branch, unchanged) ...

    # Return image_embeddings so they can be cached
    output["image_embeddings"] = image_embeddings
    return output
```

### Step 4.3 — Implement Feature Warping via Optical Flow

**File to create:** `sam_3d_body/temporal/feature_warp.py`

For intermediate frames, warp cached features using optical flow or bbox displacement:

```python
class FeatureWarper:
    """Warp cached backbone features to approximate current frame features."""

    def __init__(self, method="bbox_affine"):
        """
        Methods:
        - "bbox_affine": simple affine warp based on bbox displacement (fast, approximate)
        - "optical_flow": use RAFT or Farneback flow (accurate, slower)
        """
        self.method = method

    def warp_features(
        self,
        cached_features: torch.Tensor,    # (B, C, H, W) from keyframe
        prev_bbox: np.ndarray,             # (B, 4) bbox at keyframe
        curr_bbox: np.ndarray,             # (B, 4) bbox at current frame
        prev_img: Optional[np.ndarray] = None,
        curr_img: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        if self.method == "bbox_affine":
            # Compute affine transform from bbox displacement
            # Apply grid_sample to warp cached features
            affine = compute_bbox_affine(prev_bbox, curr_bbox)
            grid = F.affine_grid(affine, cached_features.shape)
            return F.grid_sample(cached_features, grid, mode="bilinear")

        elif self.method == "optical_flow":
            # Compute dense optical flow between prev_img and curr_img
            # Downsample flow to feature resolution
            # Apply flow-based warping via grid_sample
            ...
```

The **bbox_affine** method is strongly recommended as a starting point — it's fast (single `grid_sample` call) and handles the common case of camera/person translation well. Optical flow is a more accurate alternative but adds ~5ms per frame.

### Step 4.4 — (Optional) Lightweight Delta Network

**File to create:** `sam_3d_body/temporal/delta_net.py`

A small CNN that refines warped features without running the full ViT:

```python
class DeltaNet(nn.Module):
    """Lightweight feature refinement for non-keyframes."""

    def __init__(self, feature_dim=1280, hidden_dim=256):
        super().__init__()
        # Takes: current crop image + warped cached features
        # Outputs: refined features
        self.img_proj = nn.Sequential(
            nn.Conv2d(3, hidden_dim, 7, stride=4, padding=3),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
            nn.GELU(),
        )  # 256x192 → 16x12 (matches backbone output spatial dims)

        self.fusion = nn.Sequential(
            nn.Conv2d(feature_dim + hidden_dim, feature_dim, 1),
            nn.GELU(),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
        )

    def forward(self, current_img, warped_features):
        """
        current_img: (B, 3, 256, 192) — preprocessed crop
        warped_features: (B, 1280, 16, 12) — warped from keyframe
        """
        img_feat = self.img_proj(current_img)            # (B, 256, 16, 12)
        combined = torch.cat([warped_features, img_feat], dim=1)
        delta = self.fusion(combined)                     # (B, 1280, 16, 12)
        return warped_features + delta                    # residual
```

This is **optional** — bbox_affine warping alone (Step 4.3) may be sufficient. The delta net adds ~1ms per frame but improves accuracy on non-keyframes. It would need to be trained (requires Phase 1 training infrastructure).

---

## Phase 5: Temporal — Temporal Attention in Decoder (Level 4)

**Goal:** Add learnable temporal cross-attention inside the transformer decoder so the model can reason about motion across frames. Requires retraining on video data.

**Prerequisite:** Phase 1 (training infrastructure) + Phase 2 (video processing).

### Step 5.1 — Create Temporal Decoder Layer

**File to create:** `sam_3d_body/models/decoders/temporal_decoder.py`

Extend the existing `TransformerDecoderLayer` with a temporal cross-attention block:

```python
class TemporalTransformerDecoderLayer(TransformerDecoderLayer):
    """Decoder layer with additional cross-attention to previous frame tokens."""

    def __init__(self, *args, temporal_window=3, **kwargs):
        super().__init__(*args, **kwargs)
        dims = kwargs.get("token_dims", args[0] if args else 1024)
        num_heads = kwargs.get("num_heads", 8)
        head_dims = kwargs.get("head_dims", 64)

        # Temporal cross-attention
        self.temporal_attn = MultiHeadAttention(
            embed_dims=dims,
            num_heads=num_heads,
            head_dims=head_dims,
        )
        self.temporal_norm = nn.LayerNorm(dims)
        self.temporal_gate = nn.Parameter(torch.zeros(1))  # learnable gate, init to 0

    def forward(
        self,
        x, context, x_pe=None, context_pe=None, x_mask=None,
        temporal_kv=None,     # NEW: (B, T*N_tokens, C) from prev frames
        temporal_pe=None,     # NEW: positional encoding for temporal tokens
    ):
        # Standard self-attention + cross-attention (from parent)
        x, context = super().forward(x, context, x_pe, context_pe, x_mask)

        # Temporal cross-attention (gated residual)
        if temporal_kv is not None:
            temporal_out = self.temporal_attn(
                query=x, key=temporal_kv, value=temporal_kv
            )
            x = x + torch.tanh(self.temporal_gate) * self.temporal_norm(temporal_out)

        return x, context
```

**Key design decisions:**
- The `temporal_gate` is initialized to 0, so at the start of training the model behaves identically to the pretrained single-frame model. This means we can initialize from the pretrained checkpoint and the new parameters are "zero-initialized" — a standard practice for adding new capabilities without destroying pretrained weights.
- `temporal_kv` contains the output tokens from the previous K frames' decoder (each frame produces ~145 tokens × 1024-dim). For K=3 frames, that's 435 tokens — manageable.
- Only the **pose token** and **keypoint tokens** from previous frames should be included (not the prompt tokens, which are frame-specific).

### Step 5.2 — Create Temporal Decoder Wrapper

**File to create:** `sam_3d_body/models/decoders/temporal_promptable_decoder.py`

```python
class TemporalPromptableDecoder(nn.Module):
    """Promptable decoder with temporal attention across frames."""

    def __init__(self, base_decoder_cfg, temporal_window=3):
        super().__init__()
        self.temporal_window = temporal_window

        # Build layers using TemporalTransformerDecoderLayer
        self.layers = nn.ModuleList([
            TemporalTransformerDecoderLayer(
                token_dims=base_decoder_cfg.DIM,
                context_dims=base_decoder_cfg.CONTEXT_DIM,
                num_heads=base_decoder_cfg.NUM_HEADS,
                head_dims=base_decoder_cfg.HEAD_DIM,
                mlp_dims=base_decoder_cfg.MLP_DIM,
                temporal_window=temporal_window,
                # ... other config from base_decoder_cfg
            )
            for _ in range(base_decoder_cfg.DEPTH)
        ])
        self.norm_final = nn.LayerNorm(base_decoder_cfg.DIM)

        # Temporal position embedding (which frame in the window)
        self.temporal_pe = nn.Embedding(temporal_window, base_decoder_cfg.DIM)

    def forward(
        self,
        token_embedding,      # (B, N_tokens, C)
        image_embedding,      # (B, C, H, W)
        token_augment=None,
        image_augment=None,
        token_mask=None,
        token_to_pose_output_fn=None,
        keypoint_token_update_fn=None,
        prev_frame_tokens=None,  # NEW: list of (B, N_tokens, C) from prev K frames
    ):
        # Build temporal KV from previous frame tokens
        temporal_kv = None
        if prev_frame_tokens is not None and len(prev_frame_tokens) > 0:
            # Add temporal position embeddings
            temporal_parts = []
            for t_idx, tokens in enumerate(prev_frame_tokens):
                pe = self.temporal_pe.weight[t_idx].unsqueeze(0).unsqueeze(0)
                temporal_parts.append(tokens + pe)
            temporal_kv = torch.cat(temporal_parts, dim=1)  # (B, T*N, C)

        # Run decoder layers
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        if image_augment is not None:
            image_augment = image_augment.flatten(2).permute(0, 2, 1)

        for layer_idx, layer in enumerate(self.layers):
            token_embedding, image_embedding = layer(
                token_embedding, image_embedding,
                token_augment, image_augment, token_mask,
                temporal_kv=temporal_kv,
            )
            # ... intermediate prediction logic (same as PromptableDecoder)

        return self.norm_final(token_embedding), all_pose_outputs
```

### Step 5.3 — Integrate Temporal Decoder into SAM3DBody

**File to edit:** `sam_3d_body/models/meta_arch/sam3d_body.py`

Add a config flag `MODEL.DECODER.TEMPORAL` (default `false`). When enabled:

1. In `_initialize_model()`, replace `self.decoder = build_decoder(...)` with `TemporalPromptableDecoder(...)`.
2. Load pretrained weights for the standard decoder layers (they're architecturally identical minus the new temporal attention blocks, which are zero-initialized).
3. In `forward_decoder()`, pass `prev_frame_tokens` from a frame buffer stored on the model.

```python
# In _initialize_model:
if self.cfg.MODEL.DECODER.get("TEMPORAL", False):
    self.decoder = TemporalPromptableDecoder(
        self.cfg.MODEL.DECODER,
        temporal_window=self.cfg.MODEL.DECODER.get("TEMPORAL_WINDOW", 3),
    )
    self.frame_token_buffer = []  # stores last K frames' tokens
else:
    self.decoder = build_decoder(self.cfg.MODEL.DECODER, ...)

# In forward_decoder, before calling self.decoder():
prev_tokens = self.frame_token_buffer if hasattr(self, 'frame_token_buffer') else None

pose_token, pose_output = self.decoder(
    token_embeddings,
    image_embeddings,
    ...,
    prev_frame_tokens=prev_tokens,
)

# After decoder, store current tokens for next frame:
if hasattr(self, 'frame_token_buffer'):
    self.frame_token_buffer.append(pose_token.detach())
    if len(self.frame_token_buffer) > self.cfg.MODEL.DECODER.get("TEMPORAL_WINDOW", 3):
        self.frame_token_buffer.pop(0)
```

### Step 5.4 — Video-Aware Training DataModule

**File to create:** `sam_3d_body/data/video_datamodule.py`

For temporal training, the data loader must yield **sequences of consecutive frames** instead of single images:

```python
class VideoDataModule(pl.LightningDataModule):
    """DataModule that yields temporal windows of consecutive frames."""

    def __init__(self, video_dirs, sequence_length=4, ...):
        self.sequence_length = sequence_length  # e.g., 4 consecutive frames

    def train_dataloader(self):
        # Each sample is a sequence: {
        #     "frames": List[Dict],  # sequence_length frame batches
        #     "track_ids": (seq_len, N_persons),  # person identity across frames
        # }
```

Training sequences should come from video datasets that have per-frame 3D annotations (e.g., 3DPW, EgoExo4D, Harmony4D — all already listed in the WebDataset creation script).

### Step 5.5 — Temporal Training Step

**File to edit:** `sam_3d_body/models/meta_arch/sam3d_body.py`

Add a `training_step` variant for temporal training:

```python
def training_step_temporal(self, batch, batch_idx):
    """Training step for temporal model with sequence input."""
    total_loss = 0
    self.frame_token_buffer = []  # reset at start of each sequence

    for t in range(len(batch["frames"])):
        frame_batch = batch["frames"][t]
        pose_output = self.forward_pose_branch(frame_batch)

        # Per-frame losses (same as Phase 1)
        loss_t, loss_dict_t = self.loss_fn(pose_output["mhr"], frame_batch)
        total_loss += loss_t

        # Temporal consistency loss (for t >= 1)
        if t >= 1:
            j3d_prev = prev_output["pred_keypoints_3d"]
            j3d_curr = pose_output["mhr"]["pred_keypoints_3d"]
            # Velocity smoothness
            if t >= 2:
                j3d_prev2 = prev_prev_output["pred_keypoints_3d"]
                accel = j3d_curr - 2 * j3d_prev + j3d_prev2
                total_loss += self.cfg.LOSS.W_ACCEL * accel.norm(dim=-1).mean()

        prev_prev_output = prev_output if t >= 1 else None
        prev_output = pose_output["mhr"]

    return total_loss / len(batch["frames"])
```

---

## Phase 6: Integration & Optimization

**Goal:** Bring all temporal components together into a unified, optimized video pipeline.

**Prerequisite:** Phases 2–5.

### Step 6.1 — Unified Temporal Video Pipeline

**File to create:** `sam_3d_body/temporal_pipeline.py`

Combine all temporal components into a single high-level API:

```python
class TemporalPipeline:
    def __init__(
        self,
        model,
        detector=None,
        segmentor=None,
        fov_estimator=None,
        # Temporal settings
        keyframe_interval=5,
        adaptive_keyframes=True,
        smoothing="oneeuro",       # "oneeuro", "ema", or "none"
        feature_caching=True,
        feature_warp_method="bbox_affine",
        temporal_decoder=False,    # requires Phase 5 trained model
    ):
        self.estimator = SAM3DBodyEstimator(model, detector, segmentor, fov_estimator)
        self.keyframe_scheduler = KeyframeScheduler(keyframe_interval, adaptive_keyframes)
        self.feature_warper = FeatureWarper(feature_warp_method) if feature_caching else None
        self.tracker = IoUTracker()
        self.smoothers = {}  # track_id → per-channel OneEuroFilter instances

        # State
        self.cached_features = {}  # track_id → (features, bbox, frame_idx)
        self.prev_predictions = {}

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        batch_size: int = 1,       # process N frames at a time (future: batch temporal)
        visualize: bool = True,
    ) -> List[List[Dict]]:
        """Process a video and return per-frame, per-person predictions."""
        ...

    def process_frame(self, img, frame_idx, bboxes=None):
        """Process a single frame with all temporal components."""
        # 1. Detection + tracking
        # 2. For each tracked person:
        #    a. Check keyframe scheduler
        #    b. If keyframe: full backbone pass, cache features
        #    c. If not keyframe: warp cached features
        #    d. Run decoder (with warm-started tokens if available)
        #    e. Apply smoothing filters
        # 3. Return predictions
```

### Step 6.2 — Benchmark and Profile

**File to create:** `tools/benchmark_video.py`

Profile the temporal pipeline components:

```python
# Measure per-component timing:
# - Detection: Xms
# - Backbone (full): Xms
# - Backbone (cached + warp): Xms
# - Decoder: Xms
# - MHR head: Xms
# - Smoothing: Xms
# - Total per-frame (keyframe): Xms
# - Total per-frame (cached): Xms

# Measure accuracy:
# - MPJPE on 3DPW test set (per-frame baseline)
# - MPJPE with warm-starting
# - MPJPE with feature caching (various intervals)
# - Acceleration error (smoothness metric)
# - PCK@50mm
```

### Step 6.3 — Add Video Demo Script

**File to create:** `demo_video.py`

```python
# CLI:
# python demo_video.py \
#     --video_path input.mp4 \
#     --checkpoint_path checkpoints/sam3d_body.pth \
#     --output_path output.mp4 \
#     --keyframe_interval 5 \
#     --smoothing oneeuro \
#     --visualize
```

---

## Dependency Graph

```
Phase 0 (BEDLAM → MHR Conversion)
         │
         ▼
Phase 1 (Training)  ─────────────────────────────┐
                                                  │
Phase 2 (Warm-Starting) ──┬── Phase 3 (Smoothing) │
                          │                        │
                          ├── Phase 4 (Caching)    │
                          │                        │
                          └──────────┬─────────────┘
                                     │
                              Phase 5 (Temporal Decoder)
                                     │
                              Phase 6 (Integration)
```

- **Phase 0** must complete first (produces training data).
- **Phases 1, 2, 3** can be developed in parallel (Phase 1 needs Phase 0 data to run, but the code can be written in parallel).
- **Phase 4** requires Phase 2 (VideoEstimator).
- **Phase 5** requires Phase 1 (training) + Phase 2 (video processing).
- **Phase 6** integrates everything.

---

## Files Created / Modified Summary

| Phase | New Files | Modified Files |
|---|---|---|
| 0 | `data/scripts/bedlam_loader.py`, `data/scripts/joint_mapping.py`, `data/scripts/fit_mhr_to_smplx.py`, `data/scripts/bedlam_to_webdataset.py`, `data/scripts/validate_bedlam_conversion.py`, `data/scripts/run_bedlam_conversion.py` | — |
| 1 | `sam_3d_body/losses.py`, `sam_3d_body/data/datamodule.py`, `train.py`, `configs/train_finetune.yaml` | `sam_3d_body/models/meta_arch/sam3d_body.py` |
| 2 | `sam_3d_body/video_estimator.py` | `sam_3d_body/models/meta_arch/sam3d_body.py` |
| 3 | `sam_3d_body/temporal/filters.py` | `sam_3d_body/video_estimator.py` |
| 4 | `sam_3d_body/temporal/keyframe.py`, `sam_3d_body/temporal/feature_warp.py`, `sam_3d_body/temporal/delta_net.py` (optional) | `sam_3d_body/models/meta_arch/sam3d_body.py` |
| 5 | `sam_3d_body/models/decoders/temporal_decoder.py`, `sam_3d_body/models/decoders/temporal_promptable_decoder.py`, `sam_3d_body/data/video_datamodule.py` | `sam_3d_body/models/meta_arch/sam3d_body.py` |
| 6 | `sam_3d_body/temporal_pipeline.py`, `tools/benchmark_video.py`, `demo_video.py` | — |
