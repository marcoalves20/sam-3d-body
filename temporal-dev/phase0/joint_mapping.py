import torch
import numpy as np

# ── SMPL-X joint indices (from SMPL-X 55-joint layout) ──
# 0: pelvis, 1: L_hip, 2: R_hip, 3: spine1, 4: L_knee, 5: R_knee,
# 6: spine2, 7: L_ankle, 8: R_ankle, 9: spine3, 10: L_foot, 11: R_foot,
# 12: neck, 13: L_collar, 14: R_collar, 15: head, 16: L_shoulder,
# 17: R_shoulder, 18: L_elbow, 19: R_elbow, 20: L_wrist, 21: R_wrist,
# 22-24: jaw/eyes, 25-39: L_hand (15 joints), 40-54: R_hand (15 joints)

# ── MHR70 joint indices (from sam_3d_body/metadata/mhr70.py) ──
# 0: nose, 1: L_eye, 2: R_eye, 3: L_ear, 4: R_ear,
# 5: L_shoulder, 6: R_shoulder, 7: L_elbow, 8: R_elbow,
# 9: L_hip, 10: R_hip, 11: L_knee, 12: R_knee,
# 13: L_ankle, 14: R_ankle,
# 15-17: L_foot (big toe, small toe, heel),
# 18-20: R_foot (big toe, small toe, heel),
# 21-41: R_hand (21 joints), 42-62: L_hand (21 joints),
# 63-64: olecranons, 65-66: cubital fossa, 67-68: acromions, 69: neck

# Body joints: MHR idx -> SMPLX joint idx
SMPLX_TO_MHR70_BODY = {
    0: 87,   # nose (SMPLX extra joint: nose tip)
    1: 23,   # L eye
    2: 24,   # R eye
    5: 16,   # L shoulder
    6: 17,   # R shoulder
    7: 18,   # L elbow
    8: 19,   # R elbow
    9: 1,    # L hip
    10: 2,   # R hip
    11: 4,   # L knee
    12: 5,   # R knee
    13: 7,   # L ankle
    14: 8,   # R ankle
    62: 20,  # L wrist
    41: 21,  # R wrist
    69: 12,  # neck
}

# Head vertices: ears don't have SMPLX joint correspondences
SMPLX_VERTEX_TO_MHR70_HEAD = {
    3: 243,   # L ear
    4: 989,   # R ear
}

# Right hand: MHR indices 21-40 -> SMPLX joints 40-54 (15 joints)
# MHR right hand layout (21 joints): thumb(4), index(4), middle(4), ring(4), pinky(4), wrist(1)
# We map the non-tip joints (the ones that correspond to SMPLX joints)
# SMPLX right hand: 40=thumb0, 41=thumb1, 42=thumb2, 43=index0, 44=index1, 45=index2,
#   46=middle0, 47=middle1, 48=middle2, 49=ring0, 50=ring1, 51=ring2,
#   52=pinky0, 53=pinky1, 54=pinky2
SMPLX_TO_MHR70_RIGHT_HAND = {
    # Right thumb: MHR 22,23,24 (first, second, third joint) -> SMPLX 40,41,42
    22: 40, 23: 41, 24: 42,
    # Right index: MHR 26,27,28 -> SMPLX 43,44,45
    26: 43, 27: 44, 28: 45,
    # Right middle: MHR 30,31,32 -> SMPLX 46,47,48
    30: 46, 31: 47, 32: 48,
    # Right ring: MHR 34,35,36 -> SMPLX 49,50,51
    34: 49, 35: 50, 36: 51,
    # Right pinky: MHR 38,39,40 -> SMPLX 52,53,54
    38: 52, 39: 53, 40: 54,
}

# Left hand: MHR indices 42-61 -> SMPLX joints 25-39 (15 joints)
# SMPLX left hand: 25=thumb0, 26=thumb1, 27=thumb2, 28=index0, 29=index1, 30=index2,
#   31=middle0, 32=middle1, 33=middle2, 34=ring0, 35=ring1, 36=ring2,
#   37=pinky0, 38=pinky1, 39=pinky2
SMPLX_TO_MHR70_LEFT_HAND = {
    # Left thumb: MHR 43,44,45 -> SMPLX 25,26,27
    43: 25, 44: 26, 45: 27,
    # Left index: MHR 47,48,49 -> SMPLX 28,29,30
    47: 28, 48: 29, 49: 30,
    # Left middle: MHR 51,52,53 -> SMPLX 31,32,33
    51: 31, 52: 32, 53: 33,
    # Left ring: MHR 55,56,57 -> SMPLX 34,35,36
    55: 34, 56: 35, 57: 36,
    # Left pinky: MHR 59,60,61 -> SMPLX 37,38,39
    59: 37, 60: 38, 61: 39,
}

# SMPL-X vertex indices for feet keypoints
# Found by locating vertices nearest to foot joints in zero-pose SMPL-X
SMPLX_VERTEX_TO_MHR70_FEET = {
    # MHR idx -> SMPLX vertex idx
    15: 5776,   # L big toe tip (nearest forward vertex to L_foot ball)
    16: 5897,   # L small toe tip (nearest vertex to L_foot joint)
    17: 3757,   # L heel (backward from L_ankle)
    18: 8470,   # R big toe tip (nearest forward vertex to R_foot ball)
    19: 8591,   # R small toe tip (nearest vertex to R_foot joint)
    20: 6515,   # R heel (backward from R_ankle)
}

# SMPL-X vertex indices for fingertips
# Found by locating vertices nearest to fingertip joints in zero-pose SMPL-X
SMPLX_VERTEX_TO_MHR70_FINGERTIPS = {
    # Right hand fingertips (MHR idx -> SMPLX vertex idx)
    21: 7646,   # R thumb tip (nearest to J42)
    25: 7762,   # R index tip (nearest to J45)
    29: 8032,   # R middle tip (nearest to J48)
    33: 7914,   # R ring tip (nearest to J51)
    37: 8105,   # R pinky tip (nearest to J54)
    # Left hand fingertips (MHR idx -> SMPLX vertex idx)
    42: 4910,   # L thumb tip (nearest to J27)
    46: 5026,   # L index tip (nearest to J30)
    50: 5296,   # L middle tip (nearest to J33)
    54: 5178,   # L ring tip (nearest to J36)
    58: 5371,   # L pinky tip (nearest to J39)
}

# Weights by category
WEIGHT_BODY = 1.0
WEIGHT_HANDS = 1.0
WEIGHT_FINGERTIPS = 0.8
WEIGHT_FEET = 0.5


def get_mapping_indices(include_hands=True, include_feet=True, include_fingertips=True):
    """Build arrays of corresponding MHR and SMPLX indices with per-joint weights.

    Returns:
        mhr_indices: list of MHR70 keypoint indices
        smplx_joint_indices: list of SMPLX joint indices (for joint-based correspondences)
        smplx_vertex_indices: list of SMPLX vertex indices (for vertex-based correspondences)
        weights: tensor of per-correspondence weights
        is_vertex: list of bools indicating whether correspondence uses vertices
    """
    mhr_indices = []
    smplx_joint_indices = []
    smplx_vertex_indices = []
    weights = []
    is_vertex = []

    # Body joints (including head joints: nose, eyes)
    for mhr_idx, smplx_idx in SMPLX_TO_MHR70_BODY.items():
        mhr_indices.append(mhr_idx)
        smplx_joint_indices.append(smplx_idx)
        smplx_vertex_indices.append(-1)
        weights.append(WEIGHT_BODY)
        is_vertex.append(False)

    # Head vertices (ears — no SMPLX joint correspondence)
    for mhr_idx, vtx_idx in SMPLX_VERTEX_TO_MHR70_HEAD.items():
        mhr_indices.append(mhr_idx)
        smplx_joint_indices.append(-1)
        smplx_vertex_indices.append(vtx_idx)
        weights.append(WEIGHT_BODY)
        is_vertex.append(True)

    # Hand joints
    if include_hands:
        for mhr_idx, smplx_idx in SMPLX_TO_MHR70_RIGHT_HAND.items():
            mhr_indices.append(mhr_idx)
            smplx_joint_indices.append(smplx_idx)
            smplx_vertex_indices.append(-1)
            weights.append(WEIGHT_HANDS)
            is_vertex.append(False)
        for mhr_idx, smplx_idx in SMPLX_TO_MHR70_LEFT_HAND.items():
            mhr_indices.append(mhr_idx)
            smplx_joint_indices.append(smplx_idx)
            smplx_vertex_indices.append(-1)
            weights.append(WEIGHT_HANDS)
            is_vertex.append(False)

    # Feet (vertex-based)
    if include_feet:
        for mhr_idx, vtx_idx in SMPLX_VERTEX_TO_MHR70_FEET.items():
            mhr_indices.append(mhr_idx)
            smplx_joint_indices.append(-1)
            smplx_vertex_indices.append(vtx_idx)
            weights.append(WEIGHT_FEET)
            is_vertex.append(True)

    # Fingertips (vertex-based)
    if include_fingertips:
        for mhr_idx, vtx_idx in SMPLX_VERTEX_TO_MHR70_FINGERTIPS.items():
            mhr_indices.append(mhr_idx)
            smplx_joint_indices.append(-1)
            smplx_vertex_indices.append(vtx_idx)
            weights.append(WEIGHT_FINGERTIPS)
            is_vertex.append(True)

    return (
        mhr_indices,
        smplx_joint_indices,
        smplx_vertex_indices,
        torch.tensor(weights, dtype=torch.float32),
        is_vertex,
    )


def extract_smplx_targets(smplx_joints, smplx_vertices=None, include_hands=True,
                           include_feet=True, include_fingertips=True):
    """Extract SMPLX target positions corresponding to MHR70 keypoints.

    Args:
        smplx_joints: (B, J, 3) SMPLX joint positions (J >= 55)
        smplx_vertices: (B, V, 3) SMPLX mesh vertices (V=10475), needed for feet/fingertips
        include_hands: whether to include hand joint correspondences
        include_feet: whether to include foot vertex correspondences
        include_fingertips: whether to include fingertip vertex correspondences

    Returns:
        target_positions: (B, N, 3) target 3D positions
        mhr_indices: list of MHR70 indices
        weights: (N,) per-correspondence weights
    """
    if include_feet or include_fingertips:
        assert smplx_vertices is not None, "Need vertices for feet/fingertip targets"

    mhr_indices, smplx_jidx, smplx_vidx, weights, is_vtx = get_mapping_indices(
        include_hands=include_hands,
        include_feet=include_feet,
        include_fingertips=include_fingertips,
    )

    B = smplx_joints.shape[0]
    N = len(mhr_indices)
    device = smplx_joints.device
    target_positions = torch.zeros(B, N, 3, device=device, dtype=smplx_joints.dtype)

    for i in range(N):
        if is_vtx[i]:
            target_positions[:, i] = smplx_vertices[:, smplx_vidx[i]]
        else:
            target_positions[:, i] = smplx_joints[:, smplx_jidx[i]]

    return target_positions, mhr_indices, weights.to(device)
