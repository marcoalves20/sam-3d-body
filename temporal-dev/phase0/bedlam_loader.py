import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class BEDLAMDataset(Dataset):
    """Load BEDLAM NPZ files and provide per-person records for fitting.

    BEDLAM NPZ fields used:
        imgname, pose_cam(N,165), shape(N,10|16), trans_cam(N,3),
        cam_int(N,3,3), cam_ext(N,4,4), center(N,2), scale(N,), gender
    """

    def __init__(self, npz_dir, image_root, num_betas=10, npz_pattern="*.npz"):
        self.npz_dir = npz_dir
        self.image_root = image_root
        self.num_betas = num_betas
        self.records = self._load_npz_files(npz_pattern)

    def _load_npz_files(self, npz_pattern):
        npz_files = sorted(glob.glob(os.path.join(self.npz_dir, npz_pattern)))
        records = []
        for npz_path in npz_files:
            data = np.load(npz_path, allow_pickle=True)
            n_persons = len(data["imgname"])
            # Gender may be a single string or per-person array
            gender = data.get("gender", "neutral")
            if isinstance(gender, np.ndarray) and gender.ndim == 0:
                gender = str(gender)

            for i in range(n_persons):
                rec = {
                    "imgname": str(data["imgname"][i]),
                    "pose_cam": data["pose_cam"][i].astype(np.float32),  # (165,)
                    "trans_cam": data["trans_cam"][i].astype(np.float32),  # (3,)
                    "cam_int": data["cam_int"][i].astype(np.float32),  # (3,3)
                    "cam_ext": data["cam_ext"][i].astype(np.float32),  # (4,4)
                    "center": data["center"][i].astype(np.float32),  # (2,)
                    "scale": float(data["scale"][i]),
                    "person_idx": i,
                    "npz_file": os.path.basename(npz_path),
                }

                # Shape: may be 10 or 16 betas, pad/truncate to num_betas
                shape = data["shape"][i].astype(np.float32)
                if len(shape) < self.num_betas:
                    shape = np.pad(shape, (0, self.num_betas - len(shape)))
                else:
                    shape = shape[: self.num_betas]
                rec["shape"] = shape

                # Per-person gender
                if isinstance(gender, str):
                    rec["gender"] = gender
                elif isinstance(gender, np.ndarray):
                    rec["gender"] = str(gender[i]) if gender.ndim > 0 else str(gender)
                else:
                    rec["gender"] = "neutral"

                records.append(rec)
        return records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        return {
            "imgname": rec["imgname"],
            "pose_cam": torch.from_numpy(rec["pose_cam"]),       # (165,)
            "shape": torch.from_numpy(rec["shape"]),             # (num_betas,)
            "trans_cam": torch.from_numpy(rec["trans_cam"]),     # (3,)
            "cam_int": torch.from_numpy(rec["cam_int"]),         # (3,3)
            "cam_ext": torch.from_numpy(rec["cam_ext"]),         # (4,4)
            "center": torch.from_numpy(rec["center"]),           # (2,)
            "scale": rec["scale"],
            "person_idx": rec["person_idx"],
            "gender": rec["gender"],
            "npz_file": rec["npz_file"],
        }


def collate_bedlam_batch(records):
    """Collate list of per-person records into batched tensors."""
    batch = {}
    batch["imgname"] = [r["imgname"] for r in records]
    batch["gender"] = [r["gender"] for r in records]
    batch["npz_file"] = [r["npz_file"] for r in records]
    batch["person_idx"] = [r["person_idx"] for r in records]
    batch["scale"] = torch.tensor([r["scale"] for r in records], dtype=torch.float32)

    for key in ["pose_cam", "shape", "trans_cam", "cam_int", "cam_ext", "center"]:
        batch[key] = torch.stack([r[key] for r in records])

    return batch


def create_smplx_model(model_path, gender="neutral", num_betas=10, device="cuda"):
    """Create a frozen SMPL-X model for forward pass.

    Args:
        model_path: path to SMPLX model directory
        gender: "neutral", "male", or "female"
        num_betas: number of shape components
        device: torch device

    Returns:
        smplx model in eval mode with frozen params
    """
    import smplx

    model = smplx.create(
        model_path,
        model_type="smplx",
        gender=gender,
        num_betas=num_betas,
        use_pca=False,
        flat_hand_mean=True,
    ).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def run_smplx_forward_batch(smplx_model, pose_cam, shape, trans_cam):
    """Run SMPL-X forward pass on a batch of parameters.

    Args:
        smplx_model: SMPL-X model
        pose_cam: (B, 165) full pose in axis-angle
        shape: (B, num_betas) shape parameters
        trans_cam: (B, 3) global translation

    Returns:
        dict with 'joints' (B, J, 3) and 'vertices' (B, 10475, 3)
    """
    B = pose_cam.shape[0]
    device = pose_cam.device

    # Decompose pose_cam (165D axis-angle)
    global_orient = pose_cam[:, 0:3]      # (B, 3)
    body_pose = pose_cam[:, 3:66]         # (B, 63) = 21 joints * 3
    jaw_pose = pose_cam[:, 66:69]         # (B, 3)
    leye_pose = pose_cam[:, 69:72]        # (B, 3)
    reye_pose = pose_cam[:, 72:75]        # (B, 3)
    left_hand_pose = pose_cam[:, 75:120]  # (B, 45) = 15 joints * 3
    right_hand_pose = pose_cam[:, 120:165]  # (B, 45)

    # Pad betas if needed
    num_betas = smplx_model.num_betas
    if shape.shape[1] < num_betas:
        shape = torch.cat([shape, torch.zeros(B, num_betas - shape.shape[1], device=device)], dim=1)
    elif shape.shape[1] > num_betas:
        shape = shape[:, :num_betas]

    output = smplx_model(
        global_orient=global_orient,
        body_pose=body_pose,
        jaw_pose=jaw_pose,
        leye_pose=leye_pose,
        reye_pose=reye_pose,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        betas=shape,
        transl=trans_cam,
        expression=torch.zeros(B, smplx_model.num_expression_coeffs, device=device),
    )

    return {
        "joints": output.joints,       # (B, J, 3)  J=127 for smplx with extra joints
        "vertices": output.vertices,    # (B, 10475, 3)
    }
