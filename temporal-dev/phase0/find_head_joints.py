#!/usr/bin/env python3
import sys, torch
sys.path.insert(0, '.')
import smplx

device = torch.device('cuda:0')
model = smplx.create('./temporal-dev/phase0', model_type='smplx', gender='neutral',
                     num_betas=10, use_pca=False, flat_hand_mean=True).to(device)
model.eval()
with torch.no_grad():
    out = model(return_verts=True)
    joints = out.joints[0]
    verts = out.vertices[0]

print("Head-related SMPLX joints (zero-pose):")
labels = {12: 'neck', 15: 'head', 22: 'jaw', 23: 'left_eye', 24: 'right_eye'}
for idx, name in sorted(labels.items()):
    pos = joints[idx]
    print(f"  J[{idx:2d}] {name:15s} = ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")

# Extra joints in head region
print("\nExtra joints in head region (Y > 0.3):")
for idx in range(55, joints.shape[0]):
    pos = joints[idx]
    if pos[1].item() > 0.3:
        print(f"  J[{idx:3d}] = ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")

# Find nose tip vertex
head_pos = joints[15]
jaw_pos = joints[22]
nose_approx = (head_pos + jaw_pos) / 2
nose_approx[2] += 0.05
dists = (verts - nose_approx.unsqueeze(0)).norm(dim=-1)
nearest = dists.argsort()[:5]
print(f"\nNose vertex candidates (near {nose_approx.tolist()}):")
for vi in nearest:
    print(f"  V[{vi.item():5d}] dist={dists[vi]*1000:.1f}mm")

# Ears
leye = joints[23]
reye = joints[24]
ear_height = (leye[1] + reye[1]) / 2
head_mask = (verts[:, 1] - ear_height).abs() < 0.03
head_verts_idx = head_mask.nonzero().squeeze()
if len(head_verts_idx) > 0:
    head_x = verts[head_verts_idx, 0]
    li = head_verts_idx[head_x.argmax()].item()
    ri = head_verts_idx[head_x.argmin()].item()
    print(f"\nLeft ear vertex:  V[{li}] pos=({verts[li][0]:.4f}, {verts[li][1]:.4f}, {verts[li][2]:.4f})")
    print(f"Right ear vertex: V[{ri}] pos=({verts[ri][0]:.4f}, {verts[ri][1]:.4f}, {verts[ri][2]:.4f})")
