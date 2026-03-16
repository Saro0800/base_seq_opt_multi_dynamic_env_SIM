#!/usr/bin/env python3
import trimesh
import fast_simplification
import numpy as np

mesh = trimesh.load('/home/humans/base_pose_opt_multi_dynamic_env_SIM/src/human_control/meshes/locobot_intero_sleep_position.stl')
print(f'Original: {len(mesh.faces)} faces')

# Decimate to ~5% of original (reduce by 95%)
vertices = np.array(mesh.vertices, dtype=np.float32)
faces = np.array(mesh.faces, dtype=np.int32)
new_verts, new_faces = fast_simplification.simplify(vertices, faces, target_reduction=0.95)
simplified = trimesh.Trimesh(vertices=new_verts, faces=new_faces)
print(f'Simplified: {len(simplified.faces)} faces')

simplified.export('/home/humans/base_pose_opt_multi_dynamic_env_SIM/src/human_control/meshes/locobot_intero_sleep_position_reduced.stl')
print('Saved to locobot_intero_sleep_position_reduced.stl')
