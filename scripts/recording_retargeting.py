"""Reference-only corrections for short, grounded monocular recordings.

These functions never write the state of a dynamic policy rollout.
"""
import numpy as np
import mujoco
from scipy.signal import savgol_filter

FOOT_BODIES = {'calcn_r', 'calcn_l', 'toes_r', 'toes_l'}

def reconstruct_contact_translation(model, qpos, smooth=False):
    """Infer root XY increments from stationary candidate foot contact points.

    qpos: [frames, nq] at 100 Hz. At least one foot must remain grounded.
    A fixed material point is compared across adjacent frames, even when the
    lowest contact candidate changes. No offline trajectory is consulted.
    """
    data = mujoco.MjData(model)
    geoms = [i for i in range(model.ngeom) if model.geom_contype[i]
             and model.body(model.geom_bodyid[i]).name in FOOT_BODIES]
    centers, rotations = [], []
    for pose in qpos:
        data.qpos[:] = pose
        data.qpos[:3] = 0
        mujoco.mj_kinematics(model, data)
        centers.append(data.geom_xpos[geoms].copy())
        rotations.append(data.geom_xmat[geoms].reshape(-1, 3, 3).copy())
    centers, rotations = np.array(centers), np.array(rotations)
    increments = np.zeros((len(qpos), 2))
    confidence = []
    for frame in range(1, len(qpos)):
        current, previous, heights = [], [], []
        for k, geom in enumerate(geoms):
            size = model.geom_size[geom]
            row_z = rotations[frame, k, 2]
            if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                local = -size[0] * row_z
                local[2] -= np.sign(row_z[2]) * size[1]
            elif model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                local = -size ** 2 * row_z / max(np.linalg.norm(size * row_z), 1e-9)
            else:
                raise ValueError(f'Unsupported foot collision shape {geom}')
            p = centers[frame, k] + rotations[frame, k] @ local
            old = centers[frame - 1, k] + rotations[frame - 1, k] @ local
            current.append(p); previous.append(old); heights.append(p[2])
        heights = np.array(heights)
        # Smoothly select the lowest foot shapes, allowing double support.
        weights = np.exp(-(heights - heights.min()) / .015)
        weights /= weights.sum()
        delta = np.array(current) - np.array(previous)
        increments[frame] = -(weights[:, None] * delta[:, :2]).sum(axis=0)
        confidence.append(weights.tolist())
    if smooth:
        increments[1:] = savgol_filter(increments[1:], 21, 3, axis=0)
    result = qpos.copy()
    result[:, :2] = result[0, :2] + np.cumsum(increments, axis=0)
    return result, {'method': 'stationary lowest foot material points',
                    'foot_geometry_ids': geoms, 'weights': confidence,
                    'root_displacement_m': (result[-1, :3] - result[0, :3]).tolist(),
                    'requires_grounded_motion': True}
