"""Short recorded-pose -> IK -> unchanged policy -> MuJoCo forward simulation.

JSON on stdin; NDJSON progress/result on stdout. No training, external forces,
state projection during rollout, or camera media written to disk.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / '.venv-export/lib/python3.12/site-packages'))
import json
import time
import hashlib
import numpy as np
import mujoco
import onnxruntime as ort
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp
from scipy.signal import savgol_filter
from recording_retargeting import reconstruct_contact_translation
from recording_selection import select_rollout


def emit(kind, **data):
    print(json.dumps({'type': kind, **data}, separators=(',', ':'), allow_nan=False), flush=True)


def load(name):
    return json.loads((ROOT / 'public' / name).read_text())


def prepare_frames(payload):
    frames = payload.get('frames', [])
    if not 15 <= len(frames) <= 76:
        raise ValueError('请录制 0.5–2 秒的全身画面')
    times = np.array([f['timestamp'] / 1000 for f in frames], dtype=float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0) or times[0] < 0 or times[-1] > 2.51:
        raise ValueError('视频时间戳无效，请重新录制')
    valid = [i for i, f in enumerate(frames) if len(f.get('worldLandmarks', [])) == 33 and len(f.get('landmarks', [])) == 33]
    if len(valid) < len(frames) * .8 or len(valid) < 15:
        raise ValueError('人体识别不足 80%，请保持全身入镜、光线充足后重录')
    required = [11, 12, 23, 24, 25, 26, 27, 28]
    visible = [all(frames[i]['landmarks'][j].get('visibility', 1) >= .35 for j in required) for i in valid]
    if np.mean(visible) < .7:
        raise ValueError('肩部、膝盖或脚踝被遮挡，请让全身入镜后重录')
    world = np.array([[[p['x'], p['z'], -p['y']] for p in frames[i]['worldLandmarks']] for i in valid])
    image = np.array([[[p['x'], p['y']] for p in frames[i]['landmarks']] for i in valid])
    if not np.isfinite(world).all() or not np.isfinite(image).all() or np.abs(world).max() > 10 or np.abs(image).max() > 5:
        raise ValueError('姿态坐标无效，请重新录制')
    # Only bridge brief interior gaps. Trim unobserved ends and retain source time.
    vt = times[valid]
    if np.diff(vt).max() > .2 or vt[-1] - vt[0] < .45:
        raise ValueError('人体跟踪中断过久，请保持全身入镜后重录')
    t = np.arange(0, vt[-1] - vt[0] + .00001, 1 / 30)
    interp = lambda a: np.stack([np.interp(t + vt[0], vt, a[:, j, k]) for j in range(a.shape[1]) for k in range(a.shape[2])], axis=1).reshape(len(t), *a.shape[1:])
    return t, interp(world), interp(image), float(vt[0]), len(valid) / len(frames)


def retarget(model, manifest, payload):
    t30, world, image, source_start, coverage = prepare_frames(payload)
    ids = np.array(manifest['site_ids'])
    data = mujoco.MjData(model)
    world = savgol_filter(world, 7, 2, axis=0)
    world -= world[:, [23, 24]].mean(axis=1)[:, None, :]
    up = -world[:5, [27, 28]].mean(axis=(0, 1))
    if np.linalg.norm(up) < .2:
        raise ValueError('未识别到站立姿态，请先站稳再录制')
    up /= np.linalg.norm(up)
    lateral = (world[:5, 23] - world[:5, 24]).mean(axis=0)
    lateral -= up * np.dot(lateral, up)
    if np.linalg.norm(lateral) < .05:
        raise ValueError('身体朝向不清楚，请面向摄像头并让双腿可见')
    lateral /= np.linalg.norm(lateral)
    rotation = np.stack([np.cross(lateral, up), lateral, up])
    world = world @ rotation.T
    mujoco.mj_forward(model, data)
    neutral = data.site_xpos[ids].copy()
    leg = np.linalg.norm(neutral[9] - neutral[10]) + np.linalg.norm(neutral[10] - neutral[11])
    measured = np.mean([np.linalg.norm(world[:, a] - world[:, b], axis=1) + np.linalg.norm(world[:, b] - world[:, c], axis=1) for a, b, c in [(23, 25, 27), (24, 26, 28)]], axis=0)
    scale = leg / np.median(measured)
    if not .4 < scale < 2.5:
        raise ValueError('人体比例估计异常，请调整距离后重录')
    world *= scale
    translation = np.zeros((len(world), 3))
    translation[:, 2] = -np.min(world[:, [27, 28, 29, 30, 31, 32], 2], axis=1) + .045
    world += translation[:, None, :]
    target = np.empty((len(world), 17, 3))
    target[:, 0] = world[:, [23, 24]].mean(axis=1)
    target[:, 1] = target[:, 0] + .45 * (world[:, [11, 12]].mean(axis=1) - target[:, 0])
    target[:, 2] = world[:, [7, 8]].mean(axis=1)
    for index, landmark in {3: 11, 4: 13, 5: 15, 6: 12, 7: 14, 8: 16, 9: 23, 10: 25, 11: 27, 12: 31, 13: 24, 14: 26, 15: 28, 16: 32}.items():
        target[:, index] = world[:, landmark]
    coupled = {int(model.eq_obj1id[e]) for e in range(model.neq) if model.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT}
    independent = [j for j in range(1, model.njnt) if j not in coupled]
    addresses = np.array([model.jnt_qposadr[j] for j in independent])

    def full_qpos(x):
        q = model.qpos0.copy(); q[:3] = x[:3]
        q[3:7] = Rotation.from_rotvec(x[3:6]).as_quat()[[3, 0, 1, 2]]
        q[addresses] = x[6:]
        for _ in range(3):
            for e in range(model.neq):
                if model.eq_type[e] != mujoco.mjtEq.mjEQ_JOINT:
                    continue
                j1, j2 = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
                a1 = model.jnt_qposadr[j1]
                v = q[model.jnt_qposadr[j2]] - model.qpos0[model.jnt_qposadr[j2]] if j2 >= 0 else 0.
                q[a1] = model.qpos0[a1] + np.polynomial.polynomial.polyval(v, model.eq_data[e, :5])
        return q

    lower = np.r_[[-5, -5, .2], [-np.pi] * 3, [model.jnt_range[j, 0] for j in independent]]
    upper = np.r_[[5, 5, 2.2], [np.pi] * 3, [model.jnt_range[j, 1] for j in independent]]
    x = np.clip(np.r_[translation[0], np.zeros(3), model.qpos0[addresses]], lower + 1e-8, upper - 1e-8)
    qs, fit_errors = [], []
    for i, tgt in enumerate(target):
        previous = x.copy()

        def residual(params):
            data.qpos[:] = full_qpos(params); mujoco.mj_kinematics(model, data)
            return np.r_[(data.site_xpos[ids] - tgt).ravel(), (params - previous) * .015]

        fit = least_squares(residual, x, bounds=(lower, upper), max_nfev=35, ftol=1e-4, xtol=1e-4, gtol=1e-4)
        x = fit.x; qs.append(full_qpos(x))
        fit_errors.append(np.linalg.norm(residual(x)[:51].reshape(17, 3), axis=1).mean() * 1000)
        if i % 5 == 0:
            emit('progress', stage='retarget', progress=round(10 + 65 * (i + 1) / len(target)), message=f'拟合肌骨关节 {i + 1}/{len(target)}')
    q30 = np.array(qs)
    t = np.arange(int(np.floor(t30[-1] * 100)) + 1) / 100
    q = np.stack([np.interp(t, t30, q30[:, j]) for j in range(model.nq)], axis=1)
    q[:, 3:7] = Slerp(t30, Rotation.from_quat(q30[:, [4, 5, 6, 3]]))(t).as_quat()[:, [3, 0, 1, 2]]
    q, contact = reconstruct_contact_translation(model, q)
    feet = [i for i in range(model.ngeom) if model.geom_contype[i] and model.body(model.geom_bodyid[i]).name in ['calcn_r', 'calcn_l', 'toes_r', 'toes_l']]
    for qi in q:
        data.qpos[:] = qi; mujoco.mj_kinematics(model, data)
        bottoms = []
        for g in feet:
            row = data.geom_xmat[g].reshape(3, 3)[2]; size = model.geom_size[g]
            support = size[0] + abs(row[2]) * size[1] if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE else np.linalg.norm(row * size)
            bottoms.append(data.geom_xpos[g, 2] - support)
        qi[2] += -min(bottoms) - .0005
    q[:, :2] -= q[0, :2].copy()
    v = np.zeros((len(q), model.nv))
    for i in range(len(q)):
        a, b = max(0, i - 1), min(len(q) - 1, i + 1)
        mujoco.mj_differentiatePos(model, v[i], t[b] - t[a], q[a], q[b])
    sites = []
    for qi in q:
        data.qpos[:] = qi; mujoco.mj_kinematics(model, data); sites.append(data.site_xpos[ids].copy())
    sites = np.array(sites); relative = sites[:, 1:] - sites[:, [0]]
    goals = []
    for i in range(len(q)):
        pieces = [relative[i].ravel()]
        for offset in [20, 40, 60, 80]:
            k = min(i + offset, len(q) - 1)
            pieces += [q[k, :3] - q[i, :3], v[k, :6] - v[i, :6], relative[k].ravel()]
        goals.append(np.concatenate(pieces + [np.array([i / len(q)])]))
    return q, v, goals, sites, {'source_start': source_start, 'coverage': coverage, 'fit_mean_mm': float(np.mean(fit_errors)), 'scale': float(scale), 'camera_rotation': rotation.tolist(), 'root_displacement_m': contact['root_displacement_m']}


def observation(fields, target, manifest):
    obs = np.zeros(manifest['obs_dim'], np.float32)
    for item in manifest['observations']:
        obs[item['output']] = np.asarray(fields[item['field']]).reshape(-1)[item['source']]
    ids = np.array(manifest['site_ids'])
    pos = np.asarray(fields['site_xpos']).reshape(-1, 3)[ids]
    mat = np.asarray(fields['site_xmat']).reshape(-1, 3, 3)[ids]
    com = np.asarray(fields['subtree_com']).reshape(-1, 3)
    cvel = np.asarray(fields['cvel']).reshape(-1, 6)
    vel = []
    for sid, p in zip(ids, pos):
        body = manifest['site_bodyid'][sid]; root = manifest['body_rootid'][body]; cv = cvel[body]
        vel.append(np.r_[cv[:3], cv[3:] - np.cross(p - com[root], cv[:3])])
    vel = np.array(vel); positions, angles, velocities = [], [], []
    for i in range(1, len(ids)):
        rel = mat[0].T @ mat[i]
        positions.extend(pos[i] - pos[0]); angles.extend(Rotation.from_matrix(rel).as_rotvec())
        velocities.extend(np.r_[rel.T @ vel[i, :3] - vel[0, :3], mat[0] @ (vel[0, 3:] - vel[i, 3:])])
    obs[manifest['goal_output_indices']] = np.r_[positions, angles, velocities, target]
    return obs


def policy_bundle(directory, name):
    policy = json.loads((directory / 'policy-state.json').read_text())
    opts = ort.SessionOptions(); opts.intra_op_num_threads = 2; opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(directory / 'actor.onnx'), opts, providers=['CPUExecutionProvider'])
    return {'policy': policy, 'session': session, 'name': name,
            'sha256': hashlib.sha256((directory / 'actor.onnx').read_bytes()).hexdigest()}


def simulate(model, manifest, q, v, goals, sites, *, seed=3, noise_scale=1., bundle=None,
             progress=True, startup='first-interval', consistent_reset=False):
    bundle = bundle or policy_bundle(ROOT / 'public', manifest['checkpoint'])
    policy, session = bundle['policy'], bundle['session']
    d, scratch = mujoco.MjData(model), mujoco.MjData(model)
    initial_velocity = v[0].copy()
    if startup != 'first-interval':
        interval = min(10 if startup == 'joints-100ms' else 15, len(q) - 1)
        averaged = np.zeros(model.nv)
        mujoco.mj_differentiatePos(model, averaged, interval / 100, q[0], q[interval])
        if startup == 'joints-100ms':
            initial_velocity[6:] = averaged[6:]
        elif startup == 'window-150ms':
            initial_velocity[:] = averaged
        else:
            raise ValueError('Unknown initial velocity estimator')
    # Match the native training environment's reset caches. State is initialized once.
    scratch.qpos[:] = q[0]; scratch.qvel[:] = initial_velocity; mujoco.mj_forward(model, scratch)
    mujoco.mj_forward(model, d); d.qpos[:] = q[0]; d.qvel[:] = initial_velocity
    for key in ['xpos', 'xquat', 'cvel']:
        getattr(d, key)[:] = getattr(scratch, key)
    if consistent_reset:
        mujoco.mj_forward(model, d)
    stats = {k: np.array(policy[p], np.float32).reshape(1, -1) for k, p in [('run_mean', 'mean'), ('run_var', 'var')]}
    stats['run_count'] = np.array([policy['count']], np.float32)
    fields = set(item['field'] for item in manifest['observations']) | {'site_xpos', 'site_xmat', 'cvel', 'subtree_com'}
    steps = len(q) - 1
    # Reproducible candidates; preserve the baseline's first 150 noise samples.
    noise = np.random.default_rng(seed).standard_normal((steps, model.nu)).astype(np.float32)
    saved = np.fromfile(ROOT / 'public/noise.f32', dtype=np.float32).reshape(5, 150, model.nu)[seed]
    noise[:min(150, steps)] = saved[:min(150, steps)]
    center, delta, std = [np.array(policy[k]) for k in ['control_mean', 'control_delta', 'std']]
    std = std.astype(np.float32)
    positions, activations, errors, roots, contacts, fall_frames = [], [], [], [], [], []
    policy_times, physics_times = [], []
    started = time.perf_counter()
    for step in range(len(q)):
        scratch.qpos[:] = d.qpos; mujoco.mj_kinematics(model, scratch)
        positions.append(d.qpos.tolist()); activations.append(d.act.tolist()); contacts.append(int(d.ncon))
        errors.append(float(np.linalg.norm(scratch.site_xpos[manifest['site_ids']] - sites[step], axis=1).mean() * 1000))
        roots.append(float(np.linalg.norm(d.qpos[:3] - q[step, :3]) * 1000))
        # Diagnose collapse relative to the same intended pose, including leaning over.
        root_drop = q[step, 2] - d.qpos[2]
        head_drop = sites[step, 2, 2] - scratch.site_xpos[manifest['site_ids'][2], 2]
        fall_frames.append(bool(root_drop > .35 or head_drop > .55))
        if step == steps:
            break
        obs = observation({key: getattr(d, key) for key in fields}, goals[step], manifest)
        tick = time.perf_counter()
        mean, nm, nv, nc = session.run(None, {'obs': obs[None, :], **stats})
        policy_times.append((time.perf_counter() - tick) * 1000)
        stats = dict(run_mean=nm, run_var=nv, run_count=nc)
        sample = mean.reshape(-1) + noise_scale * std * noise[step]
        d.ctrl[:] = np.clip(sample.astype(float) * delta + center, center - delta, center + delta)
        tick = time.perf_counter()
        for _ in range(5):
            mujoco.mj_step(model, d)
        physics_times.append((time.perf_counter() - tick) * 1000)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or abs(d.time - (step + 1) * .01) > 1e-6:
            raise ValueError('本次物理仿真未稳定完成，请改为更小、更慢的动作重录')
        if progress and step % 30 == 0:
            emit('progress', stage='simulate', progress=round(78 + 20 * step / steps), message=f'肌肉驱动物理仿真 {step}/{steps}')
    mean_error, root_max = float(np.mean(errors)), max(roots)
    fall_count = sum(fall_frames)
    fallen = fall_count >= 10
    return {'qpos': positions, 'activation': activations, 'errors_mm': errors, 'root_errors_mm': roots, 'contacts': contacts,
            'metrics': {'mean': mean_error, 'final': errors[-1], 'rootMax': root_max, 'finite': True, 'complete': True,
                        'passed': mean_error < 100 and errors[-1] < 150 and root_max < 200 and not fallen and not any(d.warning.number),
                        'fallen': fallen, 'fallFrames': fall_count,
                        'firstFallSeconds': next((i / 100 for i, fell in enumerate(fall_frames) if fell), None)},
            'timing': {'simulationSeconds': time.perf_counter() - started, 'policyMs': float(np.median(policy_times)), 'physicsMs': float(np.median(physics_times))},
            'warnings': d.warning.number.tolist(),
            'initial_state': {'qpos': q[0].tolist(), 'qvel': initial_velocity.tolist(), 'activation': [0.] * model.na},
            'controller': {'checkpoint': bundle['name'], 'policy_sha256': bundle['sha256'], 'seed': seed,
                           'noise_scale': noise_scale, 'initial_velocity': startup, 'consistent_reset': consistent_reset}}


def simulate_best(model, manifest, q, v, goals, sites):
    """Small, bounded inference search, always retaining the original full rollout."""
    began = time.perf_counter()
    adapted = policy_bundle(ROOT / 'public', manifest['checkpoint'])
    proposals = [(adapted, 1., 'first-interval', False),
                 (adapted, 1.5, 'first-interval', False),
                 (adapted, .5, 'first-interval', False),
                 (adapted, 1., 'joints-100ms', False)]
    base_dir = ROOT / 'inputs/mm-fullbody-base/onnx'
    unavailable = None
    if (base_dir / 'provenance.json').is_file():
        try:
            base = policy_bundle(base_dir, 'amathislab/mm-fullbody-base')
            provenance = json.loads((base_dir / 'provenance.json').read_text())
            if base['sha256'] != provenance['onnx_sha256'] or provenance['flax_actor_max_error'] >= 1e-4:
                raise ValueError('Base policy export verification mismatch')
            proposals.extend([(base, 0., 'joints-100ms', True),
                              (base, 0., 'window-150ms', False),
                              (base, 0., 'first-interval', False)])
        except Exception as error:
            unavailable = str(error)
    candidates, failures = [], []
    for index, (bundle, scale, startup, consistent_reset) in enumerate(proposals):
        if index and time.perf_counter() - began > 12:
            break
        emit('progress', stage='simulate', progress=round(78 + 20 * index / len(proposals)),
             message=f'比较肌肉控制 {index + 1}/{len(proposals)} · 保留同一目标动作')
        try:
            result = simulate(model, manifest, q, v, goals, sites, bundle=bundle, noise_scale=scale,
                              startup=startup, consistent_reset=consistent_reset, progress=False)
            candidates.append(result)
            if result['metrics']['passed']:
                break
        except Exception as error:
            # The baseline must complete; optional alternatives may fail safely.
            if index == 0:
                raise
            failures.append({'checkpoint': bundle['name'], 'noise_scale': scale, 'initial_velocity': startup, 'error': str(error)})
    selected = select_rollout(candidates)
    selected['search'] = {'method': 'complete-rollout-selection', 'reference_unchanged': True,
                          'candidates': [{'controller': r['controller'], 'metrics': r['metrics'],
                                          'seconds': r['timing']['simulationSeconds'], 'warnings': r['warnings']} for r in candidates],
                          'selected': candidates.index(selected), 'failures': failures, 'optional_policy_error': unavailable,
                          'seconds': time.perf_counter() - began,
                          'criterion': 'No increase in mean, final, root maximum error or collapse frames relative to baseline'}
    selected['timing']['searchSeconds'] = selected['search']['seconds']
    return selected


def main():
    started = time.perf_counter()
    payload = json.load(sys.stdin)
    # Validate before loading the expensive model.
    prepare_frames(payload)
    emit('progress', stage='retarget', progress=5, message='准备肌骨模型与足地接触约束…')
    model = mujoco.MjModel.from_xml_path(str(ROOT / 'public/model/model.xml'))
    manifest = load('manifest.json')
    fit_started = time.perf_counter()
    q, v, goals, sites, fit = retarget(model, manifest, payload)
    fit_seconds = time.perf_counter() - fit_started
    result = simulate_best(model, manifest, q, v, goals, sites)
    result.update(schema=1, fps=100, duration=(len(q) - 1) / 100, sourceStart=fit['source_start'], reference_qpos=q.tolist(), fit=fit,
                  checkpoint=result['controller']['checkpoint'], mujoco=mujoco.__version__, trained=False, external_forces=False, hidden_resets=0,
                  policy_sha256=result['controller']['policy_sha256'],
                  assumptions=['Fixed camera; upright start', 'At least one foot grounded; no jumps', 'Monocular depth and contact translation are estimates', 'Frozen released policies with bounded inference search; new motion tracking is experimental'],
                  replay=True)
    result['timing'].update(retargetSeconds=fit_seconds, processingSeconds=time.perf_counter() - started)
    emit('result', result=result)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        emit('error', message=str(error))
        sys.exit(1)
