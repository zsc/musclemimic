"""Export the existing actor and model on air5; never run on the local workstation."""
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / 'inputs/musclemimic').is_dir():
    raise RuntimeError('Run this exporter in the air5 project containing the copied inputs')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
sys.path.insert(0, str(ROOT / 'inputs/scripts'))

import jax
import jax.numpy as jnp
import numpy as np
import mujoco
import musclemimic_models
import onnx
from onnx import helper as H, numpy_helper as NH, TensorProto as T
import onnxruntime as ort
from omegaconf import OmegaConf
from loco_mujoco.task_factories import CustomDatasetConf, ImitationFactory
from loco_mujoco.trajectory import Trajectory
from musclemimic.algorithms import PPOJax
from musclemimic.runner.eval_utils import load_checkpoint, align_agent_state, apply_temporal_params
from loco_mujoco.core.utils.math import calculate_relative_site_quantities
import muscle_initialization
import world_site_reward


def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if hasattr(x, 'tolist'):
        return x.tolist()
    return x


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(value), ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')


def export_actor(params, stats, config, obs_dim, destination):
    """ONNX includes the same per-observation running-statistics update as Flax."""
    nodes, constants = [], []
    def const(name, value):
        constants.append(NH.from_array(np.asarray(value, dtype=np.float32), name))
        return name
    def op(kind, inputs, name, **attrs):
        nodes.append(H.make_node(kind, inputs, [name], **attrs))
        return name
    one = const('one', [1.0])
    new_count = op('Add', ['run_count', one], 'next_count')
    delta = op('Sub', ['obs', 'run_mean'], 'delta')
    next_mean = op('Add', ['run_mean', op('Div', [delta, new_count], 'mean_increment')], 'next_mean')
    ma = op('Mul', ['run_var', 'run_count'], 'm_a')
    delta2 = op('Mul', [delta, delta], 'delta_squared')
    correction = op('Div', [op('Mul', [delta2, 'run_count'], 'delta_count'), new_count], 'variance_correction')
    m2 = op('Add', [op('Add', [ma, const('batch_epsilon', [1e-6])], 'm_ab'), correction], 'm2')
    next_var = op('Div', [m2, new_count], 'next_var')
    denom = op('Sqrt', [op('Add', [next_var, const('normalization_epsilon', [1e-8])], 'var_epsilon')], 'std')
    x = op('Div', [op('Sub', ['obs', next_mean], 'centered'), denom], 'normalized')
    actor = params['actor']
    def dense(x, name):
        block = actor[name]
        w, b = const(name + '_weight', block['kernel']), const(name + '_bias', block['bias'])
        return op('Add', [op('MatMul', [x, w], name + '_matmul'), b], name + '_out')
    def ln(x, name):
        block = actor[name]
        mean = op('ReduceMean', [x], name + '_mean', axes=[-1], keepdims=1)
        sq = op('ReduceMean', [op('Mul', [x, x], name + '_square')], name + '_mean_square', axes=[-1], keepdims=1)
        var = op('Max', [op('Sub', [sq, op('Mul', [mean, mean], name + '_square_mean')], name + '_variance'), const(name + '_zero', [0])], name + '_variance_clipped')
        norm = op('Div', [op('Sub', [x, mean], name + '_centered'), op('Sqrt', [op('Add', [var, const(name + '_epsilon', [config.layernorm_eps])], name + '_var_eps')], name + '_std')], name + '_normalized')
        return op('Add', [op('Mul', [norm, const(name + '_scale', block['scale'])], name + '_scaled'), const(name + '_bias', block['bias'])], name + '_out')
    def silu(x, name):
        return op('Mul', [x, op('Sigmoid', [x], name + '_sigmoid')], name)
    assert config.use_residual and config.residual_type == 'gated' and config.activation == 'silu'
    for i in range(len(config.actor_hidden_layers) // 2):
        prefix = f'block{i}'
        h = silu(ln(dense(x, prefix + '_layer0_dense'), prefix + '_layer0_ln'), prefix + '_activation0')
        h = ln(dense(h, prefix + '_layer1_dense'), prefix + '_layer1_ln')
        skip = dense(x, prefix + '_proj') if prefix + '_proj' in actor else x
        gate = const(prefix + '_gate', jax.nn.sigmoid(params['actor'][f'res_gate_{i}']))
        x = silu(op('Add', [skip, op('Mul', [gate, h], prefix + '_weighted')], prefix + '_sum'), prefix + '_activation1')
    if len(config.actor_hidden_layers) % 2:
        x = silu(ln(dense(x, 'tail_dense'), 'tail_ln'), 'tail_activation')
    op('Identity', [dense(x, 'output')], 'action_mean')
    inputs = [H.make_tensor_value_info(n, T.FLOAT, shape) for n, shape in [('obs', [1, obs_dim]), ('run_mean', [1, obs_dim]), ('run_var', [1, obs_dim]), ('run_count', [1])]]
    outputs = [H.make_tensor_value_info(n, T.FLOAT, shape) for n, shape in [('action_mean', [1, len(params['log_std'])]), ('next_mean', [1, obs_dim]), ('next_var', [1, obs_dim]), ('next_count', [1])]]
    model = H.make_model(H.make_graph(nodes, 'musclemimic_actor_with_running_statistics', inputs, outputs, constants), opset_imports=[H.make_opsetid('', 17)], producer_name='musclemimic-browser-export')
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, destination)


def main():
    began = time.monotonic()
    public = ROOT / 'public'
    assets = public / 'model/assets'
    assets.mkdir(parents=True, exist_ok=True)
    checkpoint = next((ROOT / 'inputs/outputs/temporal_contact_lowstd025_v1/checkpoints').glob('*/checkpoint_13095'))
    reference = ROOT / 'inputs/outputs/temporal_contact_v3/reference_native/musclemimic_trajectory.npz'
    cfg, agent_state, _ = load_checkpoint(str(checkpoint))
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.set_struct(cfg, False)
    apply_temporal_params(cfg)
    env_params = OmegaConf.to_container(cfg.experiment.env_params, resolve=True)
    env_params.update(env_name='MyoFullBody', headless=True, init_state_type='TrajInitialStateHandler', init_state_params={}, th_params=dict(random_start=False, start_from_random_step=False, fixed_start_conf=[0, 10]))
    env_params.pop('num_envs', None)
    env_params.pop('mjx_backend', None)
    traj = Trajectory.load(str(reference), backend=np)
    env = ImitationFactory.make(**env_params, custom_dataset_conf=CustomDatasetConf(traj=traj, trajectory_cache_type='sparse'))
    agent_conf = PPOJax.init_agent_conf(env, cfg)
    state = align_agent_state(agent_state, agent_conf).train_state
    obs = env.reset()
    m, d = env.model, env.data
    print('ENV', m.nq, m.nv, m.nu, len(obs), list(state.params['actor']), flush=True)
    stats = state.run_stats['RunningMeanStd_0']
    write_json(ROOT / 'reports/export_structure.json', dict(model=dict(nq=m.nq, nv=m.nv, nu=m.nu), obs_dim=len(obs), actor={k:jax.tree.map(lambda a:list(a.shape),v) for k,v in state.params['actor'].items()}, run_stats=stats, env_attributes=list(vars(env))))
    export_actor(state.params, stats, cfg.experiment, len(obs), public / 'actor.onnx')
    session = ort.InferenceSession(str(public / 'actor.onnx'), providers=['CPUExecutionProvider'])

    spec = env._mjspec
    xml = ET.fromstring(spec.to_xml())
    size = xml.find('size')
    if size is not None:
        size.attrib.pop('njmax', None)
        size.attrib.pop('nconmax', None)
        size.set('memory', '64M')
    compiler = xml.find('compiler')
    source_root = Path(musclemimic_models.get_xml_path('myofullbody')).parent
    package_root = Path(musclemimic_models.__file__).parent
    source_files = []
    for tag in xml.findall('.//asset/mesh') + xml.findall('.//asset/hfield'):
        if 'file' not in tag.attrib:
            continue
        filename = tag.attrib['file']
        candidates = [Path(filename), source_root / filename]
        meshdir = getattr(spec, 'meshdir', '')
        candidates += [source_root / meshdir / filename, Path(meshdir) / filename]
        found = next((p for p in candidates if p.is_file()), None)
        if found is None:
            matches = list(package_root.rglob(Path(filename).name))
            if len(matches) != 1:
                raise RuntimeError(f'Ambiguous asset {filename}: {matches}')
            found = matches[0]
        digest = hashlib.sha256(found.read_bytes()).hexdigest()
        target = assets / (digest[:12] + '_' + found.name)
        if not target.exists():
            shutil.copyfile(found, target)
        tag.set('file', 'assets/' + target.name)
        source_files.append(dict(path='model/assets/' + target.name, sha256=digest))
    asset = xml.find('asset')
    if asset is not None:
        for texture in list(asset.findall('texture')):
            asset.remove(texture)
        for material in asset.findall('material'):
            material.attrib.pop('texture', None)
    if compiler is not None:
        compiler.attrib.pop('meshdir', None)
        compiler.attrib.pop('texturedir', None)
    ET.ElementTree(xml).write(public / 'model/model.xml', encoding='unicode')
    exported = mujoco.MjModel.from_xml_path(str(public / 'model/model.xml'))
    assert (exported.nq, exported.nv, exported.nu) == (m.nq, m.nv, m.nu)
    differences = {}
    for name in dir(m):
        if name.startswith('_'): continue
        a,b = getattr(m,name),getattr(exported,name)
        if isinstance(a,np.ndarray) and a.shape == b.shape and a.dtype.kind in 'fiu':
            err = float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.size else 0
            if err > 1e-9: differences[name] = err
    option_differences = {}
    for name in dir(m.opt):
        if name.startswith('_'): continue
        a,b = getattr(m.opt,name),getattr(exported.opt,name)
        if isinstance(a,(int,float,np.ndarray)) and not np.array_equal(a,b): option_differences[name] = [plain(a),plain(b)]
    write_json(ROOT / 'reports/model-roundtrip-differences.json',dict(arrays=differences,options=option_differences))
    assert not option_differences, option_differences
    for field in ['body_mass','body_inertia','body_pos','body_quat','body_ipos','body_iquat','jnt_range','dof_damping','dof_armature','actuator_gainprm','actuator_biasprm','actuator_dynprm','actuator_ctrlrange','actuator_lengthrange','geom_friction','geom_solref','geom_solimp','geom_pos','geom_quat','geom_size','site_pos']:
        np.testing.assert_allclose(getattr(m,field),getattr(exported,field),rtol=1e-5,atol=1e-6,err_msg=field)
    # Mesh rendering uses the compiled local coordinates and actual geom transforms.
    geometries = []
    for mesh_id in range(m.nmesh):
        va, vn = m.mesh_vertadr[mesh_id], m.mesh_vertnum[mesh_id]
        fa, fn = m.mesh_faceadr[mesh_id], m.mesh_facenum[mesh_id]
        geometries.append(dict(vertices=m.mesh_vert[va:va+vn], faces=m.mesh_face[fa:fa+fn]))
    render_geoms = [dict(id=i, mesh=int(m.geom_dataid[i]), rgba=m.geom_rgba[i]) for i in range(m.ngeom) if int(m.geom_type[i]) == int(mujoco.mjtGeom.mjGEOM_MESH) and m.geom_rgba[i, 3] > 0]
    write_json(public / 'geometry.json', dict(meshes=geometries, geoms=render_geoms))

    goal = env.obs_container.list_all_stateful()[0]
    simple = []
    for ob in env.obs_container.values():
        if ob in env.obs_container.list_all_stateful():
            continue
        simple.append(dict(kind=type(ob).__name__, field=ob.data_type(), source=ob.data_type_ind, output=ob.obs_ind))
    site_ids = np.asarray([m.site(n).id for n in env.sites_for_mimic])
    target_goals = []
    original_carry = env._additional_carry
    for frame in range(len(traj.data.qpos)):
        carry = original_carry.replace(traj_state=original_carry.traj_state.replace(subtraj_step_no=frame))
        go, _ = goal.get_obs_and_update_state(env, m, d, carry, np)
        target_goals.append(go[192:])
    write_json(public / 'reference.json', dict(qpos=traj.data.qpos, qvel=traj.data.qvel, goal=target_goals, fps=100))

    fixtures, onnx_errors, physics_steps = [], [], []
    mean_stats, var_stats, count_stats = (np.asarray(stats[k], dtype=np.float32) for k in ['mean', 'var', 'count'])
    feeds_stats = dict(run_mean=mean_stats.reshape(1,-1), run_var=var_stats.reshape(1,-1), run_count=count_stats.reshape(1))
    raw_fields = sorted(set(x['field'] for x in simple) | {'site_xpos', 'site_xmat', 'cvel', 'subtree_com'})
    control_mean = np.asarray(env._control_func.norm_act_mean)
    control_delta = np.asarray(env._control_func.norm_act_delta)
    rng = jax.random.key(3)
    @jax.jit
    def infer(params, running, ob, key):
        (dist, _), updates = agent_conf.network.apply({'params':params, 'run_stats':running}, jnp.atleast_2d(ob), mutable=['run_stats'])
        return dist.mean(), dist.sample(seed=key), updates['run_stats']
    initial = dict(qpos=d.qpos.copy(), qvel=d.qvel.copy(), act=d.act.copy(), ctrl=d.ctrl.copy())
    for step in range(20):
        rng, key = jax.random.split(rng)
        mean, action, running = infer(state.params, state.run_stats, obs, key)
        result = session.run(None, dict(obs=np.asarray(obs, dtype=np.float32).reshape(1,-1), **feeds_stats))
        error = float(np.max(np.abs(result[0].reshape(-1)-np.asarray(mean).reshape(-1))))
        onnx_errors.append(error)
        if step in [0,1,2,10,19]:
            fixtures.append(dict(step=step, reference_frame=10+step, obs=obs, mean=mean, action=action, fields={k:np.asarray(getattr(d,k)).reshape(-1).copy() for k in raw_fields}))
        feeds_stats = dict(run_mean=result[1], run_var=result[2], run_count=result[3])
        state = state.replace(run_stats=running)
        obs, _, _, _, _ = env.step(jnp.atleast_2d(action))
        physics_steps.append(dict(ctrl=d.ctrl.copy(),qpos=d.qpos.copy(),qvel=d.qvel.copy()))
    assert max(onnx_errors) < 5e-4, onnx_errors
    write_json(public / 'validation-fixtures.json', fixtures)
    write_json(ROOT / 'reports/native-physics-34.json', dict(initial=initial, steps=physics_steps,version=mujoco.__version__))

    import distrax
    noises = []
    normal = distrax.MultivariateNormalDiag(jnp.zeros(m.nu), jnp.ones(m.nu))
    for seed in range(5):
        key = jax.random.key(seed)
        seed_noises = []
        for step in range(150):
            key, subkey = jax.random.split(key)
            seed_noises.append(np.asarray(normal.sample(seed=subkey)))
        noises.append(seed_noises)
    for fixture in fixtures:
        expected = np.asarray(fixture['mean']).reshape(-1) + np.asarray(jnp.exp(state.params['log_std'])) * noises[3][fixture['step']]
        assert np.max(np.abs(expected - np.asarray(fixture['action']).reshape(-1))) < 2e-6
    np.asarray(noises, dtype='<f4').tofile(public / 'noise.f32')
    write_json(public / 'policy-state.json', dict(mean=mean_stats, var=var_stats, count=count_stats, std=np.asarray(jnp.exp(state.params['log_std'])), control_mean=control_mean, control_delta=control_delta))
    reset_states = {}
    for start in [0,5,10]:
        ep = dict(env_params, th_params=dict(random_start=False, start_from_random_step=False, fixed_start_conf=[0,start]))
        e = ImitationFactory.make(**ep, custom_dataset_conf=CustomDatasetConf(traj=traj, trajectory_cache_type='sparse'))
        o = e.reset()
        reset_states[str(start)] = dict(qpos=e.data.qpos.copy(), qvel=e.data.qvel.copy(), act=e.data.act.copy(), ctrl=e.data.ctrl.copy(), obs=o, fields={k:np.asarray(getattr(e.data,k)).reshape(-1).copy() for k in raw_fields})
    write_json(public / 'reset-states.json', reset_states)
    files = [{'path':'model/model.xml', 'sha256':hashlib.sha256((public/'model/model.xml').read_bytes()).hexdigest()}] + list({x['path']:x for x in source_files}.values())
    manifest = dict(schema_version=1, title='MuscleMimic browser demo', native_mujoco_version=mujoco.__version__, wasm_mujoco_version='3.17.0', nq=m.nq, nv=m.nv, nu=m.nu, nbody=m.nbody, nsite=m.nsite, ngeom=m.ngeom, obs_dim=len(fixtures[0]['obs']), control_dt=.01, physics_dt=.002, substeps=5, frames=151, model_files=files, observations=simple, goal_output_indices=goal.obs_ind, site_ids=site_ids, site_bodyid=m.site_bodyid, body_rootid=m.body_rootid, body_parentid=m.body_parentid, actuator_trnid=m.actuator_trnid, initial=initial, checkpoint='checkpoint_13095', reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(), noise=dict(seeds=5,steps=150,muscles=m.nu,format='float32 little endian',description='JAX standard-normal draws only; actions are recomputed from live observations'), original_acceptance=dict(passed=13,total=15,all_passed=False), gates=dict(mean_mm=100,final_mm=150,root_max_mm=200))
    write_json(public / 'manifest.json', manifest)
    shutil.copyfile(ROOT / 'inputs/opensim_moco/inputs/pose2sim/cam01.mp4', public / 'source.mp4')
    notice_paths = list(package_root.rglob('*NOTICE*')) + list(package_root.rglob('*LICENSE*'))
    (public / 'licenses').mkdir(exist_ok=True)
    for p in notice_paths:
        if p.is_file(): shutil.copyfile(p, public / 'licenses' / p.name)
    shutil.copyfile(ROOT / 'inputs/musclemimic/LICENSE', public / 'licenses/MuscleMimic-LICENSE')
    distribution = importlib.metadata.distribution('musclemimic-models')
    for file in distribution.files:
        if '/licenses/' in str(file):
            shutil.copyfile(distribution.locate_file(file), public / 'licenses' / ('musclemimic-models-' + Path(file).name))
    validation = dict(export_complete=True, actor_onnx_max_absolute_error=max(onnx_errors), actor_onnx_tolerance=5e-4, actor_onnx_cases=len(onnx_errors), model_dimensions_preserved=True, native_mujoco=mujoco.__version__, browser_verified=False, original_reliability_passed=13, original_reliability_total=15, elapsed_seconds=time.monotonic()-began)
    write_json(public / 'validation.json', validation)
    print('EXPORT', json.dumps(validation), flush=True)


if __name__ == '__main__':
    main()
