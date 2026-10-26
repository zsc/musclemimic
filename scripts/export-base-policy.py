"""Export the released base policy unchanged; no optimizer or training loop."""
import os
os.environ['JAX_PLATFORMS']='cpu'
import ast,json,hashlib
from pathlib import Path
import jax,numpy as np,onnx,orbax.checkpoint as ocp,onnxruntime as ort
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant,orthogonal
from typing import Sequence
from onnx import helper as H,numpy_helper as NH,TensorProto as T
from omegaconf import OmegaConf
ROOT=Path(__file__).resolve().parents[1];dest=ROOT/'inputs/mm-fullbody-base/onnx';dest.mkdir(exist_ok=True)
checkpoint=ROOT/'inputs/mm-fullbody-base'
with ocp.Checkpointer(ocp.StandardCheckpointHandler()) as checkpointer:
 state=checkpointer.restore(checkpoint/'train_state',args=ocp.args.StandardRestore(item=None,fallback_sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0])))
print('state',list(state),'params',list(state['params']),flush=True)
params=state['params'];stats=state['run_stats']['RunningMeanStd_0'];config=OmegaConf.create(json.loads((checkpoint/'config/metadata').read_text()))
obs_dim=np.asarray(stats['mean']).size
print('obs',obs_dim,'stats shapes',jax.tree.map(lambda x:x.shape,stats),'layers',jax.tree.map(lambda x:x.shape,params['actor']),flush=True)
# Reuse the verified conversion function, without executing the environment exporter.
module=ast.parse((ROOT/'scripts/export_assets.py').read_text());function=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='export_actor')
exec(compile(ast.Module(body=[function],type_ignores=[]),'<export_actor>','exec'))
export_actor(params,stats,config.experiment,obs_dim,dest/'actor.onnx')
old=json.loads((ROOT/'public/policy-state.json').read_text())
for key,value in [('mean',stats['mean']),('var',stats['var']),('count',stats['count']),('std',jax.numpy.exp(params['log_std']))]:old[key]=np.asarray(value).tolist()
(dest/'policy-state.json').write_text(json.dumps(old))
# Validate against the released Flax actor implementation, not a second ONNX export.
network=ast.parse((ROOT/'inputs/musclemimic/musclemimic/algorithms/common/networks.py').read_text())
selected=[n for n in network.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ['get_activation_fn','ResidualFCNet']]
exec(compile(ast.Module(body=selected,type_ignores=[]),'<released-flax-actor>','exec'))
actor=ResidualFCNet(hidden_layer_dims=tuple(config.experiment.actor_hidden_layers),output_dim=354,activation='silu',use_layernorm=True,layernorm_eps=config.experiment.layernorm_eps,residual_type='gated')
session=ort.InferenceSession(str(dest/'actor.onnx'),providers=['CPUExecutionProvider'])
maximum=0.
for seed in range(3):
    obs=np.array(stats['mean'],np.float32)[None]+np.random.default_rng(seed).normal(size=(1,obs_dim)).astype(np.float32)*np.sqrt(np.array(stats['var'],np.float32))[None]
    result,nm,nv,nc=session.run(None,{'obs':obs,'run_mean':np.array(stats['mean'],np.float32)[None],'run_var':np.array(stats['var'],np.float32)[None],'run_count':np.array([stats['count']],np.float32)})
    normalized=(obs-nm)/np.sqrt(nv+1e-8)
    expected=np.array(actor.apply({'params':params['actor']},jnp.array(normalized)))
    maximum=max(maximum,float(np.max(np.abs(result-expected))))
assert maximum<1e-4,maximum
(dest/'provenance.json').write_text(json.dumps({'source':'https://huggingface.co/amathislab/mm-fullbody-base','trained':False,'onnx_sha256':hashlib.sha256((dest/'actor.onnx').read_bytes()).hexdigest(),'flax_actor_max_error':maximum,'obs_dim':obs_dim},indent=2))
print('PARITY',maximum,flush=True)
print('EXPORTED',dest,flush=True)
