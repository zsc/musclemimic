"""Bounded online shooting with an unchanged policy and continuous physical state."""
import importlib.util,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('processor',ROOT/'reports/tracking-fix/processor-before.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
m=p.mujoco.MjModel.from_xml_path(str(ROOT/'public/model/model.xml'));mf=p.load('manifest.json')
r=np.load(ROOT/'reports/tracking-fix/reference.npz');q,v,g,s=[r[k] for k in ['q','v','g','s']]
policy=p.load('policy-state.json');opts=p.ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
session=p.ort.InferenceSession(str(ROOT/'public/actor.onnx'),opts,providers=['CPUExecutionProvider'])
fields=set(item['field'] for item in mf['observations'])|{'site_xpos','site_xmat','cvel','subtree_com'}
center,delta,std=[np.array(policy[k]) for k in ['control_mean','control_delta','std']];std=std.astype(np.float32)
steps=len(q)-1
noise=np.random.default_rng(3).standard_normal((5,steps,m.nu)).astype(np.float32)
noise[:,:min(150,steps)]=np.fromfile(ROOT/'public/noise.f32',dtype=np.float32).reshape(5,150,m.nu)[:,:min(150,steps)]

def run(horizon=25,commit=10):
 d,scratch=p.mujoco.MjData(m),p.mujoco.MjData(m)
 scratch.qpos[:]=q[0];scratch.qvel[:]=v[0];p.mujoco.mj_forward(m,scratch)
 p.mujoco.mj_forward(m,d);d.qpos[:]=q[0];d.qvel[:]=v[0]
 for key in ['xpos','xquat','cvel']:getattr(d,key)[:]=getattr(scratch,key)
 stats={k:np.array(policy[z],np.float32).reshape(1,-1) for k,z in [('run_mean','mean'),('run_var','var')]};stats['run_count']=np.array([policy['count']],np.float32)
 runs=[];decisions=[];started=time.perf_counter()
 def record(data,i):
  scratch.qpos[:]=data.qpos;p.mujoco.mj_kinematics(m,scratch)
  error=float(np.linalg.norm(scratch.site_xpos[mf['site_ids']]-s[i],axis=1).mean()*1000)
  root=float(np.linalg.norm(data.qpos[:3]-q[i,:3])*1000)
  return dict(qpos=data.qpos.tolist(),activation=data.act.tolist(),errors_mm=error,root_errors_mm=root,contacts=int(data.ncon))
 runs.append(record(d,0))
 for start in range(0,steps,commit):
  candidates=[];count=min(commit,steps-start)
  for seed,scale in [(3,1),(0,1),(1,1),(2,1),(4,1),(3,0)]:
   trial=p.mujoco.MjData(m);p.mujoco.mj_copyData(trial,m,d);st={k:x.copy() for k,x in stats.items()};prefix=[];err=[];roots=[];save=None
   for i in range(start,min(start+horizon,steps)):
    obs=p.observation({k:getattr(trial,k) for k in fields},g[i],mf)
    mean,nm,nv,nc=session.run(None,{'obs':obs[None,:],**st});st=dict(run_mean=nm,run_var=nv,run_count=nc)
    sample=mean.reshape(-1)+scale*std*noise[seed,i]
    trial.ctrl[:]=np.clip(sample.astype(float)*delta+center,center-delta,center+delta)
    for _ in range(5):p.mujoco.mj_step(m,trial)
    frame=record(trial,i+1);err.append(frame['errors_mm']);roots.append(frame['root_errors_mm'])
    if i-start<count:prefix.append(frame)
    if i-start+1==count:
     save=p.mujoco.MjData(m);p.mujoco.mj_copyData(save,m,trial);saved_stats={k:x.copy() for k,x in st.items()}
   cost=err[-1]+.25*np.mean(err)+.3*max(roots)
   candidates.append((cost,save,saved_stats,prefix,seed,scale))
  cost,d,stats,prefix,seed,scale=min(candidates,key=lambda x:x[0]);runs.extend(prefix);decisions.append(dict(frame=start,seed=seed,scale=scale,cost=cost))
  if start%50==0:print('step',start,'selected',seed,scale,'error',round(runs[-1]['errors_mm']),flush=True)
 result={k:[x[k] for x in runs] for k in runs[0]};errors=result['errors_mm'];result['metrics']={'mean':np.mean(errors),'final':errors[-1],'rootMax':max(result['root_errors_mm'])};result['seconds']=time.perf_counter()-started;result['decisions']=decisions
 return result
r=run();(ROOT/'reports/tracking-fix/lookahead.json').write_text(json.dumps(r));print(r['metrics'],r['seconds'],flush=True)
