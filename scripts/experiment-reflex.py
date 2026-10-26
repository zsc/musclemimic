"""Muscle length/velocity feedback around a frozen neural policy."""
import importlib.util,inspect,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('p',ROOT/'reports/tracking-fix/processor-before.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p);p.emit=lambda*a,**k:None
m=p.mujoco.MjModel.from_xml_path(str(ROOT/'public/model/model.xml'));mf=p.load('manifest.json');r=np.load(ROOT/'reports/tracking-fix/reference.npz');q,v,g,s=[r[k] for k in ['q','v','g','s']]
target=p.mujoco.MjData(m);lengths=[];velocities=[]
for qi,vi in zip(q,v):
 target.qpos[:]=qi;target.qvel[:]=vi;p.mujoco.mj_forward(m,target);lengths.append(target.actuator_length.copy());velocities.append(target.actuator_velocity.copy())
lengths=np.array(lengths);velocities=np.array(velocities);ranges=np.maximum(np.diff(m.actuator_lengthrange,axis=1).ravel(),.01)
source=inspect.getsource(p.simulate);summary=[]
for kp,kd in [(k,0) for k in [.25,.5,1,2,4,8]]+[(k,.05) for k in [.5,1,2,4]]:
 code=source.replace('tick = time.perf_counter()\n        for _',f'correction=({kp}*(d.actuator_length-lengths[step])+{kd}*(d.actuator_velocity-velocities[step]))/ranges\n        d.ctrl[:]=np.clip(d.ctrl+correction,center-delta,center+delta)\n        tick = time.perf_counter()\n        for _')
 scope=dict(p.__dict__);scope.update(lengths=lengths,velocities=velocities,ranges=ranges);exec(code,scope);result=scope['simulate'](m,mf,q,v,g,s)
 name=f'reflex-{kp}-{kd}';(ROOT/f'reports/tracking-fix/{name}.json').write_text(json.dumps(result));row={'variant':name,'kp':kp,'kd':kd,**result['metrics']};summary.append(row);print(json.dumps(row),flush=True)
(ROOT/'reports/tracking-fix/reflex-summary.json').write_text(json.dumps(summary,indent=2))
