"""Rerun frozen-policy inference against an archived reference, unchanged."""
import importlib.util,inspect,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('p',ROOT/'reports/tracking-fix/processor-before.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p);p.emit=lambda*a,**k:None
file=Path(sys.argv[1]);old=json.loads(file.read_text());q=np.array(old['reference_qpos']);m=p.mujoco.MjModel.from_xml_path(str(ROOT/'public/model/model.xml'));mf=p.load('manifest.json');v=np.zeros((len(q),m.nv));d=p.mujoco.MjData(m);s=[]
for i in range(len(q)):
 a,b=max(0,i-1),min(len(q)-1,i+1);p.mujoco.mj_differentiatePos(m,v[i],(b-a)*.01,q[a],q[b]);d.qpos[:]=q[i];p.mujoco.mj_kinematics(m,d);s.append(d.site_xpos[mf['site_ids']].copy())
s=np.array(s);relative=s[:,1:]-s[:,[0]];g=[]
for i in range(len(q)):
 parts=[relative[i].ravel()]
 for offset in [20,40,60,80]:
  k=min(i+offset,len(q)-1);parts += [q[k,:3]-q[i,:3],v[k,:6]-v[i,:6],relative[k].ravel()]
 g.append(np.concatenate(parts+[np.array([i/len(q)])]))
out=ROOT/'reports/tracking-fix'/file.parent.name;out.mkdir(exist_ok=True);np.savez(out/'reference.npz',q=q,v=v,g=g,s=s)
source=inspect.getsource(p.simulate);summary=[]
for name,scale,seed,base in [('baseline',1,3,False),('half-noise',.5,3,False),('quarter-noise',.25,3,False),('more-noise',1.5,3,False),('seed1',1,1,False),('seed4',1,4,False),('base',0,3,True)]:
 code=source.replace('std * noise[step]',f'{scale} * std * noise[step]').replace('default_rng(3)',f'default_rng({seed})').replace('model.nu)[3]',f'model.nu)[{seed}]')
 scope=dict(p.__dict__)
 if base:
  code=code.replace("ROOT / 'public/actor.onnx'", "ROOT / 'reports/tracking-fix/base-policy/actor.onnx'");scope['load']=lambda name:json.loads((ROOT/'reports/tracking-fix/base-policy'/name).read_text())
 exec(code,scope);result=scope['simulate'](m,mf,q,v,g,s);(out/f'{name}.json').write_text(json.dumps(result));row={'variant':name,**result['metrics']};summary.append(row);print(json.dumps(row),flush=True)
(out/'summary.json').write_text(json.dumps(summary,indent=2))
