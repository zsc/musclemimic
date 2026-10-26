import importlib.util,inspect,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('p',ROOT/'reports/tracking-fix/processor-before.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p);p.emit=lambda*a,**k:None
folder=ROOT/'reports/tracking-fix'/sys.argv[1];r=np.load(folder/'reference.npz');q,v,g,s=[r[k] for k in ['q','v','g','s']];m=p.mujoco.MjModel.from_xml_path(str(ROOT/'public/model/model.xml'));mf=p.load('manifest.json');source=inspect.getsource(p.simulate);summary=[]
for window,root_average,correct,smooth in [(5,False,False,0),(10,False,True,0),(15,False,False,0),(20,False,False,0),(10,True,False,0),(15,True,False,0),(10,False,False,11),(10,False,False,21),(10,False,True,21),(10,True,True,21)]:
 average=np.zeros(m.nv);p.mujoco.mj_differentiatePos(m,average,window*.01,q[0],q[window]);initial=v[0].copy();initial[6:]=average[6:]
 if root_average:initial[:6]=average[:6]
 goals=np.array(g).copy()
 if smooth:
  filtered=p.savgol_filter(v[:,:6],smooth,2,axis=0)
  for i in range(len(q)):
   for k,offset in enumerate([20,40,60,80]):goals[i,48+k*57+3:48+k*57+9]=filtered[min(i+offset,len(q)-1)]-filtered[i]
 code=source.replace('qvel[:] = v[0]','qvel[:] = initial')
 if correct:code=code.replace("getattr(d, key)[:] = getattr(scratch, key)","getattr(d, key)[:] = getattr(scratch, key)\n    mujoco.mj_forward(model, d)")
 code=code.replace("ROOT / 'public/actor.onnx'", "ROOT / 'inputs/mm-fullbody-base/onnx/actor.onnx'").replace('std * noise[step]','0 * std * noise[step]')
 scope=dict(p.__dict__);scope['initial']=initial;scope['load']=lambda name:json.loads((ROOT/'inputs/mm-fullbody-base/onnx'/name).read_text());exec(code,scope);result=scope['simulate'](m,mf,q,v,goals,s)
 name=f'robust-{window}-{root_average}-{correct}-{smooth}';(folder/f'{name}.json').write_text(json.dumps(result));row={'variant':name,**result['metrics']};summary.append(row);print(json.dumps(row),flush=True)
(folder/'robust-summary.json').write_text(json.dumps(summary,indent=2))
