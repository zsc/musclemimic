import importlib.util,inspect,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('p',ROOT/'reports/tracking-fix/processor-before.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p);p.emit=lambda*a,**k:None
folder=ROOT/'reports/tracking-fix'/sys.argv[1];r=np.load(folder/'reference.npz');q,v,g,s=[r[k] for k in ['q','v','g','s']];m=p.mujoco.MjModel.from_xml_path(str(ROOT/'public/model/model.xml'));mf=p.load('manifest.json');source=inspect.getsource(p.simulate);summary=[]
for avg,activation,base in [(False,k,False) for k in [.25,.5,1]]+[(True,k,False) for k in [0,.25,.5,1]]+[(True,k,True) for k in [0,.5,1]]:
 initial=v[0].copy()
 if avg:
  average=np.zeros(m.nv);p.mujoco.mj_differentiatePos(m,average,.1,q[0],q[10]);initial[6:]=average[6:]
 code=source.replace('qvel[:] = v[0]','qvel[:] = initial')
 code=code.replace('tick = time.perf_counter()\n        for _', f'if step == 0: d.act[:] = d.ctrl * {activation}\n        tick = time.perf_counter()\n        for _')
 scope=dict(p.__dict__);scope['initial']=initial
 if base:
  code=code.replace("ROOT / 'public/actor.onnx'", "ROOT / 'inputs/mm-fullbody-base/onnx/actor.onnx'");scope['load']=lambda name:json.loads((ROOT/'inputs/mm-fullbody-base/onnx'/name).read_text());code=code.replace('std * noise[step]','0 * std * noise[step]')
 exec(code,scope);result=scope['simulate'](m,mf,q,v,g,s);name=f'warm-{avg}-{activation}-{base}';(folder/f'{name}.json').write_text(json.dumps(result));row={'variant':name,**result['metrics']};summary.append(row);print(json.dumps(row),flush=True)
(folder/'warm-summary.json').write_text(json.dumps(summary,indent=2))
