"""Compare reference fitting assumptions without changing muscle policy weights."""
import importlib.util,inspect,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('processor',ROOT/'reports/tracking-fix/processor-before.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p);p.emit=lambda *a,**k:None
m=p.mujoco.MjModel.from_xml_path(str(ROOT/'public/model/model.xml'));mf=p.load('manifest.json');payload=json.loads((ROOT/'reports/no-finetune/poses.json').read_text())
source=inspect.getsource(p.retarget)
summary=[]
for name,vertical,prior,temporal in [('vertical',True,0,.015),('prior',False,.06,.06),('vertical-prior',True,.06,.06),('vertical-strong',True,.15,.1)]:
 code=source
 if vertical:code=code.replace('up = -world[:5, [27, 28]].mean(axis=(0, 1))','up = np.array([0.,0.,1.])')
 code=code.replace('(params - previous) * .015',f'(params - previous) * {temporal}, (params[6:] - model.qpos0[addresses]) * {prior}')
 scope=dict(p.__dict__);exec(code,scope)
 q,v,g,s,f=scope['retarget'](m,mf,payload)
 r=p.simulate(m,mf,q,v,g,s);r['fit']=f
 np.savez(ROOT/f'reports/tracking-fix/{name}-reference.npz',q=q,v=v,g=g,s=s)
 (ROOT/f'reports/tracking-fix/{name}.json').write_text(json.dumps(r))
 row={'variant':name,**r['metrics'],'fit':f};summary.append(row);print(json.dumps(row),flush=True)
(ROOT/'reports/tracking-fix/retarget-summary.json').write_text(json.dumps(summary,indent=2))
