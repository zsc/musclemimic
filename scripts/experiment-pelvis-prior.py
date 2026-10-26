import importlib.util,inspect,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('p',ROOT/'reports/tracking-fix/processor-before.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p);p.emit=lambda*a,**k:None
m=p.mujoco.MjModel.from_xml_path(str(ROOT/'public/model/model.xml'));mf=p.load('manifest.json');payload=json.loads((ROOT/'reports/no-finetune/poses.json').read_text());source=inspect.getsource(p.retarget);summary=[]
for strength in [.05,.1,.2,.4]:
 code=source.replace('(params - previous) * .015',f'(params - previous) * .015, Rotation.from_rotvec(params[3:6]).as_matrix()[:2,2] * {strength}')
 scope=dict(p.__dict__);exec(code,scope);q,v,g,s,f=scope['retarget'](m,mf,payload)
 result=p.simulate(m,mf,q,v,g,s);result['fit']=f
 name=f'pelvis-{strength}';(ROOT/f'reports/tracking-fix/{name}.json').write_text(json.dumps(result));np.savez(ROOT/f'reports/tracking-fix/{name}-reference.npz',q=q,v=v,g=g,s=s)
 row={'variant':name,**result['metrics'],'fit':f['fit_mean_mm']};summary.append(row);print(json.dumps(row),flush=True)
(ROOT/'reports/tracking-fix/pelvis-summary.json').write_text(json.dumps(summary,indent=2))
