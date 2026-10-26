import importlib.util,inspect,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('p',ROOT/'reports/tracking-fix/processor-before.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p);p.emit=lambda*a,**k:None
m=p.mujoco.MjModel.from_xml_path(str(ROOT/'public/model/model.xml'));mf=p.load('manifest.json');r=np.load(ROOT/'reports/tracking-fix/reference.npz');q,v,g,s=[r[k] for k in ['q','v','g','s']]
source=inspect.getsource(p.simulate);summary=[]
for seed,scale,alpha in [(k,1,1) for k in range(5)]+[(3,k,1) for k in [.25,.5,.75,1.5,2]]+[(3,1,k) for k in [.1,.25,.5,.75]]:
 code=source.replace('default_rng(3)',f'default_rng({seed})').replace('model.nu)[3]',f'model.nu)[{seed}]').replace('std * noise[step]',f'{scale} * std * noise[step]')
 if alpha<1:code=code.replace('d.ctrl[:] = np.clip(',f'd.ctrl[:] = (1-{alpha}) * d.ctrl + {alpha} * np.clip(')
 scope=dict(p.__dict__);exec(code,scope);result=scope['simulate'](m,mf,q,v,g,s)
 name=f'search-{seed}-{scale}-{alpha}';(ROOT/f'reports/tracking-fix/{name}.json').write_text(json.dumps(result));row={'variant':name,'seed':seed,'noise_scale':scale,'alpha':alpha,**result['metrics']};summary.append(row);print(json.dumps(row),flush=True)
(ROOT/'reports/tracking-fix/search-summary.json').write_text(json.dumps(summary,indent=2))
