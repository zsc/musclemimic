import importlib.util,inspect,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('p',ROOT/'reports/tracking-fix/processor-before.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p);p.emit=lambda*a,**k:None
folder=ROOT/'reports/tracking-fix'/sys.argv[1];r=np.load(folder/'reference.npz');q,v,g,s=[r[k] for k in ['q','v','g','s']];m=p.mujoco.MjModel.from_xml_path(str(ROOT/'public/model/model.xml'));mf=p.load('manifest.json');source=inspect.getsource(p.simulate);summary=[]
for mode in ['zero','half','quarter','average5','average10','average15','root-zero','root-average10','joint-average10']:
 initial=v[0].copy()
 if mode=='zero':initial*=0
 if mode=='half':initial*=.5
 if mode=='quarter':initial*=.25
 if mode.startswith('average'):
  n=int(mode[7:]);p.mujoco.mj_differentiatePos(m,initial,n*.01,q[0],q[n])
 if mode=='root-zero':initial[:6]=0
 if mode in ['root-average10','joint-average10']:
  avg=np.zeros(m.nv);p.mujoco.mj_differentiatePos(m,avg,.1,q[0],q[10]);initial[:6] = avg[:6] if mode=='root-average10' else initial[:6];initial[6:]=avg[6:] if mode=='joint-average10' else initial[6:]
 code=source.replace('qvel[:] = v[0]','qvel[:] = initial')
 scope=dict(p.__dict__);scope['initial']=initial;exec(code,scope);result=scope['simulate'](m,mf,q,v,g,s);name=f'initial-{mode}';(folder/f'{name}.json').write_text(json.dumps(result));row={'variant':name,**result['metrics'],'rootVelocity':initial[:6].tolist()};summary.append(row);print(json.dumps(row),flush=True)
(folder/'initial-summary.json').write_text(json.dumps(summary,indent=2))
