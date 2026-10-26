import importlib.util,inspect,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('p',ROOT/'reports/tracking-fix/processor-before.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p);p.emit=lambda*a,**k:None
m=p.mujoco.MjModel.from_xml_path(str(ROOT/'public/model/model.xml'));mf=p.load('manifest.json');r=np.load(ROOT/'reports/tracking-fix/reference.npz');q,v,g,s=[r[k] for k in ['q','v','g','s']]
summary=[]
for mode in ['phase0','phase25','phase50','phase75','slow-phase','reverse-phase','short-lookahead','long-lookahead']:
 goals=g.copy()
 if mode.startswith('phase'):goals[:,-1]=int(mode[5:])/100
 if mode=='slow-phase':goals[:,-1]*=.5
 if mode=='reverse-phase':goals[:,-1]=1-goals[:,-1]
 if mode.endswith('lookahead'):
  relative=s[:,1:]-s[:,[0]]
  for i in range(len(q)):
   parts=[relative[i].ravel()]
   for offset in ([10,20,30,40] if mode=='short-lookahead' else [30,60,90,120]):
    k=min(i+offset,len(q)-1);parts += [q[k,:3]-q[i,:3],v[k,:6]-v[i,:6],relative[k].ravel()]
   goals[i]=np.concatenate(parts+[np.array([i/len(q)])])
 result=p.simulate(m,mf,q,v,goals,s)
 (ROOT/f'reports/tracking-fix/{mode}.json').write_text(json.dumps(result));row={'mode':mode,**result['metrics']};summary.append(row);print(json.dumps(row),flush=True)
(ROOT/'reports/tracking-fix/conditioning-summary.json').write_text(json.dumps(summary,indent=2))
