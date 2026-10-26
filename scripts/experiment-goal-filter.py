import importlib.util,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('p',ROOT/'scripts/process-recording.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p);p.emit=lambda*a,**k:None
folder=ROOT/'reports/tracking-fix'/sys.argv[1];r=np.load(folder/'reference.npz');q,v,g,s=[r[k] for k in ['q','v','g','s']];m=p.mujoco.MjModel.from_xml_path(str(ROOT/'public/model/model.xml'));mf=p.load('manifest.json');summary=[]
base=p.policy_bundle(ROOT/'inputs/mm-fullbody-base/onnx','base');adapted=p.policy_bundle(ROOT/'public','adapted')
for name,window in [('base',k) for k in [11,21,31,51]]+[('adapted',k) for k in [11,21,31,51]]:
 goals=p.savgol_filter(g,window,2,axis=0)
 result=p.simulate(m,mf,q,v,goals,s,bundle=base if name=='base' else adapted,noise_scale=0 if name=='base' else 1,startup='joints-100ms',consistent_reset=name=='base',progress=False)
 label=f'goal-filter-{name}-{window}';(folder/f'{label}.json').write_text(json.dumps(result));row={'variant':label,**result['metrics']};summary.append(row);print(json.dumps(row),flush=True)
(folder/'goal-filter-summary.json').write_text(json.dumps(summary,indent=2))
