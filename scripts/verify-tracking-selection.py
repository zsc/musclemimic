"""Regression checks on fixed references, including the user's two archived clips."""
import importlib.util,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('p',ROOT/'scripts/process-recording.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p);p.emit=lambda*a,**k:None
m=p.mujoco.MjModel.from_xml_path(str(ROOT/'public/model/model.xml'));mf=p.load('manifest.json');summary=[]
for id in ['example','131cbf8a-2215-482a-ae43-414c1faa28df','58f77f76-14d5-4b63-b446-283d8e200417','26b3c441-517c-49d7-8ee1-34de21d8dd4b']:
 folder=ROOT/'reports/tracking-fix';folder=folder if id=='example' else folder/id
 r=np.load(folder/'reference.npz');q,v,g,s=[r[k] for k in ['q','v','g','s']];result=p.simulate_best(m,mf,q,v,g,s)
 result.update(reference_qpos=q.tolist(),fps=100,duration=(len(q)-1)/100,trained=False,external_forces=False,hidden_resets=0)
 baseline=result['search']['candidates'][0]['metrics'];metrics=result['metrics']
 for key in ['mean','final','rootMax','fallFrames']:assert metrics[key]<=baseline[key]+1e-7,(id,key)
 assert len(result['qpos'])==len(q)
 assert np.array_equal(result['qpos'][0],q[0])
 assert not any(result['warnings'])
 (folder/'selected-result.json').write_text(json.dumps(result,separators=(',',':')))
 row={'id':id,'before':baseline,'after':metrics,'controller':result['controller'],'seconds':result['search']['seconds']};summary.append(row);print(json.dumps(row),flush=True)
(ROOT/'reports/tracking-fix/verification-summary.json').write_text(json.dumps(summary,indent=2))
