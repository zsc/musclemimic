import importlib.util,inspect,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('p',ROOT/'reports/tracking-fix/processor-before.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p);p.emit=lambda*a,**k:None
m=p.mujoco.MjModel.from_xml_path(str(ROOT/'public/model/model.xml'));mf=p.load('manifest.json');payload=json.loads((ROOT/'reports/no-finetune/poses.json').read_text());source=inspect.getsource(p.retarget);summary=[]
d=p.mujoco.MjData(m);p.mujoco.mj_forward(m,d);feet=[m.body(n).id for n in ['calcn_l','calcn_r']];normals=np.array([d.xmat[b].reshape(3,3).T@np.array([0.,0.,1.]) for b in feet])
for footweight,pelvisweight in [(.15,0),(.4,0),(.15,.3),(.0,.3)]:
 code=source.replace('previous = x.copy()', 'previous = x.copy()\n        stance = np.exp(-(tgt[[11,15],2]-min(tgt[[11,15],2]))/.04)\n        bodyup = tgt[1]-tgt[0]; bodyup /= np.linalg.norm(bodyup)')
 code=code.replace('(params - previous) * .015',f'(params - previous) * .015, (np.array([data.xmat[b].reshape(3,3)@normals[k] for k,b in enumerate(stance_bodies)])[:,:2]*stance[:,None]*{footweight}).ravel(), (Rotation.from_rotvec(params[3:6]).as_matrix()[:,2]-bodyup)*{pelvisweight}')
 scope=dict(p.__dict__);scope.update(stance_bodies=feet,normals=normals);exec(code,scope);q,v,g,s,f=scope['retarget'](m,mf,payload);result=p.simulate(m,mf,q,v,g,s);result['fit']=f
 name=f'stance-{footweight}-{pelvisweight}';(ROOT/f'reports/tracking-fix/{name}.json').write_text(json.dumps(result));np.savez(ROOT/f'reports/tracking-fix/{name}-reference.npz',q=q,v=v,g=g,s=s)
 row={'variant':name,**result['metrics'],'fit':f['fit_mean_mm']};summary.append(row);print(json.dumps(row),flush=True)
(ROOT/'reports/tracking-fix/stance-summary.json').write_text(json.dumps(summary,indent=2))
