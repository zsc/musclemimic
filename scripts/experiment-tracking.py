"""Controlled no-training inference ablations on the identical reference."""
import importlib.util, json, inspect, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('processor',ROOT/'reports/tracking-fix/processor-before.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
p.emit=lambda *args,**kwargs:None
m=p.mujoco.MjModel.from_xml_path(str(ROOT/'public/model/model.xml'));mf=p.load('manifest.json')
out=ROOT/'reports/tracking-fix';out.mkdir(exist_ok=True)
cache=out/'reference.npz'
if not cache.exists():
 q,v,g,s,f=p.retarget(m,mf,json.loads((ROOT/'reports/no-finetune/poses.json').read_text()))
 np.savez(cache,q=q,v=v,g=g,s=s)
cache=np.load(cache);q,v,g,s=[cache[k] for k in ['q','v','g','s']]
source=inspect.getsource(p.simulate)
summary=[]
for name,correct,noise,initial,stats in [('baseline',False,1,False,True),('deterministic',False,0,False,True),('consistent-reset',True,1,False,True),('consistent-deterministic',True,0,False,True),('initial-activation',True,0,True,True),('frozen-statistics',True,0,True,False)]:
 code=source.replace('std * noise[step]',f'{noise} * std * noise[step]')
 if correct:code=code.replace("getattr(d, key)[:] = getattr(scratch, key)","getattr(d, key)[:] = getattr(scratch, key)\n    mujoco.mj_forward(model, d)")
 if initial:code=code.replace('tick = time.perf_counter()\n        for _', 'if step == 0: d.act[:] = d.ctrl\n        tick = time.perf_counter()\n        for _')
 if not stats:code=code.replace('stats = dict(run_mean=nm, run_var=nv, run_count=nc)','pass # frozen')
 scope=dict(p.__dict__);exec(code,scope)
 r=scope['simulate'](m,mf,q,v,g,s)
 (out/f'{name}.json').write_text(json.dumps(r,separators=(',',':')))
 row={'variant':name,**r['metrics'],'seconds':r['timing']['simulationSeconds']};summary.append(row);print(json.dumps(row),flush=True)
(out/'ablation-summary.json').write_text(json.dumps(summary,indent=2))
