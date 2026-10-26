"""Run only on air5 in .venv-physics (MuJoCo 3.17)."""
import json
from pathlib import Path
import mujoco
import numpy as np
root=Path(__file__).resolve().parents[1]
m=mujoco.MjModel.from_xml_path(str(root/'public/model/model.xml'))
d=mujoco.MjData(m)
reset=json.loads((root/'public/reset-states.json').read_text())['10']
source=json.loads((root/'reports/native-physics-34.json').read_text())
mujoco.mj_resetData(m,d);mujoco.mj_forward(m,d)
for key in ['qpos','qvel','act','ctrl']:getattr(d,key)[:]=reset[key]
for key,values in reset['fields'].items():getattr(d,key).reshape(-1)[:]=values
steps=[]
for item in source['steps']:
 d.ctrl[:]=item['ctrl']
 for _ in range(5):mujoco.mj_step(m,d)
 steps.append(dict(qpos=d.qpos.tolist(),qvel=d.qvel.tolist()))
result=dict(version=mujoco.__version__,steps=steps,arena_bytes=m.narena,max_used_arena_bytes=d.maxuse_arena,warnings=d.warning.number.tolist())
(root/'reports/native-physics-317.json').write_text(json.dumps(result))
print({k:v for k,v in result.items() if k!='steps'})
