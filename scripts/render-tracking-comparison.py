"""Render full-length before/after evidence, including any remaining collapse."""
import json,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.append(str(ROOT/'.venv-export/lib/python3.12/site-packages'))
import mujoco,numpy as np
from PIL import Image,ImageDraw
id=sys.argv[1];out=ROOT/'reports/tracking-fix'/id;old=json.loads((ROOT/'reports/recordings'/id/'result.json').read_text());new=json.loads((out/'selected-result.json').read_text());ref=old['reference_qpos']
m=mujoco.MjModel.from_xml_path(str(ROOT/'public/model/model.xml'));d=mujoco.MjData(m);m.geom_rgba[0]=[.09,.13,.17,1]
cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.lookat[:]=[0,0,.75];cam.distance=4.;cam.azimuth=45;cam.elevation=-15
opt=mujoco.MjvOption();opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON]=True
video=out/'comparison.mp4';process=subprocess.Popen(['/opt/homebrew/bin/ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1440x540','-r','30','-i','-','-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p',str(video)],stdin=subprocess.PIPE)
count=int(np.ceil((len(ref)-1)/100*30))+1
with mujoco.Renderer(m,480,480) as renderer:
 for frame in range(count):
  step=min(round(frame*100/30),len(ref)-1);canvas=Image.new('RGB',(1440,540),'#101a23');draw=ImageDraw.Draw(canvas)
  for panel,(label,q,run) in enumerate([('Same reconstructed reference',ref[step],None),('Before: frozen adapted policy',old['qpos'][step],old),('After: frozen-policy inference selection',new['qpos'][step],new)]):
   d.qpos[:]=q;mujoco.mj_forward(m,d);m.tendon_rgba[:]=[.35,.6,.75,.55]
   if run:
    for a in range(m.nu):
     tendon=m.actuator_trnid[a,0];act=run['activation'][step][a]
     if tendon>=0:m.tendon_rgba[tendon]=[.3+.7*act,.68-.5*act,.65-.5*act,1]
   renderer.update_scene(d,camera=cam,scene_option=opt);canvas.paste(Image.fromarray(renderer.render()),(panel*480,60));draw.text((panel*480+12,12),label,fill='white')
   detail=f't = {step/100:.2f} s'
   if run:detail+=f' | frame error {run["errors_mm"][step]:.0f} mm | mean {run["metrics"]["mean"]:.0f} mm'
   draw.text((panel*480+12,32),detail,fill='white')
  process.stdin.write(np.array(canvas).tobytes())
  if step in [0,100] or frame==count-1:canvas.save(out/f'comparison-{step:03d}.png')
process.stdin.close();assert process.wait()==0
print(video)
