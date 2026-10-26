import sys,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.append(str(ROOT/'.venv-export/lib/python3.12/site-packages'))
import mujoco,numpy as np
from PIL import Image,ImageDraw
OUT=ROOT/'reports/no-finetune'/ (sys.argv[1] if len(sys.argv)>1 else '')
m=mujoco.MjModel.from_xml_path(str(ROOT/'public/model/model.xml'));ref=json.loads((OUT/'reference.json').read_text());run=json.loads((OUT/'rollout-seed3.json').read_text())
m.geom_rgba[0]=[.09,.13,.17,1]
d=mujoco.MjData(m);mujoco.mj_forward(m,d)
cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.lookat[:]=[0,0,.8];cam.distance=3.5;cam.azimuth=85;cam.elevation=-12
opt=mujoco.MjvOption();opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON]=True
process=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x520','-r','30','-i','-','-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p',str(OUT/'comparison.mp4')],stdin=subprocess.PIPE)
with mujoco.Renderer(m,480,640) as renderer:
 for frame in range(46):
  step=min(frame*100//30,150);images=[]
  for name,q in [('Monocular IK reference',ref['qpos'][step]),('Muscle-driven physics (no finetuning)',run['qpos'][step])]:
   d.qpos[:]=q;mujoco.mj_forward(m,d)
   m.tendon_rgba[:]=[.35,.6,.75,.55]
   if name.startswith('Muscle'):
    act=run['activation'][step]
    for a in range(m.nu):
     tendon=m.actuator_trnid[a,0]
     if tendon>=0:m.tendon_rgba[tendon]=[.3+.7*act[a],.68-.5*act[a],.65-.5*act[a],1]
   renderer.update_scene(d,camera=cam,scene_option=opt)
   im=Image.fromarray(renderer.render());images.append(im)
  canvas=Image.new('RGB',(1280,520),'#101a23');canvas.paste(images[0],(0,40));canvas.paste(images[1],(640,40))
  draw=ImageDraw.Draw(canvas);draw.text((16,12),f'Monocular IK reference  |  {step/100:.2f}s',fill='white');draw.text((650,12),f'Actual muscle simulation  |  error {run["errors_mm"][step]:.0f} mm',fill='white')
  process.stdin.write(np.array(canvas).tobytes())
  if frame in [0,15,30,45]:canvas.save(OUT/f'comparison-{step:03d}.png')
process.stdin.close();assert process.wait()==0
print(OUT/'comparison.mp4')
