"""Bounded monocular-recording experiment. No policy fitting or training.

Uses this checkout's Python environments; all new outputs stay in reports/.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / '.venv-export/lib/python3.12/site-packages'))
import json, time, hashlib
import numpy as np
import mujoco
import onnxruntime as ort
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
from recording_retargeting import reconstruct_contact_translation
ROOT_FILTER = '--root-filter' in sys.argv
SMOOTH = '--smooth' in sys.argv
CONTACT_ROOT = '--contact-root' in sys.argv or ROOT_FILTER
NATIVE_RESET = '--native-reset' in sys.argv or SMOOTH or CONTACT_ROOT
GROUNDED = '--grounded' in sys.argv or NATIVE_RESET
CALIBRATE = '--calibrate' in sys.argv or GROUNDED
OUT = ROOT / 'reports/no-finetune' / ('contact-filter' if ROOT_FILTER else 'contact-smooth' if SMOOTH and CONTACT_ROOT else 'contact-root' if CONTACT_ROOT else 'smooth' if SMOOTH else 'native-reset' if NATIVE_RESET else 'grounded' if GROUNDED else 'calibrated' if CALIBRATE else '')
OUT.mkdir(exist_ok=True)
load = lambda name: json.loads((ROOT / name).read_text())
write = lambda name, value: (OUT / name).write_text(json.dumps(value, separators=(',', ':'), allow_nan=False))
began = time.perf_counter()
manifest = load('public/manifest.json')
model = mujoco.MjModel.from_xml_path(str(ROOT / 'public/model/model.xml'))
data = mujoco.MjData(model)
ids = np.array(manifest['site_ids'])
poses = load('reports/no-finetune/poses.json')
assert all(len(f['worldLandmarks']) == 33 for f in poses['frames'])
world = np.array([[[p['x'],p['z'],-p['y']] for p in f['worldLandmarks']] for f in poses['frames']])
image = np.array([[[p['x'],p['y']] for p in f['landmarks']] for f in poses['frames']])
world = savgol_filter(world, 7, 2, axis=0)
world -= world[:,[23,24]].mean(axis=1)[:,None,:]
calibration=np.eye(3)
if CALIBRATE:
 # Bounded second trial: assume the first five frames are upright, face policy +X.
 # Estimate one fixed camera-to-model rotation, never track-align individual frames.
 up=-world[:5,[27,28]].mean(axis=(0,1));up/=np.linalg.norm(up)
 lateral=(world[:5,23]-world[:5,24]).mean(axis=0)
 lateral-=up*np.dot(lateral,up);lateral/=np.linalg.norm(lateral)
 forward=np.cross(lateral,up)
 calibration=np.stack([forward,lateral,up]);world=world@calibration.T
# Scale to the model's leg lengths. No old motion is used for pose fitting.
mujoco.mj_forward(model,data)
neutral = data.site_xpos[ids].copy()
leg = np.linalg.norm(neutral[9]-neutral[10])+np.linalg.norm(neutral[10]-neutral[11])
observed_leg = np.mean([np.linalg.norm(world[:,a]-world[:,b],axis=1)+np.linalg.norm(world[:,b]-world[:,c],axis=1) for a,b,c in [(23,25,27),(24,26,28)]],axis=0)
scale = leg / np.median(observed_leg)
world *= scale
# Approximate root translation for a fixed monocular camera. Depth remains fixed.
# At least one foot is assumed grounded. This is a limitation, not measured 3D truth.
hip_image=image[:,[23,24]].mean(axis=1)
image_leg = np.median(np.linalg.norm(image[:,23]-image[:,25],axis=1)+np.linalg.norm(image[:,25]-image[:,27],axis=1))
root_x=(hip_image[:,0]-hip_image[0,0]) * leg / image_leg
root_z=-np.min(world[:,[27,28,29,30,31,32],2],axis=1)+.045
translation=np.outer(root_x,calibration[:,0]);translation[:,2]=root_z
world += translation[:,None,:]
target=np.empty((len(world),17,3))
target[:,0]=world[:,[23,24]].mean(axis=1)
shoulders=world[:,[11,12]].mean(axis=1)
target[:,1]=target[:,0]+.45*(shoulders-target[:,0])
target[:,2]=world[:,[7,8]].mean(axis=1)
for index,landmark in {3:11,4:13,5:15,6:12,7:14,8:16,9:23,10:25,11:27,12:31,13:24,14:26,15:28,16:32}.items(): target[:,index]=world[:,landmark]
# Respect the model's coupled joints during IK instead of independently fitting them.
coupled={int(model.eq_obj1id[e]) for e in range(model.neq) if model.eq_type[e]==mujoco.mjtEq.mjEQ_JOINT}
independent=[j for j in range(1,model.njnt) if j not in coupled]
addresses=np.array([model.jnt_qposadr[j] for j in independent])
def full_qpos(x):
 q=model.qpos0.copy();q[:3]=x[:3]
 quat=Rotation.from_rotvec(x[3:6]).as_quat();q[3:7]=quat[[3,0,1,2]]
 q[addresses]=x[6:]
 for _ in range(3):
  for e in range(model.neq):
   if model.eq_type[e]!=mujoco.mjtEq.mjEQ_JOINT:continue
   j1,j2=int(model.eq_obj1id[e]),int(model.eq_obj2id[e]);a1=model.jnt_qposadr[j1]
   v=q[model.jnt_qposadr[j2]]-model.qpos0[model.jnt_qposadr[j2]] if j2>=0 else 0.
   q[a1]=model.qpos0[a1]+np.polynomial.polynomial.polyval(v,model.eq_data[e,:5])
 return q
lower=np.r_[[-5,-5,.2],[-np.pi]*3,[model.jnt_range[j,0] for j in independent]]
upper=np.r_[[5,5,2.2],[np.pi]*3,[model.jnt_range[j,1] for j in independent]]
x=np.clip(np.r_[translation[0],np.zeros(3),model.qpos0[addresses]],lower+1e-8,upper-1e-8)
fit_started=time.perf_counter(); qs=[];fit_errors=[];costs=[];parameters=[]
for frame,tgt in enumerate(target):
 previous=x.copy()
 def residual(params):
  data.qpos[:]=full_qpos(params);mujoco.mj_kinematics(model,data)
  r=(data.site_xpos[ids]-tgt).ravel()
  smooth=(params-previous)*.015
  return np.r_[r,smooth]
 fit=least_squares(residual,x,bounds=(lower,upper),max_nfev=35,ftol=1e-4,xtol=1e-4,gtol=1e-4)
 x=fit.x;parameters.append(x.copy());qs.append(full_qpos(x));fit_errors.append(np.linalg.norm(residual(x)[:51].reshape(17,3),axis=1).mean()*1000);costs.append(fit.nfev)
q30=np.array(qs)
ik_seconds=time.perf_counter()-fit_started
print('IK',{'seconds':ik_seconds,'mean_marker_error_mm':np.mean(fit_errors),'frames':len(qs)},flush=True)
# Resample to the policy's 100 Hz clock; preserve the quaternion manifold.
t30=np.arange(len(q30))/30;t=np.arange(161)/100
q=np.stack([np.interp(t,t30,q30[:,j]) for j in range(model.nq)],axis=1)
q[:,3:7]=Slerp(t30,Rotation.from_quat(q30[:,[4,5,6,3]]))(t).as_quat()[:,[3,0,1,2]]
if SMOOTH:
 # Recorded clips allow symmetric temporal filtering. Fit continuous parameters,
 # then reconstruct coupled joints, rather than independently filtering qpos.
 filtered=savgol_filter(np.array(parameters),11,3,axis=0)
 interpolated=CubicSpline(t30,filtered,axis=0,bc_type='natural')(t)
 q=np.array([full_qpos(np.clip(x,lower+1e-8,upper-1e-8)) for x in interpolated])
contact_reconstruction=None
if CONTACT_ROOT:
 q,contact_reconstruction=reconstruct_contact_translation(model,q,smooth=SMOOTH or ROOT_FILTER)
ground_offsets=[]
if GROUNDED:
 feet=[i for i in range(model.ngeom) if model.geom_contype[i] and model.body(model.geom_bodyid[i]).name in ['calcn_r','calcn_l','toes_r','toes_l']]
 for qi in q:
  data.qpos[:]=qi;mujoco.mj_kinematics(model,data)
  bottoms=[]
  for g in feet:
   row=data.geom_xmat[g].reshape(3,3)[2];size=model.geom_size[g]
   support=size[0]+abs(row[2])*size[1] if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_CAPSULE else np.linalg.norm(row*size)
   bottoms.append(data.geom_xpos[g,2]-support)
  offset=-min(bottoms)-.0005;qi[2]+=offset;ground_offsets.append(offset)
if SMOOTH and GROUNDED:
 # Smooth the reference floor adjustment only. Rollout state is never projected.
 old_offsets=np.array(ground_offsets)
 new_offsets=savgol_filter(old_offsets,21,3)
 q[:,2]+=new_offsets-old_offsets;ground_offsets=new_offsets.tolist()
if ROOT_FILTER:
 q[:,2]=savgol_filter(q[:,2],21,3)
v=np.zeros((len(q),model.nv))
for i in range(len(q)):
 a=max(0,i-1);b=min(len(q)-1,i+1);mujoco.mj_differentiatePos(model,v[i],t[b]-t[a],q[a],q[b])
site_positions=[]
for qi in q:
 data.qpos[:]=qi;mujoco.mj_kinematics(model,data);site_positions.append(data.site_xpos[ids].copy())
site_positions=np.array(site_positions)
relative=site_positions[:,1:]-site_positions[:,[0]]
goals=[]
for i in range(len(q)):
 pieces=[relative[i].ravel()]
 for offset in [20,40,60,80]:
  k=min(i+offset,len(q)-1);pieces += [q[k,:3]-q[i,:3],v[k,:6]-v[i,:6],relative[k].ravel()]
 pieces += [np.array([i/len(q)])]
 goals.append(np.concatenate(pieces))
# Observation port checked against the saved native fixtures before any rollout.
def observation(fields,target):
 obs=np.zeros(manifest['obs_dim'],np.float32)
 for item in manifest['observations']:
  obs[item['output']]=np.asarray(fields[item['field']]).reshape(-1)[item['source']]
 pos=np.asarray(fields['site_xpos']).reshape(-1,3)[ids]
 mat=np.asarray(fields['site_xmat']).reshape(-1,3,3)[ids]
 com=np.asarray(fields['subtree_com']).reshape(-1,3)
 cvel=np.asarray(fields['cvel']).reshape(-1,6)
 vel=[]
 for sid,p in zip(ids,pos):
  body=manifest['site_bodyid'][sid];root=manifest['body_rootid'][body];cv=cvel[body]
  vel.append(np.r_[cv[:3],cv[3:]-np.cross(p-com[root],cv[:3])])
 vel=np.array(vel);positions=[];angles=[];velocities=[]
 for i in range(1,len(ids)):
  rel=mat[0].T@mat[i]
  positions.extend(pos[i]-pos[0]);angles.extend(Rotation.from_matrix(rel).as_rotvec())
  velocities.extend(np.r_[rel.T@vel[i,:3]-vel[0,:3],mat[0]@(vel[0,3:]-vel[i,3:])])
 obs[manifest['goal_output_indices']]=np.r_[positions,angles,velocities,target]
 return obs
oldref=load('public/reference.json');fixtures=load('public/validation-fixtures.json')
parity=max(np.max(np.abs(observation(f['fields'],oldref['goal'][f['reference_frame']])-np.asarray(f['obs'],np.float32))) for f in fixtures)
assert parity<1e-5,parity
policy=load('public/policy-state.json');noise=np.fromfile(ROOT/'public/noise.f32',dtype=np.float32).reshape(5,150,model.nu)
opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
session=ort.InferenceSession(str(ROOT/'public/actor.onnx'),opts,providers=['CPUExecutionProvider'])
fields=set(item['field'] for item in manifest['observations'])|{'site_xpos','site_xmat','cvel','subtree_com'}
scratch=mujoco.MjData(model)
def rollout(ref_q,ref_v,ref_goals,seed,baseline=False):
 d=mujoco.MjData(model);mujoco.mj_resetData(model,d)
 if baseline:
  initial=load('public/reset-states.json')['10'];mujoco.mj_forward(model,d)
  for key in ['qpos','qvel','act','ctrl']:getattr(d,key)[:]=initial[key]
  for key,values in initial['fields'].items():getattr(d,key).reshape(-1)[:]=values
 else:
  if NATIVE_RESET:
   trajectory_state=mujoco.MjData(model)
   trajectory_state.qpos[:]=ref_q[0];trajectory_state.qvel[:]=ref_v[0];mujoco.mj_forward(model,trajectory_state)
   mujoco.mj_forward(model,d)
   d.qpos[:]=ref_q[0];d.qvel[:]=ref_v[0]
   for key in ['xpos','xquat','cvel']:getattr(d,key)[:]=getattr(trajectory_state,key)
  else:
   d.qpos[:]=ref_q[0];d.qvel[:]=ref_v[0];mujoco.mj_forward(model,d)
 stats={k:np.array(policy[p],np.float32).reshape(1,-1) for k,p in [('run_mean','mean'),('run_var','var')]};stats['run_count']=np.array([policy['count']],np.float32)
 positions=[d.qpos.copy()];activations=[d.act.copy()];errors=[];root_errors=[];times=[]
 start=time.perf_counter();finite=True;min_height=float(d.qpos[2]);peak_warnings=np.zeros(7,dtype=int)
 for step in range(151):
  scratch.qpos[:]=d.qpos;mujoco.mj_kinematics(model,scratch);actual=scratch.site_xpos[ids].copy()
  scratch.qpos[:]=ref_q[step];mujoco.mj_kinematics(model,scratch)
  errors.append(float(np.linalg.norm(actual-scratch.site_xpos[ids],axis=1).mean()*1000));root_errors.append(float(np.linalg.norm(d.qpos[:3]-ref_q[step,:3])*1000))
  if step==150:break
  obs=observation({key:getattr(d,key) for key in fields},ref_goals[step]);tick=time.perf_counter()
  mean,nm,nv,nc=session.run(None,{'obs':obs[None,:],**stats})
  stats=dict(run_mean=nm,run_var=nv,run_count=nc)
  sample=mean.reshape(-1)+np.array(policy['std'],np.float32)*noise[seed,step]
  delta=np.array(policy['control_delta']);center=np.array(policy['control_mean'])
  d.ctrl[:]=np.clip(sample.astype(float)*delta+center,center-delta,center+delta)
  for _ in range(5):mujoco.mj_step(model,d)
  times.append(time.perf_counter()-tick);positions.append(d.qpos.copy());activations.append(d.act.copy())
  min_height=min(min_height,float(d.qpos[2]));peak_warnings=np.maximum(peak_warnings,d.warning.number)
  if not np.isfinite(d.qpos).all() or abs(d.time-(step+1)*.01)>1e-6:
   finite=False;break
 elapsed=time.perf_counter()-start
 metrics={'seed':seed,'seconds':elapsed,'complete':len(errors)==151,'finite':finite,'mean_mm':float(np.mean(errors)),'final_mm':errors[-1],'root_max_mm':max(root_errors),'min_pelvis_height_m':min_height,'warnings':peak_warnings.tolist(),'p50_step_ms':float(np.median(times)*1000)}
 metrics['passed']=metrics['complete'] and finite and metrics['mean_mm']<100 and metrics['final_mm']<150 and metrics['root_max_mm']<200
 return metrics,{'qpos':np.array(positions).tolist(),'activation':np.array(activations).tolist(),'errors_mm':errors}
# Existing validated clip catches errors in the inference/observation port.
oldq=np.array(oldref['qpos'][10:161]);oldq[:,:2]-=np.array(oldref['qpos'][10])[:2]
baseline,_=rollout(oldq,np.array(oldref['qvel'][10:161]),oldref['goal'][10:161],3,True)
assert abs(baseline['final_mm']-173.4)<1.,baseline
print('BASELINE',baseline,flush=True)
reset_check=None
if NATIVE_RESET:
 reset_check,_=rollout(oldq,np.array(oldref['qvel'][10:161]),oldref['goal'][10:161],3)
 print('REBUILT_NATIVE_RESET_BASELINE',reset_check,flush=True)
runs=[]
for seed in range(5):
 metrics,record=rollout(q,v,goals,seed);runs.append(metrics);write(f'rollout-seed{seed}.json',record);print('RUN',metrics,flush=True)
write('reference.json',{'qpos':q.tolist(),'qvel':v.tolist(),'goal':np.array(goals).tolist(),'fps':100,'target_sites_30fps':target.tolist()})
summary={'root_filter':ROOT_FILTER,'temporal_smoothing':SMOOTH,'contact_root_reconstruction':contact_reconstruction,'native_reset':NATIVE_RESET,'rebuilt_reset_baseline':reset_check,'grounded_reference':GROUNDED,'ground_offsets_m':ground_offsets,'calibrated':CALIBRATE,'camera_rotation':calibration.tolist(),'policy_sha256':hashlib.sha256((ROOT/'public/actor.onnx').read_bytes()).hexdigest(),'trained':False,'source':'public/source.mp4; same short motion as the previous demo, new monocular reconstruction; not a novel-motion generalization test','mujoco':mujoco.__version__,'pose':{k:poses[k] for k in ['modelLoadMs','decodeAndPoseMs','totalMs']},'ik':{'seconds':ik_seconds,'frames':len(q30),'mean_marker_error_mm':float(np.mean(fit_errors)),'max_frame_mean_mm':max(fit_errors),'scale':scale},'observation_max_abs_error':float(parity),'baseline':baseline,'runs':runs,'passed':sum(r['passed'] for r in runs),'assumptions':['Fixed camera; upright first frames estimate gravity and canonical heading' if CALIBRATE else 'Fixed camera, camera vertical treated as gravity','Root translation estimated by stationary support feet' if CONTACT_ROOT else 'Root translation from image horizontal displacement; single fixed camera rotation; depth is not reconstructed','At least one foot grounded; no airborne-motion support; optional collision-geometry floor correction applies only to reference' ,'Root initialized once from fitted pose and velocity; zero muscle activation','No external forces, hidden resets, pose overwrites or policy finetuning during rollout','Future reference samples clamp at the recorded clip end'], 'native_pipeline_seconds':time.perf_counter()-began}
write('summary.json',summary)
print('SUMMARY',json.dumps(summary),flush=True)
