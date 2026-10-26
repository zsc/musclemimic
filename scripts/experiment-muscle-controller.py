"""Inference-only muscle allocation experiment, no external root forces."""
import sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT/'.venv-export/lib/python3.12/site-packages'))
import numpy as np,mujoco
from scipy.sparse import csr_matrix
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
m=mujoco.MjModel.from_xml_path(str(ROOT/'public/model/model.xml'))
ref=np.load(ROOT/'reports/tracking-fix/reference.npz');q=ref['q'];v=ref['v'];sites=ref['s'];acc=np.gradient(v,.01,axis=0)
mf=json.loads((ROOT/'public/manifest.json').read_text());ids=mf['site_ids']
ind=[j for j in range(1,m.njnt) if j not in set(m.eq_obj1id[m.eq_type==mujoco.mjtEq.mjEQ_JOINT])]
dofs=m.jnt_dofadr[ind]
print('independent',len(dofs),'cone',m.opt.cone,flush=True)

def dense(values,nnz,adr,cols,rows,ncols):
 out=np.zeros((rows,ncols))
 for i in range(rows):
  sl=slice(adr[i],adr[i]+nnz[i]);out[i,cols[sl]]=values[sl]
 return out

def run(kp=100,kd=20,rootweight=3,actreg=.1):
 d=mujoco.MjData(m);d.qpos[:]=q[0];d.qvel[:]=v[0];mujoco.mj_forward(m,d)
 scratch=mujoco.MjData(m);M=np.zeros((m.nv,m.nv));err=np.zeros(m.nv)
 errors=[];roots=[];qs=[];acts=[];ncons=[];u=np.zeros(m.nu);start=time.perf_counter()
 for step in range(len(q)):
  mujoco.mj_forward(m,d)
  errors.append(float(np.linalg.norm(d.site_xpos[ids]-sites[step],axis=1).mean()*1000));roots.append(float(np.linalg.norm(d.qpos[:3]-q[step,:3])*1000));qs.append(d.qpos.tolist());acts.append(d.act.tolist());ncons.append(int(d.ncon))
  if step==len(q)-1:break
  mujoco.mj_differentiatePos(m,err,1,d.qpos,q[step])
  target=acc[step]+kp*err+kd*(v[step]-d.qvel)
  moment=dense(d.actuator_moment,d.moment_rownnz,d.moment_rowadr,d.moment_colind,m.nu,m.nv)
  gains=np.array([mujoco.mju_muscleGain(d.actuator_length[i],d.actuator_velocity[i],m.actuator_lengthrange[i],m.actuator_acc0[i],m.actuator_gainprm[i,:9]) for i in range(m.nu)])
  B=moment.T*gains
  mujoco.mj_fullM(m,d,M)
  J=dense(d.efc_J,d.efc_J_rownnz,d.efc_J_rowadr,d.efc_J_colind,d.nefc,m.nv)
  D=d.efc_D*(d.efc_state==mujoco.mjtConstraintState.mjCNSTRSTATE_QUADRATIC)
  H=M+J.T@(D[:,None]*J)
  A=cho_solve(cho_factor(H),B)
  a0=d.qacc-A@d.act
  indices=np.r_[np.arange(6),dofs]
  weights=np.r_[np.full(6,rootweight),np.ones(len(dofs))]
  W=A[indices]*weights[:,None];b=(target-a0)[indices]*weights
  # Bounded muscle activations, a weak minimum-effort penalty.
  def fun(x):
   residual=W@x-b
   return .5*np.dot(residual,residual)+actreg*np.dot(x,x),W.T@residual+2*actreg*x
  fit=minimize(fun,u,jac=True,method='L-BFGS-B',bounds=[(0,1)]*m.nu,options={'maxiter':35,'ftol':1e-6,'maxls':10})
  u=fit.x
  if step==0:d.act[:]=u
  d.ctrl[:]=np.clip(u+(u-d.act)*.5,0,1)
  for _ in range(5):mujoco.mj_step(m,d)
  if step%50==0:print('frame',step,'error',round(errors[-1]),'root_z',round(d.qpos[2],3),flush=True)
 r={'qpos':qs,'activation':acts,'errors_mm':errors,'root_errors_mm':roots,'contacts':ncons,'metrics':{'mean':np.mean(errors),'final':errors[-1],'rootMax':max(roots)},'seconds':time.perf_counter()-start,'warnings':d.warning.number.tolist()}
 return r
for kp,kd,w in [(100,20,3),(400,40,3),(100,20,10)]:
 r=run(kp,kd,w);name=f'allocation-{kp}-{kd}-{w}';(ROOT/f'reports/tracking-fix/{name}.json').write_text(json.dumps(r));print(name,r['metrics'],r['seconds'],flush=True)
