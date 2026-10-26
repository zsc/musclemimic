// Exact port of loco_mujoco/core/utils/math.py; preserve its frame conventions.
const sub = (a,b) => a.map((x,i)=>x-b[i]);
const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const mv = (m,v,transpose=false) => [0,1,2].map(i=>v.reduce((s,x,j)=>s+m[transpose?j*3+i:i*3+j]*x,0));
const relativeMatrix = (a,b) => Array.from({length:9},(_,n)=>[0,1,2].reduce((s,k)=>s+a[k*3+Math.floor(n/3)]*b[k*3+n%3],0));
export function rotationVector(m) {
  // Stable matrix -> quaternion -> shortest rotation vector, including near pi.
  const q = [0,0,0,0];
  const choice = [m[0],m[4],m[8],m[0]+m[4]+m[8]];
  const i = choice.indexOf(Math.max(...choice));
  if (i===3) {
    q[0]=m[7]-m[5]; q[1]=m[2]-m[6]; q[2]=m[3]-m[1]; q[3]=1+choice[3];
  } else {
    const j=(i+1)%3,k=(j+1)%3;
    q[i]=1-choice[3]+2*m[i*3+i]; q[j]=m[j*3+i]+m[i*3+j]; q[k]=m[k*3+i]+m[i*3+k]; q[3]=m[k*3+j]-m[j*3+k];
  }
  const norm=Math.hypot(...q)*(q[3]<0?-1:1);
  for(let j=0;j<4;j++)q[j]/=norm;
  const angle=2*Math.atan2(Math.hypot(...q.slice(0,3)),q[3]);
  const scale=angle<1e-3?2+angle*angle/12+7*angle**4/2880:angle/Math.sin(angle/2);
  return q.slice(0,3).map(x=>x*scale);
}
export function createObservation(data, manifest, target) {
  const obs=new Float32Array(manifest.obs_dim);
  for(const item of manifest.observations) {
    const values=data[item.field];
    for(let j=0;j<item.source.length;j++)obs[item.output[j]]=values[item.source[j]];
  }
  const ids=manifest.site_ids;
  const vec=(name,id,size)=>Array.from(data[name].slice(id*size,(id+1)*size));
  const pos=ids.map(id=>vec('site_xpos',id,3));
  const mat=ids.map(id=>vec('site_xmat',id,9));
  const vel=ids.map((id,i)=>{
    const body=manifest.site_bodyid[id], root=manifest.body_rootid[body];
    const cvel=vec('cvel',body,6), offset=sub(pos[i],vec('subtree_com',root,3));
    return [...cvel.slice(0,3),...sub(cvel.slice(3),cross(offset,cvel.slice(0,3)))];
  });
  const positions=[],angles=[],velocities=[];
  for(let i=1;i<ids.length;i++) {
    const rel=relativeMatrix(mat[0],mat[i]);
    positions.push(...sub(pos[i],pos[0])); angles.push(...rotationVector(rel));
    velocities.push(...sub(mv(rel,vel[i].slice(0,3),true),vel[0].slice(0,3)),...mv(mat[0],sub(vel[0].slice(3),vel[i].slice(3))));
  }
  const goal=[...positions,...angles,...velocities,...target];
  if(goal.length!==manifest.goal_output_indices.length)throw Error('Goal dimension mismatch');
  goal.forEach((x,i)=>obs[manifest.goal_output_indices[i]]=x);
  return obs;
}
export function acceptance(errors, roots, gates, complete, finite=true) {
  const mean=errors.reduce((a,b)=>a+b,0)/errors.length;
  const final=errors.at(-1),rootMax=Math.max(...roots);
  return {mean,final,rootMax,complete,finite,passed:complete&&finite&&mean<gates.mean_mm&&final<gates.final_mm&&rootMax<gates.root_max_mm};
}
