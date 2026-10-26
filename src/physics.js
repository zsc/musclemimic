import {createObservation,acceptance} from './observation.js';
export class Physics {
  constructor(mj,model,manifest,reference,resets) {
    Object.assign(this,{mj,model,manifest,reference,resets});
    this.data=new mj.MjData(model); this.display=new mj.MjData(model); this.ghost=new mj.MjData(model);
  }
  reset(start=10,seed=3) {
    if(![0,5,10].includes(start)||!Number.isInteger(seed)||seed<0||seed>4)throw Error('Invalid run configuration');
    this.start=start;this.seed=seed;this.stepIndex=0;this.errors=[];this.roots=[];this.finite=true;
    const {mj,model,data}=this;
    mj.mj_resetData(model,data); mj.mj_forward(model,data);
    const initial=this.resets[start];
    // Original native reset retains derived fields; refresh only on separate display data.
    for(const field of ['qpos','qvel','act','ctrl'])data[field].set(initial[field]);
    for(const [field,values] of Object.entries(initial.fields))data[field].set(values);
    this.measure();
  }
  observation(){return createObservation(this.data,this.manifest,this.reference.goal[this.start+this.stepIndex]);}
  advance(action) {
    const {data,model,mj,manifest}=this;
    data.ctrl.set(action);
    for(let i=0;i<manifest.substeps;i++)mj.mj_step(model,data);
    this.stepIndex++;
    this.finite=Array.from(data.qpos).every(Number.isFinite)&&Array.from(data.qvel).every(Number.isFinite);
    if(!this.finite||Math.abs(data.time-this.stepIndex*.01)>1e-7)throw Error('Simulation became invalid or MuJoCo reset itself');
    this.measure();
  }
  measure() {
    const {mj,model,display,ghost,data,reference,manifest}=this;
    display.qpos.set(data.qpos); mj.mj_kinematics(model,display);
    const qpos=reference.qpos[this.start+this.stepIndex].slice();
    qpos[0]-=reference.qpos[this.start][0];qpos[1]-=reference.qpos[this.start][1];
    ghost.qpos.set(qpos); mj.mj_kinematics(model,ghost);
    const errors=manifest.site_ids.map(id=>Math.hypot(...[0,1,2].map(i=>display.site_xpos[id*3+i]-ghost.site_xpos[id*3+i]))*1000);
    this.errors.push(errors.reduce((a,b)=>a+b,0)/errors.length);
    this.roots.push(Math.hypot(...[0,1,2].map(i=>data.qpos[i]-qpos[i]))*1000);
  }
  metrics(){return {...acceptance(this.errors,this.roots,this.manifest.gates,this.stepIndex===150,this.finite),step:this.stepIndex,time:this.stepIndex*.01,seed:this.seed,start:this.start/100};}
  snapshot(muscles=true) {
    const {mj,model,display,ghost,data,manifest}=this;
    const lines=[],colors=[];
    if(muscles){
      mj.mj_tendon(model,display);
      const pos=display.wrap_xpos;
      for(let a=0;a<manifest.nu;a++){
        const t=manifest.actuator_trnid[a][0]; if(t<0)continue;
        const adr=display.ten_wrapadr[t],n=display.ten_wrapnum[t];
        const activation=Math.min(1,Math.max(0,data.act[a]));
        const color=[.3+.7*activation,.68-.5*activation,.65-.5*activation];
        for(let j=1;j<n;j++){
          for(const k of [adr+j-1,adr+j]){lines.push(pos[k*3],pos[k*3+1],pos[k*3+2]);colors.push(...color);}
        }
      }
    }
    return {metrics:this.metrics(),actual:{pos:Array.from(display.geom_xpos),mat:Array.from(display.geom_xmat)},reference:{pos:Array.from(ghost.geom_xpos),mat:Array.from(ghost.geom_xmat)},muscles:new Float32Array(lines),colors:new Float32Array(colors),contacts:data.ncon,activationMean:Array.from(data.act).reduce((a,b)=>a+b,0)/manifest.nu};
  }
}
