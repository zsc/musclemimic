import * as ort from 'onnxruntime-web/webgpu';
import {Physics} from './physics.js';
import {RecordedPlayback} from './recorded-playback.js';
import {loadAsset,loadJSONAsset,unpack} from './assets.js';
let actorBytes,ortBytes;
let recording;
const current = () => recording || physics;
const limit = () => recording ? recording.limit : 150;
let physics,session,stats,policy,manifest,noise,base,backend,adapterInfo,playing=false,speed=1,showMuscles=true,busy=false;
let policyTimes=[],physicsTimes=[],stepTimes=[],wallStarted=null,wallEnded=null;
let timer=null,playClock=0,lastSnapshotAt=0;
const send=(type,payload={})=>postMessage({type,...payload});
const fetchJSON=async p=>{const r=await fetch(new URL(p,base));if(!r.ok)throw Error(`${p}: HTTP ${r.status}`);return r.json();};
const buffer=async p=>{const r=await fetch(new URL(p,base));if(!r.ok)throw Error(`${p}: HTTP ${r.status}`);return r.arrayBuffer();};
const tensor=(x,dims)=>new ort.Tensor('float32',Float32Array.from(x),dims);
function releaseStats(){if(stats)Object.values(stats).forEach(x=>x.dispose());}
function reset(start=10,seed=3){
  playing=false;clearTimeout(timer);
  if(recording){recording.at(0);sendSnapshot();return;}
  physics.reset(start,seed);releaseStats();
  stats={run_mean:tensor(policy.mean,[1,manifest.obs_dim]),run_var:tensor(policy.var,[1,manifest.obs_dim]),run_count:tensor([policy.count],[1])};
  policyTimes=[];physicsTimes=[];stepTimes=[];wallStarted=null;wallEnded=null;sendSnapshot();
}
async function load(config){
  playing=false;base=config.base;backend=config.backend;
  send('progress',{text:'载入模型、参考轨迹和策略权重…'});
  const loadPhysics=async()=>{
    if(physics)return;
    const [m,reference,resets,p,n,mod,wasmBinary,archive]=await Promise.all([
      fetchJSON('manifest.json'),loadJSONAsset('reference',base),loadJSONAsset('resets',base),
      fetchJSON('policy-state.json'),buffer('noise.f32'),
      import(/* @vite-ignore */new URL('vendor/mujoco.js',base).href),loadAsset('mujoco',base),loadAsset('model',base)]);
    manifest=m;policy=p;noise=new Float32Array(n);
    const mj=await mod.default({wasmBinary,locateFile:name=>new URL('vendor/'+name,base).href});
    mj.FS.mkdir('/model');mj.FS.mkdir('/model/assets');
    let {header,offset}=unpack(archive);
    for(const file of header.files){mj.FS.writeFile('/'+file.path,new Uint8Array(archive,offset,file.bytes));offset+=file.bytes;}
    const model=mj.MjModel.from_xml_path('/model/model.xml');
    if(model.nq!==manifest.nq||model.nu!==manifest.nu)throw Error('Model dimension mismatch');
    physics=new Physics(mj,model,manifest,reference,resets);
  };
  if(session){await session.release();session=null;}
  ort.env.wasm.numThreads=1;
  ort.env.wasm.wasmPaths=new URL('vendor/',base).href;
  if(backend==='webgpu'){
    if(!navigator.gpu)throw Error('WebGPU 不可用；可选择 WASM CPU 策略后端');
    const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw Error('没有可用的 WebGPU adapter；可选择 WASM CPU');
    adapterInfo={...adapter.info,vendor:adapter.info?.vendor,architecture:adapter.info?.architecture,device:adapter.info?.device,description:adapter.info?.description,isFallbackAdapter:adapter.info?.isFallbackAdapter};
  }else adapterInfo={description:'CPU · single-thread WASM'};
  const loadPolicy=async()=>{
    const [actor,wasmBinary]=await Promise.all([actorBytes??=loadAsset('actor',base,(done,total)=>send('progress',{text:`准备策略权重 ${(done/1e6).toFixed(1)} / ${(total/1e6).toFixed(1)} MB · 首次下载后保存在此浏览器`})),ortBytes??=loadAsset('ort',base)]);
    ort.env.wasm.wasmBinary=wasmBinary;
    send('progress',{text:`下载完成，编译 ${backend==='webgpu'?'WebGPU':'WASM CPU'} 策略…`});
    session=await ort.InferenceSession.create(new Uint8Array(actor),{executionProviders:[backend],graphOptimizationLevel:'all'});
  };
  await Promise.all([loadPhysics(),loadPolicy()]);
  reset(config.start??10,config.seed??3);
  // Warm-up does not change the simulation or the saved running statistics.
  const warmObs=tensor(physics.observation(),[1,manifest.obs_dim]);
  const outputs=await session.run({obs:warmObs,...stats});Object.values(outputs).forEach(x=>x.dispose());warmObs.dispose();
  send('ready',{backend,adapterInfo,manifest:{nq:manifest.nq,nu:manifest.nu,physicsVersion:physics.mj.mj_versionString()}});
}
async function step(){
  if(recording){recording.at(recording.stepIndex+1);return;}
  const began=performance.now();if(wallStarted===null)wallStarted=began;
  const obs=tensor(physics.observation(),[1,manifest.obs_dim]);
  const t=performance.now();const out=await session.run({obs,...stats});policyTimes.push(performance.now()-t);
  const means=out.action_mean.data,action=new Float64Array(manifest.nu);
  const offset=(physics.seed*150+physics.stepIndex)*manifest.nu;
  for(let i=0;i<action.length;i++){
    const sample=Math.fround(means[i]+Math.fround(policy.std[i]*noise[offset+i]));
    const value=sample*policy.control_delta[i]+policy.control_mean[i];
    action[i]=Math.max(policy.control_mean[i]-policy.control_delta[i],Math.min(policy.control_mean[i]+policy.control_delta[i],value));
  }
  obs.dispose();out.action_mean.dispose();releaseStats();
  stats={run_mean:out.next_mean,run_var:out.next_var,run_count:out.next_count};
  const p=performance.now();physics.advance(action);physicsTimes.push(performance.now()-p);stepTimes.push(performance.now()-began);wallEnded=performance.now();
}
function timing(a){if(!a.length)return {p50:0,p95:0,mean:0};const s=a.slice().sort((a,b)=>a-b);return {p50:s[Math.floor((s.length-1)*.5)],p95:s[Math.floor((s.length-1)*.95)],mean:a.reduce((a,b)=>a+b,0)/a.length};}
function report(){if(recording)return {...recording.recording,backend:'native-recording',metrics:recording.metrics(),live_policy:false,notes:'Native forward simulation computed once after recording; this is playback.'};return {schema:1,created:new Date().toISOString(),backend,adapterInfo,mujoco:physics.mj.mj_versionString(),checkpoint:manifest.checkpoint,metrics:physics.metrics(),gates:manifest.gates,timing:{policy:timing(policyTimes),physicsAndMeasurement:timing(physicsTimes),controlStep:timing(stepTimes),wallMs:wallEnded===null?0:wallEnded-wallStarted,wallIncludesPauses:true},errors_mm:physics.errors,root_errors_mm:physics.roots,live_policy:true,hidden_resets:0,external_forces:false,notes:'IK reference computed offline. Policy observes live state; only standard-normal noise is precomputed. Renderer uses separate scratch data.'};}
function sendSnapshot(){if(recording){const t=recording.recording.timing;send('snapshot',{snapshot:recording.snapshot(showMuscles),timing:{policy:{p50:t.policyMs},physics:{p50:t.physicsMs},step:{mean:0}}});return;}send('snapshot',{snapshot:physics.snapshot(showMuscles),timing:{policy:timing(policyTimes),physics:timing(physicsTimes),step:timing(stepTimes)}});}
async function loop(){
  if(busy||!playing)return;busy=true;const started=performance.now();
  try{await step();if(performance.now()-lastSnapshotAt>=16||current().stepIndex===limit()){sendSnapshot();lastSnapshotAt=performance.now();}if(current().stepIndex===limit()){playing=false;send('complete',{report:report()});}}
  catch(e){playing=false;send('error',{message:e.message,stack:e.stack});}
  finally{busy=false;}
  if(playing)timer=setTimeout(loop,Math.max(0,playClock+current().stepIndex*10/speed-performance.now()));
}
async function validateActor(){
  const fixtures=await fetchJSON('validation-fixtures.json');
  // First fixture uses initial running statistics; other fixtures use updated statistics.
  const obs=tensor(fixtures[0].obs,[1,manifest.obs_dim]);
  const st={run_mean:tensor(policy.mean,[1,manifest.obs_dim]),run_var:tensor(policy.var,[1,manifest.obs_dim]),run_count:tensor([policy.count],[1])};
  const out=await session.run({obs,...st});const expected=fixtures[0].mean.flat();
  const error=Math.max(...out.action_mean.data.map((x,i)=>Math.abs(x-expected[i])));
  [obs,...Object.values(st),...Object.values(out)].forEach(x=>x.dispose());return {maxAbsoluteError:error,tolerance:5e-4,passed:error<5e-4};
}
let chain=Promise.resolve();
onmessage=({data:msg})=>{
  if(msg.type==='pause'){playing=false;clearTimeout(timer);return;}
  if(msg.type==='speed'){speed=msg.value;playClock=performance.now()-(current()?.stepIndex??0)*10/speed;return;}
  if(msg.type==='muscles'){showMuscles=msg.value;return;}
  chain=chain.then(async()=>{
    while(busy)await new Promise(r=>setTimeout(r,5));
    let result;
    if(msg.type==='load')await load(msg);
    else if(msg.type==='reset')reset(msg.start,msg.seed);
    else if(msg.type==='loadRecording'){playing=false;clearTimeout(timer);recording?.dispose();recording=new RecordedPlayback(physics,msg.recording);sendSnapshot();}
    else if(msg.type==='seek'){if(!recording)throw Error('Only recorded results can be scrubbed');playing=false;clearTimeout(timer);recording.at(Math.round(Math.max(0,Math.min(recording.limit,Number(msg.frame)||0))));sendSnapshot();}
    else if(msg.type==='restoreExample'){recording?.dispose();recording=null;reset(msg.start,msg.seed);}
    else if(msg.type==='play'){if(current().stepIndex<limit()){clearTimeout(timer);playing=true;playClock=performance.now()-current().stepIndex*10/speed;loop();}}
    else if(msg.type==='report')result=report();
    else if(msg.type==='validateActor')result=await validateActor();
    else if(msg.type==='runAll'){
      playing=false;const runs=[];
      for(const start of [0,5,10])for(let seed=0;seed<5;seed++){
        reset(start,seed);for(let i=0;i<150;i++)await step();runs.push(report());send('auditProgress',{completed:runs.length,total:15,result:runs.at(-1).metrics});
      }
      sendSnapshot();result=runs;
    }else if(msg.type==='run'){
      reset(msg.start,msg.seed);for(let i=0;i<150;i++)await step();sendSnapshot();result=report();send('complete',{report:result});
    }
    if(msg.id)send('reply',{id:msg.id,result});
  }).catch(e=>{playing=false;send('error',{message:e.message,stack:e.stack,id:msg.id});});
};
