import './style.css';
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {VideoSync} from './video-sync.js';
import {setupCamera} from './camera.js';
import {setupRecording} from './recording.js';
import {savedResults,saveResult,deleteResult,archiveResult} from './result-store.js';
import {loadGeometry} from './assets.js';
const staticSite=import.meta.env.VITE_STATIC_SITE==='1';
const $=id=>document.getElementById(id);
const base=new URL(import.meta.env.BASE_URL,location.href).href;
const videoSync=new VideoSync($('source-video'),$('video-status'));
const worker=new Worker(new URL('./simulation.worker.js',import.meta.url),{type:'module'});
let ready=false,playing=false,completed=false,lastReport=null,lastMetrics=null,seq=0,frames=0,lastFpsTime=performance.now();
let recordingResult=null,duration=1.5,switching=false,bufferedSnapshot=null,selectedId='original';
let history=[];const historyURLs=new Map();
let sceneReady=false;
const pending=new Map();
const log=s=>{$('log').textContent+='['+new Date().toLocaleTimeString()+'] '+s+'\n';};
const send=(type,data={})=>worker.postMessage({type,...data});
const rpc=(type,data={})=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});send(type,{...data,id});});
const config=()=>({base,backend:$('backend').value,start:Math.round(Number($('start').value)*100),seed:Number($('seed').value)});
const scene=new THREE.Scene();scene.background=new THREE.Color('#111a23');scene.fog=new THREE.Fog('#111a23',7,16);
const renderer=new THREE.WebGLRenderer({canvas:$('scene'),antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.outputColorSpace=THREE.SRGBColorSpace;
const camera=new THREE.PerspectiveCamera(38,1,.02,50);camera.up.set(0,0,1);camera.position.set(3,-6,2.8);
const controls=new OrbitControls(camera,renderer.domElement);controls.target.set(.2,0,.85);controls.enableDamping=true;controls.minDistance=1;controls.maxDistance=12;controls.update();
scene.add(new THREE.HemisphereLight(0xe3f5ff,0x405363,2.3));
const light=new THREE.DirectionalLight(0xffffff,3);light.position.set(-3,-4,6);scene.add(light);
const floor=new THREE.Mesh(new THREE.PlaneGeometry(100,100),new THREE.MeshStandardMaterial({color:0x14212b,roughness:1}));floor.position.z=-.008;scene.add(floor);
const grid=new THREE.GridHelper(14,70,0x345360,0x223743);grid.rotation.x=Math.PI/2;grid.position.z=-.005;scene.add(grid);
const actual=new THREE.Group(),reference=new THREE.Group();actual.position.x=1.05;reference.position.x=-1.05;scene.add(actual,reference);
const actualMeshes=[],referenceMeshes=[];
const muscleGeometry=new THREE.BufferGeometry();
const muscleLines=new THREE.LineSegments(muscleGeometry,new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:.8}));muscleLines.frustumCulled=false;actual.add(muscleLines);
const matrix=new THREE.Matrix4();
function transform(mesh,state,id){
 const p=id*3,r=id*9,m=state.mat;
 matrix.set(m[r],m[r+1],m[r+2],state.pos[p],m[r+3],m[r+4],m[r+5],state.pos[p+1],m[r+6],m[r+7],m[r+8],state.pos[p+2],0,0,0,1);
 mesh.matrix.copy(matrix);mesh.matrixWorldNeedsUpdate=true;
}
function updateScene(snapshot){
 actualMeshes.forEach(({mesh,id})=>transform(mesh,snapshot.actual,id));referenceMeshes.forEach(({mesh,id})=>transform(mesh,snapshot.reference,id));
 muscleGeometry.setAttribute('position',new THREE.BufferAttribute(snapshot.muscles,3));muscleGeometry.setAttribute('color',new THREE.BufferAttribute(snapshot.colors,3));
 const m=snapshot.metrics;lastMetrics=m;
 $('sim-time').textContent=m.time.toFixed(2);$('progress').style.width=`${m.time/duration*100}%`;
 const summary=recordingResult?.metrics||m;
 for(const [id,v] of [['mean-error',summary.mean],['current-error',m.final],['root-error',summary.rootMax]])$(id).textContent=v.toFixed(1)+' mm';
 $('quality').textContent=recordingResult?(summary.passed?'达到阈值':'未达阈值'):m.complete?(m.passed?'达到阈值':'未达阈值'):(m.step?'运行中':'待播放');$('quality').className='status '+((recordingResult||m.complete)?(summary.passed?'pass':'fail'):'neutral');
 $('seek').value=m.step;
 if(recordingResult)completed=m.complete;
 videoSync.update(m);
 if(!recordingResult)chartPoints[m.step]=m.final;drawChart();
}
let chartPoints=[];
function drawChart(){
 const canvas=$('error-chart'),ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height;
 ctx.clearRect(0,0,w,h);const max=Math.max(200,...chartPoints.filter(Number.isFinite));
 ctx.strokeStyle='#704d47';ctx.setLineDash([6,5]);ctx.beginPath();ctx.moveTo(0,h-150/max*(h-10));ctx.lineTo(w,h-150/max*(h-10));ctx.stroke();ctx.setLineDash([]);
 ctx.strokeStyle='#69d5bd';ctx.lineWidth=3;ctx.beginPath();chartPoints.forEach((v,i)=>{const x=i/(duration*100)*w,y=h-v/max*(h-10);if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y)});ctx.stroke();
 if(lastMetrics){const x=lastMetrics.time/duration*w;ctx.strokeStyle='#bfd5df';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();}
}
new ResizeObserver(()=>{const rect=$('viewport').getBoundingClientRect();if(!rect.width||!rect.height)return;renderer.setSize(rect.width,rect.height,false);camera.aspect=rect.width/rect.height;camera.zoom=rect.width>600?1.3:.9;camera.updateProjectionMatrix();}).observe($('viewport'));
function render(){requestAnimationFrame(render);if($('simulation-panel').hidden)return;controls.update();renderer.render(scene,camera);frames++;const now=performance.now();if(now-lastFpsTime>1000){$('render-fps').textContent=Math.round(frames*1000/(now-lastFpsTime))+' fps';frames=0;lastFpsTime=now;}}
render();
function playLabel(){
 $('play').textContent=playing?'❚❚ 暂停':completed?'↻ 重播':'▶ 播放';
 $('play').setAttribute('aria-label',playing?'暂停':completed?'重播':'播放');
 $('play').disabled=!ready||switching;$('seek').disabled=!ready||switching||!recordingResult;
}
function pausePlayback(){videoSync.stop();send('pause');playing=false;playLabel();}
function failure(message){pausePlayback();ready=false;$('loading').hidden=false;$('loading-title').textContent='模型未能启动';$('loading-text').textContent=message+'。可在左侧参数中切换 WASM CPU 重试。';$('loading').classList.add('failed');playLabel();$('backend-pill').textContent='运行错误';log(message);}
function showSnapshot(m){if(!sceneReady){bufferedSnapshot=m;return;}updateScene(m.snapshot);$('policy-ms').textContent=m.timing.policy.p50.toFixed(2)+' ms';$('physics-ms').textContent=m.timing.physics.p50.toFixed(2)+' ms';$('realtime-factor').textContent=m.timing.step.mean?(10/m.timing.step.mean).toFixed(2)+'×':'—';}
worker.onerror=e=>failure(e.message);
worker.onmessage=({data:m})=>{
 if(m.type==='progress'){$('loading-text').textContent=m.text;log(m.text);}
 if(m.type==='ready'){ready=sceneReady;$('loading').hidden=ready;$('export-report').disabled=false;playLabel();$('backend-pill').textContent=(m.backend==='webgpu'?'WebGPU':'WASM CPU')+' · MuJoCo';$('device-info').textContent=[m.adapterInfo.vendor,m.adapterInfo.architecture,m.adapterInfo.description].filter(Boolean).join(' · ')+' · 控制 100 Hz / 物理 500 Hz';log('后端就绪：'+JSON.stringify(m));}
 if(m.type==='snapshot'){if(switching)bufferedSnapshot=m;else showSnapshot(m);}
 if(m.type==='complete'){videoSync.stop();playing=false;completed=true;lastReport=m.report;chartPoints=m.report.errors_mm.slice();drawChart();playLabel();log('完成：'+JSON.stringify(m.report.metrics));}
 if(m.type==='auditProgress')log(`验收 ${m.completed}/${m.total}：${JSON.stringify(m.result)}`);
 if(m.type==='reply'){pending.get(m.id)?.resolve(m.result);pending.delete(m.id);}
 if(m.type==='error'){if(m.id){pending.get(m.id)?.reject(Error(m.message));pending.delete(m.id);}failure(m.message);}
};
async function reset(){
 pausePlayback();completed=false;chartPoints=recordingResult?.errors_mm.slice()||[];lastReport=null;
 await rpc('reset',config());playLabel();
}
async function play(){
 if(!ready||switching)return;
 if(playing){pausePlayback();return;}
 if(completed)await reset();
 playing=true;videoSync.start();send('play');playLabel();
}
$('play').onclick=()=>play().catch(e=>failure(e.message));
for(const id of ['seed','start'])$(id).onchange=()=>{if(ready)reset().catch(e=>failure(e.message));};
$('speed').onchange=()=>{const value=Number($('speed').value);videoSync.setSpeed(value);send('speed',{value});};
$('show-muscles').onchange=()=>{muscleLines.visible=$('show-muscles').checked;send('muscles',{value:$('show-muscles').checked});};
let seekFrame;
$('seek').oninput=()=>{
 if(!recordingResult)return;
 pausePlayback();cancelAnimationFrame(seekFrame);const frame=Number($('seek').value);
 seekFrame=requestAnimationFrame(async()=>{try{await rpc('seek',{frame});completed=frame===Math.round(duration*100);playLabel();}catch(e){failure(e.message);}});
};
$('backend').onchange=async()=>{
 pausePlayback();ready=false;completed=false;chartPoints=[];$('loading').hidden=false;$('loading').classList.remove('failed');$('loading-title').textContent='准备肌骨模型';playLabel();
 try{await rpc('load',config());}catch(e){log(e.message);}
};
$('export-report').onclick=async()=>{
 const report=recordingResult?{...recordingResult,backend:'native-recording',live_policy:false}:await rpc('report');
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));a.download=`musclemimic-${selectedId}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);document.querySelector('.result-menu').open=false;
};
window.__demo={get ready(){return ready},get metrics(){return lastMetrics},get recording(){return recordingResult},get history(){return history.map(({id,title})=>({id,title}))},rpc,reset,config};
async function boot(){
 const loading=rpc('load',config());loading.catch(()=>{});
 const g=await loadGeometry(base);
 const geometries=g.meshes.map(item=>{const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(item.vertices,3));geometry.setIndex(new THREE.BufferAttribute(item.faces,1));geometry.computeVertexNormals();return geometry;});
 const bone=new THREE.MeshStandardMaterial({color:0xd9cfb6,roughness:.8,metalness:.05});
 const ghost=new THREE.MeshStandardMaterial({color:0x84b9cc,roughness:.8,transparent:true,opacity:.42,depthWrite:false});
 for(const item of g.geoms){const a=new THREE.Mesh(geometries[item.mesh],bone),r=new THREE.Mesh(geometries[item.mesh],ghost);a.matrixAutoUpdate=false;r.matrixAutoUpdate=false;actual.add(a);reference.add(r);actualMeshes.push({mesh:a,id:item.id});referenceMeshes.push({mesh:r,id:item.id});}
 sceneReady=true;
 $('source-video').src=new URL('source.mp4',base).href;
 await loading;
 if(bufferedSnapshot){showSnapshot(bufferedSnapshot);bufferedSnapshot=null;}
 ready=true;$('loading').hidden=true;playLabel();
}
const requested=new URLSearchParams(location.search).get('backend');if(['wasm','webgpu'].includes(requested))$('backend').value=requested;
const cameraController=setupCamera(base,pausePlayback);
const initialBoot=boot();initialBoot.catch(e=>failure(e.message));
function setDuration(seconds){duration=seconds;$('duration-label').textContent=' / '+seconds.toFixed(2)+' s';$('chart-duration').textContent=seconds.toFixed(2)+' s';$('seek').max=Math.round(seconds*100);}
function renderHistory(){
 const select=$('result-select');select.replaceChildren(new Option('内置示例 · 原始仿真实验','original'));
 for(const entry of history){const time=new Date(entry.created).toLocaleTimeString('zh-CN',{hour:'2-digit',minute:'2-digit'});select.add(new Option(`${entry.title} · ${time} · ${entry.result.duration.toFixed(2)} s`,entry.id));}
 select.value=selectedId;$('history-count').textContent=history.length?`${history.length}/6 条历史 · 本机`:'历史保存在本机';
}
function releaseURL(id){if(historyURLs.has(id)){URL.revokeObjectURL(historyURLs.get(id));historyURLs.delete(id);}}
async function selectResult(id,autoplay=false){
 await initialBoot;
 const entry=history.find(x=>x.id===id);if(id!=='original'&&!entry)throw Error('历史结果不存在');
 pausePlayback();cancelAnimationFrame(seekFrame);switching=true;bufferedSnapshot=null;playLabel();$('result-select').disabled=true;
 selectedId=id;recordingResult=entry?.result||null;completed=false;lastReport=null;chartPoints=recordingResult?.errors_mm.slice()||[];
 try{
  if(entry){
   const result=entry.result;setDuration(result.duration);
   await rpc('loadRecording',{recording:result});
   if(entry.video&&!historyURLs.has(id))historyURLs.set(id,URL.createObjectURL(entry.video));
   $('source-video').src=historyURLs.get(id)||new URL('source.mp4',base).href;
   videoSync.update({start:result.sourceStart,time:0});
   $('backend-pill').textContent='本机 MuJoCo · 结果回放';$('result-kind').textContent='已生成';
   $('recording-result').classList.toggle('warning',!result.metrics.passed);
   $('recording-summary').textContent=`${result.duration.toFixed(2)} 秒 · 生成 ${result.timing.endToEndSeconds.toFixed(1)} 秒 · `+(result.metrics.passed?'本次达到跟踪阈值':'未达跟踪阈值，可能偏离或失稳');
   $('device-info').textContent='此处为本机已计算轨迹的回放，策略与物理耗时来自生成过程。';
  }else{
   setDuration(1.5);$('source-video').src=new URL('source.mp4',base).href;
   await rpc('restoreExample',config());$('backend-pill').textContent=($('backend').value==='webgpu'?'WebGPU':'WASM CPU')+' · MuJoCo';
   $('result-kind').textContent='实时仿真';$('recording-result').classList.remove('warning');
   $('recording-summary').textContent=staticSite?'浏览器实时计算肌肉控制与物理仿真。':'内置参考已就绪，点击播放；或在左侧生成你的动作。';
   $('device-info').textContent='控制 100 Hz · 物理 500 Hz · 此浏览器实时计算。';
  }
  $('result-select').value=id;$('delete-result').disabled=!entry;
  for(const control of ['seed','start','backend'])$(control).disabled=Boolean(entry);
  if(bufferedSnapshot){showSnapshot(bufferedSnapshot);bufferedSnapshot=null;}drawChart();
 }finally{switching=false;$('result-select').disabled=false;playLabel();}
 if(autoplay)await play();
}
$('result-select').onchange=()=>selectResult($('result-select').value).catch(e=>log(e.message));
$('delete-result').onclick=async()=>{
 const id=selectedId;if(id==='original')return;
 document.querySelector('.result-menu').open=false;
 try{await deleteResult(id);}catch(e){log('历史删除失败：'+e.message);return;}
 history=history.filter(x=>x.id!==id);renderHistory();await selectResult(history[0]?.id||'original');releaseURL(id);
};
const historyReady=(async()=>{
 try{history=await savedResults();renderHistory();if(history.length)await selectResult(history[0].id);
  if(!staticSite)void (async()=>{for(const entry of history.slice())try{await archiveResult(base,entry);}catch(e){log('本机自动归档失败：'+e.message);}})();
 }
 catch(e){log('历史记录暂不可用：'+e.message);$('history-count').textContent='本次会话';}
})();
if(staticSite){
 $('input-source').replaceChildren(new Option('内置示例 · 实时肌骨仿真','example'));
 $('input-placeholder').hidden=true;$('upload-zone').hidden=true;
 $('input-video').hidden=false;$('input-video').src=new URL('source.mp4',base).href;
 $('input-name').textContent='公开示例 · 点击启动同步播放';
 $('recording-status').textContent='在线版支持示例仿真。摄像头／上传视频生成请使用本机版。';
 $('recording-summary').textContent='浏览器实时计算肌肉控制与物理仿真。';
 document.querySelector('footer span:nth-child(2)').textContent='在线示例 · 无需后端';
 $('run').disabled=true;initialBoot.then(()=>{$('run').disabled=false;}).catch(()=>{});
 $('run').onclick=()=>selectResult('original',true).catch(e=>failure(e.message));
}else setupRecording(base,cameraController,async(result,clip,signal)=>{
 await initialBoot;await historyReady;signal.throwIfAborted();
 const entry={id:result.recordingId||crypto.randomUUID(),created:Date.now(),title:clip.title,result,video:clip.blob||null};
 let expired=[];
 try{expired=await saveResult(entry);}catch(e){log('本次结果未保存到磁盘：'+e.message);entry.volatile=true;}
 if(signal.aborted){if(!entry.volatile)await deleteResult(entry.id);signal.throwIfAborted();}
 history.unshift(entry);const removed=history.slice(6);history=history.slice(0,6);renderHistory();
 await selectResult(entry.id,true);
 void archiveResult(base,entry).then(()=>log('已自动保存到本机 reports/recordings/'+entry.id)).catch(e=>log('本机自动归档失败：'+e.message));
 for(const id of new Set([...expired,...removed.map(x=>x.id)]))releaseURL(id);
 if(entry.volatile)$('history-count').textContent='本次结果仅在会话中';
 log('录制处理完成：'+JSON.stringify({duration:result.duration,timing:result.timing,metrics:result.metrics,trained:result.trained}));
},pausePlayback);
