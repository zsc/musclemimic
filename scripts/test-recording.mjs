import {chromium} from 'playwright';
import assert from 'node:assert/strict';
import {writeFile,readFile,stat} from 'node:fs/promises';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {setTimeout as delay} from 'node:timers/promises';
async function waitAPI(page,route,ready) {
 const deadline=Date.now()+30000;
 while(Date.now()<deadline){
  const response=await page.request.get(`http://127.0.0.1:4187/musclelab/api/${route}`);
  if(response.ok()&&ready(await response.json()))return;
  await delay(100);
 }
 throw Error(`API condition timed out: ${route}`);
}
execFileSync('ffmpeg',['-hide_banner','-loglevel','error','-y','-i','public/source.mp4','-vf','scale=480:640,fps=30','-pix_fmt','yuv420p','reports/camera-input.y4m']);
const browser=await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,args:['--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream',`--use-file-for-fake-video-capture=${path.resolve('reports/camera-input.y4m')}`]});
const report={testedAt:new Date().toISOString(),camera:'Chrome fake camera; no physical webcam accessed'};
try{
 const page=await browser.newPage({viewport:{width:1440,height:900},extraHTTPHeaders:{'X-Demo-Test':'fake-camera'}}),errors=[];let processingRequests=0;
 page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(r.method()==='POST'&&r.url().endsWith('/api/recording'))processingRequests++;});
 await page.goto('http://127.0.0.1:4187/musclelab/?backend=wasm');
 await page.waitForFunction(()=>window.__demo?.ready,null,{timeout:60000});
 assert.equal(await page.locator('#advanced').getAttribute('open'),null);
 await page.screenshot({path:'reports/workbench-initial.png',fullPage:true});
 report.desktopHeight=await page.evaluate(()=>document.documentElement.scrollHeight);
 assert.ok(report.desktopHeight<=930,`Desktop UI too tall: ${report.desktopHeight}`);
 await page.locator('#input-source').selectOption('example');await page.locator('#run').click();
 await page.waitForFunction(()=>!window.__recording.state.busy,null,{timeout:60000});
 const first=await page.evaluate(()=>({result:window.__demo.recording,id:document.querySelector('#result-select').value}));
 assert.ok(first.result,await page.locator('#recording-status').textContent());assert.equal(first.result.trained,false);assert.equal(first.result.external_forces,false);
 await page.waitForFunction(()=>window.__demo.metrics.complete);
 await page.waitForFunction(()=>document.querySelector('#play').getAttribute('aria-label')==='重播');
 assert.equal(await page.locator('#play').getAttribute('aria-label'),'重播');
 assert.equal(await page.locator('#result-select option').count(),2);
 report.example={timing:first.result.timing,metrics:first.result.metrics};report.oneClickAutoplay=true;
 console.log('EXAMPLE',JSON.stringify(report.example));
 // Range input drives native snapshots and the original video to the same timestamp.
 await page.locator('#seek').evaluate(el=>{el.value='50';el.dispatchEvent(new Event('input',{bubbles:true}));});
 await page.waitForFunction(()=>window.__demo.metrics.step===50&&!document.querySelector('#source-video').seeking);
 assert.equal(await page.locator('#source-video').evaluate(v=>v.paused),true);
 assert.ok(await page.evaluate(()=>Math.abs(document.querySelector('#source-video').currentTime-.5)<.025));
 assert.equal(await page.locator('#play').getAttribute('aria-label'),'播放');
 const stableMean=await page.locator('#mean-error').textContent();
 await page.screenshot({path:'reports/workbench-result.png',fullPage:true});
 await page.locator('#seek').evaluate(el=>{el.value='125';el.dispatchEvent(new Event('input',{bubbles:true}));});
 await page.waitForFunction(()=>window.__demo.metrics.step===125);
 assert.equal(await page.locator('#mean-error').textContent(),stableMean);
 await page.locator('#play').click();await page.waitForFunction(()=>window.__demo.metrics.complete);
 report.scrubAndReplay=true;
 // Same button cancels the countdown and releases the acquired track.
 await page.locator('#input-source').selectOption('camera');await page.locator('#run').click();
 await page.waitForFunction(()=>window.__recording.state.stage==='countdown');
 await page.evaluate(()=>{window.savedTrack=document.querySelector('#camera-video').srcObject.getVideoTracks()[0];});
 await page.screenshot({path:'reports/workbench-countdown.png',fullPage:true});
 await page.locator('#run').click();await page.waitForFunction(()=>!window.__recording.state.busy);
 assert.equal(await page.evaluate(()=>window.savedTrack.readyState),'ended');assert.equal(await page.locator('#result-select').inputValue(),first.id);
 report.countdownCancellation=true;
 // One click from a closed camera completes every stage, while the old result stays visible.
 await page.locator('#run').click();await page.waitForFunction(()=>window.__recording.state.stage==='recording');
 await page.evaluate(()=>{window.savedTrack=document.querySelector('#camera-video').srcObject.getVideoTracks()[0];});
 await page.waitForFunction(()=>window.__recording.state.stage==='processing');
 assert.equal(await page.locator('#result-select').inputValue(),first.id);assert.equal(await page.locator('#simulation-panel').isVisible(),true);
 await page.locator('#result-select').selectOption('original');
 await page.waitForFunction(()=>!window.__recording.state.busy,null,{timeout:60000});
 const second=await page.evaluate(()=>({result:window.__demo.recording,id:document.querySelector('#result-select').value}));
 assert.notEqual(second.id,first.id,await page.locator('#recording-status').textContent());
 assert.equal(await page.evaluate(()=>window.savedTrack.readyState),'ended');assert.match(await page.locator('#source-video').getAttribute('src'),/^blob:/);
 report.camera={timing:second.result.timing,metrics:second.result.metrics};console.log('CAMERA',JSON.stringify(report.camera));
 const archive=path.resolve('reports/recordings',second.id);
 await waitAPI(page,`recordings/${second.id}`,s=>s.metadata&&s.video);
 assert.ok((await stat(path.join(archive,'source.video'))).size>1000);
 assert.equal(JSON.parse(await readFile(path.join(archive,'result.json'),'utf8')).metrics.mean,second.result.metrics.mean);
 assert.ok(JSON.parse(await readFile(path.join(archive,'poses.json'),'utf8')).frames.length>=15);
 report.automaticLocalArchive={id:second.id,video:true,poses:true,result:true};
 await page.waitForFunction(()=>window.__demo.metrics.complete);
 // History includes its video blob and survives refresh, without posting another job.
 const requestsBefore=processingRequests;await page.reload();
 await page.waitForFunction(()=>window.__demo?.ready&&window.__demo.history.length===2&&window.__demo.recording,null,{timeout:60000});
 assert.equal(await page.locator('#result-select').inputValue(),second.id);
 await page.waitForFunction(()=>document.querySelector('#source-video').readyState>=2);
 assert.match(await page.locator('#source-video').getAttribute('src'),/^blob:/);
 assert.equal(processingRequests,requestsBefore);
 await page.locator('#result-select').selectOption(first.id);await page.waitForFunction(()=>window.__demo.metrics.step===0&&document.querySelector('#source-video').src.endsWith('/source.mp4'));
 assert.equal(processingRequests,requestsBefore);report.persistentHistory=true;
 // Upload chooses input; the only primary action generates it.
 await page.locator('#input-source').selectOption('upload');await page.locator('#recording-file').setInputFiles('public/source.mp4');
 assert.equal(await page.evaluate(()=>window.__recording.state.busy),false);
 await page.locator('#run').click();await page.waitForFunction(()=>!window.__recording.state.busy,null,{timeout:60000});
 assert.equal(await page.evaluate(()=>window.__demo.history.length),3);
 const thirdId=await page.locator('#result-select').inputValue();report.upload=true;
 await page.waitForFunction(()=>window.__demo.metrics.complete);
 await page.locator('.result-menu summary').click();const downloadWait=page.waitForEvent('download');await page.locator('#export-report').click();const download=await downloadWait;await download.saveAs('reports/workbench-export.json');
 // Native cancellation leaves the result and history intact.
 await page.locator('#input-source').selectOption('example');await page.locator('#run').click();
 await page.waitForFunction(()=>/拟合肌骨关节/.test(window.__recording.state.message),null,{timeout:30000});
 await page.locator('#run').click();await page.waitForFunction(()=>!window.__recording.state.busy);
 await waitAPI(page,'recording',s=>!s.busy);
 assert.equal(await page.locator('#result-select').inputValue(),thirdId);assert.equal(await page.evaluate(()=>window.__demo.history.length),3);
 report.nativeCancellation=true;
 // Permission errors and cancellation while permission is pending recover immediately.
 await page.evaluate(()=>{navigator.mediaDevices.getUserMedia=async()=>{throw new DOMException('test','NotAllowedError');};});
 await page.locator('#input-source').selectOption('camera');await page.locator('#run').click();await page.waitForFunction(()=>!window.__recording.state.busy);
 assert.match(await page.locator('#recording-status').textContent(),/权限未允许/);
 await page.evaluate(()=>{navigator.mediaDevices.getUserMedia=()=>new Promise(resolve=>{window.permissionResolve=resolve;});});
 await page.locator('#run').click();await page.waitForFunction(()=>window.__recording.state.stage==='starting');
 await page.locator('#run').click();await page.waitForFunction(()=>!window.__recording.state.busy);
 await page.evaluate(()=>{const c=document.createElement('canvas');window.lateStream=c.captureStream(30);window.permissionResolve(window.lateStream);});
 await page.waitForFunction(()=>window.lateStream.getTracks()[0].readyState==='ended');report.permissionRecovery=true;
 // Delete only a result created in this isolated test browser.
 await page.locator('.result-menu summary').click();await page.locator('#delete-result').click();
 await page.waitForFunction(()=>window.__demo.history.length===2);assert.notEqual(await page.locator('#result-select').inputValue(),thirdId);
 await page.reload();await page.waitForFunction(()=>window.__demo?.recording&&window.__demo.history.length===2);
 assert.equal(await page.locator(`#result-select option[value="${thirdId}"]`).count(),0);report.deletePersists=true;
 // An old browser-only entry is archived on refresh, without rerunning physics.
 const legacyId=await page.evaluate(async id=>{
   const db=await new Promise((resolve,reject)=>{const r=indexedDB.open('musclemimic-results',1);r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);});
   const entry=await new Promise((resolve,reject)=>{const r=db.transaction('results').objectStore('results').get(id);r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);});
   entry.id=crypto.randomUUID();entry.created=Date.now();entry.title='Legacy archive regression';
   await new Promise((resolve,reject)=>{const tx=db.transaction('results','readwrite');tx.objectStore('results').put(entry);tx.oncomplete=resolve;tx.onerror=()=>reject(tx.error);});
   db.close();return entry.id;
 },second.id);
 const beforeBackfill=processingRequests;await page.reload();
 await waitAPI(page,`recordings/${legacyId}`,s=>s.metadata&&s.video);
 assert.equal(processingRequests,beforeBackfill);
 assert.ok((await stat(path.resolve('reports/recordings',legacyId,'source.video'))).size>1000);
 report.legacyBackfill=true;
 await page.setViewportSize({width:390,height:900});await page.screenshot({path:'reports/workbench-mobile.png',fullPage:true});
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false);report.mobileNoOverflow=true;
 assert.deepEqual(errors,[]);report.errors=errors;
 await writeFile('reports/workbench-test.json',JSON.stringify(report,null,2));console.log('PASS',JSON.stringify(report,null,2));
}finally{await browser.close();}
process.exit(0);
