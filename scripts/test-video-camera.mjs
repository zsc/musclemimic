import {chromium} from 'playwright';
import assert from 'node:assert/strict';
import {writeFile,readFile} from 'node:fs/promises';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
const fixture = path.resolve('reports/camera-input.y4m');
execFileSync('ffmpeg', ['-hide_banner','-loglevel','error','-y','-i','public/source.mp4','-vf','scale=480:640,fps=30','-pix_fmt','yuv420p',fixture]);
const browser = await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless:true,
  args:['--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream',`--use-file-for-fake-video-capture=${fixture}`]});
const report = {testedAt:new Date().toISOString(),cameraInput:'Recorded source.mp4 via Chrome fake camera; no physical camera accessed',video:[]};
try {
  const page = await browser.newPage({viewport:{width:1440,height:1100}});
  const errors = []; page.on('pageerror', e=>errors.push(e.message));
  page.on('console', m=>{if(m.type()==='error') console.log('CONSOLE',m.text());});
  const mediaURL='http://127.0.0.1:4187/musclelab/source.mp4';
  const originalVideo=await readFile('public/source.mp4');
  const range=await page.request.get(mediaURL,{headers:{Range:'bytes=100-199'}});
  assert.equal(range.status(),206);assert.equal(range.headers()['content-length'],'100');
  assert.deepEqual(await range.body(),originalVideo.subarray(100,200));
  const suffix=await page.request.get(mediaURL,{headers:{Range:'bytes=-100'}});
  assert.equal(suffix.status(),206);assert.deepEqual(await suffix.body(),originalVideo.subarray(-100));
  const invalid=await page.request.get(mediaURL,{headers:{Range:`bytes=${originalVideo.length}-`}});
  assert.equal(invalid.status(),416);
  const head=await page.request.head(mediaURL);assert.equal(Number(head.headers()['content-length']),originalVideo.length);
  report.httpRange=true;
  await page.goto('http://127.0.0.1:4187/musclelab/?backend=wasm');
  await page.waitForFunction(()=>window.__demo?.ready && document.querySelector('#source-video').readyState >= 2);
  await page.evaluate(()=>{
    window.videoSamples=[];
    const video=document.querySelector('#source-video');
    const canvas=document.createElement('canvas');canvas.width=32;canvas.height=32;
    const ctx=canvas.getContext('2d',{willReadFrequently:true});
    function frame(now,meta){
      const m=window.__demo.metrics;
      if(m && !video.paused && !video.seeking && m.step>10 && m.step<140){
        ctx.drawImage(video,0,0,32,32);
        const pixels=ctx.getImageData(0,0,32,32).data;
        let hash=0;for(const x of pixels)hash=(hash*31+x)|0;
        window.videoSamples.push({mediaTime:meta.mediaTime,sim:m.start+m.time,drift:Math.abs(meta.mediaTime-m.start-m.time),hash});
      }
      video.requestVideoFrameCallback(frame);
    }
    video.requestVideoFrameCallback(frame);
  });
  for(const speed of ['1','0.5','0.25']){
    await page.evaluate(()=>window.__demo.reset());
    await page.waitForFunction(()=>window.__demo.metrics.step===0);
    await page.locator('#speed').selectOption(speed);
    await page.evaluate(()=>{window.videoSamples=[];});
    await page.locator('#play').click();
    await page.waitForFunction(()=>window.__demo.metrics?.complete);
    await page.waitForFunction(()=>!document.querySelector('#source-video').seeking);
    const result=await page.evaluate(()=>{
      const v=document.querySelector('#source-video'),m=window.__demo.metrics;
      return {samples:window.videoSamples,paused:v.paused,finalDrift:Math.abs(v.currentTime-m.start-m.time)};
    });
    await writeFile('reports/video-sync-debug.json',JSON.stringify({speed,...result},null,2));
    console.log('VIDEO',speed,result.samples.length,new Set(result.samples.map(s=>s.hash)).size,Math.max(...result.samples.map(s=>s.drift)));
    assert.ok(result.samples.length>15,`No presented video frames at ${speed}x`);
    assert.ok(new Set(result.samples.map(s=>s.hash)).size>10,'Video pixels did not change');
    const maxDrift=Math.max(...result.samples.map(s=>s.drift));
    assert.ok(maxDrift<.15,`Video drift ${maxDrift}s at ${speed}x`);
    assert.ok(result.paused && result.finalDrift<.02,'Video not stopped at final simulation frame');
    report.video.push({speed:Number(speed),presentedFrames:result.samples.length,maxDriftSeconds:maxDrift,finalDriftSeconds:result.finalDrift});
  }
  await page.evaluate(()=>window.__demo.reset());
  await page.waitForFunction(()=>window.__demo.metrics.step===0);
  await page.locator('#play').click();await page.waitForFunction(()=>window.__demo.metrics.step>15);
  await page.locator('#play').click();await page.waitForTimeout(200);
  const paused=await page.evaluate(()=>({step:window.__demo.metrics.step,time:document.querySelector('#source-video').currentTime}));
  await page.waitForTimeout(200);
  assert.deepEqual(await page.evaluate(()=>({step:window.__demo.metrics.step,time:document.querySelector('#source-video').currentTime})),paused);
  await page.locator('#play').click();await page.waitForFunction(step=>window.__demo.metrics.step>step+10,paused.step);
  assert.equal(await page.locator('#source-video').evaluate(v=>v.paused),false);
  await page.locator('#input-source').selectOption('camera');
  await page.locator('#run').click();
  await page.waitForFunction(()=>window.__camera.state.detectedFrames>=8 && window.__camera.state.active || window.__recording.state.stage==='error',null,{timeout:90000});
  assert.ok(await page.evaluate(()=>window.__camera.state.detectedFrames>=8),await page.locator('#recording-status').textContent());
  report.camera=await page.evaluate(()=>window.__camera.state);
  await page.screenshot({path:'reports/camera-desktop.png',fullPage:true});
  await page.setViewportSize({width:390,height:1100});
  await page.screenshot({path:'reports/camera-mobile.png',fullPage:true});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false);
  await page.evaluate(()=>{window.cameraTrack=document.querySelector('#camera-video').srcObject.getVideoTracks()[0];});
  await page.locator('#run').click();
  await page.waitForFunction(()=>!window.__recording.state.busy);
  assert.equal(await page.evaluate(()=>window.cameraTrack.readyState),'ended');
  assert.equal(await page.evaluate(()=>document.querySelector('#camera-video').srcObject),null);
  await page.locator('#run').click();await page.waitForFunction(()=>window.__camera.state.active);
  await page.evaluate(()=>{window.cameraTrack=document.querySelector('#camera-video').srcObject.getVideoTracks()[0];});
  await page.locator('#run').click();await page.waitForFunction(()=>!window.__recording.state.busy);
  assert.equal(await page.evaluate(()=>window.cameraTrack.readyState),'ended');
  await page.evaluate(()=>{navigator.mediaDevices.getUserMedia=async()=>{throw new DOMException('Test denial','NotAllowedError');};});
  await page.locator('#run').click();await page.waitForFunction(()=>!window.__recording.state.busy);
  assert.match(await page.locator('#recording-status').textContent(),/权限未允许/);
  assert.equal(await page.locator('#run').isEnabled(),true);
  assert.equal(await page.evaluate(()=>document.querySelector('#camera-video').srcObject),null);
  assert.deepEqual(errors,[]);
  report.controls={pauseResume:true,cancelReleasesCamera:true,restart:true,permissionDenial:true,mobileNoOverflow:true};
  report.pageErrors=errors;
  await writeFile('reports/video-camera.json',JSON.stringify(report,null,2));
  console.log(JSON.stringify(report,null,2));
} finally {await browser.close();}
