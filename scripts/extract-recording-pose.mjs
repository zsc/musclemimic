import {chromium} from 'playwright';
import {writeFile} from 'node:fs/promises';
const browser=await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true});
try {
 const page=await browser.newPage();
 await page.goto('http://127.0.0.1:4187/musclelab/ATTRIBUTION.html');
 const result=await page.evaluate(async()=>{
  const began=performance.now(),base=new URL('./',location.href).href;
  const worker=new Worker(new URL('pose.worker.js',base));
  const reply=()=>new Promise((resolve,reject)=>{worker.onmessage=({data})=>data.type==='error'?reject(Error(data.message)):resolve(data);worker.onerror=e=>reject(Error(e.message));});
  let wait=reply();worker.postMessage({type:'load',base});await wait;
  const loaded=performance.now();
  const v=document.createElement('video');v.muted=true;v.preload='auto';v.src=new URL('source.mp4',base);document.body.append(v);
  await new Promise((resolve,reject)=>{v.onloadeddata=resolve;v.onerror=reject;});
  const frames=[];
  for(let i=0;i<50;i++){
   const t=i/30;
   if(Math.abs(v.currentTime-t)>.0001){const seek=new Promise(resolve=>v.onseeked=resolve);v.currentTime=t;await seek;}
   const bitmap=await createImageBitmap(v);wait=reply();worker.postMessage({type:'frame',bitmap,timestamp:t*1000},[bitmap]);
   const r=await wait;r.bitmap.close();delete r.bitmap;frames.push(r);
  }
  worker.terminate();return {input:'public/source.mp4',fps:30,sourceDuration:v.duration,modelLoadMs:loaded-began,decodeAndPoseMs:performance.now()-loaded,totalMs:performance.now()-began,frames};
 });
 await writeFile('reports/no-finetune/poses.json',JSON.stringify(result));
 console.log({frames:result.frames.length,detected:result.frames.filter(f=>f.worldLandmarks.length===33).length,modelLoadMs:result.modelLoadMs,decodeAndPoseMs:result.decodeAndPoseMs,totalMs:result.totalMs});
}finally{await browser.close();}
