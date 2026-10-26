import {chromium} from 'playwright';
import {readFile,writeFile} from 'node:fs/promises';
import path from 'node:path';
const folder=path.resolve(process.argv[2]);
const metadata=JSON.parse(await readFile(path.join(folder,'metadata.json'),'utf8'));
const bytes=await readFile(path.join(folder,'source.video'));
const browser=await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true});
try {
 const page=await browser.newPage();await page.goto('http://127.0.0.1:4187/musclelab/ATTRIBUTION.html');
 const result=await page.evaluate(async input=>{
  const base=new URL('./',location.href).href,worker=new Worker(new URL('pose.worker.js',base));
  const reply=()=>new Promise((resolve,reject)=>{worker.onmessage=({data})=>data.type==='error'?reject(Error(data.message)):resolve(data);worker.onerror=e=>reject(Error(e.message));});
  let pending=reply();worker.postMessage({type:'load',base});await pending;
  const video=document.createElement('video');video.muted=true;video.preload='auto';
  video.src=URL.createObjectURL(new Blob([Uint8Array.from(atob(input.base64),c=>c.charCodeAt(0))],{type:input.type}));document.body.append(video);
  await new Promise((resolve,reject)=>{video.onloadeddata=resolve;video.onerror=reject;});
  const seek=async time=>{if(Math.abs(video.currentTime-time)<.0001)return;const done=new Promise(resolve=>video.onseeked=resolve);video.currentTime=time;await done;};
  if(!Number.isFinite(video.duration)){await seek(1e6);await seek(0);}
  const count=Math.floor((Math.min(video.duration,2)-1/60)*30)+1,frames=[];
  for(let i=0;i<count;i++){
   await seek(i/30);const bitmap=await createImageBitmap(video);pending=reply();worker.postMessage({type:'frame',bitmap,timestamp:i/30*1000},[bitmap]);
   const r=await pending;r.bitmap.close();frames.push({timestamp:i/30*1000,landmarks:r.landmarks,worldLandmarks:r.worldLandmarks});
  }
  worker.terminate();return {frames};
 },{base64:bytes.toString('base64'),type:metadata.videoType});
 await writeFile(path.join(folder,'poses-reextracted.json'),JSON.stringify(result));console.log({id:metadata.id,frames:result.frames.length,detected:result.frames.filter(f=>f.worldLandmarks.length===33).length});
}finally{await browser.close();}
process.exit(0);
