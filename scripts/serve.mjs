import http from 'node:http';
import {recordingAPI} from './recording-api.mjs';
import {stat} from 'node:fs/promises';
import {createReadStream} from 'node:fs';
import path from 'node:path';
const root=path.resolve('dist');
const mime={'.html':'text/html','.js':'text/javascript','.mjs':'text/javascript','.css':'text/css','.wasm':'application/wasm','.json':'application/json','.mp4':'video/mp4','.svg':'image/svg+xml'};
http.createServer(async(req,res)=>{try{
 if(await recordingAPI(req,res))return;
 if(!['GET','HEAD'].includes(req.method)){res.writeHead(405,{'Allow':'GET, HEAD'});res.end();return;}
 let pathname=decodeURIComponent(new URL(req.url,'http://localhost').pathname);
 if(pathname.startsWith('/musclelab/'))pathname=pathname.slice('/musclelab'.length);
 if(pathname.endsWith('/'))pathname+='index.html';
 const file=path.resolve(root,'.'+pathname);if(!file.startsWith(root+path.sep))throw Error('Invalid path');
 const s=await stat(file);if(!s.isFile())throw Error('Not a file');
 const headers={'Content-Type':mime[path.extname(file)]||'application/octet-stream','Cache-Control':'no-cache','Accept-Ranges':'bytes'};
 let start=0,end=s.size-1,status=200;
 if(req.headers.range && req.method==='GET'){
   const match=/^bytes=(\d*)-(\d*)$/.exec(req.headers.range);
   if(match && (match[1] || match[2])){
     start=match[1]?Number(match[1]):Math.max(0,s.size-Number(match[2]));
     end=match[1] && match[2]?Math.min(Number(match[2]),s.size-1):s.size-1;
     if(!Number.isSafeInteger(start)||!Number.isSafeInteger(end)||start>end||start>=s.size){
       res.writeHead(416,{...headers,'Content-Range':`bytes */${s.size}`});res.end();return;
     }
     status=206;headers['Content-Range']=`bytes ${start}-${end}/${s.size}`;
   }
 }
 headers['Content-Length']=Math.max(0,end-start+1);
 res.writeHead(status,headers);
 if(req.method==='HEAD'||s.size===0){res.end();return;}
 const body=createReadStream(file,{start,end});body.on('error',()=>res.destroy());
 res.on('close',()=>body.destroy());body.pipe(res);
 }catch{res.writeHead(404);res.end('Not found');}
}).listen(4187,'127.0.0.1',()=>console.log('http://127.0.0.1:4187/musclelab/'));
