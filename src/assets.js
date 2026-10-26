import assets from './generated-assets.json';
let cachePromise;
async function assetCache(base){
 if(!globalThis.caches)return null;
 return cachePromise??=(async()=>{
  try{
   const cache=await caches.open('musclemimic-packed-v1');
   const current=new Set(Object.values(assets).flatMap(a=>a.parts.map(p=>new URL(p.path,base).href)));
   const prefix=new URL('packed/',base).href;
   await Promise.all((await cache.keys()).filter(r=>r.url.startsWith(prefix)&&!current.has(r.url)).map(r=>cache.delete(r)));
   return cache;
  }catch{return null;}
 })();
}
export async function loadAsset(name,base,onProgress) {
 const item=assets[name],cache=await assetCache(base),result=new Uint8Array(item.decodedBytes);
 let next=0,completed=0,lastProgress=0;
 const offsets=[];let offset=0;
 for(const part of item.parts){offsets.push(offset);offset+=part.decodedBytes;}
 async function part(index){
  const info=item.parts[index],url=new URL(info.path,base).href;
  let response;try{response=await cache?.match(url);}catch{}
  const cached=Boolean(response);response??=await fetch(url);
  if(!response.ok)throw Error(`${name}: HTTP ${response.status}`);
  // Content-addressed URLs keep cached weights tied to this exact build.
  const saving=!cached&&cache?cache.put(url,response.clone()).catch(()=>{}):Promise.resolve();
  const counted=response.body.pipeThrough(new TransformStream({transform(chunk,controller){
   completed+=chunk.length;
   if(onProgress&&(performance.now()-lastProgress>200||completed===item.bytes)){lastProgress=performance.now();onProgress(completed,item.bytes);}
   controller.enqueue(chunk);
  }}));
  const data=await new Response(counted.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
  if(data.byteLength!==info.decodedBytes)throw Error(`${name}: incomplete asset`);
  result.set(new Uint8Array(data),offsets[index]);await saving;
 }
 await Promise.all(Array.from({length:Math.min(4,item.parts.length)},async()=>{while(next<item.parts.length)await part(next++);}));
 return result.buffer;
}
export async function loadJSONAsset(name,base){return JSON.parse(new TextDecoder().decode(await loadAsset(name,base)));}
export function unpack(data){
 const size=new DataView(data).getUint32(0,true);
 return {header:JSON.parse(new TextDecoder().decode(new Uint8Array(data,4,size))),offset:4+Math.ceil(size/4)*4};
}
export async function loadGeometry(base){
 const data=await loadAsset('geometry',base);let {header,offset}=unpack(data);
 const meshes=header.meshes.map(m=>{
  const vertices=new Float32Array(data,offset,m.vertices);offset+=m.vertices*4;
  const faces=new Uint32Array(data,offset,m.faces);offset+=m.faces*4;
  return {vertices,faces};
 });
 return {geoms:header.geoms,meshes};
}
