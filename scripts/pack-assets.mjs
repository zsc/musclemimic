import {readFile,writeFile,mkdir,rm} from 'node:fs/promises';
import {gzipSync} from 'node:zlib';
import {createHash} from 'node:crypto';
const root='public/packed';await rm(root,{recursive:true,force:true});await mkdir(root,{recursive:true});
const assets={};
async function save(name,data){
 const parts=[];
 const size=name==='actor'?4*1024*1024:data.length;
 for(let offset=0;offset<data.length;offset+=size){
  const chunk=data.subarray(offset,offset+size),packed=gzipSync(chunk,{level:9});
  const hash=createHash('sha256').update(packed).digest('hex').slice(0,16);
  const path=`packed/${name}.${hash}.gz`;await writeFile(`public/${path}`,packed);
  parts.push({path,bytes:packed.length,decodedBytes:chunk.length});
 }
 assets[name]={parts,bytes:parts.reduce((n,p)=>n+p.bytes,0),decodedBytes:data.length};
}
function pack(header,chunks){
 const json=Buffer.from(JSON.stringify(header)),padded=Math.ceil(json.length/4)*4,head=Buffer.alloc(4+padded);
 head.writeUInt32LE(json.length);json.copy(head,4);return Buffer.concat([head,...chunks]);
}
const geometry=JSON.parse(await readFile('public/geometry.json','utf8'));
const chunks=[],meshes=[];
for(const m of geometry.meshes){
 const v=new Float32Array(m.vertices.flat()),f=new Uint32Array(m.faces.flat());
 meshes.push({vertices:v.length,faces:f.length});chunks.push(Buffer.from(v.buffer),Buffer.from(f.buffer));
}
await save('geometry',pack({geoms:geometry.geoms,meshes},chunks));
const manifest=JSON.parse(await readFile('public/manifest.json','utf8')),files=[],modelChunks=[];
for(const f of manifest.model_files){const b=await readFile(`public/${f.path}`);files.push({path:f.path,bytes:b.length});modelChunks.push(b);}
await save('model',pack({files},modelChunks));
for(const [name,path] of Object.entries({actor:'actor.onnx',reference:'reference.json',resets:'reset-states.json',mujoco:'vendor/mujoco.wasm',ort:'vendor/ort-wasm-simd-threaded.asyncify.wasm'}))await save(name,await readFile(`public/${path}`));
await writeFile('src/generated-assets.json',JSON.stringify(assets));
console.log('Packed startup assets:',JSON.stringify(assets));
