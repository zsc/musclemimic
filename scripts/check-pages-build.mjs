import {readFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
const html=await readFile('dist/index.html','utf8');
assert.ok(!html.includes('./src/'),'Pages must publish compiled output');
assert.match(html,/assets\/index-[\w-]+\.js/);
const files=JSON.parse(await readFile('src/generated-assets.json','utf8'));
for(const {path,bytes} of Object.values(files))assert.equal((await readFile(`dist/${path}`)).length,bytes);
console.log('Pages artifact validated');
// Compressed model/policy resources must round-trip byte for byte.
const {gunzipSync}=await import('node:zlib');
for(const [name,path] of Object.entries({actor:'actor.onnx',reference:'reference.json',resets:'reset-states.json',mujoco:'vendor/mujoco.wasm',ort:'vendor/ort-wasm-simd-threaded.asyncify.wasm'})){
 assert.deepEqual(gunzipSync(await readFile(`dist/${files[name].path}`)),await readFile(`public/${path}`));
}
function unpack(b){const size=b.readUInt32LE();return {header:JSON.parse(b.subarray(4,4+size)),offset:4+Math.ceil(size/4)*4};}
const model=gunzipSync(await readFile(`dist/${files.model.path}`));let {header,offset}=unpack(model);
for(const f of header.files){assert.deepEqual(model.subarray(offset,offset+f.bytes),await readFile(`public/${f.path}`));offset+=f.bytes;}
assert.equal(offset,model.length);
const geometry=gunzipSync(await readFile(`dist/${files.geometry.path}`)),original=JSON.parse(await readFile('public/geometry.json','utf8'));
({header,offset}=unpack(geometry));assert.deepEqual(header.geoms,original.geoms);
for(let i=0;i<header.meshes.length;i++){
 for(const v of original.meshes[i].vertices.flat()){assert.equal(geometry.readFloatLE(offset),Math.fround(v));offset+=4;}
 for(const f of original.meshes[i].faces.flat()){assert.equal(geometry.readUInt32LE(offset),f);offset+=4;}
}
assert.equal(offset,geometry.length);console.log('All weights, physics files and rendered float32 geometry preserved exactly');
