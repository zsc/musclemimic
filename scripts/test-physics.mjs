import fs from 'node:fs';import assert from 'node:assert/strict';import load from '@mujoco/mujoco';
import {Physics} from '../src/physics.js';
const json=p=>JSON.parse(fs.readFileSync(p));const manifest=json('public/manifest.json');const mj=await load();
mj.FS.mkdir('/model');mj.FS.mkdir('/model/assets');for(const f of manifest.model_files)mj.FS.writeFile('/'+f.path,fs.readFileSync('public/'+f.path));
const model=mj.MjModel.from_xml_path('/model/model.xml');const p=new Physics(mj,model,manifest,json('public/reference.json'),json('public/reset-states.json'));
p.reset(10,3);const inputs=json('reports/native-physics-34.json'),expected=json('reports/native-physics-317.json');
let qposError=0,qvelError=0,versionDrift=0;
for(let i=0;i<inputs.steps.length;i++){
 p.advance(inputs.steps[i].ctrl);
 for(let j=0;j<model.nq;j++){qposError=Math.max(qposError,Math.abs(p.data.qpos[j]-expected.steps[i].qpos[j]));versionDrift=Math.max(versionDrift,Math.abs(p.data.qpos[j]-inputs.steps[i].qpos[j]));}
 for(let j=0;j<model.nv;j++)qvelError=Math.max(qvelError,Math.abs(p.data.qvel[j]-expected.steps[i].qvel[j]));
}
const result={wasm:mj.mj_versionString(),native:expected.version,steps:inputs.steps.length,substeps:5,qposMaxAbs:qposError,qvelMaxAbs:qvelError,source34QposMaxAbs:versionDrift,arenaBytes:model.narena,maxUsedArenaBytes:expected.max_used_arena_bytes,warnings:expected.warnings};
fs.writeFileSync('reports/physics-parity.json',JSON.stringify(result,null,2));console.log(result);
assert.ok(qposError<1e-8);assert.ok(qvelError<1e-7);assert.ok(expected.warnings.every(v=>v===0));
