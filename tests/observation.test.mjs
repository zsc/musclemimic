import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createObservation,rotationVector,acceptance} from '../src/observation.js';
const read=p=>JSON.parse(readFileSync(new URL('../public/'+p,import.meta.url)));
const manifest=read('manifest.json'),ref=read('reference.json'),fixtures=read('validation-fixtures.json');
for(const fixture of fixtures)test(`Native observation parity, step ${fixture.step}`,()=>{
 const actual=createObservation(fixture.fields,manifest,ref.goal[fixture.reference_frame]);
 let max=0,ix=0;actual.forEach((v,i)=>{const e=Math.abs(v-Math.fround(fixture.obs[i]));if(e>max){max=e;ix=i}});
 assert.ok(max<2e-5,`max ${max} at ${ix}: JS ${actual[ix]}, native ${fixture.obs[ix]}`);
});
test('Rotation vectors remain finite at pi',()=>assert.deepEqual(rotationVector([1,0,0,0,-1,0,0,0,-1]),[Math.PI,0,0]));
test('Incomplete and failed trajectories cannot pass',()=>{
 const gates={mean_mm:100,final_mm:150,root_max_mm:200};
 assert.equal(acceptance([1],[1],gates,false).passed,false);
 assert.equal(acceptance([90,160],[10,20],gates,true).passed,false);
});
