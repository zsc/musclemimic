import {mkdir,copyFile,readdir,readFile,writeFile} from 'node:fs/promises';
await mkdir('public/vendor',{recursive:true});await mkdir('public/licenses',{recursive:true});
for(const file of ['mujoco.js','mujoco.wasm'])await copyFile('node_modules/@mujoco/mujoco/'+file,'public/vendor/'+file);
for(const file of await readdir('node_modules/onnxruntime-web/dist'))if(/^ort-wasm.*\.(wasm|mjs)$/.test(file))await copyFile('node_modules/onnxruntime-web/dist/'+file,'public/vendor/'+file);
for(const [pkg,prefix] of [['@mujoco/mujoco','mujoco'],['onnxruntime-web','onnxruntime'],['three','three']]){
 for(const file of await readdir('node_modules/'+pkg))if(/^(LICENSE|NOTICE)/i.test(file))await copyFile(`node_modules/${pkg}/${file}`,`public/licenses/${prefix}-${file}`);
}
await writeFile('public/.nojekyll','');

await mkdir('public/vendor/mediapipe/wasm',{recursive:true});
await copyFile('node_modules/@mediapipe/tasks-vision/vision_bundle.mjs','public/vendor/mediapipe/vision_bundle.mjs');
for(const file of await readdir('node_modules/@mediapipe/tasks-vision/wasm'))await copyFile('node_modules/@mediapipe/tasks-vision/wasm/'+file,'public/vendor/mediapipe/wasm/'+file);
await copyFile('src/pose.worker.js','public/pose.worker.js');
