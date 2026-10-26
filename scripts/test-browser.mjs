import {chromium} from 'playwright';
import {writeFile,mkdir} from 'node:fs/promises';
const backend=process.argv[2]||'webgpu';
const browser=await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,args:[]});
const page=await browser.newPage({viewport:{width:1440,height:1100}});const errors=[];
page.on('pageerror',e=>{errors.push(e.message);console.log('PAGE ERROR',e.message)});
page.on('console',m=>{if(m.type()==='error')console.log('CONSOLE ERROR',m.text())});
await mkdir('reports',{recursive:true});
try{
 await page.goto(`http://127.0.0.1:4187/musclelab/?backend=${backend}`,{waitUntil:'domcontentloaded'});
 await page.waitForFunction(()=>window.__demo?.ready||document.querySelector('#loading')?.classList.contains('failed'),{},{timeout:180000});
 console.log('UI',await page.locator('#log').textContent());
 if(!await page.evaluate(()=>window.__demo.ready))throw Error(await page.locator('#loading-text').textContent());
 const actor=await page.evaluate(()=>window.__demo.rpc('validateActor'));console.log('ACTOR',actor);
 const run=await page.evaluate(()=>window.__demo.rpc('run',{start:10,seed:3}));console.log('RUN',JSON.stringify(run.metrics),JSON.stringify(run.timing));
 await page.waitForTimeout(100);await page.screenshot({path:`reports/demo-${backend}-desktop.png`,fullPage:true});
 await page.evaluate(()=>window.__demo.reset());await page.waitForFunction(()=>window.__demo.metrics?.step===0);
 await page.locator('#play').click();await page.waitForFunction(()=>window.__demo.metrics?.step>=3);await page.locator('#play').click();
 await page.waitForTimeout(100);const paused=await page.evaluate(()=>window.__demo.metrics.step);await page.waitForTimeout(150);if(paused!==await page.evaluate(()=>window.__demo.metrics.step))throw Error('Pause failed');
 await page.setViewportSize({width:390,height:844});await page.screenshot({path:`reports/demo-${backend}-mobile.png`,fullPage:true});
 const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1);if(overflow)throw Error('Mobile overflow');
 await page.setViewportSize({width:1440,height:1000});await page.evaluate(()=>window.__demo.reset());await page.waitForFunction(()=>window.__demo.metrics?.step===0);await page.locator('#play').click();await page.waitForFunction(()=>window.__demo.metrics?.complete);const paced=await page.evaluate(()=>window.__demo.rpc('report'));
 await page.locator('.result-menu summary').click();const downloadWait=page.waitForEvent('download');await page.locator('#export-report').click();const download=await downloadWait;await download.saveAs(`reports/download-${backend}.json`);
 const resources=await page.evaluate(async()=>{const paths=['validation.json','ATTRIBUTION.html','licenses/musclemimic-models-NOTICE','licenses/musclemimic-models-LICENSE','licenses/MuscleMimic-LICENSE','licenses/onnxruntime-LICENSE','licenses/mujoco-LICENSE'];return Promise.all(paths.map(async p=>[p,(await fetch(new URL(p,location.href))).status]));});if(resources.some(x=>x[1]!==200))throw Error('Missing attribution or validation resource: '+JSON.stringify(resources));
 await page.locator('#advanced summary').click();const other=backend==='webgpu'?'wasm':'webgpu';await page.locator('#backend').selectOption(other);await page.waitForFunction(()=>window.__demo.ready,{},{timeout:180000});const switched=await page.evaluate(()=>window.__demo.rpc('validateActor'));if(!switched.passed)throw Error('Backend switch invalid');
 const result={backend,actor,run,paced,errors,resources,backendSwitch:true,download:true,controls:{pause:true,reset:true,mobileNoOverflow:!overflow},testedAt:new Date().toISOString()};
 await writeFile(`reports/browser-${backend}.json`,JSON.stringify(result,null,2));if(errors.length||!actor.passed)throw Error('Browser validation failed');
 console.log('PASS browser smoke');
}finally{await browser.close();}
