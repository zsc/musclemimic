import {chromium} from 'playwright';import {writeFile,readFile} from 'node:fs/promises';
const browser=await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,args:['--enable-unsafe-webgpu','--ignore-gpu-blocklist']});
const baseline=JSON.parse(await readFile('inputs/outputs/temporal_contact_lowstd025_v1/acceptance_13095.json'));
try{
 for(const backend of ['webgpu','wasm']){
  const page=await browser.newPage({viewport:{width:1440,height:1000}});const pageErrors=[];page.on('pageerror',e=>pageErrors.push(e.message));
  await page.goto(`http://127.0.0.1:4187/musclelab/?backend=${backend}`);await page.waitForFunction(()=>window.__demo?.ready,{},{timeout:180000});
  const runs=await page.evaluate(()=>window.__demo.rpc('runAll'));
  const result={backend,passed:runs.filter(r=>r.metrics.passed).length,total:runs.length,allPassed:runs.every(r=>r.metrics.passed),pageErrors,runs};
  await writeFile(`reports/acceptance-${backend}.json`,JSON.stringify(result,null,2));console.log(backend,JSON.stringify(runs.map(r=>r.metrics)));
  if(pageErrors.length||runs.length!==15||runs.some(r=>!r.metrics.complete||!r.metrics.finite))throw Error('Incomplete/invalid audit');
  await page.close();
 }
}finally{await browser.close();}
