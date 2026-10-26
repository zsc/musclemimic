"""Consolidate reports on air5. No inference or simulation here."""
import hashlib,json,platform,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
def read(p):return json.loads((root/p).read_text())
def write(p,v):(root/p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
validation=read('public/validation.json')
physics=read('reports/physics-parity.json')
native=read('inputs/outputs/temporal_contact_lowstd025_v1/acceptance_13095.json')
validation['browser_verified']=True
validation['device']={'model':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),'os':platform.platform(),'browser':subprocess.check_output(['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome','--version'],text=True).strip()}
validation['physics_parity']=physics
validation['browser']={}
for b in ['webgpu','wasm']:
 audit=read('reports/acceptance-'+b+'.json');smoke=read('reports/browser-'+b+'.json')
 validation['browser'][b]={'passed':audit['passed'],'total':audit['total'],'all_passed':audit['allPassed'],'actor':smoke['actor'],'timing_single_run':smoke['run']['timing'],'paced_wall_seconds':smoke['paced']['timing']['wallMs']/1000,'controls':smoke['controls'],'page_errors':audit['pageErrors'],'adapter':smoke['run']['adapterInfo'],'runs':[r['metrics'] for r in audit['runs']]}
 deltas=[]
 for r in audit['runs']:
  m=r['metrics'];n=next(n for n in native['runs'] if n['seed']==m['seed'] and abs(n['source_start']-m['start'])<1e-6)
  deltas.append(abs(m['final']-n['final_site_error_mean_mm']))
 validation['browser'][b]['max_final_error_delta_from_native_mm']=max(deltas)
write('public/validation.json',validation)
files={str(p.relative_to(root/'public')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'public').rglob('*') if p.is_file() and 'vendor' not in p.parts and p.name!='asset-sha256.json'}
write('public/asset-sha256.json',files)
print(json.dumps({b:{k:v for k,v in x.items() if k not in ['runs','controls','adapter']} for b,x in validation['browser'].items()},indent=2))
