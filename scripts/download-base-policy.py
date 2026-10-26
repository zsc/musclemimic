"""Fetch the public, unmodified MuscleMimic base checkpoint (data only)."""
import json,urllib.request,concurrent.futures
from pathlib import Path
root=Path(__file__).resolve().parents[1]
dest=root/'inputs/mm-fullbody-base'
files=json.loads((root/'reports/tracking-fix/base-model-files.json').read_text())
def download(info):
 if info['type']!='file':return
 target=dest/info['path'];target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists() and target.stat().st_size==info['size']:return
 url='https://huggingface.co/amathislab/mm-fullbody-base/resolve/main/'+info['path']
 temporary=target.with_name(target.name+'.partial')
 urllib.request.urlretrieve(url,temporary)
 assert temporary.stat().st_size==info['size']
 temporary.rename(target);print(info['path'],info['size'],flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(download,files))
