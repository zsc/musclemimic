// Browser history plus an archive on the same-machine, loopback-only server.
const LIMIT = 6;
let dbPromise;
function database() {
  if (!dbPromise) dbPromise = new Promise((resolve, reject) => {
    const request = indexedDB.open('musclemimic-results', 1);
    request.onupgradeneeded = () => request.result.createObjectStore('results', {keyPath:'id'});
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
    request.onblocked = () => reject(Error('历史记录被另一页面占用'));
  });
  return dbPromise;
}
export async function savedResults() {
  const db = await database();
  return new Promise((resolve, reject) => {
    const request = db.transaction('results').objectStore('results').getAll();
    request.onsuccess = () => resolve(request.result.sort((a,b) => b.created - a.created));
    request.onerror = () => reject(request.error);
  });
}
export async function saveResult(entry) {
  const entries = await savedResults(), db = await database();
  const expired = [entry, ...entries.filter(x => x.id !== entry.id)].sort((a,b) => b.created - a.created).slice(LIMIT);
  await new Promise((resolve, reject) => {
    const tx = db.transaction('results', 'readwrite'), store = tx.objectStore('results');
    store.put(entry);expired.forEach(x => store.delete(x.id));
    tx.oncomplete = resolve;tx.onerror = () => reject(tx.error);tx.onabort = () => reject(tx.error);
  });
  return expired.map(x => x.id);
}
export async function deleteResult(id) {
  const db = await database();
  await new Promise((resolve, reject) => {
    const tx = db.transaction('results', 'readwrite');tx.objectStore('results').delete(id);
    tx.oncomplete = resolve;tx.onerror = () => reject(tx.error);
  });
}

export async function archiveResult(base, entry) {
  const url = new URL(`api/recordings/${entry.id}`, base);
  const check = await fetch(url);
  if (!check.ok) throw Error('本机归档服务不可用');
  const status = await check.json();
  if (!status.metadata) {
    const response = await fetch(url, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:entry.id,created:entry.created,title:entry.title,result:entry.result,hasVideo:Boolean(entry.video),videoType:entry.video?.type})});
    if (!response.ok) throw Error('运行记录保存失败');
  }
  if (entry.video && !status.video) {
    const response = await fetch(`${url}/video`, {method:'POST',headers:{'Content-Type':entry.video.type||'application/octet-stream'},body:entry.video});
    if (!response.ok) throw Error('录制视频保存失败');
  }
}
