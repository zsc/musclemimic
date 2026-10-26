import {mkdir, writeFile, rename, stat, copyFile} from 'node:fs/promises';
import {randomUUID} from 'node:crypto';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const archiveRoot = path.join(root, 'reports/recordings');
const validId = id => /^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/i.test(id);
export async function archiveFile(id, name, value) {
  if (!validId(id)) throw Error('Invalid recording ID');
  const folder = path.join(archiveRoot, id);
  await mkdir(folder, {recursive:true});
  const temporary = path.join(folder, `.${randomUUID()}.tmp`);
  await writeFile(temporary, value);
  await rename(temporary, path.join(folder, name));
}
const exists = async file => {try {await stat(file);return true;} catch {return false;}};
export async function archiveAPI(req, res, pathname) {
  const match = /^\/(?:musclelab\/)?api\/recordings\/([^/]+)(\/video)?$/.exec(pathname);
  if (!match) return false;
  const reply = (code,value) => {res.writeHead(code, {'Content-Type':'application/json','Cache-Control':'no-store'});res.end(JSON.stringify(value));};
  if (req.headers.origin && req.headers.origin !== `http://${req.headers.host}`) {reply(403,{error:'仅允许本地页面调用'});return true;}
  const id = match[1], media = Boolean(match[2]);
  if (!validId(id)) {reply(400,{error:'Invalid recording ID'});return true;}
  const folder = path.join(archiveRoot, id);
  if (req.method === 'GET' && !media) {
    reply(200,{metadata:await exists(path.join(folder,'metadata.json')),video:await exists(path.join(folder,'source.video'))});return true;
  }
  if (req.method !== 'POST') {reply(405,{error:'Method not allowed'});return true;}
  try {
    if (!media && !req.headers['content-type']?.startsWith('application/json')) {reply(415,{error:'JSON required'});return true;}
    const limit = (media ? 100 : 20) * 1024 * 1024, chunks = [];let size = 0;
    for await (const chunk of req) {size += chunk.length;if(size > limit){reply(413,{error:'Recording archive too large'});return true;}chunks.push(chunk);}
    const body = Buffer.concat(chunks);
    if (media) await archiveFile(id,'source.video',body);
    else {
      const entry = JSON.parse(body);
      if (entry.id !== id || !Array.isArray(entry.result?.qpos) || !Number.isFinite(entry.created)) throw Error('Invalid recording');
      await archiveFile(id,'result.json',JSON.stringify(entry.result));
      if (!entry.hasVideo) {await mkdir(folder,{recursive:true});await copyFile(path.join(root,'public/source.mp4'),path.join(folder,'source.video'));}
      await archiveFile(id,'metadata.json',JSON.stringify({id,created:entry.created,title:entry.title,videoType:entry.videoType||'video/mp4',source:'browser-history',test:req.headers['x-demo-test']==='fake-camera',archivedAt:new Date().toISOString()}));
    }
    reply(200,{saved:true});
  } catch (error) {reply(500,{error:`本机归档失败：${error.message}`});}
  return true;
}
