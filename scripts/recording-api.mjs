import {spawn} from 'node:child_process';
import {access} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {randomUUID} from 'node:crypto';
import {archiveAPI, archiveFile} from './recording-archive.mjs';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const python = path.join(root, '.venv-physics/bin/python');
let active = null;
const json = (res, code, value) => {res.writeHead(code, {'Content-Type':'application/json', 'Cache-Control':'no-store'});res.end(JSON.stringify(value));};
export async function recordingAPI(req, res) {
  const pathname = new URL(req.url, 'http://localhost').pathname;
  if (await archiveAPI(req, res, pathname)) return true;
  if (!['/api/recording', '/musclelab/api/recording'].includes(pathname)) return false;
  // This endpoint only runs on the loopback server and accepts same-origin JSON.
  if (req.headers.origin && req.headers.origin !== `http://${req.headers.host}`) {
    json(res, 403, {error:'仅允许本地页面调用'}); return true;
  }
  if (req.method === 'GET') {
    try {await access(python);json(res, 200, {available:true, maxSeconds:2, busy:Boolean(active)});}
    catch {json(res, 503, {error:'本地 Python 环境未安装，请参阅 README'});}
    return true;
  }
  if (req.method !== 'POST') {json(res, 405, {error:'Method not allowed'});return true;}
  if (!req.headers['content-type']?.startsWith('application/json')) {json(res, 415, {error:'需要 JSON 姿态数据'});return true;}
  if (active) {json(res, 409, {error:'另一段录制正在处理，请稍后重试'});return true;}
  const job = {};active = job;
  let child, timer, finished = false;
  const cleanup = () => {clearTimeout(timer);if (child && child.exitCode === null) child.kill('SIGKILL');if(active === job)active = null;};
  res.on('close', cleanup);
  const send = value => {if (!res.destroyed && !res.writableEnded) res.write(JSON.stringify(value) + '\n');};
  try {
    let size = 0;const chunks = [];
    for await (const chunk of req) {
      size += chunk.length;
      if (size > 4 * 1024 * 1024) {json(res, 413, {error:'片段过大，请限制为 2 秒'});cleanup();return true;}
      chunks.push(chunk);
    }
    const body = Buffer.concat(chunks).toString('utf8');
    const data = JSON.parse(body);
    if (!Array.isArray(data.frames) || data.frames.length < 15 || data.frames.length > 76) {
      json(res, 400, {error:'需要 15–76 帧姿态数据'});cleanup();return true;
    }
    await access(python);
    const recordingId = randomUUID();
    await archiveFile(recordingId, 'poses.json', body);
    res.writeHead(200, {'Content-Type':'application/x-ndjson', 'Cache-Control':'no-store', 'X-Content-Type-Options':'nosniff'});
    send({type:'progress', progress:1, message:'启动本机处理…'});
    child = spawn(python, ['-u', path.join(root, 'scripts/process-recording.py')], {cwd:root, stdio:['pipe','pipe','pipe']});
    child.stdin.on('error', () => {});child.stdin.end(body);
    let buffer = '', stderr = '', pending = Promise.resolve();
    child.stdout.setEncoding('utf8');child.stderr.setEncoding('utf8');
    child.stdout.on('data', chunk => {
      buffer += chunk;
      let end;
      while ((end = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, end);buffer = buffer.slice(end + 1);
        try {
          const item = JSON.parse(line);
          if (['progress','result','error'].includes(item.type)) {
            if (item.type === 'result') {
              item.result.recordingId = recordingId;
              pending = archiveFile(recordingId, 'result.json', JSON.stringify(item.result))
                .then(() => send(item)).catch(error => {item.result.archiveError = error.message;send(item);});
            } else send(item);
            if (item.type !== 'progress') finished = true;
          }
        } catch {/* Native diagnostic lines aren't protocol messages. */}
      }
    });
    child.stderr.on('data', text => {stderr = (stderr + text).slice(-2000);});
    child.on('error', error => {finished = true;send({type:'error', message:'本地处理启动失败：' + error.message});res.end();});
    child.on('close', async () => {
      await pending;
      if (!finished) {console.error('Recording process failed:', stderr);send({type:'error', message:'本地处理未完成，请重试或查看终端诊断'});}
      res.end();cleanup();
    });
    timer = setTimeout(() => {finished = true;send({type:'error', message:'处理超过 45 秒，已停止。请缩短片段或减少遮挡后重试'});cleanup();res.end();}, 45000);
  } catch (error) {
    if (!res.headersSent) json(res, 400, {error:'无法处理录制：' + error.message});
    else {send({type:'error', message:error.message});res.end();}
    cleanup();
  }
  return true;
}
