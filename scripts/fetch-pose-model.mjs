import {createHash} from 'node:crypto';
import {existsSync, mkdirSync, readFileSync, writeFileSync} from 'node:fs';

const expected = '59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a';
const destination = 'public/models/pose_landmarker_lite.task';
const url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
let bytes;
if (existsSync(destination) && sha256(readFileSync(destination)) === expected) {
  bytes = readFileSync(destination);
} else {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`pose model download failed: HTTP ${response.status}`);
  bytes = Buffer.from(await response.arrayBuffer());
}
if (sha256(bytes) !== expected) throw new Error('pose model SHA-256 mismatch');
mkdirSync('public/models', {recursive: true});
writeFileSync(destination, bytes);
console.log(`pose model verified: ${bytes.length} bytes`);
