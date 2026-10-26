export function setupCamera(base, pauseSimulation) {
  const $ = id => document.getElementById(id);
  const video = $('camera-video'), overlay = $('camera-overlay');
  const status = $('camera-status');
  let stream, worker, generation = 0, active = false, callbackId, fallbackId, startupTimer;
  let connections = [], stats = {}, lastPresented = 0, frameStarted = 0, rejectStartup;
  const clear = () => {
    for (const canvas of [overlay]) canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
  };
  function stop(message = '摄像头已关闭') {
    generation++; active = false;
    rejectStartup?.(new Error(message));rejectStartup = null;
    clearTimeout(startupTimer);
    if (callbackId !== undefined) video.cancelVideoFrameCallback?.(callbackId);
    cancelAnimationFrame(fallbackId);
    worker?.terminate(); worker = null;
    stream?.getTracks().forEach(track => track.stop()); stream = null;
    video.pause(); video.srcObject = null;
    clear();
    $('camera-empty').hidden = false;$('camera-panel').hidden = true;
    status.textContent = message;
    document.dispatchEvent(new Event('camerastate'));
    $('camera-fps').textContent = '—'; $('camera-latency').textContent = '—';
    $('camera-points').textContent = '—';
  }
  function draw(canvas, bitmap, landmarks, withVideo) {
    if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
      canvas.width = bitmap.width; canvas.height = bitmap.height;
    }
    const ctx = canvas.getContext('2d'), w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h); ctx.save();
    if ($('camera-mirror').checked) { ctx.translate(w, 0); ctx.scale(-1, 1); }
    if (withVideo) ctx.drawImage(bitmap, 0, 0);
    const visible = p => p && (p.visibility ?? 1) >= .5;
    ctx.lineWidth = Math.max(2, w / 220); ctx.strokeStyle = '#77f2cf';
    for (const {start, end} of connections) {
      const a = landmarks[start], b = landmarks[end];
      if (!visible(a) || !visible(b)) continue;
      ctx.beginPath(); ctx.moveTo(a.x * w, a.y * h); ctx.lineTo(b.x * w, b.y * h); ctx.stroke();
    }
    landmarks.forEach((p, i) => {
      if (!visible(p)) return;
      ctx.fillStyle = i % 2 ? '#ffb48d' : '#b8f5ff';
      ctx.beginPath(); ctx.arc(p.x * w, p.y * h, Math.max(2.5, w / 150), 0, Math.PI * 2); ctx.fill();
    });
    ctx.restore();
  }
  function schedule(token) {
    if (!active || token !== generation) return;
    const capture = async (_now, metadata) => {
      if (!active || token !== generation) return;
      try {
        frameStarted = performance.now();
        const timestamp = metadata ? metadata.mediaTime * 1000 : performance.now();
        const bitmap = await createImageBitmap(video);
        if (!active || token !== generation) { bitmap.close(); return; }
        worker.postMessage({type: 'frame', bitmap, timestamp}, [bitmap]);
      } catch (error) { if (token === generation) stop('画面读取失败：' + error.message); }
    };
    if (video.requestVideoFrameCallback) callbackId = video.requestVideoFrameCallback(capture);
    else fallbackId = requestAnimationFrame(now => capture(now));
  }
  async function start() {
    stop('正在请求摄像头权限…');
    const token = generation;
    $('camera-panel').hidden = false;$('input-placeholder').hidden = true;
    pauseSimulation();
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw Error('请使用 localhost 或 HTTPS，并在浏览器中允许摄像头');
      const acquired = await navigator.mediaDevices.getUserMedia({audio: false, video: {
        facingMode: 'user', width: {ideal: 640}, height: {ideal: 480}, frameRate: {ideal: 30, max: 30},
      }});
      if (token !== generation) { acquired.getTracks().forEach(track => track.stop()); throw new DOMException('已取消','AbortError'); }
      stream = acquired;
      stream.getVideoTracks()[0].addEventListener('ended', () => { if (token === generation) stop('摄像头已断开，可重新开启'); });
      video.srcObject = stream; await video.play();
      if (token !== generation) throw new DOMException('已取消','AbortError');
      status.textContent = '正在加载本地骨架模型…';
      worker = new Worker(new URL('pose.worker.js', base));
      let resolveStartup;
      const startup = new Promise((resolve,reject)=>{resolveStartup=resolve;rejectStartup=reject;});
      startupTimer = setTimeout(() => { if (token === generation) stop('模型加载超时，请重试'); }, 60000);
      worker.onerror = event => { if (token === generation) stop('骨架识别启动失败：' + event.message); };
      worker.onmessage = ({data}) => {
        if (token !== generation) { data.bitmap?.close(); return; }
        if (data.type === 'error') { stop('骨架识别失败：' + data.message); return; }
        if (data.type === 'ready') {
          clearTimeout(startupTimer); connections = data.connections; active = true;
          stats = {frames: 0, detectedFrames: 0}; lastPresented = 0;
          document.dispatchEvent(new Event('camerastate'));
          resolveStartup();rejectStartup=null;schedule(token);
        }
        if (data.type === 'result') {
          // Draw the exact input bitmap returned with its landmarks.
          // Never draw old landmarks over a newer live video frame.
          draw(overlay, data.bitmap, data.landmarks, true);
          data.bitmap.close();
          const now = performance.now();
          stats = {...stats, frames: stats.frames + 1, detectedFrames: stats.detectedFrames + Number(data.landmarks.length > 0),
            timestamp: data.timestamp, landmarkCount: data.landmarks.length,
            latencyMs: now - frameStarted, inferenceMs: data.inferenceMs,
            fps: lastPresented ? 1000 / (now - lastPresented) : 0};
          lastPresented = now;
          $('camera-empty').hidden = true;
          status.textContent = data.landmarks.length ? '正在跟踪 · 画面与骨架同帧' : '未检测到人体 · 请让全身进入画面';
          $('camera-fps').textContent = stats.fps.toFixed(0) + ' fps';
          $('camera-latency').textContent = stats.latencyMs.toFixed(0) + ' ms';
          $('camera-points').textContent = data.landmarks.length ? '33 个关键点' : '未检测到';
          schedule(token);
        }
      };
      worker.postMessage({type: 'load', base});
      await startup;
    } catch (error) {
      if (token !== generation) throw error;
      const messages = {NotAllowedError: '摄像头权限未允许，请在地址栏授权后重试',
        NotFoundError: '没有找到摄像头，请连接设备后重试', NotReadableError: '摄像头无法读取，可能被其他应用占用'};
      const message=messages[error.name] || error.message;stop(message);throw Error(message);
    }
  }
  document.addEventListener('visibilitychange', () => { if (document.hidden && stream) stop('页面已隐藏，摄像头已关闭'); });
  window.addEventListener('pagehide', () => stop());
  window.__camera = {get state() { return {active, ...stats}; }};
  return {start, stop, get active() {return active}, get stream() {return stream}};
}
