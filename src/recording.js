// Media stays local. One action owns camera startup, countdown, capture and processing.
export function setupRecording(base, camera, onResult, pausePlayback) {
  const $ = id => document.getElementById(id), status = $('recording-status');
  let controller, recorder, timer, elapsedTimer, busy = false, recording = false, stage = 'idle';
  let selectedFile, inputURL;
  function progress(message, value = 0) {
    status.textContent = message;$('recording-progress').value = value;
    $('recording-progress').hidden = !busy;
  }
  function setStage(value, label) {stage = value;$('job-label').textContent = label;}
  function setBusy(value) {
    busy = value;$('run').textContent = value ? '取消' : '一键启动';$('run').classList.toggle('busy',value);
    $('input-source').disabled = value;$('recording-file').disabled = value;
    $('upload-zone').classList.toggle('disabled',value);
    $('recording-progress').hidden = !value;
    if(value){const started=performance.now();$('job-time').textContent='0.0 s';elapsedTimer=setInterval(()=>{$('job-time').textContent=((performance.now()-started)/1000).toFixed(1)+' s';},100);}
    else clearInterval(elapsedTimer);
    document.dispatchEvent(new CustomEvent('recordingstate',{detail:{busy}}));
  }
  function refreshPreview() {
    const source=$('input-source').value, video=$('input-video');
    $('input-placeholder').hidden=source!=='camera';$('upload-zone').hidden=source!=='upload'||Boolean(selectedFile);
    video.hidden=source==='camera'||(source==='upload'&&!selectedFile);
    if(source==='example'){video.src=new URL('source.mp4',base).href;$('input-name').textContent='公开示例 · 1.67 秒 · 点击启动重新重建';}
    else if(source==='upload'&&selectedFile){video.src=inputURL;$('input-name').textContent=selectedFile.name+' · 点击预览下方文件名可更换';}
    else{video.pause();video.removeAttribute('src');video.load();$('input-name').textContent=source==='camera'?'固定机位 · 缓慢动作 · 至少一只脚着地':'选择一个短视频，自动截取前 2 秒';}
    $('input-name').classList.toggle('replace-file',source==='upload'&&Boolean(selectedFile));
  }
  function cancel() {
    controller?.abort();clearTimeout(timer);
    if (recorder?.state === 'recording') recorder.stop();
    camera.stop();$('camera-countdown').hidden=true;
  }
  const delay = (ms, signal) => new Promise((resolve,reject)=>{
    const abort=()=>{clearTimeout(timeout);signal.removeEventListener('abort',abort);reject(new DOMException('已取消','AbortError'));};
    const timeout=setTimeout(()=>{signal.removeEventListener('abort',abort);resolve();},ms);
    signal.addEventListener('abort',abort,{once:true});if(signal.aborted)abort();
  });
  function waitEvent(target, event, signal, timeout = 15000) {
    return new Promise((resolve, reject) => {
      const done = error => {clearTimeout(timeoutId);target.removeEventListener(event, success);target.removeEventListener('error', failure);signal.removeEventListener('abort', abort);error ? reject(error) : resolve();};
      const success = () => done(), failure = () => done(Error('无法解码视频，请换用 MP4 或 WebM')), abort = () => done(new DOMException('已取消', 'AbortError'));
      const timeoutId = setTimeout(() => done(Error('读取视频超时，请换用更短的视频')), timeout);
      target.addEventListener(event, success, {once:true});target.addEventListener('error', failure, {once:true});signal.addEventListener('abort', abort, {once:true});
      if (signal.aborted) abort();
    });
  }
  async function extract(url, knownDuration, signal) {
    const video = document.createElement('video');video.muted = true;video.playsInline = true;video.preload = 'auto';
    let worker;
    try {
      const loaded = waitEvent(video, 'loadeddata', signal);video.src = url;await loaded;
      let duration = Number.isFinite(video.duration) ? video.duration : knownDuration;
      if (!duration || duration < .6) throw Error('片段太短，请至少录制 0.6 秒');
      duration = Math.min(duration, 2);
      // MediaRecorder WebM can omit duration. Seeking past its end discovers it.
      if (!Number.isFinite(video.duration)) {
        const seek = waitEvent(video, 'seeked', signal);video.currentTime = 1e6;await seek;
        if (Number.isFinite(video.duration)) duration = Math.min(duration, video.duration);
      }
      worker = new Worker(new URL('pose.worker.js', base));
      const reply = () => new Promise((resolve, reject) => {
        let timeoutId;
        const cleanup = () => {clearTimeout(timeoutId);signal.removeEventListener('abort', abort);worker.onmessage = null;worker.onerror = null;};
        const abort = () => {cleanup();reject(new DOMException('已取消', 'AbortError'));};
        worker.onmessage = ({data}) => {cleanup();data.type === 'error' ? reject(Error(data.message)) : resolve(data);};
        worker.onerror = event => {cleanup();reject(Error(event.message));};
        timeoutId = setTimeout(() => {cleanup();reject(Error('姿态识别超时，请重试'));}, 30000);
        signal.addEventListener('abort', abort, {once:true});if (signal.aborted) abort();
      });
      progress('加载姿态模型…', 3);
      let pending = reply();worker.postMessage({type:'load', base});await pending;
      const frames = [], count = Math.floor((duration - 1 / 60) * 30) + 1;
      for (let i = 0; i < count; i++) {
        signal.throwIfAborted();const time = i / 30;
        if (Math.abs(video.currentTime - time) > .0001) {const seek = waitEvent(video, 'seeked', signal);video.currentTime = time;await seek;}
        const bitmap = await createImageBitmap(video);
        if(signal.aborted){bitmap.close();signal.throwIfAborted();}pending = reply();
        worker.postMessage({type:'frame', bitmap, timestamp:time * 1000}, [bitmap]);
        const result = await pending;result.bitmap.close();
        frames.push({timestamp:time * 1000, landmarks:result.landmarks, worldLandmarks:result.worldLandmarks});
        progress(`识别录制姿态 ${i + 1}/${count}`, 5 + 30 * (i + 1) / count);
      }
      return {frames};
    } finally {worker?.terminate();video.removeAttribute('src');video.load();}
  }
  async function processClip(clip, signal) {
    const began=performance.now();setStage('processing','生成中');
    const payload=await extract(clip.url,clip.duration,signal), poseMs=performance.now()-began;
    progress('连接本机肌骨处理…',36);
    const response=await fetch(new URL('api/recording',base),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload),signal});
    if(!response.ok){let error;try{error=(await response.json()).error;}catch{}throw Error(error||'本地处理服务未启动，请运行 npm run demo');}
    if(!response.headers.get('content-type')?.includes('ndjson'))throw Error('本地处理服务未启动，请运行 npm run demo');
    const reader=response.body.getReader(),decoder=new TextDecoder();let text='',result;
    while(true){
      const {value,done}=await reader.read();text+=decoder.decode(value,{stream:!done});let end;
      while((end=text.indexOf('\n'))>=0){const item=JSON.parse(text.slice(0,end));text=text.slice(end+1);
        if(item.type==='error')throw Error(item.message);
        if(item.type==='progress')progress(item.message,36+item.progress*.63);
        if(item.type==='result')result=item.result;
      }
      if(done)break;
    }
    signal.throwIfAborted();if(!result)throw Error('本地服务没有返回结果');
    result.timing.poseMs=poseMs;result.timing.endToEndSeconds=(performance.now()-began)/1000;
    await onResult(result,clip,signal);
    setStage('complete','已完成');progress(`${result.duration.toFixed(2)} 秒结果 · 生成用时 ${result.timing.endToEndSeconds.toFixed(1)} 秒`,100);
  }
  async function capture(signal) {
    setStage('starting','启动摄像头');progress('请允许摄像头访问，随后自动倒数录制…');pausePlayback();
    await new Promise((resolve,reject)=>{
      const abort=()=>reject(new DOMException('已取消','AbortError'));
      signal.addEventListener('abort',abort,{once:true});
      camera.start().then(resolve,reject).finally(()=>signal.removeEventListener('abort',abort));
      if(signal.aborted)abort();
    });signal.throwIfAborted();
    $('camera-countdown').hidden=false;$('camera-countdown').classList.remove('recording');
    setStage('countdown','准备录制');
    for(let n=3;n>0;n--){
      if(!camera.active)throw Error('摄像头已关闭，请重新启动');
      $('camera-countdown').textContent=String(n);progress(`${n} 秒后开始 · 全身入镜，站稳准备`,0);
      await delay(1000,signal);
    }
    if(!camera.active)throw Error('摄像头已关闭，请重新启动');
    if(!window.MediaRecorder)throw Error('浏览器不支持录制，请选择本地视频');
    const mimeType=['video/webm;codecs=vp8','video/webm','video/mp4'].find(type=>MediaRecorder.isTypeSupported(type));
    if(!mimeType)throw Error('没有可用的录制格式，请选择本地视频');
    const began=performance.now(),chunks=[];
    recording=true;setStage('recording','录制中');progress('正在录制 2 秒，随后自动生成…',0);
    $('camera-countdown').textContent='● 录制中';$('camera-countdown').classList.add('recording');
    const blob=await new Promise((resolve,reject)=>{
      recorder=new MediaRecorder(camera.stream,{mimeType,videoBitsPerSecond:1500000});
      const abort=()=>{if(recorder.state==='recording')recorder.stop();};
      recorder.ondataavailable=event=>{if(event.data.size)chunks.push(event.data);};
      recorder.onerror=()=>{signal.removeEventListener('abort',abort);reject(Error('录制失败，请重新启动'));};
      recorder.onstop=()=>{clearTimeout(timer);signal.removeEventListener('abort',abort);signal.aborted?reject(new DOMException('已取消','AbortError')):resolve(new Blob(chunks,{type:mimeType}));};
      signal.addEventListener('abort',abort,{once:true});recorder.start();
      timer=setTimeout(()=>{if(recorder.state==='recording')recorder.stop();},2000);
    });
    recording=false;$('camera-countdown').hidden=true;camera.stop();
    return {url:URL.createObjectURL(blob),blob,duration:(performance.now()-began)/1000,title:'摄像头录制',ownURL:true};
  }
  async function run() {
    if(busy){cancel();return;}
    const source=$('input-source').value;
    if(source==='upload'&&!selectedFile){$('recording-file').click();return;}
    controller=new AbortController();const signal=controller.signal;
    $('job-state').classList.remove('error');setBusy(true);$('input-video').pause();
    let clip;
    try{
      clip=source==='camera'?await capture(signal):source==='upload'?{url:inputURL,blob:selectedFile,title:selectedFile.name}:{url:new URL('source.mp4',base).href,title:'示例视频'};
      await processClip(clip,signal);
    }catch(error){
      const cancelled=signal.aborted||error.name==='AbortError';setStage(cancelled?'cancelled':'error',cancelled?'已取消':'未完成');
      $('job-state').classList.toggle('error',!cancelled);progress(cancelled?'已取消，之前的结果仍可回放。':error.message);
    }finally{
      camera.stop();recording=false;clearTimeout(timer);$('camera-countdown').hidden=true;
      if(clip?.ownURL)URL.revokeObjectURL(clip.url);
      controller=null;setBusy(false);refreshPreview();
    }
  }
  function chooseFile(file){
    if(!file||busy)return;
    if(file.size>100*1024*1024){progress('请选择小于 100 MB 的短视频');return;}
    if(!file.type.startsWith('video/')){progress('请选择 MP4 或 WebM 视频');return;}
    if(inputURL)URL.revokeObjectURL(inputURL);selectedFile=file;inputURL=URL.createObjectURL(file);$('input-source').value='upload';refreshPreview();
    setStage('idle','准备就绪');progress('视频已选好，点击一键启动生成。');
  }
  $('run').onclick=run;
  $('input-source').onchange=()=>{camera.stop();refreshPreview();setStage('idle','准备就绪');progress('点击一键启动，自动处理并播放结果。');};
  $('recording-file').onchange=()=>{chooseFile($('recording-file').files[0]);$('recording-file').value='';};
  $('input-name').onclick=()=>{if(!busy&&$('input-source').value==='upload')$('recording-file').click();};
  $('input-preview').ondragover=event=>{event.preventDefault();if(!busy)$('input-preview').classList.add('dragover');};
  $('input-preview').ondragleave=()=>{$('input-preview').classList.remove('dragover');};
  $('input-preview').ondrop=event=>{event.preventDefault();$('input-preview').classList.remove('dragover');chooseFile(event.dataTransfer.files[0]);};
  document.addEventListener('visibilitychange',()=>{if(document.hidden&&['starting','countdown','recording'].includes(stage)&&busy)cancel();});
  window.addEventListener('pagehide',cancel);
  window.__recording={get state(){return {busy,recording,stage,message:status.textContent};}};
  refreshPreview();
}
