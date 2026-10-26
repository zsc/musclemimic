// Decode continuously. Seeking on every simulation snapshot can starve presentation.
export class VideoSync {
  constructor(video, status) {
    this.video = video;
    this.status = status;
    this.target = 0;
    this.speed = 1;
    this.playing = false;
    video.addEventListener('loadedmetadata', () => this.correct(true));
    video.addEventListener('seeked', () => { if (!this.playing) this.correct(); });
    video.addEventListener('error', () => { status.textContent = '原视频加载失败'; });
  }
  correct(force = false) {
    const v = this.video;
    if (!Number.isFinite(v.duration) || v.seeking) return;
    const target = Math.min(this.target, v.duration);
    if (Math.abs(v.currentTime - target) > (force || !this.playing ? .008 : .1)) v.currentTime = target;
  }
  update(metrics) {
    this.target = metrics.start + metrics.time;
    this.correct();
  }
  start() {
    this.playing = true;
    this.correct(true);
    this.video.playbackRate = this.speed;
    this.video.play().then(() => {
      if (!this.playing) this.video.pause();
      else this.status.textContent = '原始视频 · 同步播放';
    }).catch(error => {
      if (this.playing && error.name !== 'AbortError') this.status.textContent = '视频播放失败，请暂停后重试';
    });
  }
  stop() {
    this.playing = false;
    this.video.pause();
    this.correct(true);
    this.status.textContent = '原始视频 · 同步时间';
  }
  setSpeed(speed) {
    this.speed = speed;
    this.video.playbackRate = speed;
    this.correct(true);
  }
}
