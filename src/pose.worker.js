// A classic worker keeps importScripts available to the MediaPipe WASM loader.
let detector;
self.onmessage = async ({data}) => {
  try {
    if (data.type === 'load') {
      const {FilesetResolver, PoseLandmarker} = await import(new URL('vendor/mediapipe/vision_bundle.mjs', data.base).href);
      const files = await FilesetResolver.forVisionTasks(new URL('vendor/mediapipe/wasm/', data.base).href);
      detector = await PoseLandmarker.createFromOptions(files, {
        baseOptions: {modelAssetPath: new URL('models/pose_landmarker_lite.task', data.base).href, delegate: 'CPU'},
        canvas: new OffscreenCanvas(640, 480), runningMode: 'VIDEO', numPoses: 1,
        minPoseDetectionConfidence: .5, minPosePresenceConfidence: .5, minTrackingConfidence: .5,
      });
      self.postMessage({type: 'ready', connections: PoseLandmarker.POSE_CONNECTIONS});
    } else if (data.type === 'frame') {
      const began = performance.now();
      const result = detector.detectForVideo(data.bitmap, data.timestamp);
      self.postMessage({type: 'result', bitmap: data.bitmap, timestamp: data.timestamp,
        landmarks: result.landmarks[0] || [], worldLandmarks: result.worldLandmarks[0] || [],
        inferenceMs: performance.now() - began}, [data.bitmap]);
    }
  } catch (error) {
    data.bitmap?.close();
    self.postMessage({type: 'error', message: error.message});
  }
};
