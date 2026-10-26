import {Physics} from './physics.js';
import {acceptance} from './observation.js';
// The native process already integrated every frame. This class only reconstructs
// render geometry; it never presents stored qpos as a new forward simulation.
export class RecordedPlayback extends Physics {
  constructor(physics, recording) {
    super(physics.mj, physics.model, physics.manifest, {qpos:recording.reference_qpos}, null);
    this.recording = recording;this.start = 0;this.seed = 3;this.finite = recording.metrics.finite;
    this.limit = recording.qpos.length - 1;this.at(0);
  }
  at(step) {
    this.stepIndex = Math.max(0, Math.min(this.limit, step));
    const i = this.stepIndex, r = this.recording;
    this.data.qpos.set(r.qpos[i]);this.data.act.set(r.activation[i]);
    this.display.qpos.set(r.qpos[i]);this.ghost.qpos.set(r.reference_qpos[i]);
    this.mj.mj_kinematics(this.model, this.display);this.mj.mj_kinematics(this.model, this.ghost);
    this.errors = r.errors_mm.slice(0, i + 1);this.roots = r.root_errors_mm.slice(0, i + 1);
  }
  metrics() {
    return {...acceptance(this.errors, this.roots, this.manifest.gates, this.stepIndex === this.limit, this.finite),
      step:this.stepIndex, time:this.stepIndex / 100, duration:this.limit / 100, start:this.recording.sourceStart, seed:3, replay:true};
  }
  snapshot(muscles) {
    return {...super.snapshot(muscles), contacts:this.recording.contacts[this.stepIndex]};
  }
  dispose() {this.data.delete();this.display.delete();this.ghost.delete();}
}
