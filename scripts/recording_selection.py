"""Select one complete physical rollout against an identical reference.

No state stitching, trajectory trimming, weight updates or relaxed thresholds.
"""
import math

ERROR_KEYS = ('mean', 'final', 'rootMax')


def eligible(candidate, baseline):
    metrics, original = candidate['metrics'], baseline['metrics']
    return (metrics.get('finite', False) and metrics.get('complete', False)
            and not any(candidate.get('warnings', []))
            and all(math.isfinite(metrics[k]) and metrics[k] <= original[k] + 1e-7 for k in ERROR_KEYS)
            and metrics.get('fallFrames', 0) <= original.get('fallFrames', 0))


def select_rollout(candidates):
    baseline = candidates[0]
    allowed = [c for c in candidates if eligible(c, baseline)]
    if not allowed:
        return baseline
    return min(allowed, key=lambda c: c['metrics']['mean'] / 100
               + c['metrics']['final'] / 150 + c['metrics']['rootMax'] / 200
               + c['metrics'].get('fallFrames', 0))
