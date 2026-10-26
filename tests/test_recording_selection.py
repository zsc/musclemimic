"""Reject apparent improvements that worsen another physical tracking measure."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from recording_selection import select_rollout


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.baseline = {'metrics': {'mean': 150., 'final': 220., 'rootMax': 300.,
                                    'fallFrames': 20, 'finite': True, 'complete': True},
                         'warnings': [0, 0]}
        self.improved = copy.deepcopy(self.baseline)
        self.improved['metrics'].update(mean=90., final=130., rootMax=180., fallFrames=0)

    def test_selects_entire_improved_result(self):
        self.assertIs(select_rollout([self.baseline, self.improved]), self.improved)

    def test_rejects_each_regression(self):
        for key, value in [('mean', 151.), ('final', 221.), ('rootMax', 301.),
                           ('fallFrames', 21), ('mean', float('nan')),
                           ('finite', False), ('complete', False)]:
            with self.subTest(key=key, value=value):
                candidate = copy.deepcopy(self.improved)
                candidate['metrics'][key] = value
                self.assertIs(select_rollout([self.baseline, candidate]), self.baseline)

    def test_rejects_engine_warning(self):
        self.improved['warnings'] = [0, 1]
        self.assertIs(select_rollout([self.baseline, self.improved]), self.baseline)

    def test_keeps_original_if_every_trial_is_invalid(self):
        self.baseline['warnings'] = [1]
        self.improved['metrics']['complete'] = False
        self.assertIs(select_rollout([self.baseline, self.improved]), self.baseline)


if __name__ == '__main__':
    unittest.main()
