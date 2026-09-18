import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import publish_progress


class ProgressTests(unittest.TestCase):
    def test_missing_final_report_does_not_claim_test_improvement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(publish_progress,'ROOT',root):
                text=publish_progress.analyze(root)
            self.assertIn('**incomplete**',text)
            self.assertIn('no held-out improvement claim',text)
            self.assertNotIn('## Held-out test evidence',text)

    def test_trial_counts_and_validation_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            baseline={'objective':2.,'families':{'example':{'accuracy':0.5}}}
            selected={'objective':1.,'families':{'example':{'accuracy':0.75}}}
            (root/'baseline-metrics.json').write_text(json.dumps(baseline))
            (root/'best.json').write_text(json.dumps({'adapter':'local','trial':'trial1','step':50,'validation':selected}))
            (root/'events.jsonl').write_text(json.dumps({'phase':'train','trial':'trial1','step':1})+'\n')
            with patch.object(publish_progress,'ROOT',root):text=publish_progress.analyze(root)
            self.assertIn('decrease 1.000000',text)
            self.assertIn('50.00% | 75.00%',text)
            self.assertIn('Completed optimizer updates: **1**',text)
            self.assertIn('**incomplete**',text)


if __name__=='__main__':unittest.main()
