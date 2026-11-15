import base64
import json
import tempfile
import unittest
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from examples.soccer.player_review import prepare, apply_decisions, build_batch, load_workspace
from examples.soccer.review_tracking import fingerprint, identity_label, validate_assignments
from examples.soccer.temporal_player_filter import temporal_report, apply_exclusions


class PlayerReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.analysis = self.root/'tracks.jsonl'
        rows = []
        for index in range(10):
            boxes = [(1, [0, 0, 10, 20]), (2, [100, 0, 110, 20]),
                     (3, [1, 0, 11, 20])]
            rows.append(dict(schema='player-tracking-v1', frame_index=index, fps=10,
                             width=160, height=100, timestamp=index/10, source_timestamp=index/10,
                             detections=[dict(tracker_id=tid, bbox=box) for tid, box in boxes]))
        self.analysis.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        image = np.zeros((100, 40, 3), np.uint8)
        image[:] = (220, 70, 30)
        ok, encoded = cv2.imencode('.jpg', image)
        self.assertTrue(ok)
        data = dict(analysis_sha256=fingerprint(self.analysis), fps=10, frame_count=10,
                    source_time_offset=0, tracks=[
                        dict(id=tid, start=0, end=10,
                             samples=[dict(frame=0, image=base64.b64encode(encoded).decode())])
                        for tid in (1, 2, 3)])
        self.review = self.root/'review.html'
        self.review.write_text('const data = '+json.dumps(data)+';\nconst status = null;')
        self.workspace = self.root/'workspace'
        prepare(self.review, self.workspace, blue_only=True)

    def decisions(self, entries):
        data, _ = load_workspace(self.workspace)
        return dict(analysis_sha256=data['analysis_sha256'], decisions=entries)

    def test_prepare_build_merge_resume_and_named_render_labels(self):
        doc = apply_decisions(self.workspace, self.decisions([
            dict(track_id=1, decision='yes'), dict(track_id=2, decision='no')]), 'Alex')
        grouped = validate_assignments(doc, self.analysis, 10, 10)
        self.assertEqual(identity_label({'tracker_id': 1}, 2, grouped), 'Alex (ID 1)')
        page = self.root/'buttons.html'
        self.assertEqual(build_batch(self.workspace, 'Alex', page), 1)
        html = page.read_text()
        self.assertIn('"track_id": 3', html)
        self.assertNotIn('__PAYLOAD__', html)
        self.assertIn('sendFollowUpMessage', html)
        # Reapplying a downloaded batch is idempotent and explicit corrections override exclusions.
        doc = apply_decisions(self.workspace, self.decisions([dict(track_id=2, decision='yes')]), 'Alex')
        doc = apply_decisions(self.workspace, self.decisions([dict(track_id=2, decision='yes')]), 'Alex')
        self.assertEqual(sum(a['tracker_id'] == 2 for a in doc['assignments']), 1)
        self.assertFalse(doc['rejected_matches'])

    def test_invalid_batch_is_atomic_and_wrong_run_is_rejected(self):
        before = (self.workspace/'named-assignments.json').read_text()
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            apply_decisions(self.workspace, self.decisions([
                dict(track_id=1, decision='yes'), dict(track_id=999, decision='no')]), 'Alex')
        self.assertEqual(before, (self.workspace/'named-assignments.json').read_text())
        with self.assertRaisesRegex(ValueError, 'another tracking run'):
            apply_decisions(self.workspace, dict(analysis_sha256='wrong', decisions=[]), 'Alex')

    def test_temporal_separation_does_not_exclude_near_duplicates(self):
        doc = apply_decisions(self.workspace, self.decisions([dict(track_id=1, decision='yes')]), 'Alex')
        report = temporal_report(self.analysis, doc, 'Alex')
        self.assertEqual(report['confirmed_player_conflicts'], [])
        self.assertEqual(report['proposed_exclusions'], [2])
        self.assertEqual(apply_exclusions(self.workspace, report), 1)
        self.assertEqual(apply_exclusions(self.workspace, report), 0)
        _, saved = load_workspace(self.workspace)
        self.assertTrue(saved['automatic_exclusions'][0]['reversible'])
        # A user's explicit match overrides an automatic exclusion.
        doc = apply_decisions(self.workspace, self.decisions([dict(track_id=2, decision='yes')]), 'Alex')
        self.assertFalse(doc['automatic_exclusions'])
        report = temporal_report(self.analysis, doc, 'Alex')
        self.assertEqual(set(report['confirmed_player_conflicts']), {1, 2})
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            apply_exclusions(self.workspace, report)

    def test_temporal_known_identity_respects_frame_range(self):
        doc = apply_decisions(self.workspace, self.decisions([dict(track_id=1, decision='yes')]), 'Alex')
        doc['assignments'][0]['end_frame'] = 4
        report = temporal_report(self.analysis, doc, 'Alex')
        self.assertEqual(report['proposed_exclusions'], [])

    def test_player_name_is_script_safe(self):
        page = self.root/'escaped.html'
        build_batch(self.workspace, "Alex </script><script>alert(1)</script>", page)
        self.assertNotIn('Alex </script>', page.read_text())

    def test_ranking_audits_held_out_tracks_and_binds_cached_features(self):
        data, doc = load_workspace(self.workspace)
        template = data['tracks'][0]
        candidates = []
        data['tracks'] = []
        for index in range(20):
            tid = index + 1
            track = dict(template, id=tid, start=index*900, end=index*900+10)
            data['tracks'].append(track)
            image = np.full((100, 40, 3), (220, 70, 30) if index % 2 else (50, 180, 80), np.uint8)
            cv2.imwrite(str(self.workspace/'images'/f'track-{tid}-frame-0.jpg'), image)
            candidates.append(dict(track_id=tid, image=f'track-{tid}-frame-0.jpg'))
            if index < 18:
                entry = dict(tracker_id=tid, name='Alex')
                if index % 2:
                    entry.update(start_frame=track['start'], end_frame=track['end'])
                    doc['assignments'].append(entry)
                else:
                    doc['rejected_matches'].append(entry)
        (self.workspace/'review.json').write_text(json.dumps(data))
        (self.workspace/'named-assignments.json').write_text(json.dumps(doc))
        (self.workspace/'candidates.json').write_text(json.dumps(dict(candidates=candidates)))
        command = [sys.executable, 'examples/soccer/learn_player_filter.py',
                   '--workspace', str(self.workspace), '--player', 'Alex']
        subprocess.run(command, check=True, capture_output=True)
        report = json.loads((self.workspace/'learned-filter-report.json').read_text())
        self.assertTrue(report['regression_gate']['accepted'])
        self.assertEqual({p['track_id'] for p in report['predictions']}, {19, 20})
        self.assertEqual(report['held_out_player_tracks_rejected'], 0)
        self.assertGreaterEqual(report['held_out_average_precision'], .9)
        # Cached features cannot silently cross tracking runs.
        data['analysis_sha256'] = doc['analysis_sha256'] = 'different'
        (self.workspace/'review.json').write_text(json.dumps(data))
        (self.workspace/'named-assignments.json').write_text(json.dumps(doc))
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('another tracking run', result.stderr)


if __name__ == '__main__':
    unittest.main()
