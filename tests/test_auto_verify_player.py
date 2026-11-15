import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

from examples.soccer.auto_verify_player import (
    apply_report, revert_automatic_verification, verify_anchored_intervals,
)
from examples.soccer.review_tracking import fingerprint


class AutomaticAnchorVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video, self.analysis = self.root/'source.mp4', self.root/'tracks.jsonl'
        self.box = [40, 20, 80, 120]

    def recording(self, swap_at=None, competitor_at=None, missing=(), jump_at=None, lower_swap_at=None,
                  filtered_without_context=False, ball=False, neutral_lower=False):
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*'mp4v'), 10, (320, 160))
        self.assertTrue(writer.isOpened())
        rows = []
        for index in range(20):
            box = [210, 20, 250, 120] if jump_at is not None and index >= jump_at else self.box
            frame = np.full((160, 320, 3), (20, 100, 20), np.uint8)
            colour = (245, 245, 245) if swap_at is not None and index >= swap_at else (220, 40, 20)
            x1, y1, x2, y2 = box
            frame[y1:y2, x1:x2] = colour
            if neutral_lower:
                frame[y1+int((y2-y1)*.60):y2, x1:x2] = (245, 245, 245)
            if lower_swap_at is not None and index >= lower_swap_at:
                frame[y1+int((y2-y1)*.60):y2, x1:x2] = (10, 10, 10) if neutral_lower else (245, 245, 245)
            detections = [] if index in missing else [dict(tracker_id=1, bbox=box, class_id=2)]
            all_boxes = [] if index in missing else [box]
            if competitor_at is not None and index >= competitor_at:
                competitor = [65, 20, 105, 120]
                frame[20:120, 65:105] = (220, 40, 20)
                detections.append(dict(tracker_id=2, bbox=competitor, class_id=2))
                all_boxes.append(competitor)
            if ball:
                detections.append(dict(tracker_id=99, bbox=[50, 40, 55, 45], class_id=0))
            row = dict(schema='player-tracking-v1', frame_index=index, fps=10,
                       width=320, height=160, timestamp=index/10, source_timestamp=index/10,
                       detections=detections, tracking_team='blue' if filtered_without_context else 'all')
            if not filtered_without_context and not ball:
                row['all_person_boxes'] = all_boxes
            rows.append(row)
            writer.write(frame)
        writer.release()
        self.analysis.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        return dict(schema='named-player-assignments-v1', analysis_sha256=fingerprint(self.analysis),
                    fps=10, frame_count=20, assignments=[dict(tracker_id=1, name='Alex',
                        start_frame=0, end_frame=20, confirmed_sample_frame=5,
                        continuity_status='unverified')], rejected_matches=[])

    def audit(self, doc):
        return verify_anchored_intervals(self.video, self.analysis, doc, 'Alex')

    def test_safe_short_extension_is_auditable_reversible_and_does_not_create_an_identity(self):
        doc = self.recording()
        report = self.audit(doc)
        proposal = report['proposals'][0]
        self.assertTrue(proposal['extended'])
        self.assertEqual((proposal['start_frame'], proposal['end_frame']), (0, 16))
        updated = apply_report(doc, report)
        self.assertEqual(len(updated['assignments']), 1)
        entry = updated['assignments'][0]
        self.assertEqual(entry['tracker_id'], 1)
        self.assertEqual(entry['confirmed_sample_frame'], 5)
        self.assertEqual(entry['continuity_status'], 'verified')
        self.assertEqual(entry['reported_track_range'], [0, 20])
        self.assertIn('automatic-anchor-gates', entry['verification_source'])
        self.assertEqual(doc['assignments'][0]['end_frame'], 20)
        restored = revert_automatic_verification(updated)
        self.assertEqual(restored['assignments'], doc['assignments'])
        # Subsequent manual edits are never overwritten by rollback.
        updated['assignments'][0]['end_frame'] = 9
        self.assertEqual(revert_automatic_verification(updated)['assignments'][0]['end_frame'], 9)

    def test_same_id_blue_to_white_swap_stops_before_the_swap(self):
        doc = self.recording(swap_at=8)
        proposal = self.audit(doc)['proposals'][0]
        self.assertEqual(proposal['end_frame'], 8)
        self.assertEqual(proposal['stopped']['after'], 'blue_shirt_gate_failed')
        doc = self.recording(lower_swap_at=8)
        proposal = self.audit(doc)['proposals'][0]
        self.assertEqual(proposal['end_frame'], 8)
        self.assertEqual(proposal['stopped']['after'], 'clothing_colour_inconsistency')

    def test_same_team_crossing_and_motion_jump_stop_extensions(self):
        doc = self.recording(competitor_at=8)
        proposal = self.audit(doc)['proposals'][0]
        self.assertEqual(proposal['end_frame'], 8)
        self.assertEqual(proposal['stopped']['after'], 'close_competing_person')
        doc = self.recording(jump_at=8)
        proposal = self.audit(doc)['proposals'][0]
        self.assertEqual(proposal['end_frame'], 8)
        self.assertEqual(proposal['stopped']['after'], 'motion_or_scale_jump')

    def test_two_frame_gap_can_bridge_but_three_frames_cannot(self):
        doc = self.recording(missing=(8, 9))
        proposal = self.audit(doc)['proposals'][0]
        self.assertEqual(proposal['end_frame'], 16)
        self.assertEqual(proposal['gaps'][0]['missing_frames'], 2)
        doc = self.recording(missing=(8, 9, 10))
        proposal = self.audit(doc)['proposals'][0]
        self.assertEqual(proposal['end_frame'], 8)
        self.assertEqual(proposal['stopped']['after'], 'missing_frame_gap_exceeded')

    def test_negative_sample_blocks_extension_and_mixed_flags_skip_the_track(self):
        doc = self.recording()
        doc['rejected_matches'] = [dict(tracker_id=1, name='Alex', sample_frame=8)]
        proposal = self.audit(doc)['proposals'][0]
        self.assertEqual(proposal['end_frame'], 8)
        self.assertEqual(proposal['stopped']['after'], 'known_negative_sample')
        doc['continuity_flags'] = [dict(tracker_id=1, status='user-confirmed-ID-switch')]
        proposal = self.audit(doc)['proposals'][0]
        self.assertFalse(proposal['extended'])
        self.assertEqual(proposal['stopped']['anchor'], 'explicit_mixed_or_unresolved_track')

    def test_user_verified_ranges_are_immutable_and_report_is_bound_to_decisions(self):
        doc = self.recording()
        doc['assignments'].append(dict(tracker_id=1, name='Alex', start_frame=0,
                                      end_frame=3, continuity_status='verified', verification_source='user-reviewed'))
        report = self.audit(doc)
        updated = apply_report(doc, report)
        self.assertEqual(updated['assignments'], doc['assignments'])
        changed = deepcopy(doc)
        changed['assignments'][0]['name'] = 'Different player'
        with self.assertRaisesRegex(ValueError, 'changed'):
            apply_report(changed, report)
        doc['assignments'][0]['continuity_status'] = 'verified'
        self.assertFalse(self.audit(doc)['proposals'])

    def test_blue_only_export_needs_all_person_context_and_balls_are_not_competitors(self):
        doc = self.recording(filtered_without_context=True)
        proposal = self.audit(doc)['proposals'][0]
        self.assertFalse(proposal['extended'])
        self.assertEqual(proposal['stopped']['anchor'], 'all_person_competition_coverage_missing')
        doc = self.recording(ball=True)
        self.assertTrue(self.audit(doc)['proposals'][0]['extended'])

    def test_source_frame_count_must_match_tracking_export(self):
        doc = self.recording()
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*'mp4v'), 10, (320, 160))
        self.assertTrue(writer.isOpened())
        for _ in range(10):
            writer.write(np.zeros((160, 320, 3), np.uint8))
        writer.release()
        with self.assertRaisesRegex(ValueError, 'Video frame count differs'):
            self.audit(doc)

    def test_neutral_garment_brightness_change_and_old_reports_are_rejected(self):
        doc = self.recording(lower_swap_at=8, neutral_lower=True)
        report = self.audit(doc)
        proposal = report['proposals'][0]
        self.assertEqual(proposal['end_frame'], 8)
        self.assertEqual(proposal['stopped']['after'], 'clothing_colour_inconsistency')
        self.assertEqual(report['source'], 'automatic-anchor-gates-v2')
        old = deepcopy(report)
        old.update(source='automatic-anchor-gates-v1', schema='anchor-verification-report-v1')
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            apply_report(doc, old)


if __name__ == '__main__':
    unittest.main()
