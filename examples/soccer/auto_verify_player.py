"""Conservatively extend exact player samples into short, auditable same-ID intervals.

No model identity score is used. Extensions must pass clothing, motion, gap and
all-person competition gates. This is an evidence rule, not a calibrated probability.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

try:
    from .review_tracking import fingerprint, records, open_video
    from sports.common.tracking_uniform import blue_shirt_fraction
except ImportError:
    from review_tracking import fingerprint, records, open_video
    from sports.common.tracking_uniform import blue_shirt_fraction

SOURCE = 'automatic-anchor-gates-v2'


def document_hash(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()


def iou(a, b):
    width = max(0, min(a[2], b[2])-max(a[0], b[0]))
    height = max(0, min(a[3], b[3])-max(a[1], b[1]))
    intersection = width*height
    return intersection/max(1e-9, (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])-intersection)


def near_competitor(box, boxes, remove_target=True):
    """Reject close people, including other teams and nearly duplicate detections."""
    others = list(boxes)
    if any(len(other) != 4 or not np.isfinite(other).all()
           or other[2] <= other[0] or other[3] <= other[1] for other in others):
        return True
    if remove_target:
        if not others:
            return True
        matches = [iou(box, other) for other in others]
        index = int(np.argmax(matches))
        if matches[index] < .85:
            return True
        others.pop(index)
    x1, y1, x2, y2 = box
    width, height = x2-x1, y2-y1
    expanded = [x1-width*.25, y1-height*.15, x2+width*.25, y2+height*.15]
    for other in others:
        cx, cy = (x1+x2)/2, (y1+y2)/2
        ox, oy = (other[0]+other[2])/2, (other[1]+other[3])/2
        separation = math.hypot(cx-ox, cy-oy)/max(height, other[3]-other[1], 1)
        if iou(expanded, other) > 0 or separation < .8:
            return True
    return False


def descriptors(frame, box, blue_minimum):
    box = np.asarray(box, dtype=float)
    if not np.isfinite(box).all() or box[2]-box[0] < 16 or box[3]-box[1] < 48:
        return None, 'insufficient_crop_detail'
    x1, y1, x2, y2 = box.astype(int)
    if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
        return None, 'clipped_person_box'
    if blue_shirt_fraction(frame, box) < blue_minimum:
        return None, 'blue_shirt_gate_failed'
    crop = frame[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    result = []
    for top, bottom in ((.20, .60), (.60, .95)):
        region = crop[int(h*top):int(h*bottom), int(w*.18):int(w*.82)]
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256]).flatten()
        value = cv2.calcHist([hsv], [2], None, [12], [0, 256]).flatten()
        # H/S alone cannot distinguish white and black garments. Modest value
        # weight adds that guard without treating brightness as identity proof.
        combined = np.concatenate((.8*hist/max(1, hist.sum()), .2*value/max(1, value.sum())))
        result.append(np.sqrt(combined))
    return result, None


def mixed_track(document, track_id):
    """Treat explicit mixed/unresolved continuity flags as a stop, regardless of format."""
    for field in ('mixed_tracks', 'unresolved_tracks', 'continuity_flags'):
        value = document.get(field, [])
        if isinstance(value, dict):
            entry = value.get(str(track_id), value.get(track_id))
            if entry:
                return True
        elif isinstance(value, list):
            for entry in value:
                if entry == track_id or isinstance(entry, dict) and entry.get('tracker_id', entry.get('track_id')) == track_id:
                    return True
    for entry in document.get('assignments', []):
        if entry['tracker_id'] == track_id and (entry.get('continuity_issue') or
                entry.get('continuity_status') in ('mixed', 'mixed-needs-range-review', 'unresolved', 'conflicted')):
            return True
    return False


def verify_anchored_intervals(video_path, analysis_path, document, player,
                             max_seconds=1.0, max_gap=2, competitor_analysis=None):
    if document.get('schema') != 'named-player-assignments-v1' or not player.strip():
        raise ValueError('Named player decisions and a player name are required.')
    if fingerprint(analysis_path) != document['analysis_sha256']:
        raise ValueError('Tracking analysis belongs to a different decision file.')
    if not math.isfinite(max_seconds) or not 0 < max_seconds <= 2 or not 0 <= max_gap <= 2:
        raise ValueError('Use a window in (0, 2] seconds and at most two missing frames.')
    anchors = [(index, entry) for index, entry in enumerate(document['assignments'])
               if entry.get('name') == player and type(entry.get('confirmed_sample_frame')) is int
               and entry.get('continuity_status') != 'verified']
    # Explicit user-verified ranges are immutable and are not candidates for extension.
    rows, metadata, count = {}, None, 0
    wanted = set()
    first = next(records(analysis_path), None)
    if first is None:
        raise ValueError('Tracking export is empty.')
    if first['fps'] != document['fps']:
        raise ValueError('Decision FPS differs from tracking export.')
    radius = max(1, int(math.floor(max_seconds*first['fps'])))
    for _, anchor in anchors:
        frame = anchor['confirmed_sample_frame']
        wanted.update(range(max(0, frame-radius), min(document['frame_count'], frame+radius+1)))
    for row in records(analysis_path):
        metadata = metadata or row
        count += 1
        if row['frame_index'] in wanted:
            rows[row['frame_index']] = row
    if count != document['frame_count']:
        raise ValueError('Decision frame count differs from tracking export.')
    if competitor_analysis:
        context_count = 0
        for row in records(competitor_analysis):
            context_count += 1
            if (row['fps'], row['width'], row['height']) != (metadata['fps'], metadata['width'], metadata['height']):
                raise ValueError('Competitor analysis must have matching frame rate and dimensions.')
            if row['frame_index'] in rows:
                boxes = row.get('all_person_boxes')
                if boxes is None:
                    if row.get('tracking_team', 'all') != 'all':
                        raise ValueError('Competitor export must include all people, including other teams.')
                    boxes = [d['bbox'] for d in row['detections'] if d.get('class_id', 2) in (1, 2, 3)]
                rows[row['frame_index']]['all_person_boxes'] = boxes
        if context_count != count:
            raise ValueError('Competitor analysis frame count differs from tracking export.')
    video = open_video(video_path, metadata)
    source_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    if not math.isfinite(source_count) or int(round(source_count)) != count:
        video.release()
        raise ValueError('Video frame count differs from the tracking export; use the exact matching segment.')
    output = []
    negatives = {}
    for entry in document.get('rejected_matches', []):
        if entry.get('name') == player:
            negatives.setdefault(entry['tracker_id'], []).append(entry.get('sample_frame'))

    def observe(frame_index, track_id, previous_box=None):
        row = rows.get(frame_index)
        if row is None:
            return None, 'outside_review_window'
        matches = [d for d in row['detections'] if d['tracker_id'] == track_id]
        boxes = row.get('all_person_boxes')
        if boxes is None:
            if row.get('tracking_team', 'all') != 'all':
                return None, 'all_person_competition_coverage_missing'
            boxes = [d['bbox'] for d in row['detections'] if d.get('class_id', 2) in (1, 2, 3)]
        if not matches:
            if previous_box is not None and near_competitor(previous_box, boxes, remove_target=False):
                return None, 'competing_person_during_gap'
            return None, 'missing_detection'
        if len(matches) != 1:
            return None, 'duplicate_target_id'
        box = matches[0]['bbox']
        if near_competitor(box, boxes):
            return None, 'close_competing_person'
        if int(video.get(cv2.CAP_PROP_POS_FRAMES)) != frame_index:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = video.read()
        if not ok:
            return None, 'unreadable_video_frame'
        values, reason = descriptors(frame, box, .12)
        if reason:
            return None, reason
        return dict(frame=frame_index, box=box, descriptors=values), None

    try:
        for index, anchor in anchors:
            track_id, frame = anchor['tracker_id'], anchor['confirmed_sample_frame']
            result = dict(assignment_index=index, tracker_id=track_id,
                          confirmed_sample_frame=frame, original_assignment=deepcopy(anchor),
                          start_frame=frame, end_frame=frame+1, extended=False,
                          source=SOURCE, max_seconds=max_seconds, max_gap=max_gap)
            if mixed_track(document, track_id):
                result['stopped'] = {'anchor': 'explicit_mixed_or_unresolved_track'}
                output.append(result)
                continue
            if None in negatives.get(track_id, []):
                result['stopped'] = {'anchor': 'negative_sample_provenance_missing'}
                output.append(result)
                continue
            if frame in negatives.get(track_id, []):
                result['stopped'] = {'anchor': 'anchor_conflicts_with_negative_sample'}
                output.append(result)
                continue
            origin, reason = observe(frame, track_id)
            if reason:
                result['stopped'] = {'anchor': reason}
                output.append(result)
                continue
            stopped, checked, gaps = {}, 1, []
            minimum = {'upper_anchor': 1.0, 'lower_anchor': 1.0,
                       'upper_adjacent': 1.0, 'lower_adjacent': 1.0}
            for direction, label in ((-1, 'before'), (1, 'after')):
                previous, missing = origin, 0
                limit = min(count-1, frame+radius) if direction == 1 else max(0, frame-radius)
                position = frame+direction
                while 0 <= position < count and (position <= limit if direction == 1 else position >= limit):
                    if position in negatives.get(track_id, []):
                        stopped[label] = 'known_negative_sample'
                        break
                    current, reason = observe(position, track_id, previous['box'])
                    if reason == 'missing_detection':
                        missing += 1
                        if missing > max_gap:
                            stopped[label] = 'missing_frame_gap_exceeded'
                            break
                        position += direction
                        continue
                    if reason:
                        stopped[label] = reason
                        break
                    box, old = current['box'], previous['box']
                    height = max(1, (box[3]-box[1]+old[3]-old[1])/2)
                    step = abs(current['frame']-previous['frame'])
                    motion = math.hypot((box[0]+box[2]-old[0]-old[2])/2,
                                        (box[1]+box[3]-old[1]-old[3])/2)/height/step
                    scale = (box[3]-box[1])/max(1, origin['box'][3]-origin['box'][1])
                    if motion > .30 or not .75 <= scale <= 1.33:
                        stopped[label] = 'motion_or_scale_jump'
                        break
                    similarities = {
                        'upper_anchor': float(current['descriptors'][0] @ origin['descriptors'][0]),
                        'lower_anchor': float(current['descriptors'][1] @ origin['descriptors'][1]),
                        'upper_adjacent': float(current['descriptors'][0] @ previous['descriptors'][0]),
                        'lower_adjacent': float(current['descriptors'][1] @ previous['descriptors'][1])}
                    if any(similarities[key] < threshold for key, threshold in (
                            ('upper_anchor', .92), ('lower_anchor', .85),
                            ('upper_adjacent', .96), ('lower_adjacent', .90))):
                        stopped[label] = 'clothing_colour_inconsistency'
                        break
                    for key, value in similarities.items():
                        minimum[key] = min(minimum[key], value)
                    if missing:
                        gaps.append(dict(between=[previous['frame'], position], missing_frames=missing))
                    missing = 0
                    checked += 1
                    if direction == -1:
                        result['start_frame'] = position
                    else:
                        result['end_frame'] = position+1
                    previous = current
                    position += direction
                else:
                    stopped[label] = 'short_window_boundary'
            result.update(extended=result['end_frame']-result['start_frame'] > 1,
                          stopped=stopped, checked_detections=checked,
                          minimum_colour_similarity=minimum, gaps=gaps)
            output.append(result)
    finally:
        video.release()
    return dict(schema='anchor-verification-report-v2', analysis_sha256=document['analysis_sha256'],
                decisions_sha256=document_hash(document), player=player, source=SOURCE,
                metadata={'fps': metadata['fps'], 'frame_count': count,
                          'width': metadata['width'], 'height': metadata['height']},
                colour_descriptor='80% hue/saturation + 20% brightness histogram; square-root overlap',
                rationale='Short same-ID extensions gated by blue clothing, colour/brightness stability, smooth motion, small gaps and no close competing people; not a probability.',
                proposals=output)


def apply_report(document, report):
    """Apply short extensions reversibly, preserving exact samples and user-verified ranges."""
    if report.get('schema') != 'anchor-verification-report-v2' or report.get('source') != SOURCE:
        raise ValueError('Unsupported automatic verification report.')
    if document_hash(document) != report.get('decisions_sha256') or document['analysis_sha256'] != report['analysis_sha256']:
        raise ValueError('Decisions changed since this report; regenerate it.')
    output = deepcopy(document)
    audit = output.setdefault('automatic_verification_audit', [])
    for proposal in report['proposals']:
        if not proposal['extended']:
            continue
        index = proposal['assignment_index']
        entry = output['assignments'][index]
        if entry.get('continuity_status') == 'verified' or entry != proposal['original_assignment']:
            raise ValueError('Cannot alter an existing verified range or changed sample.')
        start, end = proposal['start_frame'], proposal['end_frame']
        anchor = entry['confirmed_sample_frame']
        if not 0 <= start <= anchor < end <= document['frame_count']:
            raise ValueError('Invalid automatic interval.')
        # Do not create overlaps with another saved verified identity interval.
        conflict = any(other_index != index and other['tracker_id'] == entry['tracker_id']
                       and other.get('continuity_status') == 'verified'
                       and max(start, other['start_frame']) < min(end, other['end_frame'])
                       for other_index, other in enumerate(output['assignments']))
        if conflict:
            continue
        original = deepcopy(entry)
        entry.setdefault('reported_track_range', [entry['start_frame'], entry['end_frame']])
        entry.update(start_frame=start, end_frame=end, continuity_status='verified',
                     verification_source=SOURCE,
                     verification_evidence={k: v for k, v in proposal.items() if k != 'original_assignment'})
        audit.append(dict(assignment_index=index, original_assignment=original,
                          applied_assignment=deepcopy(entry), source=SOURCE))
    return output


def revert_automatic_verification(document):
    """Restore unchanged automatic edits; never overwrite later user corrections."""
    output = deepcopy(document)
    for audit in reversed(output.get('automatic_verification_audit', [])):
        index = audit['assignment_index']
        if index < len(output['assignments']) and output['assignments'][index] == audit['applied_assignment']:
            output['assignments'][index] = deepcopy(audit['original_assignment'])
            audit['reverted'] = True
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path)
    parser.add_argument('--analysis', type=Path)
    parser.add_argument('--assignments', type=Path, required=True)
    parser.add_argument('--player')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--competitor-analysis', type=Path)
    parser.add_argument('--max-seconds', type=float, default=1.0)
    parser.add_argument('--max-gap', type=int, default=2)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--revert', action='store_true', help='Restore unchanged automatic edits from the saved audit.')
    args = parser.parse_args()
    try:
        document = json.loads(args.assignments.read_text())
        if args.revert:
            output = revert_automatic_verification(document)
        else:
            if not all((args.video, args.analysis, args.player, args.report)):
                raise ValueError('--video, --analysis, --player and --report are required unless reverting.')
            report = verify_anchored_intervals(args.video, args.analysis, document, args.player,
                                               args.max_seconds, args.max_gap, args.competitor_analysis)
            protected = [args.assignments, args.analysis, args.video]
            if args.competitor_analysis:
                protected.append(args.competitor_analysis)
            if args.report.resolve() in [path.resolve() for path in protected]:
                raise ValueError('Report must use a separate path from video, tracking data and decisions.')
            args.report.write_text(json.dumps(report, indent=2)+'\n')
            print('Proposed short extensions:', sum(p['extended'] for p in report['proposals']))
            output = apply_report(document, report) if args.apply else None
        if output is not None:
            temporary = args.assignments.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(output, indent=2)+'\n')
            temporary.replace(args.assignments)
            print('Saved changes with reversible audit.')
    except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
