"""Prepare, review and merge named-player decisions without rerunning detection."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def load_workspace(workspace):
    data = json.loads((workspace / 'review.json').read_text())
    doc = json.loads((workspace / 'named-assignments.json').read_text())
    if doc.get('schema') != 'named-player-assignments-v1':
        raise ValueError('Expected named-player-assignments-v1 decisions.')
    for key in ('analysis_sha256', 'fps', 'frame_count'):
        if data[key] != doc.get(key):
            raise ValueError('Decisions belong to another tracking run.')
    return data, doc


def sample_decisions(document, player):
    """Legacy whole-track decisions have no sample authority and are not propagated."""
    decided = set()
    for field in ('assignments', 'rejected_matches', 'skipped_matches', 'automatic_exclusions'):
        for entry in document.get(field, []):
            if entry.get('name') != player:
                continue
            frame = entry.get('sample_frame', entry.get('confirmed_sample_frame'))
            if type(frame) is int:
                decided.add((entry['tracker_id'], frame))
    return decided


def prepare(review_path, workspace, blue_only=False):
    """Extract an existing review page into an offline workspace."""
    text = review_path.read_text()
    data = json.loads(text.split('const data = ', 1)[1].split(';\nconst status', 1)[0])
    workspace.mkdir(parents=True, exist_ok=False)
    images = workspace / 'images'
    images.mkdir()
    candidates = []
    for track in data['tracks']:
        if type(track['id']) is not int:
            raise ValueError('Invalid track ID.')
        for sample in track['samples']:
            if type(sample['frame']) is not int:
                raise ValueError('Invalid sample frame.')
            content = base64.b64decode(sample['image'], validate=True)
            image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue
            path = images / f"track-{track['id']}-frame-{sample['frame']}.jpg"
            path.write_bytes(content)
            h, w = image.shape[:2]
            crop = image[int(h*.2):max(int(h*.65), int(h*.2)+1),
                         int(w*.15):max(int(w*.85), int(w*.15)+1)]
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            blue = float(((hsv[:, :, 0] >= 95) & (hsv[:, :, 0] <= 130)
                          & (hsv[:, :, 1] >= 100) & (hsv[:, :, 2] >= 45)).mean())
            if not blue_only or blue >= .35:
                candidates.append({'track_id': track['id'], 'sample_frame': sample['frame'],
                                   'image': path.name, 'blue_fraction': blue})
    # Keep only metadata in review.json; images stay in the workspace.
    for track in data['tracks']:
        for sample in track['samples']:
            sample.pop('image', None)
    (workspace / 'review.json').write_text(json.dumps(data))
    (workspace / 'candidates.json').write_text(json.dumps({'candidates': candidates}))
    doc = {k: data[k] for k in ('analysis_sha256', 'fps', 'frame_count')}
    doc.update(schema='named-player-assignments-v1', assignments=[],
               rejected_matches=[], skipped_matches=[], automatic_exclusions=[])
    (workspace / 'named-assignments.json').write_text(json.dumps(doc, indent=2))
    return len(candidates)


def apply_decisions(workspace, document, player):
    """Merge exact-sample decisions; never extend them over a track's duration."""
    data, doc = load_workspace(workspace)
    if not player.strip() or len(player) > 60 or document.get('player', player) != player:
        raise ValueError('Invalid or mismatched player name.')
    if document.get('analysis_sha256') != data['analysis_sha256']:
        raise ValueError('Decisions belong to another tracking run.')
    tracks = {t['id']: t for t in data['tracks']}
    decisions = document['decisions']
    seen = set()
    for item in decisions:
        tid, frame = item['track_id'], item.get('sample_frame')
        if type(tid) is not int or tid not in tracks:
            raise ValueError('Unknown track ID.')
        if type(frame) is not int or frame not in {s['frame'] for s in tracks[tid]['samples']}:
            raise ValueError('Exact displayed sample_frame is required; rebuild old review pages.')
        if (tid, frame) in seen:
            raise ValueError('Duplicate sample decision.')
        if item['decision'] not in ('yes', 'no', 'skip'):
            raise ValueError('Decision must be yes, no or skip.')
        seen.add((tid, frame))
    # Validate the entire input before changing any saved decision.
    for field in ('assignments', 'rejected_matches', 'skipped_matches', 'automatic_exclusions'):
        doc[field] = [a for a in doc.get(field, [])
                      if not (a.get('name') == player and
                              (a['tracker_id'], a.get('sample_frame', a.get('confirmed_sample_frame'))) in seen)]
    for entry in doc['assignments']:
        if entry.get('continuity_status') != 'verified':
            continue
        for item in decisions:
            if entry['tracker_id'] != item['track_id'] or not (
                    entry['start_frame'] <= item['sample_frame'] < entry['end_frame']):
                continue
            contradicts = ((item['decision'] == 'no' and entry.get('name') == player)
                           or (item['decision'] == 'yes' and entry.get('name') != player))
            if contradicts:
                entry['continuity_status'] = 'unverified'
                entry['continuity_issue'] = f"Contradicted by a sample decision at frame {item['sample_frame']}; review this range again."
    for item in decisions:
        tid, decision = item['track_id'], item['decision']
        frame = item['sample_frame']
        entry = {'tracker_id': tid, 'name': player, 'sample_frame': frame,
                 'continuity_status': 'unverified'}
        field = {'yes': 'assignments', 'no': 'rejected_matches', 'skip': 'skipped_matches'}[decision]
        if decision == 'yes':
            entry.update(start_frame=frame, end_frame=frame+1, confirmed_sample_frame=frame)
        doc[field].append(entry)
    path = workspace / 'named-assignments.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(doc, indent=2) + '\n')
    temporary.replace(path)
    return doc


def verify_range(workspace, player, track_id, start_frame, end_frame):
    """Record an explicitly reviewed continuous identity interval, never inferred from a thumbnail."""
    data, doc = load_workspace(workspace)
    tracks = {t['id']: t for t in data['tracks']}
    if not player.strip() or len(player) > 60 or track_id not in tracks:
        raise ValueError('Valid player and track ID required.')
    track = tracks[track_id]
    if not track['start'] <= start_frame < end_frame <= track['end']:
        raise ValueError('Verified range must lie within the observed track interval.')
    # Retain sample confirmations; replace only overlapping verified intervals for this player.
    kept = []
    for entry in doc['assignments']:
        overlap = (entry['tracker_id'] == track_id and
                   max(entry['start_frame'], start_frame) < min(entry['end_frame'], end_frame))
        if overlap and entry.get('name') != player:
            raise ValueError('Range conflicts with another saved player; correct that assignment first.')
        if overlap and entry.get('continuity_status') == 'verified':
            if entry['start_frame'] < start_frame:
                kept.append(dict(entry, end_frame=start_frame))
            if end_frame < entry['end_frame']:
                kept.append(dict(entry, start_frame=end_frame))
        else:
            kept.append(entry)
    kept.append(dict(tracker_id=track_id, name=player, start_frame=start_frame,
                     end_frame=end_frame, continuity_status='verified', verification_source='user-reviewed'))
    doc['assignments'] = kept
    path = workspace/'named-assignments.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(doc, indent=2)+'\n')
    temporary.replace(path)
    return doc


def context_images(video_path, analysis_path, data, candidates):
    """Render enlarged scene context around the exact reviewed detection frame."""
    try:
        from .review_tracking import fingerprint, records, open_video
    except ImportError:
        from review_tracking import fingerprint, records, open_video
    if fingerprint(analysis_path) != data['analysis_sha256']:
        raise ValueError('Context tracking data belongs to another run.')
    wanted = {(c['track_id'], c['sample_frame']) for c in candidates}
    boxes, metadata = {}, None
    for row in records(analysis_path):
        metadata = metadata or row
        for detection in row['detections']:
            key = (detection['tracker_id'], row['frame_index'])
            if key in wanted:
                boxes[key] = detection['bbox']
    if metadata is None or set(boxes) != wanted:
        raise ValueError('Exact sample detection missing from context tracking data.')
    video = open_video(video_path, metadata)
    output = {}
    try:
        for key, (x1, y1, x2, y2) in boxes.items():
            video.set(cv2.CAP_PROP_POS_FRAMES, key[1])
            ok, frame = video.read()
            if not ok:
                raise ValueError(f'Cannot read context frame {key[1]}.')
            h, w = frame.shape[:2]
            ch = min(h, max(225, int((y2-y1)*2.5)))
            cw = min(w, max(300, int(ch*4/3), int((x2-x1)*4)))
            left = max(0, min(w-cw, int((x1+x2-cw)/2)))
            top = max(0, min(h-ch, int((y1+y2-ch)/2)))
            crop = frame[top:top+ch, left:left+cw].copy()
            cv2.rectangle(crop, (int(x1-left), int(y1-top)),
                          (int(x2-left), int(y2-top)), (0, 200, 255), 2)
            scale = min(640/cw, 480/ch)
            crop = cv2.resize(crop, (int(cw*scale), int(ch*scale)))
            ok, encoded = cv2.imencode('.jpg', crop)
            if not ok:
                raise ValueError('Cannot encode contextual sample.')
            output[key] = 'data:image/jpeg;base64,' + base64.b64encode(encoded).decode()
    finally:
        video.release()
    return output


def build_batch(workspace, player, output, count=40, reference=None, ranking=None,
                context_video=None, analysis=None):
    data, doc = load_workspace(workspace)
    if not player.strip() or len(player) > 60 or count <= 0:
        raise ValueError('Player name and positive batch size are required.')
    tracks = {t['id']: t for t in data['tracks']}
    candidates = json.loads((workspace / 'candidates.json').read_text())['candidates']
    if any(type(c.get('sample_frame')) is not int for c in candidates):
        raise ValueError('Legacy workspace has no exact sample provenance; prepare a new workspace from the review HTML.')
    decided = sample_decisions(doc, player)
    if ranking:
        scores = json.loads(ranking.read_text())
        if scores.get('analysis_sha256') != data['analysis_sha256'] or scores.get('player') != player:
            raise ValueError('Ranking belongs to another player or tracking run.')
        if scores.get('feature_version') != 'sample-colour-v2':
            raise ValueError('Legacy whole-track ranking is unsafe; rerun learn_player_filter.py.')
        order = {(a['track_id'], a['sample_frame']): a['appearance_score'] for a in scores['predictions']}
        candidates.sort(key=lambda c: order.get((c['track_id'], c['sample_frame']), -1), reverse=True)
    remaining = [c for c in candidates if (c['track_id'], c['sample_frame']) not in decided][:count]
    if not remaining:
        raise ValueError('No unreviewed candidates remain; skipped tracks remain unresolved.')
    if bool(context_video) != bool(analysis):
        raise ValueError('Context review requires both --context-video and --analysis.')
    contexts = context_images(context_video, analysis, data, remaining) if context_video else {}

    def encoded(filename):
        return 'data:image/jpeg;base64,' + base64.b64encode((workspace/'images'/filename).read_bytes()).decode()

    cards = []
    for candidate in remaining:
        track = tracks[candidate['track_id']]
        cards.append({'track_id': track['id'], 'sample_frame': candidate['sample_frame'], 'start_frame': track['start'],
                      'end_frame': track['end'], 'image': contexts.get(
                          (track['id'], candidate['sample_frame'])) or encoded(candidate['image'])})
    confirmed = next((a for a in doc['assignments'] if a['name'] == player and
                      type(a.get('confirmed_sample_frame')) is int and
                      (reference is None or a['tracker_id'] == reference)), None)
    reference_image = None
    if reference is not None and confirmed is None:
        raise ValueError('Reference must have an exact confirmed sample for this player.')
    if confirmed is not None:
        reference = confirmed['tracker_id']
        frame = confirmed['confirmed_sample_frame']
        reference_image = encoded(f"track-{reference}-frame-{frame}.jpg")
    key = hashlib.sha256(json.dumps([player, data['analysis_sha256'],
                                   [(c['track_id'], c['sample_frame']) for c in cards]]).encode()).hexdigest()
    payload = dict(player=player, baseline=doc, cards=cards,
                   reference=reference_image, reference_id=reference, batch_key=key,
                   contextual=bool(context_video))
    template = Path(__file__).with_name('player_review.html').read_text()
    with output.open('x') as destination:
        destination.write(template.replace('__PAYLOAD__', json.dumps(payload).replace('<', '\\u003c')))
    return len(cards)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare_parser = commands.add_parser('prepare')
    prepare_parser.add_argument('--review', type=Path, required=True)
    prepare_parser.add_argument('--blue-only', action='store_true')
    for name in ('apply', 'build', 'verify-range'):
        command = commands.add_parser(name)
        command.add_argument('--player', required=True)
        if name == 'apply':
            command.add_argument('--decisions', type=Path, required=True)
        elif name == 'build':
            command.add_argument('--output', type=Path, required=True)
            command.add_argument('--count', type=int, default=40)
            command.add_argument('--reference', type=int)
            command.add_argument('--ranking', type=Path)
            command.add_argument('--context-video', type=Path)
            command.add_argument('--analysis', type=Path)
        else:
            command.add_argument('--track-id', type=int, required=True)
            command.add_argument('--start-frame', type=int, required=True)
            command.add_argument('--end-frame', type=int, required=True)
    for command in commands.choices.values():
        command.add_argument('--workspace', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            print('Prepared candidates:', prepare(args.review, args.workspace, args.blue_only))
        elif args.command == 'apply':
            doc = apply_decisions(args.workspace, json.loads(args.decisions.read_text()), args.player)
            print('Confirmed tracks:', sum(a['name'] == args.player for a in doc['assignments']))
        elif args.command == 'verify-range':
            verify_range(args.workspace, args.player, args.track_id, args.start_frame, args.end_frame)
            print('Saved explicitly verified identity range.')
        else:
            print('Review cards:', build_batch(args.workspace, args.player, args.output,
                                             args.count, args.reference, args.ranking,
                                             args.context_video, args.analysis))
    except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
