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
        choices = []
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
            choices.append((blue, path.name))
        if choices:
            score, filename = max(choices)
            if not blue_only or score >= .35:
                candidates.append({'track_id': track['id'], 'image': filename,
                                   'blue_fraction': score})
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
    """Merge one downloaded document or a batch of yes/no/skip decisions."""
    data, doc = load_workspace(workspace)
    if not player.strip() or len(player) > 60 or document.get('player', player) != player:
        raise ValueError('Invalid or mismatched player name.')
    if document.get('analysis_sha256') != data['analysis_sha256']:
        raise ValueError('Decisions belong to another tracking run.')
    tracks = {t['id']: t for t in data['tracks']}
    decisions = document['decisions']
    seen = set()
    for item in decisions:
        tid = item['track_id']
        if type(tid) is not int or tid not in tracks or tid in seen:
            raise ValueError('Unknown or duplicate track ID.')
        if item['decision'] not in ('yes', 'no', 'skip'):
            raise ValueError('Decision must be yes, no or skip.')
        seen.add(tid)
    # Validate the entire input before changing any saved decision.
    for field in ('assignments', 'rejected_matches', 'skipped_matches', 'automatic_exclusions'):
        doc[field] = [a for a in doc.get(field, [])
                      if not (a.get('name') == player and a['tracker_id'] in seen)]
    for item in decisions:
        tid, decision = item['track_id'], item['decision']
        entry = {'tracker_id': tid, 'name': player}
        field = {'yes': 'assignments', 'no': 'rejected_matches', 'skip': 'skipped_matches'}[decision]
        if decision == 'yes':
            entry.update(start_frame=tracks[tid]['start'], end_frame=tracks[tid]['end'])
        doc[field].append(entry)
    path = workspace / 'named-assignments.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(doc, indent=2) + '\n')
    temporary.replace(path)
    return doc


def build_batch(workspace, player, output, count=40, reference=None, ranking=None):
    data, doc = load_workspace(workspace)
    if not player.strip() or len(player) > 60 or count <= 0:
        raise ValueError('Player name and positive batch size are required.')
    tracks = {t['id']: t for t in data['tracks']}
    candidates = json.loads((workspace / 'candidates.json').read_text())['candidates']
    decided = {a['tracker_id'] for field in ('assignments', 'rejected_matches',
                                           'skipped_matches', 'automatic_exclusions')
               for a in doc.get(field, []) if a.get('name') == player}
    if ranking:
        scores = json.loads(ranking.read_text())
        if scores.get('analysis_sha256') != data['analysis_sha256'] or scores.get('player') != player:
            raise ValueError('Ranking belongs to another player or tracking run.')
        order = {a['track_id']: a['appearance_score'] for a in scores['predictions']}
        candidates.sort(key=lambda c: order.get(c['track_id'], -1), reverse=True)
    remaining = [c for c in candidates if c['track_id'] not in decided][:count]
    if not remaining:
        raise ValueError('No unreviewed candidates remain; skipped tracks remain unresolved.')

    def encoded(filename):
        return 'data:image/jpeg;base64,' + base64.b64encode((workspace/'images'/filename).read_bytes()).decode()

    cards = []
    for candidate in remaining:
        track = tracks[candidate['track_id']]
        cards.append({'track_id': track['id'], 'start_frame': track['start'],
                      'end_frame': track['end'], 'image': encoded(candidate['image'])})
    if reference is None:
        reference = next((a['tracker_id'] for a in doc['assignments'] if a['name'] == player), None)
    reference_image = None
    if reference is not None:
        track = tracks[reference]
        if track['samples']:
            reference_image = encoded(f"track-{reference}-frame-{track['samples'][0]['frame']}.jpg")
    key = hashlib.sha256(json.dumps([player, data['analysis_sha256'],
                                   [c['track_id'] for c in cards]]).encode()).hexdigest()
    payload = dict(player=player, baseline=doc, cards=cards,
                   reference=reference_image, reference_id=reference, batch_key=key)
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
    for name in ('apply', 'build'):
        command = commands.add_parser(name)
        command.add_argument('--player', required=True)
        if name == 'apply':
            command.add_argument('--decisions', type=Path, required=True)
        else:
            command.add_argument('--output', type=Path, required=True)
            command.add_argument('--count', type=int, default=40)
            command.add_argument('--reference', type=int)
            command.add_argument('--ranking', type=Path)
    for command in commands.choices.values():
        command.add_argument('--workspace', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            print('Prepared candidates:', prepare(args.review, args.workspace, args.blue_only))
        elif args.command == 'apply':
            doc = apply_decisions(args.workspace, json.loads(args.decisions.read_text()), args.player)
            print('Confirmed tracks:', sum(a['name'] == args.player for a in doc['assignments']))
        else:
            print('Review cards:', build_batch(args.workspace, args.player, args.output,
                                             args.count, args.reference, args.ranking))
    except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
