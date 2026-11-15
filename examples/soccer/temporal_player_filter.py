"""Propose reversible exclusions for tracks far from a confirmed player at the same time."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

try:
    from .player_review import load_workspace, sample_decisions
    from .review_tracking import fingerprint, records
except ImportError:
    from player_review import load_workspace, sample_decisions
    from review_tracking import fingerprint, records


def temporal_report(analysis, document, player):
    if fingerprint(analysis) != document['analysis_sha256']:
        raise ValueError('Tracking data belongs to another review run.')
    ranges = defaultdict(list)
    for assignment in document['assignments']:
        if assignment['name'] == player and assignment.get('continuity_status') == 'verified':
            ranges[assignment['tracker_id']].append((assignment['start_frame'], assignment['end_frame']))
    yes = set(ranges)
    no = {a['tracker_id'] for a in document.get('rejected_matches', []) if a['name'] == player}
    stats = defaultdict(lambda: dict(observations=0, shared=0, near=0, far=0))
    far_frames = defaultdict(list)
    for row in records(analysis):
        frame = row['frame_index']
        known = [d for d in row['detections'] if any(
            start <= frame < end for start, end in ranges.get(d['tracker_id'], []))]
        for detection in row['detections']:
            tid = detection['tracker_id']
            s = stats[tid]
            s['observations'] += 1
            others = [a for a in known if a['tracker_id'] != tid]
            if not others:
                continue
            x1, y1, x2, y2 = detection['bbox']
            distances = []
            for other in others:
                bx1, by1, bx2, by2 = other['bbox']
                height = max(1, (y2-y1+by2-by1)/2)
                distances.append(math.hypot((x1+x2-bx1-bx2)/2, (y1+y2-by1-by2)/2)/height)
            distance = min(distances)
            s['shared'] += 1
            s['near'] += int(distance < 1)
            s['far'] += int(distance > 2)
            if distance > 2:
                far_frames[tid].append(frame)
    rejected = [tid for tid, s in stats.items() if s['shared'] >= 5 and s['near'] == 0
                and s['far'] == s['shared'] and s['shared']/s['observations'] >= .6]
    def intervals(frames):
        output = []
        for frame in frames:
            if output and output[-1][1] == frame:
                output[-1][1] = frame+1
            else:
                output.append([frame, frame+1])
        return output
    return {'analysis_sha256': document['analysis_sha256'], 'player': player,
            'evidence_version': 'verified-ranges-v2',
            'method': 'Simultaneous separation; self-ID excluded from comparison.',
            'confirmed_player_conflicts': [tid for tid in rejected if tid in yes],
            'manual_negative_matches': sum(tid in no for tid in rejected),
            'proposed_exclusions': [tid for tid in rejected if tid not in yes | no],
            'far_frame_ranges': {tid: intervals(far_frames[tid]) for tid in rejected},
            'tracks': dict(stats)}


def apply_exclusions(workspace, report):
    data, doc = load_workspace(workspace)
    if report['analysis_sha256'] != data['analysis_sha256']:
        raise ValueError('Report belongs to another tracking run.')
    if report.get('evidence_version') != 'verified-ranges-v2':
        raise ValueError('Legacy temporal reports used unverified whole tracks; regenerate the report.')
    if report['confirmed_player_conflicts']:
        raise ValueError('Temporal evidence conflicts with confirmed tracks; inspect the report first.')
    player = report['player']
    decided = sample_decisions(doc, player)
    candidates = json.loads((workspace/'candidates.json').read_text())['candidates']
    added = 0
    for candidate in candidates:
        tid, frame = candidate['track_id'], candidate['sample_frame']
        if (tid, frame) in decided or tid not in report['proposed_exclusions']:
            continue
        intervals = report['far_frame_ranges'].get(tid, report['far_frame_ranges'].get(str(tid), []))
        if not any(start <= frame < end for start, end in intervals):
            continue
        doc.setdefault('automatic_exclusions', []).append({
            'tracker_id': tid, 'sample_frame': frame, 'name': player, 'method': 'temporal-separation',
            'reason': 'Over two player-heights apart in at least 5 shared frames and 60% of observations; no near matches.',
            'evidence': report['tracks'].get(tid, report['tracks'].get(str(tid))), 'reversible': True})
        added += 1
    path = workspace/'named-assignments.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(doc, indent=2)+'\n')
    temporary.replace(path)
    return added


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--player', required=True)
    parser.add_argument('--apply', action='store_true', help='Save reversible exclusions if no confirmed-track conflicts exist.')
    args = parser.parse_args()
    try:
        _, doc = load_workspace(args.workspace)
        report = temporal_report(args.analysis, doc, args.player)
        (args.workspace/'temporal-filter-report.json').write_text(json.dumps(report, indent=2)+'\n')
        print('Confirmed-track conflicts:', report['confirmed_player_conflicts'])
        print('Proposed exclusions:', len(report['proposed_exclusions']))
        if args.apply:
            print('Saved exclusions:', apply_exclusions(args.workspace, report))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
