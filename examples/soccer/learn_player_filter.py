"""Rank saved track crops using user-confirmed yes/no decisions."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score
def features(image):
    # Generic colour/texture only: no face embeddings or demographic inference.
    h, w = image.shape[:2]
    pieces = []
    for y1, y2, x1, x2 in [(0, .20, .25, .75), (.20, .62, .15, .85),
                           (.62, 1, .15, .85)]:
        crop = image[int(h*y1):max(int(h*y2), int(h*y1)+1),
                     int(w*x1):max(int(w*x2), int(w*x1)+1)]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        hist = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]).flatten()
        pieces.extend(hist / max(1, hist.sum()))
        for channel in cv2.split(lab):
            hist = cv2.calcHist([channel], [0], None, [12], [0, 256]).flatten()
            pieces.extend(hist / max(1, hist.sum()))
        pieces.extend(np.percentile(lab.reshape(-1, 3), [20, 50, 80], axis=0).flatten()/255)
    # Coarse clothing layout; excludes the head.
    torso = image[max(0, int(h*.20)):]
    lab = cv2.cvtColor(cv2.resize(torso, (6, 12)), cv2.COLOR_BGR2LAB)
    pieces.extend(lab.flatten()/255)
    return np.asarray(pieces, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--player', required=True)
    args = parser.parse_args()
    ROOT = args.workspace
    data = json.loads((ROOT/'review.json').read_text())
    tracks = {t['id']: t for t in data['tracks']}
    doc = json.loads((ROOT/'named-assignments.json').read_text())
    if doc['analysis_sha256'] != data['analysis_sha256']:
        raise ValueError('Assignments belong to another tracking run.')
    blue = json.loads((ROOT/'candidates.json').read_text())['candidates']
    cache = ROOT/'appearance-features.npz'
    source_hash = data['analysis_sha256']
    if cache.exists():
        with np.load(cache) as saved:
            if str(saved['source_hash']) != source_hash:
                raise ValueError('Feature cache belongs to another tracking run; remove it.')
            ids, matrix = saved['ids'], saved['features']
    else:
        rows, ids = [], []
        for candidate in blue:
            track = tracks[candidate['track_id']]
            samples = []
            for sample in track['samples']:
                image = cv2.imread(str(ROOT/'images'/f"track-{track['id']}-frame-{sample['frame']}.jpg"))
                if image is not None:
                    samples.append(features(image))
            if samples:
                rows.append(np.concatenate((np.mean(samples, axis=0), np.std(samples, axis=0))))
                ids.append(track['id'])
        ids, matrix = np.asarray(ids), np.asarray(rows)
        np.savez_compressed(cache, ids=ids, features=matrix, source_hash=source_hash)
    yes = {a['tracker_id'] for a in doc['assignments'] if a['name'] == args.player}
    no = {a['tracker_id'] for a in doc['rejected_matches'] if a['name'] == args.player}
    labelled = np.asarray([i for i, tid in enumerate(ids) if tid in yes or tid in no], dtype=int)
    x = matrix[labelled]
    y = np.asarray([int(ids[i] in yes) for i in labelled], dtype=int)
    # Hold nearby tracks together to reduce leakage from adjacent views.
    groups = np.asarray([int(tracks[int(ids[i])]['start']/data['fps']/30) for i in labelled])
    if min(np.bincount(y, minlength=2)) < 5 or len(set(groups)) < 5:
        raise ValueError('Ranking needs at least five positives, five negatives, and five 30-second groups.')
    report_path = ROOT/'learned-filter-report.json'
    previous = json.loads(report_path.read_text()) if report_path.exists() else {}
    if previous.get('analysis_sha256') != source_hash or previous.get('player') != args.player:
        previous = {}
    current_ids = np.asarray([int(ids[i]) for i in labelled])
    prior_ids = set(previous.get('active_training_track_ids', current_ids.tolist()))
    prior_mask = np.asarray([tid in prior_ids for tid in current_ids])
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    candidate_oof, baseline_oof = np.zeros(len(y)), np.zeros(len(y))
    comparable = True
    for train, test in cv.split(x, y, groups):
        model = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2,
                                     class_weight='balanced', n_jobs=2, random_state=42)
        if len(set(y[train])) != 2:
            raise ValueError('A validation fold has only one class; collect more varied examples.')
        model.fit(x[train], y[train])
        candidate_oof[test] = model.predict_proba(x[test])[:, 1]
        old_train = train[prior_mask[train]]
        if len(set(y[old_train])) != 2:
            comparable = False
        else:
            baseline = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2,
                                            class_weight='balanced', n_jobs=2, random_state=42)
            baseline.fit(x[old_train], y[old_train])
            baseline_oof[test] = baseline.predict_proba(x[test])[:, 1]
    def metrics(values):
        top = np.argsort(-values, kind='stable')[:min(40, len(y))]
        return {'average_precision': float(average_precision_score(y, values)),
                'top40_precision': float(y[top].mean())}
    candidate_metrics = metrics(candidate_oof)
    baseline_metrics = metrics(baseline_oof) if comparable else None
    accepted = comparable and all(candidate_metrics[key] + 1e-12 >= baseline_metrics[key]
                                  for key in candidate_metrics)
    active_mask = np.ones(len(y), dtype=bool) if accepted else prior_mask
    if len(set(y[active_mask])) != 2:
        raise ValueError('Previous training set is no longer usable; remove the ranking report and retrain.')
    oof = candidate_oof if accepted else baseline_oof
    # Reject only below the weakest held-out positive, with an extra margin.
    threshold = min(.20, float(oof[y == 1].min()) * .75) if comparable else 0
    top = np.argsort(-oof, kind='stable')[:min(40, len(y))]
    model = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2,
                                 class_weight='balanced', n_jobs=2, random_state=42)
    model.fit(x[active_mask], y[active_mask])
    scores = model.predict_proba(matrix)[:, 1]
    decided = yes | no | {a['tracker_id'] for k in ('skipped_matches', 'automatic_exclusions')
                          for a in doc.get(k, []) if a.get('name') == args.player}
    predictions = [{'track_id': int(tid), 'appearance_score': float(score),
                    'proposed_exclusion': bool(score < threshold)}
                   for tid, score in zip(ids, scores) if tid not in decided]
    report = {'method': 'Generic clothing, hair-region and colour features; no face recognition.',
              'validation': '5 folds grouped by 30-second track-start windows',
              'regression_gate': {'accepted': accepted, 'comparable': comparable,
                                  'candidate': candidate_metrics, 'baseline': baseline_metrics},
              'active_training_track_ids': current_ids[active_mask].tolist(),
              'manual_positive_tracks': int(y.sum()), 'manual_negative_tracks': int((y == 0).sum()),
              'threshold': threshold, 'held_out_positive_minimum': float(oof[y == 1].min()),
              'held_out_average_precision': float(average_precision_score(y, oof)) if comparable else None,
              'held_out_top40_precision': float(y[top].mean()) if comparable else None,
              'held_out_player_tracks_rejected': int((oof[y == 1] < threshold).sum()),
              'held_out_other_tracks_rejected': int((oof[y == 0] < threshold).sum()),
              'remaining_tracks': len(predictions),
              'proposed_exclusions': sum(p['proposed_exclusion'] for p in predictions),
              'validation_note': 'Threshold is chosen from these folds, not an independent test; future accuracy is not guaranteed.',
              'analysis_sha256': source_hash, 'player': args.player, 'predictions': predictions}
    (ROOT/'learned-filter-report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('predictions', 'active_training_track_ids')}, indent=2))


if __name__ == '__main__':
    main()
