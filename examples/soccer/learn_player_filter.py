"""Rank saved track crops using user-confirmed yes/no decisions."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score
try:
    from .player_review import sample_decisions
except ImportError:
    from player_review import sample_decisions


def sample_training_labels(document, player):
    """Only exact reviewed crops are supervised; a name never labels another frame of its track."""
    labels = {}
    for field, value in [('assignments', 1), ('rejected_matches', 0)]:
        for entry in document.get(field, []):
            if entry.get('name') != player:
                continue
            frame = entry.get('confirmed_sample_frame') if value else entry.get('sample_frame')
            if type(frame) is not int:
                continue
            key = (entry['tracker_id'], frame)
            if key in labels and labels[key] != value:
                raise ValueError('Conflicting labels for the same sample; correct the decisions first.')
            labels[key] = value
    return labels


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
    cache = ROOT/'sample-appearance-features.npz'
    source_hash = data['analysis_sha256']
    feature_version = 'sample-colour-v2'
    if cache.exists():
        with np.load(cache) as saved:
            if str(saved['source_hash']) != source_hash or str(saved['feature_version']) != feature_version:
                raise ValueError('Feature cache belongs to another tracking run/version; remove it.')
            keys, matrix = saved['keys'], saved['features']
    else:
        rows, keys = [], []
        for track in data['tracks']:
            for sample in track['samples']:
                image = cv2.imread(str(ROOT/'images'/f"track-{track['id']}-frame-{sample['frame']}.jpg"))
                if image is not None:
                    rows.append(features(image))
                    keys.append((track['id'], sample['frame']))
        keys, matrix = np.asarray(keys, dtype=int).reshape(-1, 2), np.asarray(rows)
        np.savez_compressed(cache, keys=keys, features=matrix, source_hash=source_hash,
                            feature_version=feature_version)
    labels = sample_training_labels(doc, args.player)
    labelled = np.asarray([i for i, key in enumerate(keys) if tuple(key) in labels], dtype=int)
    x = matrix[labelled]
    y = np.asarray([labels[tuple(keys[i])] for i in labelled], dtype=int)
    # Hold nearby tracks together to reduce leakage from adjacent views.
    groups = np.asarray([int(tracks[int(keys[i][0])]['start']/data['fps']/30) for i in labelled])
    if min(np.bincount(y, minlength=2)) < 5 or len(set(groups)) < 5:
        raise ValueError('Ranking needs at least five positives, five negatives, and five 30-second groups.')
    report_path = ROOT/'learned-filter-report.json'
    previous = json.loads(report_path.read_text()) if report_path.exists() else {}
    if (previous.get('analysis_sha256') != source_hash or previous.get('player') != args.player
            or previous.get('feature_version') != feature_version):
        previous = {}
    current_keys = [f'{keys[i][0]}:{keys[i][1]}' for i in labelled]
    prior_keys = set(previous.get('active_training_sample_keys', current_keys))
    prior_mask = np.asarray([key in prior_keys for key in current_keys])
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
    decided = sample_decisions(doc, args.player)
    candidates = {(c['track_id'], c['sample_frame']) for c in
                  json.loads((ROOT/'candidates.json').read_text())['candidates']}
    predictions = [{'track_id': int(key[0]), 'sample_frame': int(key[1]),
                    'appearance_score': float(score), 'proposed_exclusion': bool(score < threshold)}
                   for key, score in zip(keys, scores)
                   if tuple(key) in candidates and tuple(key) not in decided]
    report = {'method': 'Generic clothing, hair-region and colour features; no face recognition.',
              'validation': '5 folds grouped by 30-second track-start windows',
              'regression_gate': {'accepted': accepted, 'comparable': comparable,
                                  'candidate': candidate_metrics, 'baseline': baseline_metrics},
              'feature_version': feature_version,
              'active_training_sample_keys': [key for key, active in zip(current_keys, active_mask) if active],
              'manual_positive_samples': int(y.sum()), 'manual_negative_samples': int((y == 0).sum()),
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
    print(json.dumps({k: v for k, v in report.items() if k not in ('predictions', 'active_training_sample_keys')}, indent=2))


if __name__ == '__main__':
    main()
