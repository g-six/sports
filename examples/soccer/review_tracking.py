"""Build an offline track review page or render manually assigned identities."""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path

import cv2


def fingerprint(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def records(path):
    metadata = None
    with open(path) as source:
        for index, line in enumerate(source):
            row = json.loads(line)
            if row.get('schema') != 'player-tracking-v1' or row['frame_index'] != index:
                raise ValueError('Expected a contiguous player-tracking-v1 export starting at frame 0.')
            current = (row['fps'], row['width'], row['height'])
            if not math.isfinite(current[0]) or min(current) <= 0:
                raise ValueError('Invalid video metadata.')
            if metadata is not None and current != metadata:
                raise ValueError('Video metadata changed within tracking export.')
            metadata = current
            yield row


def open_video(path, first):
    video = cv2.VideoCapture(str(path))
    if not video.isOpened():
        raise ValueError(f'Cannot open video: {path}')
    actual = (video.get(cv2.CAP_PROP_FRAME_WIDTH), video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if actual != (first['width'], first['height']) or abs(
            video.get(cv2.CAP_PROP_FPS) - first['fps']) > 0.05:
        video.release()
        raise ValueError('Video dimensions/FPS differ from the tracking export.')
    return video


def build_review(video_path, analysis_path, output_path):
    tracks = {}
    first = None
    count = 0
    for row in records(analysis_path):
        first = first or row
        count += 1
        for detection in row['detections']:
            track_id = detection['tracker_id']
            sample = dict(detection, frame=row['frame_index'])
            if track_id not in tracks:
                tracks[track_id] = {'first': sample, 'best': sample, 'last': sample}
            track = tracks[track_id]
            track['last'] = sample
            if sample['confidence'] > track['best']['confidence']:
                track['best'] = sample
    if first is None:
        raise ValueError('Tracking export is empty.')
    video = open_video(video_path, first)
    cards = []
    try:
        for track_id, track in sorted(tracks.items()):
            samples = []
            unique = {sample['frame']: sample for sample in track.values()}
            for frame_index, sample in sorted(unique.items()):
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = video.read()
                if not ok:
                    raise ValueError(f'Cannot read frame {frame_index}; use the matching tracking output video.')
                x1, y1, x2, y2 = [int(value) for value in sample['bbox']]
                crop = frame[max(0, y1):min(frame.shape[0], y2),
                             max(0, x1):min(frame.shape[1], x2)]
                if crop.size == 0:
                    continue
                scale = min(1, 240 / crop.shape[0])
                crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)),
                                         max(1, int(crop.shape[0] * scale))))
                ok, encoded = cv2.imencode('.jpg', crop)
                if not ok:
                    raise ValueError('Failed to encode review image.')
                samples.append({'frame': frame_index, 'image': base64.b64encode(encoded).decode()})
            cards.append({'id': track_id, 'start': track['first']['frame'],
                          'end': track['last']['frame'] + 1, 'samples': samples})
    finally:
        video.release()
    data = {'analysis_sha256': fingerprint(analysis_path), 'fps': first['fps'],
            'frame_count': count, 'source_time_offset': first['source_timestamp'],
            'tracks': cards}
    payload = json.dumps(data).replace('<', '\\u003c')
    with open(output_path, 'x') as output:
        output.write(REVIEW_HTML.replace('__TRACKING_DATA__', payload))
    print(f'Review page: {Path(output_path).resolve()}')
    print('Open it in your browser, assign labels, then download assignments.json.')


def validate_assignments(document, analysis_path, fps, frame_count):
    if document.get('schema') not in ('player-assignments-v1', 'named-player-assignments-v1'):
        raise ValueError('Unsupported assignments format.')
    if document.get('analysis_sha256') != fingerprint(analysis_path):
        raise ValueError('Assignments belong to a different tracking run.')
    if document.get('fps') != fps or document.get('frame_count') != frame_count:
        raise ValueError('Assignments metadata does not match the tracking export.')
    grouped = {}
    for item in document['assignments']:
        track_id, start, end = item['tracker_id'], item['start_frame'], item['end_frame']
        if any(type(value) is not int for value in (track_id, start, end)):
            raise ValueError('Track IDs and frame ranges must be integers.')
        if not 0 <= start < end <= frame_count:
            raise ValueError('Invalid assignment frame range.')
        if document['schema'] == 'named-player-assignments-v1':
            name = item.get('name')
            if not isinstance(name, str) or not name.strip() or len(name) > 60:
                raise ValueError('Player name must contain 1–60 characters.')
            if item.get('continuity_status') != 'verified':
                frame = item.get('confirmed_sample_frame')
                if (item.get('continuity_status') != 'unverified' or type(frame) is not int
                        or start != frame or end != frame + 1):
                    raise ValueError('Named track continuity is unverified. Confirm the displayed sample_frame only, '
                                     'or review the video and use player_review.py verify-range before rendering a range.')
        else:
            team, jersey = item['team'], item['jersey']
            if not isinstance(team, str) or not team.strip() or len(team) > 60:
                raise ValueError('Team label must contain 1–60 characters.')
            if not isinstance(jersey, str) or not jersey.isascii() or not jersey.isdigit() or len(jersey) > 3:
                raise ValueError('Jersey number must contain 1–3 digits.')
        grouped.setdefault(track_id, []).append(item)
    for items in grouped.values():
        items.sort(key=lambda item: item['start_frame'])
        if document['schema'] == 'named-player-assignments-v1':
            # A verified interval may cover an existing exact-sample confirmation.
            merged = []
            for item in items:
                if merged and merged[-1]['end_frame'] > item['start_frame']:
                    if merged[-1]['name'] != item['name']:
                        raise ValueError('Different players overlap on the same track; correct their ranges.')
                    merged[-1] = dict(merged[-1], end_frame=max(merged[-1]['end_frame'], item['end_frame']))
                else:
                    merged.append(dict(item))
            items[:] = merged
        if any(a['end_frame'] > b['start_frame'] for a, b in zip(items, items[1:])):
            raise ValueError('Assignments for the same track overlap; split their frame ranges.')
    return grouped


def identity_label(detection, frame_index, assignments):
    track_id = detection['tracker_id']
    for assignment in assignments.get(track_id, []):
        if assignment['start_frame'] <= frame_index < assignment['end_frame']:
            if 'name' in assignment:
                return f"{assignment['name']} (ID {track_id})"
            return f"{assignment['team']} #{assignment['jersey']} (ID {track_id})"
    return f'ID {track_id}'


def render(video_path, analysis_path, assignments_path, output_path):
    if Path(output_path).exists():
        raise ValueError('Output already exists; choose a new filename.')
    first = next(records(analysis_path), None)
    if first is None:
        raise ValueError('Tracking export is empty.')
    frame_count = sum(1 for _ in records(analysis_path))
    with open(assignments_path) as source:
        document = json.load(source)
    assignments = validate_assignments(document, analysis_path, first['fps'], frame_count)
    known_ids = {d['tracker_id'] for row in records(analysis_path) for d in row['detections']}
    if set(assignments) - known_ids:
        raise ValueError('Assignments contain unknown track IDs.')
    video = open_video(video_path, first)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'),
                             first['fps'], (first['width'], first['height']))
    if not writer.isOpened():
        video.release()
        raise ValueError(f'Cannot create output video: {output_path}')
    try:
        for row in records(analysis_path):
            ok, frame = video.read()
            if not ok:
                raise ValueError('Video ended before tracking data; use the matching video segment.')
            for detection in row['detections']:
                x1, y1, x2, y2 = [int(value) for value in detection['bbox']]
                label = identity_label(detection, row['frame_index'], assignments)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)
                origin = (max(0, x1), max(18, y1 - 8))
                cv2.putText(frame, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                cv2.putText(frame, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            writer.write(frame)
    finally:
        video.release()
        writer.release()
    print(f'Labelled video: {Path(output_path).resolve()}')


REVIEW_HTML = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Player identity review</title>
<style>
body{font:16px system-ui;background:#f4f6f8;color:#17212b;margin:24px;max-width:1200px}
header{position:sticky;top:0;background:#f4f6f8;padding:12px 0;z-index:1}
button,input{font:inherit;padding:7px;margin:4px}button{cursor:pointer}
article{background:white;border:1px solid #ccd4dd;border-radius:10px;padding:18px;margin:18px 0}
.samples{display:flex;gap:12px;overflow:auto}figure{margin:8px}img{height:200px;object-fit:contain}
label{display:inline-flex;flex-direction:column;margin:4px}input{width:130px}
.assignment{border-top:1px solid #ddd;padding-top:8px}#status{color:#274b70}
</style>
<header><h1>Assign player identities</h1>
<button id="save">Download assignments.json</button>
<label>Load saved assignments<input type="file" id="load" accept=".json"></label>
<p id="status" role="status"></p></header>
<p>Review the matching video when needed. Samples show first, highest-confidence and last observations;
they do not verify an entire track. Leave unknown players blank. Add separate ranges if an ID switches players.
Multiple track IDs can share the same team and jersey.</p>
<p>Ranges use <b>segment frame numbers</b>: start included, end excluded. Frame 0 is the segment's start.
Edits remain in this page until you download them. Keep the JSON file to save your work.</p>
<div id="tracks"></div>
<script>
const data = __TRACKING_DATA__;
const status = document.querySelector('#status');
status.textContent = `${data.tracks.length} tracks · ${data.fps.toFixed(3)} FPS · original video offset ${data.source_time_offset.toFixed(3)} seconds`;
function input(label,value,type='text') {
 const wrapper=document.createElement('label'); wrapper.textContent=label;
 const field=document.createElement('input'); field.type=type; field.value=value;
 wrapper.append(field); return [wrapper,field];
}
function addAssignment(card,track,entry={}) {
 const row=document.createElement('div'); row.className='assignment'; row.dataset.id=track.id;
 const specs=[['Team / colour',entry.team??'','text'],['Jersey number',entry.jersey??'','text'],
 ['Start frame',entry.start_frame??track.start,'number'],['End frame (excluded)',entry.end_frame??track.end,'number']];
 for(const [name,value,type] of specs){ const [label,field]=input(name,value,type); row.append(label);
 if(type==='number'){field.min=0;field.max=data.frame_count;field.step=1;} }
 const remove=document.createElement('button');remove.textContent='Remove range'; remove.onclick=()=>row.remove();
 row.append(remove);card.append(row);
}
const cards=new Map();
for(const track of data.tracks){
 const card=document.createElement('article'); cards.set(track.id,card);
 const title=document.createElement('h2');title.textContent=`Track ID ${track.id}`;card.append(title);
 const info=document.createElement('p');
 info.textContent=`Frames ${track.start}–${track.end-1} · segment ${(track.start/data.fps).toFixed(2)}–${(track.end/data.fps).toFixed(2)} s`;
 card.append(info);const samples=document.createElement('div');samples.className='samples';
 for(const sample of track.samples){const figure=document.createElement('figure'),image=document.createElement('img'),caption=document.createElement('figcaption');
 image.src='data:image/jpeg;base64,'+sample.image;image.alt=`Track ${track.id} at frame ${sample.frame}`;
 caption.textContent=`Frame ${sample.frame} · source ${(data.source_time_offset+sample.frame/data.fps).toFixed(2)} s`;
 figure.append(image,caption);samples.append(figure);}card.append(samples);
 const add=document.createElement('button');add.textContent='Add identity range';add.onclick=()=>addAssignment(card,track);
 card.append(add);addAssignment(card,track);document.querySelector('#tracks').append(card);
}
document.querySelector('#save').onclick=()=>{
 try{
 const assignments=[];
 for(const row of document.querySelectorAll('.assignment')){
 const [team,jersey,start,end]=[...row.querySelectorAll('input')].map(field=>field.value.trim());
 if(!team&&!jersey)continue;
 const a=Number(start),b=Number(end),id=Number(row.dataset.id);
 if(!team||team.length>60||!/^\d{1,3}$/.test(jersey)||!start||!end||!Number.isInteger(a)||!Number.isInteger(b)||a<0||a>=b||b>data.frame_count)
 throw Error(`Check identity and frame range for track ${id}.`);
 assignments.push({tracker_id:id,team,jersey,start_frame:a,end_frame:b});
 }
 assignments.sort((a,b)=>a.tracker_id-b.tracker_id||a.start_frame-b.start_frame);
 for(let i=1;i<assignments.length;i++){const a=assignments[i-1],b=assignments[i];
 if(a.tracker_id===b.tracker_id&&a.end_frame>b.start_frame)throw Error(`Overlapping ranges for track ${a.tracker_id}.`);}
 const documentData={schema:'player-assignments-v1',analysis_sha256:data.analysis_sha256,fps:data.fps,frame_count:data.frame_count,assignments};
 const url=URL.createObjectURL(new Blob([JSON.stringify(documentData,null,2)],{type:'application/json'}));
 const link=document.createElement('a');link.href=url;link.download='assignments.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
 status.textContent=`Downloaded ${assignments.length} identity ranges. Move the JSON next to your tracking files.`;
 }catch(error){status.textContent=error.message;}
};
document.querySelector('#load').onchange=async event=>{
 try{
 const saved=JSON.parse(await event.target.files[0].text());
 if(saved.schema!=='player-assignments-v1'||saved.analysis_sha256!==data.analysis_sha256||!Array.isArray(saved.assignments))
 throw Error('This file does not belong to this tracking run.');
 for(const item of saved.assignments)if(!cards.has(item.tracker_id))throw Error('Unknown track ID.');
 document.querySelectorAll('.assignment').forEach(row=>row.remove());
 for(const item of saved.assignments)addAssignment(cards.get(item.tracker_id),data.tracks.find(t=>t.id===item.tracker_id),item);
 status.textContent='Loaded saved assignments.';
 }catch(error){status.textContent=error.message;}
};
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('build', 'render'):
        command = commands.add_parser(name)
        command.add_argument('--video', required=True, type=Path,
                             help='Exact segment used for tracking, or its annotated output (frame 0 aligned).')
        command.add_argument('--analysis', required=True, type=Path)
        command.add_argument('--output', required=True, type=Path)
        if name == 'render':
            command.add_argument('--assignments', required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == 'build':
            build_review(args.video, args.analysis, args.output)
        else:
            render(args.video, args.analysis, args.assignments, args.output)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
