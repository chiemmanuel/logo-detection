import os
import json
import time
from datetime import timedelta
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
from ultralytics import YOLO

from sort import Sort 
from dc_utils import mysql_execute_insert

# ---------- SQL for inserting track-level rows ----------
INSERT_SQL_VIDEO = """
INSERT INTO yolo_video_detection
(study_id, media_id, plateform_id, logo, size, area, areaPercentage, timeBegin, timeEnd, confidence, x1, y1, x2, y2)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
"""

# ---------- Configurable thresholds ----------
CONF_THRESHOLD = 0.35
MIN_CONF_FOR_VALID_TRACK = 0.35
MIN_VISIBLE_FRAMES = 3
MAX_INACTIVE_FRAMES = 20
IOU_MATCH_THRESHOLD = 0.3

# ---------- Helpers ----------
def parse_filename(file_name: str) -> Tuple[int, str, int]:
    """Parse filename like 53_33731469523_1219967756810240_6.mp4 -> (study_id, media_id, plateform_id)"""
    base_name = os.path.splitext(file_name)[0]
    parts = base_name.split("_")
    if len(parts) < 3:
        raise ValueError(f"Filename not in expected format: {file_name}")
    study_id = int(parts[0])
    plateform_id = int(parts[-1])
    media_id = "_".join(parts[1:-1])
    return study_id, media_id, plateform_id

def iou_bbox(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    xx1 = max(a[0], b[0])
    yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2])
    yy2 = min(a[3], b[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_b = (b[2]-b[0])*(b[3]-b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def match_tracks_to_detections(tracks: np.ndarray, detections: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Match each SORT track to the best YOLO detection based on IoU."""
    mapping: Dict[int, Dict[str, Any]] = {}
    det_boxes = [(d['box']['x1'], d['box']['y1'], d['box']['x2'], d['box']['y2']) for d in detections]
    for t in tracks:
        tx1, ty1, tx2, ty2, tid = float(t[0]), float(t[1]), float(t[2]), float(t[3]), int(t[4])
        best_iou, best_idx = 0.0, -1
        for i, db in enumerate(det_boxes):
            iou_val = iou_bbox((tx1, ty1, tx2, ty2), db)
            if iou_val > best_iou:
                best_iou, best_idx = iou_val, i
        if best_idx >= 0 and best_iou >= IOU_MATCH_THRESHOLD:
            mapping[tid] = detections[best_idx]
        else:
            mapping[tid] = {'box': {'x1': tx1,'y1': ty1,'x2': tx2,'y2': ty2}, 'name': None, 'confidence': 0.0}
    return mapping

# ---------- Track data management ----------
def make_empty_track_entry(track_id: int, first_frame_idx: int, first_time_s: float) -> Dict[str, Any]:
    return {
        'track_id': track_id,
        'start_frame': first_frame_idx,
        'end_frame': first_frame_idx,
        'start_time_s': first_time_s,
        'end_time_s': first_time_s,
        'last_seen_frame': first_frame_idx,
        'frames_seen': 0,
        'positions': [],
        'confidences': [],
        'logos': [],
        'max_confidence': 0.0
    }

def update_track_with_detection(track: Dict[str, Any], det: Dict[str, Any], frame_idx: int, time_s: float):
    box = det.get('box', {})
    x1, y1, x2, y2 = box.get('x1',0.0), box.get('y1',0.0), box.get('x2',0.0), box.get('y2',0.0)
    conf = float(det.get('confidence', 0.0) or 0.0)
    name = det.get('name')
    track['end_frame'] = frame_idx
    track['end_time_s'] = time_s
    track['last_seen_frame'] = frame_idx
    track['frames_seen'] += 1
    track['positions'].append((x1, y1, x2, y2))
    track['confidences'].append(conf)
    track['logos'].append(name)
    if conf > track['max_confidence']:
        track['max_confidence'] = conf

def aggregate_track_for_db(track: Dict[str, Any], img_width: int, img_height: int,
                           study_id: int, media_id: str, plateform_id: int) -> Tuple:
    max_idx = int(np.argmax(track['confidences'])) if track['confidences'] else 0
    x1, y1, x2, y2 = track['positions'][max_idx] if track['positions'] else (0,0,0,0)
    x1 = max(0, x1); y1 = max(0,y1)
    x2 = min(img_width, x2); y2 = min(img_height, y2)
    width, height = max(0.0, x2-x1), max(0.0, y2-y1)
    area = width*height
    frame_area = img_width*img_height if img_width>0 and img_height>0 else 1
    area_percentage = (area/frame_area)*100.0
    if area_percentage >= 10.0:
        size_str = "large"
    elif area_percentage >= 1.0:
        size_str = "meduim"
    elif area_percentage >= 0.1:
        size_str = "small"
    else:
        size_str = "tiny"
    confidence = track.get('max_confidence',0.0)
    logos = [l for l in track['logos'] if l is not None]
    logo_value = max(set(logos), key=logos.count) if logos else None
    # convert seconds to time(3) string hh:mm:ss.ms
    def sec_to_timestr(s: float) -> str:
        ms = int((s - int(s))*1000)
        t = time.gmtime(s)
        return f"{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}.{ms:03d}"
    time_begin_str = sec_to_timestr(track['start_time_s'])
    time_end_str = sec_to_timestr(track['end_time_s'])
    return (study_id, media_id, plateform_id, logo_value, size_str, area, area_percentage,
            time_begin_str, time_end_str, confidence, x1, y1, x2, y2)

def insert_track_to_db(values: Tuple, specific_config='default'):
    try:
        mysql_execute_insert(INSERT_SQL_VIDEO, values, specific_config=specific_config)
    except Exception as e:
        print(f"DB insert failed: {e}, values={values}")

# ---------- Main function ----------
def run_yolo_videos_to_db(videos_folder: str, model_weights_path: str, imgsz: int = 640, db_config='default'):
    start_all = time.time()
    if not os.path.exists(videos_folder):
        print(f"Folder not found: {videos_folder}")
        return
    model = YOLO(model_weights_path)
    video_files = [f for f in os.listdir(videos_folder) if f.lower().endswith(('.mp4','.mov','.avi'))]
    if not video_files:
        print("No videos found")
        return
    for vid_idx, video_file in enumerate(video_files, start=1):
        video_start = time.time()
        print(f"\n Processing video {vid_idx}/{len(video_files)}: {video_file}")
        video_path = os.path.join(videos_folder, video_file)
        try:
            study_id, media_id, plateform_id = parse_filename(video_file)
        except Exception as e:
            print(f"Filename parsing failed: {e}")
            continue
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open {video_file}")
            continue
        inserted_rows = 0
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        tracker = Sort(max_age=MAX_INACTIVE_FRAMES, min_hits=2, iou_threshold=IOU_MATCH_THRESHOLD)
        active_tracks: Dict[int, Dict[str, Any]] = {}
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_idx +=1
            frame_time_s = (frame_idx-1)/fps
            results = model(frame, imgsz=imgsz, verbose=False)
            det = results[0]
            try:
                det_list = json.loads(det.to_json())
            except Exception:
                det_list = []
            # filtered detections
            detections_for_sort, filtered_detections = [], []
            for d in det_list:
                conf = float(d.get('confidence',0.0) or 0.0)
                if conf >= CONF_THRESHOLD:
                    b = d['box']
                    x1,y1,x2,y2 = float(b['x1']),float(b['y1']),float(b['x2']),float(b['y2'])
                    detections_for_sort.append([x1,y1,x2,y2,conf])
                    filtered_detections.append({'box':{'x1':x1,'y1':y1,'x2':x2,'y2':y2}, 'name':d.get('name'),'confidence':conf})
            tracks_np = tracker.update(np.array(detections_for_sort)) if detections_for_sort else tracker.update()
            mapping = match_tracks_to_detections(tracks_np, filtered_detections) if len(tracks_np)>0 else {}
            seen_ids = set()
            for tid, det_info in mapping.items():
                seen_ids.add(tid)
                if tid not in active_tracks:
                    active_tracks[tid] = make_empty_track_entry(tid, frame_idx, frame_time_s)
                update_track_with_detection(active_tracks[tid], det_info, frame_idx, frame_time_s)
            # finalize inactive tracks
            to_finalize = [tid for tid,t in active_tracks.items() if tid not in seen_ids and (frame_idx - t['last_seen_frame'])>MAX_INACTIVE_FRAMES]
            for tid in to_finalize:
                t = active_tracks.pop(tid)
                if t['frames_seen']>=MIN_VISIBLE_FRAMES and t['max_confidence']>=MIN_CONF_FOR_VALID_TRACK:
                    values = aggregate_track_for_db(t, width, height, study_id, media_id, plateform_id)
                    insert_track_to_db(values, db_config)
                    inserted_rows += 1
        # finalize remaining tracks
        for tid, t in list(active_tracks.items()):
            if t['frames_seen']>=MIN_VISIBLE_FRAMES and t['max_confidence']>=MIN_CONF_FOR_VALID_TRACK:
                values = aggregate_track_for_db(t, width, height, study_id, media_id, plateform_id)
                insert_track_to_db(values, db_config)
                inserted_rows += 1
            active_tracks.pop(tid)
        cap.release()
        duration = time.time() - video_start
        print(f" Finished {video_file}: inserted {inserted_rows} track rows."
              f" Time taken: {duration:.2f} seconds. \n")
    total_duration = time.time() - start_all
    print(f"All videos processed in {total_duration:.2f} seconds.")

if __name__=="__main__":
    video_path = "test_insert_db/videos"
    model_path = "models/team_chambe_3L_fine_tune_v2/weights/best.pt"
    run_yolo_videos_to_db(videos_folder=video_path, model_weights_path=model_path, imgsz=640, db_config='default')
