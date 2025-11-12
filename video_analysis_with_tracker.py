import os
import cv2
import time
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from sort import Sort 


def run_yolo_detections_on_videos(videos_folder: str, model_weights_path: str, imgsz: int = 640):
    """
    Runs YOLO detections on all videos in a folder,
    applies SORT tracking (for temporal stability) but keeps only YOLO boxes & labels.
    Filters out detections with confidence < 0.35.
    """

    if not os.path.exists(videos_folder):
        print(f"Folder not found: {videos_folder}")
        return

    global_start_time = time.time()

    detected_videos_dir = os.path.join(os.path.dirname(videos_folder), "detected_videos_with_tracking_fine_tuneV2")
    os.makedirs(detected_videos_dir, exist_ok=True)

    print(f"Loading YOLO model from: {model_weights_path}")
    model = YOLO(model_weights_path)

    video_files = [f for f in os.listdir(videos_folder) if f.lower().endswith(('.mp4', '.mov', '.avi'))]
    total_videos = len(video_files)
    if not total_videos:
        print(f"No videos found in {videos_folder}")
        return

    print(f"Running detections on {total_videos} videos...")

    for idx, video_file in enumerate(video_files, start=1):
        video_path = os.path.join(videos_folder, video_file)
        print(f"[{idx}/{total_videos}] Processing video: {video_file}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_file}")
            continue

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        output_video_path = os.path.join(detected_videos_dir, video_file)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        start_time = time.time()
        frame_count = 0

        tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.3)
        CONF_THRESHOLD = 0.35

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO detection
            results = model(frame, imgsz=imgsz, verbose=False)
            det = results[0]

            detections_for_sort = []
            filtered_boxes = []

            if det.boxes is not None and len(det.boxes) > 0:
                for box in det.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf)
                    if conf >= CONF_THRESHOLD:
                        detections_for_sort.append([x1, y1, x2, y2, conf])
                        filtered_boxes.append(box)

            # Update tracker (for smoother temporal behavior)
            if len(detections_for_sort) > 0:
                tracker.update(np.array(detections_for_sort))
            else:
                tracker.update()

            # Replace YOLO’s boxes with the filtered subset
            if filtered_boxes:
                stacked = torch.stack([b.data[0] for b in filtered_boxes])
                det.boxes = Boxes(stacked, det.orig_img.shape[:2]) 
            else:
                det.boxes = Boxes(torch.empty((0, 6)), det.orig_img.shape[:2])

            # Plot YOLO boxes & class names only
            annotated_frame = det.plot()

            out.write(annotated_frame)
            frame_count += 1

        total_time = time.time() - start_time
        print(f"\nFinished {video_file}: {frame_count} frames in {total_time:.2f}s "
              f"(avg {total_time/frame_count:.3f}s/frame)")

        cap.release()
        out.release()

    global_time_taken = (time.time() - global_start_time) / 60
    print(f"\n{total_videos} videos processed in {global_time_taken:.2f} min.")
    print(f"Annotated videos saved in: {detected_videos_dir}")


# === CONFIG ===
video_path = "media_for_detection_53/videos"
model_path = "models/team_chambe_3L_fine_tune_v2/weights/best.pt" 

run_yolo_detections_on_videos(videos_folder=video_path, model_weights_path=model_path)
