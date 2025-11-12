import os
import cv2
import time
from ultralytics import YOLO


def run_yolo_detections_on_videos(videos_folder: str, model_weights_path: str, imgsz: int = 640):
    """
    Runs YOLO detections on all videos in a folder.
    Saves annotated videos in a dedicated subfolder and prints progress/timing.

    Args:
        videos_folder (str): Path to folder containing videos.
        model_weights_path (str): Path to YOLO model weights.
        imgsz (int, optional): Image size for YOLO inference (default: 640).
    """

    if not os.path.exists(videos_folder):
        print(f"Folder not found: {videos_folder}")
        return

    global_start_time = time.time()

    # Create output folder for annotated videos
    detected_videos_dir = os.path.join(os.path.dirname(videos_folder), "detected_videos_fine_tune_v2")
    os.makedirs(detected_videos_dir, exist_ok=True)

    # Load YOLO model
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

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO detection (with configurable image size)
            results = model(frame, imgsz=imgsz)

            # Use YOLO's built-in visualization (shows colors + class names)
            annotated_frame = results[0].plot()

            # Write the annotated frame to output video
            out.write(annotated_frame)
            frame_count += 1

        total_time = time.time() - start_time
        print(f"\nFinished video {video_file}: {frame_count} frames in {total_time:.2f}s "
              f"(avg {total_time/frame_count:.3f}s/frame)")

        cap.release()
        out.release()

    global_time_taken = (time.time() - global_start_time) / 60
    print(f"\n{total_videos} videos processed in {global_time_taken:.2f} min.")
    print(f"Annotated videos saved in: {detected_videos_dir}")


video_path = "media_for_detection_53/videos"
model_path = "models/team_chambe_3L_fine_tune_v2/weights/best.pt" 

run_yolo_detections_on_videos(videos_folder=video_path, model_weights_path=model_path)