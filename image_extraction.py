import cv2
import os
import random
import time


# ---------- Basic Helpers ----------
def extract_frames_standard(cap, output_folder, selected_frames, base_name, start_sec=None, step=None):
    """
    Save frames in a descriptive way:
    videoName_startXXs_stepYY_frame_XXXXX.jpg
    """
    os.makedirs(output_folder, exist_ok=True)
    saved_frame_count = 0
    frame_count = 0

    # Labels for naming
    part_label = f"start{int(start_sec)}s" if start_sec is not None else "full"
    step_label = f"step{step}" if step else "step1"

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count in selected_frames:
            filename = f"{base_name}_-{saved_frame_count:05d}.jpg"
            filepath = os.path.join(output_folder, filename)
            cv2.imwrite(filepath, frame)
            saved_frame_count += 1
        frame_count += 1

    return saved_frame_count


def get_video_info(video_path):
    """Return FPS, total frames, duration in seconds."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps else 0
    cap.release()
    return fps, total_frames, duration_sec


# ---------- Helper to Apply Range ----------
def filter_range_by_time(total_frames, fps, start_sec, end_sec):
    start_frame = max(0, int(start_sec * fps))
    end_frame = min(total_frames - 1, int(end_sec * fps))
    return range(start_frame, end_frame + 1)


# ---------- Mode Frame Selection ----------
def mode_percentage(frames_in_range, percent):
    desired_count = max(1, int((percent / 100) * len(frames_in_range)))
    step = max(1, len(frames_in_range) // desired_count)
    return list(frames_in_range[0:len(frames_in_range):step])


def mode_step(frames_in_range, step):
    return list(frames_in_range[0:len(frames_in_range):step])


def mode_time(frames_in_range, fps, time_interval):
    step = max(1, int(fps * time_interval))
    return list(frames_in_range[0:len(frames_in_range):step])


# ---------- Strategies ----------
def apply_strategy(frames_list, strategy, n_target=None):
    """Reduce list to n_target frames with either equal spacing or random pick."""
    if strategy == "equal":
        if n_target and n_target < len(frames_list):
            step = len(frames_list) / n_target
            return [frames_list[int(round(i * step))] for i in range(n_target)]
        else:
            return frames_list
    elif strategy == "random":
        if n_target and n_target < len(frames_list):
            return sorted(random.sample(frames_list, n_target))
        else:
            return frames_list
    return frames_list


# ---------- Main Extraction Controller ----------
def video_to_frames(video_path, output_folder, mode, value=None,
                    fps=None, total_frames=None, start_sec=None, end_sec=None,
                    strategy="equal"):
    base_name = os.path.splitext(os.path.basename(video_path))[0]  # video name without extension

    # Determine working frame list
    if start_sec is not None and end_sec is not None:
        frames_in_range = list(filter_range_by_time(total_frames, fps, start_sec, end_sec))
    else:
        frames_in_range = list(range(total_frames))

    # Mode selection
    if mode == "all":
        selected = frames_in_range
    elif mode == "percentage":
        selected = mode_percentage(frames_in_range, value)
    elif mode == "step":
        selected = mode_step(frames_in_range, value)
    elif mode == "time":
        selected = mode_time(frames_in_range, fps, value)
    else:
        selected = frames_in_range

    # Calculate effective frame step for naming
    if mode == "all":
        step_frames = 1
    elif mode == "percentage":
        desired_count = max(1, int((value / 100) * len(frames_in_range)))
        step_frames = max(1, len(frames_in_range) // desired_count)
    elif mode == "step":
        step_frames = value
    elif mode == "time":
        step_frames = max(1, int(fps * value))
    else:
        step_frames = 1

    # Apply strategy where applicable
    if mode in ["percentage", "time", "step"] and strategy in ["equal", "random"]:
        if mode == "percentage":
            target_n = max(1, int((value / 100) * len(frames_in_range)))
        else:
            target_n = len(selected)
        selected = apply_strategy(selected, strategy, target_n)

    # Perform extraction
    cap = cv2.VideoCapture(video_path)
    count = extract_frames_standard(cap, output_folder, set(selected), base_name, start_sec, step=step_frames)
    cap.release()

    print(f"Extracted {count} frames to folder: {output_folder}")


# ---------- Script Entry ----------
if __name__ == "__main__":
    # Fixed paths
    video_path = './datasets/dataset_53/videos_to_be_cut/53_18199678675311224_4.mp4'
    output_folder = './datasets/dataset_53'

    fps, total_frames, duration = get_video_info(video_path)
    print(f"\nVideo FPS: {fps:.2f}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Total frames: {total_frames}\n")

    # Select mode
    print("Modes:")
    print("1 - all (every frame)")
    print("2 - percentage")
    print("3 - step")
    print("4 - time")
    choice = input("Enter choice: ").strip()
    mode_map = {"1": "all", "2": "percentage", "3": "step", "4": "time"}
    mode = mode_map.get(choice, "all")

    value = None
    if mode == "percentage":
        value = float(input("Enter percentage to extract: ").strip())
    elif mode == "step":
        value = int(input("Enter step size in frames: ").strip())
    elif mode == "time":
        value = float(input("Enter seconds between each frame: ").strip())

    # Optional range
    use_range = input("Apply a start/end time range? (y/n): ").strip().lower()
    start_sec = end_sec = None
    if use_range == "y":
        start_sec = float(input("Enter start time (sec): ").strip())
        end_sec = float(input("Enter end time (sec): ").strip())

    # Optional strategy
    strategy = "equal"
    if mode in ["percentage", "step", "time"]:
        strategy = input("Strategy (equal/random): ").strip().lower()

    # Calculate elapsed time
    start_time = time.perf_counter()

    # Run extraction
    video_to_frames(video_path, output_folder, mode, value,
                    fps=fps, total_frames=total_frames,
                    start_sec=start_sec, end_sec=end_sec,
                    strategy=strategy)

    elasped = time.perf_counter() - start_time
    print(f"\nElapsed time: {elasped:.2f} seconds")


    """
    3, 6, 11, 18, 26-28
    """