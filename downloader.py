import os
import cv2
import time
import shutil
import requests
from urllib.parse import urlparse


def download_images_for_study(study_id: str, purpose: str = "dataset"):
    """
    Downloads images for a study.
    - If purpose == "dataset": behaves normally, saves images in datasets/dataset_{id}/
    - If purpose == "media_analysis": saves images in media_for_detection_{id}/images/
    """

    # Define target folder based on purpose
    if purpose == "dataset":
        dataset_folder = f"datasets/dataset_{study_id}_10"
    else:
        dataset_folder = os.path.join(f"media_detection_{study_id}_10", "images") # 10 correspond au mois de l'année sur le quel la requete du link construct est lancée

    os.makedirs(dataset_folder, exist_ok=True)

    image_folder = f"images_{study_id}_10" # 10 correspond au mois de l'année sur le quel la requete du link construct est lancée
    aws_file = os.path.join(image_folder, "aws_links.txt")
    twitter_file = os.path.join(image_folder, "twitter_links.txt")

    error_log = []
    total_downloaded = 0

    def download_and_save(url, target_path):
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            with open(target_path, 'wb') as f:
                f.write(resp.content)
            print(f" Saved: {os.path.abspath(target_path)}")
            return True
        except Exception as ex:
            error_log.append(f"FAILED: {url} -> {target_path}, error: {ex}")
            return False

    # --- Download from AWS file ---
    if os.path.exists(aws_file):
        with open(aws_file, "r") as f:
            for line in f:
                url = line.strip()
                if not url:
                    continue
                filename_base = url.split('/')[-1]
                base = filename_base.rsplit('.', 1)[0]
                filename = f"{study_id}_{base}.jpg"
                target_path = os.path.join(dataset_folder, filename)
                if download_and_save(url, target_path):
                    total_downloaded += 1

    # --- Download from Twitter file ---
    if os.path.exists(twitter_file):
        with open(twitter_file, "r") as f:
            for line in f:
                url = line.strip()
                if not url:
                    continue
                parsed = urlparse(url)
                pub_id = parsed.path.rstrip('/').split('/')[-1]
                filename = f"{study_id}_{pub_id}_3.jpg"
                target_path = os.path.join(dataset_folder, filename)
                if download_and_save(url, target_path):
                    total_downloaded += 1

    print(f"\n Downloaded {total_downloaded} images for study {study_id} over month 10")

    if error_log:
        error_logfile = os.path.join(dataset_folder, f"errors_{study_id}.log")
        with open(error_logfile, "w") as logf:
            for line in error_log:
                print(line)
                logf.write(line + "\n")
        print(f" Some downloads failed. Check {error_logfile} for details.")


def download_and_process_videos(study_id: str, purpose: str = "dataset", frame_threshold: int = 150):
    """
    Downloads videos for a study.
    - If purpose == "dataset": keeps full logic (frame extraction + moving long videos)
    - If purpose == "media_analysis": downloads only videos (no frame extraction)
      into media_for_detection_{id}/videos/
    """

    # Define folders based on purpose
    if purpose == "dataset":
        dataset_folder = f"datasets/dataset_{study_id}"
        cut_folder = os.path.join(dataset_folder, "videos_to_be_cut")
    else:
        dataset_folder = os.path.join(f"media_detection_{study_id}", "videos")
        cut_folder = None  # not used

    os.makedirs(dataset_folder, exist_ok=True)
    if cut_folder:
        os.makedirs(cut_folder, exist_ok=True)

    video_folder = f"videos_{study_id}_10" # 10 correspond au mois de l'année sur lequel la requete de link construct est lancée
    video_files = {
        "aws": os.path.join(video_folder, "aws_links.txt"),
        "twitter": os.path.join(video_folder, "twitter_links.txt"),
    }

    error_log = []
    total_downloaded = 0
    processed_frames = 0

    def download_and_save(url, target_path):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            with open(target_path, 'wb') as f:
                f.write(resp.content)
            print(f" Saved: {os.path.abspath(target_path)}")
            return True
        except Exception as ex:
            error_log.append(f"FAILED: {url} -> {target_path}, error: {ex}")
            return False

    for source, file_path in video_files.items():
        if not os.path.exists(file_path):
            continue

        with open(file_path, "r") as f:
            for line in f:
                url = line.strip()
                if not url:
                    continue

                if source == "aws":
                    filename_base = url.split('/')[-1]
                    base = filename_base.rsplit('.', 1)[0]
                    out_name = f"{study_id}_{base}.mp4"
                else:
                    parsed = urlparse(url)
                    pub_id = parsed.path.rstrip('/').split('/')[-1]
                    out_name = f"{study_id}_{pub_id}_3.mp4"

                video_path = os.path.join(dataset_folder, out_name)

                if not download_and_save(url, video_path):
                    continue

                total_downloaded += 1

                # --- If purpose == "media_analysis", skip all processing ---
                if purpose != "dataset":
                    continue

                # --- Dataset processing logic ---
                cap = None
                try:
                    cap = cv2.VideoCapture(video_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration_sec = total_frames / fps if fps else 0

                    print(f"\n Video: {video_path}\nFPS: {fps:.2f}, Frames: {total_frames}, Duration: {duration_sec:.1f}s")

                    if total_frames <= frame_threshold and fps > 0:
                        frame_output_folder = dataset_folder
                        base_name = os.path.splitext(out_name)[0]
                        frame_count = 0
                        saved_frames = 0
                        sample_rate = max(1, total_frames // max(1, total_frames // 20))

                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            if frame_count % sample_rate == 0:
                                filename = f"{base_name}-{saved_frames:05d}.jpg"
                                filepath = os.path.join(frame_output_folder, filename)
                                cv2.imwrite(filepath, frame)
                                saved_frames += 1
                            frame_count += 1

                        print(f" Extracted {saved_frames} frames (~5%) from {out_name}")
                        processed_frames += saved_frames
                    else:
                        # Move long videos to cut folder
                        if cut_folder:
                            dest_path = os.path.join(cut_folder, out_name)
                            try:
                                if os.path.exists(dest_path):
                                    os.remove(dest_path)
                                cap.release()
                                time.sleep(0.1)
                                shutil.move(video_path, dest_path)
                                print(f" Moved long video to videos_to_be_cut: {out_name}")
                                cap = None
                            except Exception as move_ex:
                                error_log.append(f"ERROR moving video {video_path}: {move_ex}")

                except Exception as ex:
                    error_log.append(f"ERROR processing video {video_path}: {ex}")
                finally:
                    if cap is not None:
                        cap.release()
                        time.sleep(0.2)

    print(f"\nSummary for study {study_id}: {total_downloaded} videos downloaded.")
    if purpose == "dataset":
        print(f"Frames extracted: {processed_frames}")

    if error_log:
        error_logfile = os.path.join(dataset_folder, f"errors_{study_id}.log")
        with open(error_logfile, "w") as logf:
            for line in error_log:
                print(line)
                logf.write(line + "\n")
        print(f" Some videos failed. See {error_logfile} for details.")


def retry_failed_downloads(study_id: int):
    """
    Retry downloads from the error log by replacing .mp4 with .jpg in URLs.
    Save the images in the same folder as the error log.
    """
    # Locate the error log relative to current working directory
    error_log_path = os.path.join("datasets", f"dataset_{study_id}", f"errors_{study_id}.log")

    if not os.path.exists(error_log_path):
        print(f"No error log found for study {study_id} at {error_log_path}")
        return

    # Folder where the error log is
    log_folder = os.path.dirname(error_log_path)

    with open(error_log_path, "r") as f:
        lines = f.readlines()

    remaining_lines = []
    total_fixed = 0
    total_attempts = 0

    for line in lines:
        if "FAILED:" not in line or "->" not in line:
            remaining_lines.append(line)
            continue

        total_attempts += 1
        try:
            url_part = line.split("FAILED: ")[1].split(" -> ")[0].strip()
            save_path = line.split("->")[1].split(",")[0].strip()
        except Exception:
            remaining_lines.append(line)
            continue

        # Replace .mp4 with .jpg
        new_url = url_part.replace(".mp4", ".jpg")
        # Use same folder as error log
        filename = os.path.basename(save_path).replace(".mp4", ".jpg")
        new_save_path = os.path.join(log_folder, filename)

        try:
            resp = requests.get(new_url, timeout=10)
            if resp.status_code == 200:
                with open(new_save_path, "wb") as f:
                    f.write(resp.content)
                print(f"   Saved to: {os.path.abspath(new_save_path)}")
                total_fixed += 1
            else:
                print(f" Failed ({resp.status_code}): {new_url}")
                remaining_lines.append(line)
        except Exception as e:
            print(f" Failed ({e}): {new_url}")
            remaining_lines.append(line)

    # Rewrite remaining failed entries to log
    with open(error_log_path, "w") as f:
        f.writelines(remaining_lines)

    print(f"\n Retry complete for study {study_id}: {total_fixed}/{total_attempts} fixed successfully.")


download_images_for_study("53", purpose="dataset")
#download_and_process_videos("53", purpose="media_analysis")
#retry_failed_downloads(study_id=53)