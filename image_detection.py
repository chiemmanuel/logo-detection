import os
import json
import time
from ultralytics import YOLO


def run_yolo_detections_on_folder(images_folder: str, model_weights_path: str):
    """
    Runs YOLO detections on all images in a folder.
    Saves annotated images and detection JSONs in separate subfolders.

    Args:
        images_folder (str): Path to folder containing images.
        model_weights_path (str): Path to YOLO model weights (e.g. 'runs/segment/.../best.pt').
    """

    # --- Validate and prepare paths ---
    if not os.path.exists(images_folder):
        print(f"Folder not found: {images_folder}")
        return

    detection_img_dir = os.path.join(os.path.dirname(images_folder), "detections_images")
    detection_json_dir = os.path.join(os.path.dirname(images_folder), "detection_jsons")
    os.makedirs(detection_img_dir, exist_ok=True)
    os.makedirs(detection_json_dir, exist_ok=True)

    # --- Load YOLO model ---
    print(f"Loading YOLO model from: {model_weights_path}")
    model = YOLO(model_weights_path)

    # --- Process all images ---
    image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total = len(image_files)

    if not total:
        print(f"No images found in {images_folder}")
        return

    print(f"Running detections on {total} images...\n")
    total_start = time.time()
    times = []

    for idx, image_file in enumerate(image_files, start=1):
        image_start = time.time()
        image_path = os.path.join(images_folder, image_file)
        print(f"[{idx}/{total}] Processing: {image_file}")

        try:
            # --- Run detection ---
            results = model(image_path)

            # --- Save annotated image ---
            save_path = os.path.join(detection_img_dir, image_file)
            results[0].save(filename=save_path)
            print(f"Saved annotated image: {save_path}")

            # --- Save detection JSON ---
            json_data = results[0].to_json()
            json_filename = os.path.splitext(image_file)[0] + ".json"
            json_path = os.path.join(detection_json_dir, json_filename)
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(json.loads(json_data), jf, indent=2, ensure_ascii=False)
            print(f"Saved detection JSON: {json_path}")

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

        image_elapsed = time.time() - image_start
        times.append(image_elapsed)
        print(f"Time for {image_file}: {image_elapsed:.2f}s\n")

    total_elapsed = time.time() - total_start
    avg_time = sum(times) / len(times) if times else 0

    print("\nDetection complete.")
    print(f"Annotated images: {detection_img_dir}")
    print(f"JSON results: {detection_json_dir}")
    print(f"Total time: {total_elapsed:.2f}s")
    print(f"Average per image: {avg_time:.2f}s\n")

image_path = "media_detection_53_9/images"
model_path = "models/team_chambe_3L_fine_tune_v2/weights/best.pt" 

run_yolo_detections_on_folder(images_folder=image_path, model_weights_path=model_path)