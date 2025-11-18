import os
import json
import time
from ultralytics import YOLO
from dc_utils import mysql_execute_insert

# --- SQL statement ---
INSERT_SQL = """
INSERT INTO yolo_image_detection 
(study_id, media_id, plateform_id, logo, size, area, areaPercentage, confidence, x1, y1, x2, y2)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
"""

def parse_filename(file_name: str):
    """
    Extract study_id, media_id, and plateform_id from filename.
    Example: 53_33731469523_1219967756810240_6.jpg
    Returns: (53, '33731469523_1219967756810240', 6)
    """
    base_name = os.path.splitext(file_name)[0]
    parts = base_name.split("_")
    study_id = int(parts[0])
    plateform_id = int(parts[-1])
    media_id = "_".join(parts[1:-1])
    return study_id, media_id, plateform_id

def insert_yolo_results(file_name: str, results, specific_config='default'):
    """
    Insert YOLO detection results into the MySQL table.
    One row per detected object.
    """
    study_id, media_id, plateform_id = parse_filename(file_name)
    
    img_height, img_width = results[0].orig_shape[:2]
    detections = json.loads(results[0].to_json())  # list of dicts

    for det in detections:
        name = det['name']
        confidence = det['confidence']
        box = det['box']
        x1 = box['x1']
        y1 = box['y1']
        x2 = box['x2']
        y2 = box['y2']

        width = x2 - x1
        height = y2 - y1
        area = width * height
        area_percentage = area / (img_width * img_height) * 100
        if area_percentage >= 10.0:
            size_str = "small"
        elif area_percentage >= 1.0:
            size_str = "meduim"
        elif area_percentage >= 0.1:
            size_str = "large"

        values = (
            study_id,
            media_id,
            str(plateform_id),
            name,
            size_str,
            area,
            area_percentage,
            confidence,
            x1,
            y1,
            x2,
            y2
        )

        mysql_execute_insert(INSERT_SQL, values, specific_config=specific_config)

def run_yolo_detections_on_folder(images_folder: str, model_weights_path: str):
    """
    Runs YOLO detections on all images in a folder and inserts results into MySQL.
    """
    if not os.path.exists(images_folder):
        print(f"Folder not found: {images_folder}")
        return

    detection_img_dir = os.path.join(os.path.dirname(images_folder), "detections_images")
    detection_json_dir = os.path.join(os.path.dirname(images_folder), "detection_jsons")
    os.makedirs(detection_img_dir, exist_ok=True)
    os.makedirs(detection_json_dir, exist_ok=True)

    print(f"Loading YOLO model from: {model_weights_path}")
    model = YOLO(model_weights_path)

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
            results = model(image_path)

            # Save annotated image
            save_path = os.path.join(detection_img_dir, image_file)
            results[0].save(filename=save_path)
            print(f"Saved annotated image: {save_path}")

            # Save detection JSON
            json_data = results[0].to_json()
            json_filename = os.path.splitext(image_file)[0] + ".json"
            json_path = os.path.join(detection_json_dir, json_filename)
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(json.loads(json_data), jf, indent=2, ensure_ascii=False)
            print(f"Saved detection JSON: {json_path}")

            # Insert detection results into DB
            insert_yolo_results(image_file, results)
            print(f"Inserted detection results into DB for: {image_file}")

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


image_path = "test_insert_db/images"
model_path = "models/team_chambe_3L_fine_tune_v2/weights/best.pt" 

run_yolo_detections_on_folder(images_folder=image_path, model_weights_path=model_path)
