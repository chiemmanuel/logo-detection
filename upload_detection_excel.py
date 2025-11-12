import json
import openpyxl
import pandas as pd
from ultralytics import YOLO
import time


test_model1 = YOLO("runs/detect/liquimoly_model_V2/weights/best.pt")
test_model2 = YOLO("runs/detect/liquimoly_model_V3/weights/best.pt")

# ---- Model 1 results ----
results1 = []
start_frame = 300
num_images = 288
start_time = time.perf_counter()
for i in range(num_images):
    frame_num = start_frame + i
    filename = f"./team_chambe_analysis/J1_ChambÃ©ry_v_Nantes_Le_rÃ©sumÃ©_start3s_step10_frame_{frame_num:05d}.jpg"
    results = test_model1(filename)  # First model
    result_json = results[0].to_json()
    detections = json.loads(result_json)
    confidences = [d["confidence"] for d in detections]
    results1.append({
        "image": f"frame_{frame_num:05d}",
        "num_detections": len(confidences),
        "confidences": ", ".join([f"{c:.3f}" for c in confidences]),
        "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0
    })

# ---- Model 2 results ----
results2 = []
for i in range(num_images):
    frame_num = start_frame + i
    filename = f"./team_chambe_analysis/J1_ChambÃ©ry_v_Nantes_Le_rÃ©sumÃ©_start3s_step10_frame_{frame_num:05d}.jpg"
    results = test_model2(filename)  # Second model
    result_json = results[0].to_json()
    detections = json.loads(result_json)
    confidences = [d["confidence"] for d in detections]
    results2.append({
        "image": f"frame_{frame_num:05d}",
        "num_detections": len(confidences),
        "confidences": ", ".join([f"{c:.3f}" for c in confidences]),
        "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0
    })

# ---- Save as Excel with two named tables ----
df1 = pd.DataFrame(results1)
df2 = pd.DataFrame(results2)

with pd.ExcelWriter("model_results_separated_V2_vs_V3.xlsx") as writer:
    df1.to_excel(writer, index=False, sheet_name="Model1_Results")
    df2.to_excel(writer, index=False, sheet_name="Model2_Results")

end_time = time.perf_counter() - start_time
print(f"Processing time for {num_images} images: {end_time:.2f} seconds")
print("Saved: model_results_separated.xlsx")
