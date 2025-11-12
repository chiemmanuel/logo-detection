from ultralytics import YOLO
import datetime

model = YOLO("yolo11n.pt")

#test_model = YOLO("runs/segment/liquimoly_seg_model/weights/best.pt")
start_time = datetime.datetime.now()

results = model.train(data="./data.yaml", 
                      epochs=100, 
                      imgsz=640, 
                      batch=8,
                      name="team_chambe_3L_fine_tune_v2",)

metrics = model.val()

end_time = datetime.datetime.now()
print(f"Training started at: {start_time}, ended at: {end_time}, duration: {end_time - start_time}")

# Starting frame number
"""start_frame = 439
num_images = 1

for i in range(num_images):
    frame_num = start_frame + i
    filename = f"./team_chambe_analysis/J1_ChambÃ©ry_v_Nantes_Le_rÃ©sumÃ©_start3s_step10_frame_{frame_num:05d}.jpg"

    results = test_model(filename)

    # Show results
    results[0].show()

    # Print results
    result_json = results[0].to_json()

    print(result_json)"""

