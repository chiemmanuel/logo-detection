from ultralytics import YOLO
import datetime

model = YOLO("yolo11n.pt")

start_time = datetime.datetime.now()

results = model.train(data="./data.yaml", 
                      epochs=100, 
                      imgsz=640, 
                      batch=8,
                      name="etude_60",)

metrics = model.val()

end_time = datetime.datetime.now()
print(f"Training started at: {start_time}, ended at: {end_time}, duration: {end_time - start_time}")


