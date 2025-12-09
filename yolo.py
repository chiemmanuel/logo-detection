import os
from ultralytics import YOLO
import datetime

# Force execution from this script's folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("CWD:", os.getcwd())

model = YOLO("yolo11n.pt")

start_time = datetime.datetime.now()

results = model.train(
    data="./data.yaml", 
    epochs=100, 
    imgsz=640, 
    batch=8,
    name="etude_60_test",
    project="./runs"   # ensures runs/ is created next to this file
)

metrics = model.val()

end_time = datetime.datetime.now()
print(f"Training started at: {start_time}, ended at: {end_time}, duration: {end_time - start_time}")
