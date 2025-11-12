from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("runs/detect/team_chambe_3L_raw/weights/best.pt")

# Define path to video file
source = "53_17852331975533658_4.mp4"

# Run inference on the source
results = model(source, save=True)  # generator of Results objects