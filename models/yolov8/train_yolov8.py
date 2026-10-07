from ultralytics import YOLO

model = YOLO("yolov8m.pt")

results = model.train(
    data=".../data.yaml",
    epochs=25,
    imgsz=800,
    batch=8,
    save=True,
    save_period=1,
    patience=10,
    project=".../yolov8m_baseline",
    name="train"
)
