from ultralytics import YOLO

data_yaml = "/content/drive/MyDrive/Steel detection & segmentation/severstal-steel-defect-detection/yolo_dataset/data.yaml"

model = YOLO("yolo11m.pt")

results = model.train(
    data=data_yaml,
    epochs=25,
    imgsz=800,
    batch=8,
    save=True,
    save_period=1,
    patience=10
)
