from ultralytics import YOLO
def main():

    model = YOLO("yolo26m.pt")

    model.train(
        data=r"D:\Under_water_dehazing\GAN\data.yaml",
        epochs=25,
        imgsz=800,
        batch=8,
        device=0,
        workers=6,
        patience=10,
        save=True,
        save_period=1,
        project=r"D:\Under_water_dehazing\GAN\yolo26\runs",
        name="yolo26m_baseline",
        optimizer="MuSGD",
        amp=True,
        seed=0,
        deterministic=True
    )


if __name__ == "__main__":
    main()
