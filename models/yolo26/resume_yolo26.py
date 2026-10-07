# ============================================================
# YOLO26m BASELINE - RESUME TRAINING
# Steel Surface Defect Detection
# ============================================================

from ultralytics import YOLO
def main():

    # Load the latest checkpoint from the interrupted run
    model = YOLO(
        r"D:\Under_water_dehazing\GAN\yolo26\runs\yolo26m_baseline-7\weights\last.pt"
    )

    # Resume training from the saved checkpoint
    model.train(
        resume=True
    )


if __name__ == "__main__":
    main()
