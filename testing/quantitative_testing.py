"""
Quantitative evaluation of a trained YOLO model.

The script evaluates a trained checkpoint on the validation dataset
and reports:

    - Precision
    - Recall
    - mAP@50
    - mAP@50-95

It can be used for the YOLO26m baseline or the proposed
YOLO26m + MSC model.
"""

from ultralytics import YOLO


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

MODEL_PATH = "path/to/best.pt"
DATA_YAML = "path/to/data.yaml"


# ------------------------------------------------------------------
# Evaluation configuration
# ------------------------------------------------------------------

IMAGE_SIZE = 800
BATCH_SIZE = 8
DEVICE = 0


# ------------------------------------------------------------------
# Quantitative evaluation
# ------------------------------------------------------------------

def evaluate_model():
    """Evaluate the trained YOLO model on the validation dataset."""

    model = YOLO(MODEL_PATH)

    print("\n========================================")
    print("        QUANTITATIVE EVALUATION")
    print("========================================")

    print(f"Model : {MODEL_PATH}")
    print(f"Data  : {DATA_YAML}")
    print(f"Image size : {IMAGE_SIZE}")

    results = model.val(
        data=DATA_YAML,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        plots=True,
    )

    # --------------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------------

    precision = results.box.mp
    recall = results.box.mr
    map50 = results.box.map50
    map50_95 = results.box.map

    print("\n========================================")
    print("             OVERALL RESULTS")
    print("========================================")

    print(f"Precision : {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall    : {recall:.4f} ({recall * 100:.2f}%)")
    print(f"mAP@50    : {map50:.4f} ({map50 * 100:.2f}%)")
    print(f"mAP@50-95 : {map50_95:.4f} ({map50_95 * 100:.2f}%)")

    # --------------------------------------------------------------
    # Class-wise metrics
    # --------------------------------------------------------------

    print("\n========================================")
    print("           CLASS-WISE RESULTS")
    print("========================================")

    class_names = model.names

    for class_id, class_name in class_names.items():

        class_precision = results.box.p[class_id]
        class_recall = results.box.r[class_id]
        class_map50 = results.box.ap50[class_id]
        class_map50_95 = results.box.ap[class_id]

        print(f"\nClass: {class_name}")
        print(
            f"  Precision : {class_precision:.4f} "
            f"({class_precision * 100:.2f}%)"
        )
        print(
            f"  Recall    : {class_recall:.4f} "
            f"({class_recall * 100:.2f}%)"
        )
        print(
            f"  mAP@50    : {class_map50:.4f} "
            f"({class_map50 * 100:.2f}%)"
        )
        print(
            f"  mAP@50-95 : {class_map50_95:.4f} "
            f"({class_map50_95 * 100:.2f}%)"
        )

    print("\n========================================")
    print("       QUANTITATIVE TESTING COMPLETE")
    print("========================================")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    evaluate_model()
