# ============================================================
# QUALITATIVE TESTING
# YOLO Steel Defect Detection
# ============================================================
#
# PURPOSE:
#   This script performs qualitative testing on any trained
#   Ultralytics YOLO .pt model using any uploaded test image.
#
# INPUT:
#   1. Trained YOLO .pt model
#   2. Test image (.jpg / .jpeg / .png)
#
# OUTPUT:
#   - Bounding boxes
#   - Predicted defect class
#   - Confidence score
#   - Total number of detections
#   - Saved prediction image
#
# DEFECT CLASSES:
#   0 - pitted_surface
#   1 - crazing
#   2 - scratches
#   3 - patches
#
# NOTE:
#   This is QUALITATIVE TESTING only.
#   It is intended for visual inspection of model predictions.
#
# ============================================================


# ============================================================
# 1. INSTALL / IMPORT
# ============================================================

# If running in Google Colab, uncomment:
# !pip install -q ultralytics

from ultralytics import YOLO
import cv2
import os
import matplotlib.pyplot as plt


# ============================================================
# 2. ENTER MODEL AND IMAGE PATH
# ============================================================
#
# Replace these two paths with:
#   - any trained .pt model
#   - any test image
#
# Example:
# MODEL_PATH = "yolo26_best.pt"
# IMAGE_PATH = "test_image.jpg"
#
# ============================================================

MODEL_PATH = "your_model.pt"
IMAGE_PATH = "your_test_image.jpg"


# ============================================================
# 3. CLASS NAMES
# ============================================================

CLASS_NAMES = {
    0: "pitted_surface",
    1: "crazing",
    2: "scratches",
    3: "patches"
}


# ============================================================
# 4. CHECK INPUT FILES
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}"
    )

print("Model:", MODEL_PATH)
print("Image:", IMAGE_PATH)


# ============================================================
# 5. LOAD MODEL
# ============================================================

model = YOLO(MODEL_PATH)

print("\nYOLO model loaded successfully.")


# ============================================================
# 6. RUN QUALITATIVE INFERENCE
# ============================================================

results = model.predict(
    source=IMAGE_PATH,
    imgsz=800,
    conf=0.25,
    verbose=True
)

result = results[0]


# ============================================================
# 7. PRINT DETECTION INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("QUALITATIVE PREDICTION")
print("=" * 60)

total_detections = len(result.boxes)

print("Total detections:", total_detections)


if total_detections == 0:

    print("No defects detected.")

else:

    for i, box in enumerate(result.boxes):

        # Bounding-box coordinates
        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
            .astype(int)
        )

        # Class ID
        class_id = int(
            box.cls[0].cpu().numpy()
        )

        # Confidence
        confidence = float(
            box.conf[0].cpu().numpy()
        )

        # Class name
        if class_id in CLASS_NAMES:
            class_name = CLASS_NAMES[class_id]
        else:
            class_name = f"class_{class_id}"

        print(
            f"Detection {i + 1}: "
            f"{class_name} | "
            f"Confidence = {confidence:.3f}"
        )


# ============================================================
# 8. LOAD ORIGINAL IMAGE
# ============================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise ValueError(
        "Unable to read the selected image."
    )

image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# 9. DRAW BOUNDING BOXES
# ============================================================

for box in result.boxes:

    # Coordinates
    x1, y1, x2, y2 = (
        box.xyxy[0]
        .cpu()
        .numpy()
        .astype(int)
    )

    # Class
    class_id = int(
        box.cls[0].cpu().numpy()
    )

    # Confidence
    confidence = float(
        box.conf[0].cpu().numpy()
    )

    # Class name
    if class_id in CLASS_NAMES:
        class_name = CLASS_NAMES[class_id]
    else:
        class_name = f"class_{class_id}"

    # Bounding box
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        3
    )

    # Label
    label = f"{class_name} {confidence:.2f}"

    cv2.putText(
        image,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )


# ============================================================
# 10. DISPLAY RESULT
# ============================================================

plt.figure(figsize=(16, 6))

plt.imshow(image)

plt.title(
    "YOLO Qualitative Testing"
)

plt.axis("off")

plt.show()


# ============================================================
# 11. SAVE RESULT
# ============================================================

os.makedirs(
    "qualitative_results",
    exist_ok=True
)

output_path = os.path.join(
    "qualitative_results",
    "prediction_result.jpg"
)

plt.imsave(
    output_path,
    image
)

print("\nPrediction saved to:")
print(output_path)

print("\nQualitative testing completed successfully.")
