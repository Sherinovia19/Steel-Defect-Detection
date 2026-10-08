"""
Generate a Grad-CAM++-style visualization for a YOLO26m + MSC model.

This script:
    1. Loads the trained YOLO26m + MSC checkpoint.
    2. Loads a steel surface image.
    3. Runs object detection.
    4. Selects the strongest confident detection.
    5. Computes a Grad-CAM++-style heatmap.
    6. Overlays the heatmap on the original image.

The visualization does not draw detection bounding boxes.
"""

import argparse

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

from xai.gradcam_pp import GradCAMPlusPlus


# ------------------------------------------------------------------
# Class configuration
# ------------------------------------------------------------------

CLASS_NAMES = {
    0: "Pitted Surface",
    1: "Crazing",
    2: "Scratches",
    3: "Patches",
}


# ------------------------------------------------------------------
# Image preprocessing
# ------------------------------------------------------------------

def preprocess_image(image_path, image_size=800, device="cpu"):
    """
    Load and preprocess an image using Ultralytics-compatible
    letterbox preprocessing.

    Returns:
        original_image: Original BGR image.
        input_tensor: Preprocessed tensor.
    """

    original_image = cv2.imread(image_path)

    if original_image is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    image_rgb = cv2.cvtColor(
        original_image,
        cv2.COLOR_BGR2RGB,
    )

    letterbox = LetterBox(
        new_shape=(image_size, image_size),
        auto=False,
        stride=32,
        scaleup=True,
        center=True,
    )

    processed = letterbox(image=image_rgb)

    input_tensor = (
        torch.from_numpy(processed)
        .permute(2, 0, 1)
        .float()
        / 255.0
    )

    input_tensor = input_tensor.unsqueeze(0).to(device)

    input_tensor.requires_grad_(True)

    return original_image, input_tensor


# ------------------------------------------------------------------
# Detection
# ------------------------------------------------------------------

def get_strongest_detection(
    model,
    image_path,
    image_size=800,
    confidence_threshold=0.25,
):
    """
    Run YOLO inference and select the strongest confident detection.

    Returns:
        class_id
        confidence
    """

    results = model.predict(
        source=image_path,
        imgsz=image_size,
        conf=confidence_threshold,
        verbose=False,
    )

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return None, None

    confidences = result.boxes.conf.detach().cpu()

    best_index = torch.argmax(confidences).item()

    confidence = confidences[best_index].item()

    class_id = int(
        result.boxes.cls[best_index].item()
    )

    return class_id, confidence


# ------------------------------------------------------------------
# Grad-CAM++-style visualization
# ------------------------------------------------------------------

def generate_gradcam(
    model,
    image_path,
    class_id,
    confidence,
    image_size=800,
):
    """
    Generate a Grad-CAM++-style heatmap for the selected detection.
    """

    device = next(model.model.parameters()).device

    _, input_tensor = preprocess_image(
        image_path,
        image_size=image_size,
        device=device,
    )

    # Target layer: high-level feature extraction block
    # immediately before the detection head.
    target_layer = model.model.model[22]

    cam_generator = GradCAMPlusPlus(
        model.model,
        target_layer,
    )

    # The Grad-CAM utility performs the forward/backward
    # feature extraction required for the heatmap.
    cam = cam_generator.generate(
        input_tensor
    )

    return cam


# ------------------------------------------------------------------
# Heatmap overlay
# ------------------------------------------------------------------

def create_heatmap_overlay(
    image_path,
    cam,
    output_path,
):
    """
    Convert the CAM into a heatmap and overlay it
    on the original image.

    No detection bounding boxes are drawn.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    height, width = image.shape[:2]

    # Resize CAM to original image size.
    cam = cv2.resize(
        cam,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    # Robust normalization.
    low = np.percentile(cam, 60)
    high = np.percentile(cam, 99)

    if high > low:
        cam = np.clip(
            (cam - low) / (high - low),
            0,
            1,
        )
    else:
        cam = np.zeros_like(cam)

    # Smooth the activation map.
    cam = cv2.GaussianBlur(
        cam,
        (0, 0),
        sigmaX=3,
    )

    cam = np.clip(cam, 0, 1)

    # Convert to 8-bit heatmap.
    heatmap = np.uint8(
        cam * 255
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET,
    )

    # Overlay heatmap on the original image.
    overlay = cv2.addWeighted(
        image,
        0.45,
        heatmap,
        0.55,
        0,
    )

    cv2.imwrite(
        output_path,
        overlay,
    )

    return overlay


# ------------------------------------------------------------------
# Main visualization workflow
# ------------------------------------------------------------------

def visualize(
    model_path,
    image_path,
    output_path,
    image_size=800,
    confidence_threshold=0.25,
):
    """
    Complete Grad-CAM++-style visualization workflow.
    """

    print("\n========================================")
    print("       GRAD-CAM++-STYLE XAI")
    print("========================================")

    print(f"Model : {model_path}")
    print(f"Image : {image_path}")

    # Load trained model.
    model = YOLO(model_path)

    # Detect strongest confident defect.
    class_id, confidence = get_strongest_detection(
        model,
        image_path,
        image_size=image_size,
        confidence_threshold=confidence_threshold,
    )

    # Handle images without confident detections.
    if class_id is None:
        print("\nNo confident defect detected.")
        print("Grad-CAM visualization was not generated.")
        return

    class_name = CLASS_NAMES.get(
        class_id,
        f"Class {class_id}",
    )

    print("\nSelected detection:")
    print(f"Class      : {class_name}")
    print(f"Class ID   : {class_id}")
    print(f"Confidence : {confidence:.4f}")

    # Generate CAM.
    cam = generate_gradcam(
        model,
        image_path,
        class_id,
        confidence,
        image_size=image_size,
    )

    # Generate overlay.
    overlay = create_heatmap_overlay(
        image_path,
        cam,
        output_path,
    )

    # Display result.
    overlay_rgb = cv2.cvtColor(
        overlay,
        cv2.COLOR_BGR2RGB,
    )

    plt.figure(figsize=(12, 5))
    plt.imshow(overlay_rgb)
    plt.axis("off")
    plt.title(
        f"Grad-CAM++-style Visualization — {class_name}"
    )
    plt.tight_layout()
    plt.show()

    print("\nHeatmap saved to:")
    print(output_path)


# ------------------------------------------------------------------
# Command-line interface
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Grad-CAM++-style visualization "
            "for YOLO26m + MSC."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to the trained YOLO26m + MSC best.pt",
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input steel image",
    )

    parser.add_argument(
        "--output",
        default="gradcam_output.jpg",
        help="Path for the generated heatmap",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=800,
        help="YOLO input image size",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )

    args = parser.parse_args()

    visualize(
        model_path=args.model,
        image_path=args.image,
        output_path=args.output,
        image_size=args.imgsz,
        confidence_threshold=args.conf,
    )


if __name__ == "__main__":
    main()
