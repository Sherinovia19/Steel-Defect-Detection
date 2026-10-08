"""
Create the proposed YOLO26m + MSC model.

This script loads a trained YOLO26m checkpoint, replaces the
target intermediate 3x3 convolution with the custom MSC module,
and saves the modified model.

The MSC module is defined in:
    models/msc_module.py
"""

from ultralytics import YOLO

from models.msc_module import MSC


# Input YOLO26m checkpoint.
# Replace this with the path to the YOLO26m checkpoint when running.
BASE_MODEL = "path/to/yolo26m/best.pt"

# Output path for the modified model.
OUTPUT_MODEL = "path/to/output/yolo26_msc_init.pt"


def modify_model():
    """Replace the target YOLO26m convolution with MSC."""

    model = YOLO(BASE_MODEL)

    # Target layer:
    # Layer 6 → m.0 → m.0 → cv1
    layer6 = model.model.model[6]
    target_block = layer6.m[0].m[0]

    print("Original target layer:")
    print(target_block.cv1)

    # Replace the original 128 → 128, 3×3 convolution
    # with the 128 → 128 MSC module.
    target_block.cv1 = MSC(channels=128)

    print("\nModified target layer:")
    print(target_block.cv1)

    # Display model information for verification.
    model.info()

    # Save the modified architecture.
    model.save(OUTPUT_MODEL)

    print("\nModified YOLO26m saved to:")
    print(OUTPUT_MODEL)


if __name__ == "__main__":
    modify_model()
