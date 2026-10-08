"""
Modify a YOLO26m model by replacing one intermediate
3x3 convolution with the proposed Multi-Scale Convolution (MSC) module.

The MSC module contains:
    - 3x3 convolution branch
    - 5x5 convolution branch

The two outputs are concatenated to preserve the
original 128-channel feature representation.
"""

from ultralytics import YOLO

from models.msc_module import MSC


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

BASE_MODEL = "path/to/yolo26m/best.pt"
OUTPUT_MODEL = "path/to/output/yolo26_msc_init.pt"


# ------------------------------------------------------------------
# Model modification
# ------------------------------------------------------------------

def modify_model():
    """Load YOLO26m and replace the target convolution with MSC."""

    model = YOLO(BASE_MODEL)

    # Target layer used in the proposed architecture.
    layer6 = model.model.model[6]

    # Target intermediate block.
    target_block = layer6.m[0].m[0]

    print("\n===== ORIGINAL TARGET LAYER =====")
    print(target_block.cv1)

    # Replace the original 3x3 convolution.
    target_block.cv1 = MSC(channels=128)

    print("\n===== MODIFIED TARGET LAYER =====")
    print(target_block.cv1)

    # Display model information.
    print("\n===== MODIFIED MODEL INFORMATION =====")
    model.info()

    # Save the modified architecture.
    model.save(OUTPUT_MODEL)

    print("\nModified YOLO26m model saved to:")
    print(OUTPUT_MODEL)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    modify_model()
