
"""
Train the proposed YOLO26m + Multi-Scale Convolution (MSC) model.

The custom trainer explicitly loads the already-modified YOLO26m
architecture so that the MSC module is preserved during training.
"""

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

MSC_MODEL = "path/to/yolo26_msc_init.pt"
DATA_YAML = "path/to/data.yaml"


# ------------------------------------------------------------------
# Custom trainer
# ------------------------------------------------------------------

class MSCTrainer(DetectionTrainer):
    """
    Custom Ultralytics trainer for the YOLO26m + MSC model.

    The modified model is loaded directly instead of allowing
    Ultralytics to reconstruct the original YOLO26 architecture.
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return the YOLO26m model containing the MSC module."""

        model = YOLO(MSC_MODEL).model

        # Set dataset information.
        model.nc = self.data["nc"]
        model.names = self.data["names"]

        # Enable training for all parameters.
        for parameter in model.parameters():
            parameter.requires_grad = True

        # Verify that MSC is still present.
        print("\n===== MSC MODEL CHECK =====")
        print(model.model[6].m[0].m[0].cv1)
        print("===== END MSC CHECK =====\n")

        return model


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train_proposed_model():
    """Train the proposed YOLO26m + MSC model."""

    trainer = MSCTrainer(
        overrides={
            # Model and dataset
            "model": MSC_MODEL,
            "data": DATA_YAML,

            # Training configuration
            "epochs": 25,
            "imgsz": 800,
            "batch": 8,
            "device": 0,
            "workers": 6,

            # Early stopping and checkpoints
            "patience": 10,
            "save": True,
            "save_period": 1,

            # Output directory
            "project": "runs",
            "name": "yolo26m_MSC_proposed",

            # Do not load standard pretrained YOLO weights.
            # The modified model already contains the initialized MSC.
            "pretrained": False,

            # Optimizer
            "optimizer": "MuSGD",
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,

            # Warmup
            "warmup_epochs": 3.0,

            # Mixed precision
            "amp": True,

            # Reproducibility
            "seed": 0,
            "deterministic": True,
        }
    )

    trainer.train()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    train_proposed_model()
