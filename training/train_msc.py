"""
Train the proposed YOLO26m + MSC model.

A custom DetectionTrainer is used to ensure that the saved
YOLO26m architecture retains the MSC module during training.
"""

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer


MSC_MODEL = "path/to/yolo26_msc_init.pt"
DATA_YAML = "path/to/data.yaml"


class MSCTrainer(DetectionTrainer):
    """Custom trainer that preserves the MSC architecture."""

    def get_model(self, cfg=None, weights=None, verbose=True):

        model = YOLO(MSC_MODEL).model

        # Set dataset-specific information.
        model.nc = self.data["nc"]
        model.names = self.data["names"]

        # Train all parameters.
        for parameter in model.parameters():
            parameter.requires_grad = True

        # Verify that MSC is still present.
        print("\n===== MSC MODEL CHECK =====")
        print(model.model[6].m[0].m[0].cv1)
        print("===== END MSC CHECK =====\n")

        return model


def main():

    trainer = MSCTrainer(
        overrides={
            "model": MSC_MODEL,
            "data": DATA_YAML,

            "epochs": 25,
            "imgsz": 800,
            "batch": 8,

            "device": 0,
            "workers": 6,

            "patience": 10,

            "save": True,
            "save_period": 1,

            "project": "runs",
            "name": "yolo26m_MSC_proposed",

            "pretrained": False,

            "optimizer": "MuSGD",
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,

            "warmup_epochs": 3.0,

            "amp": True,

            "seed": 0,
            "deterministic": True,
        }
    )

    trainer.train()


if __name__ == "__main__":
    main()
