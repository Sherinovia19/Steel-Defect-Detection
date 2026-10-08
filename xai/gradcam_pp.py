"""
Grad-CAM++-style visual explanation for YOLO26m + MSC.

This module generates a class-specific heatmap from an intermediate
YOLO feature layer. It is intended for visual explanation only and
does not modify the detector or calculate detection metrics.

Note:
The current implementation is a practical Grad-CAM++ approximation
using gradient-based weighting. It should not be described as an
exact canonical higher-order-derivative Grad-CAM++ implementation.
"""

import cv2
import numpy as np
import torch

from ultralytics.data.augment import LetterBox


class GradCAMPlusPlus:
    """
    Practical Grad-CAM++-style implementation for YOLO detection.
    """

    def __init__(self, model, target_layer, device="cuda"):
        self.model = model
        self.target_layer = target_layer
        self.device = device

        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(
            self._forward_hook
        )

        self.backward_handle = target_layer.register_full_backward_hook(
            self._backward_hook
        )

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        """Remove registered PyTorch hooks."""
        self.forward_handle.remove()
        self.backward_handle.remove()

    def compute_cam(self):
        """
        Compute the Grad-CAM++-style activation map.
        """

        activation = self.activations[0]
        gradient = self.gradients[0]

        grad_2 = gradient ** 2
        grad_3 = gradient ** 3

        denominator = (
            2.0 * grad_2
            + (activation * grad_3).sum(dim=(1, 2), keepdim=True)
        )

        denominator = torch.where(
            denominator != 0.0,
            denominator,
            torch.ones_like(denominator),
        )

        alpha = grad_2 / denominator

        positive_gradient = torch.relu(gradient)

        weights = (
            alpha * positive_gradient
        ).sum(dim=(1, 2))

        cam = (
            weights[:, None, None] * activation
        ).sum(dim=0)

        cam = torch.relu(cam)

        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        return cam.detach().cpu().numpy()


def preprocess_image(image_path, imgsz=800):
    """
    Apply Ultralytics-compatible letterbox preprocessing.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    letterbox = LetterBox(
        new_shape=(imgsz, imgsz),
        auto=False,
        stride=32,
        scaleup=True,
        center=True,
    )

    processed = letterbox(image=image_rgb)

    tensor = (
        torch.from_numpy(processed)
        .permute(2, 0, 1)
        .float()
        / 255.0
    )

    tensor = tensor.unsqueeze(0)

    return image, tensor


def create_heatmap(
    cam,
    original_image,
    threshold_percentile=60,
    blur_kernel=(11, 11),
):
    """
    Convert the CAM into a heatmap and overlay it on the image.

    No detection bounding boxes are drawn.
    """

    cam = np.asarray(cam)

    low = np.percentile(cam, threshold_percentile)
    high = np.percentile(cam, 99)

    if high <= low:
        normalized = np.zeros_like(cam, dtype=np.float32)
    else:
        normalized = np.clip(
            (cam - low) / (high - low),
            0,
            1,
        )

    normalized = cv2.GaussianBlur(
        normalized,
        blur_kernel,
        0,
    )

    heatmap = np.uint8(normalized * 255)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET,
    )

    heatmap = cv2.resize(
        heatmap,
        (original_image.shape[1], original_image.shape[0]),
    )

    overlay = cv2.addWeighted(
        original_image,
        0.45,
        heatmap,
        0.55,
        0,
    )

    return heatmap, overlay
