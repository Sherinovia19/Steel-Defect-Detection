# ============================================================
# YOLO26m + MSC — GRAD-CAM++
# CLEAR HEATMAP VERSION
# ============================================================

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from google.colab import files
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox


# ============================================================
# 1. SETTINGS
# ============================================================

MODEL_PATH = "/content/drive/MyDrive/Steel detection & segmentation/yolo26m_proposed/yolo26/runs/yolo26m_MSC_proposed_final/weights/best.pt"

IMG_SIZE = 800
CAM_LAYER = 16
CONF_THRESHOLD = 0.10

class_names = {
    0: "pitted_surface",
    1: "crazing",
    2: "scratches",
    3: "patches"
}


# ============================================================
# 2. LOAD MODEL
# ============================================================

model = YOLO(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"

model.model.to(device)

print("Model loaded")
print("Device:", device)


# ============================================================
# 3. UPLOAD IMAGE
# ============================================================

uploaded = files.upload()

image_path = list(uploaded.keys())[0]

image = cv2.imread(image_path)

if image is None:
    raise ValueError("Could not read image.")

image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

original_h, original_w = image.shape[:2]

print("Original image:", image.shape)


# ============================================================
# 4. LETTERBOX
# ============================================================

letterbox = LetterBox(
    new_shape=(IMG_SIZE, IMG_SIZE),
    auto=False,
    scale_fill=False,
    scaleup=True
)

image_resized = letterbox(
    image=image
)


# ============================================================
# 5. IMAGE TENSOR
# ============================================================

input_tensor = torch.from_numpy(
    image_resized.transpose(2, 0, 1)
).float() / 255.0

input_tensor = input_tensor.unsqueeze(0)

input_tensor = input_tensor.to(device)

input_tensor.requires_grad_(True)


# ============================================================
# 6. TARGET LAYER
# ============================================================

target_layer = model.model.model[CAM_LAYER]

print("Grad-CAM++ layer:", CAM_LAYER)


# ============================================================
# 7. HOOKS
# ============================================================

activations = None
gradients = None


def forward_hook(module, input, output):

    global activations

    activations = output


def backward_hook(module, grad_input, grad_output):

    global gradients

    gradients = grad_output[0]


target_layer.register_forward_hook(
    forward_hook
)

target_layer.register_full_backward_hook(
    backward_hook
)


# ============================================================
# 8. FORWARD PASS
# ============================================================

model.model.zero_grad(
    set_to_none=True
)

output = model.model(
    input_tensor
)


if isinstance(output, (tuple, list)):

    prediction = output[0]

else:

    prediction = output


print(
    "Prediction shape:",
    prediction.shape
)


# ============================================================
# 9. GET BOXES + CLASS SCORES
# ============================================================

boxes = prediction[
    0,
    :4
]

scores = prediction[
    0,
    4:8
]


# ============================================================
# 10. FIND STRONGEST DETECTION
# ============================================================

best_scores, best_classes = scores.max(
    dim=0
)

best_detection = best_scores.argmax()

target_class = int(
    best_classes[best_detection].item()
)

target_score = best_scores[
    best_detection
]


print(
    "Target class:",
    class_names[target_class]
)

print(
    "Target confidence:",
    float(
        target_score.detach().cpu()
    )
)


# ============================================================
# 11. DETECTION BOX
# ============================================================

box = boxes[
    :,
    best_detection
]

cx, cy, w, h = box

box_x1 = cx - w / 2
box_y1 = cy - h / 2

box_x2 = cx + w / 2
box_y2 = cy + h / 2


print(
    "Detection box:",
    [
        round(float(box_x1.detach().cpu()), 2),
        round(float(box_y1.detach().cpu()), 2),
        round(float(box_x2.detach().cpu()), 2),
        round(float(box_y2.detach().cpu()), 2)
    ]
)


# ============================================================
# 12. GRAD-CAM++ TARGET
# ============================================================

target = scores[
    target_class,
    best_detection
]

print(
    "Backpropagation target:",
    float(
        target.detach().cpu()
    )
)


# ============================================================
# 13. BACKPROPAGATION
# ============================================================

model.model.zero_grad(
    set_to_none=True
)

target.backward()


# ============================================================
# 14. CHECK GRADIENTS
# ============================================================

if activations is None:

    raise RuntimeError(
        "No activations captured."
    )

if gradients is None:

    raise RuntimeError(
        "No gradients captured."
    )


print(
    "Activation maximum:",
    float(
        activations.abs()
        .max()
        .detach()
        .cpu()
    )
)

print(
    "Gradient maximum:",
    float(
        gradients.abs()
        .max()
        .detach()
        .cpu()
    )
)


# ============================================================
# 15. GRAD-CAM++
# ============================================================

activation = activations[0]

gradient = gradients[0]


gradient_2 = gradient ** 2

gradient_3 = gradient ** 3


activation_sum = activation.sum(
    dim=(1, 2),
    keepdim=True
)


denominator = (
    2 * gradient_2
    +
    activation_sum * gradient_3
)


denominator = torch.where(
    denominator != 0,
    denominator,
    torch.ones_like(denominator)
)


alpha = (
    gradient_2 /
    denominator
)


weights = (
    alpha *
    torch.relu(gradient)
).sum(
    dim=(1, 2)
)


cam = (
    weights[:, None, None]
    *
    activation
).sum(
    dim=0
)


cam = torch.relu(cam)


# ============================================================
# 16. CONVERT CAM TO NUMPY
# ============================================================

cam = cam.detach().cpu().numpy()


print(
    "CAM minimum:",
    cam.min()
)

print(
    "CAM maximum:",
    cam.max()
)


if cam.max() <= 0:

    raise RuntimeError(
        "Grad-CAM++ produced zero activation."
    )


# ============================================================
# 17. NORMALIZE
# ============================================================

cam = cam / cam.max()


# ============================================================
# 18. SMOOTH
# ============================================================

cam = cv2.GaussianBlur(
    cam,
    (0, 0),
    sigmaX=2
)


# Normalize again

if cam.max() > 0:

    cam = cam / cam.max()


# ============================================================
# 19. RESIZE CAM TO 800x800
# ============================================================

cam = cv2.resize(
    cam,
    (IMG_SIZE, IMG_SIZE),
    interpolation=cv2.INTER_LINEAR
)


# ============================================================
# 20. LETTERBOX PARAMETERS
# ============================================================

r = min(
    IMG_SIZE / original_h,
    IMG_SIZE / original_w
)

new_w = round(
    original_w * r
)

new_h = round(
    original_h * r
)

left = round(
    (IMG_SIZE - new_w) / 2
)

top = round(
    (IMG_SIZE - new_h) / 2
)


# ============================================================
# 21. REMOVE LETTERBOX
# ============================================================

cam = cam[
    top:top + new_h,
    left:left + new_w
]


cam = cv2.resize(
    cam,
    (original_w, original_h),
    interpolation=cv2.INTER_LINEAR
)


cam = np.clip(
    cam,
    0,
    1
)


# ============================================================
# 22. CREATE STRONG JET HEATMAP
# ============================================================

heatmap = cv2.applyColorMap(
    np.uint8(cam * 255),
    cv2.COLORMAP_JET
)

heatmap = cv2.cvtColor(
    heatmap,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# 23. HEATMAP OVER ORIGINAL IMAGE
# ============================================================

image_float = (
    image.astype(np.float32)
    / 255.0
)

heatmap_float = (
    heatmap.astype(np.float32)
    / 255.0
)


# IMPORTANT:
# Use a mostly FIXED alpha so the heatmap
# is clearly visible.

alpha = 0.65


result = (
    image_float * (1 - alpha)
    +
    heatmap_float * alpha
)


result = np.clip(
    result * 255,
    0,
    255
).astype(np.uint8)


# ============================================================
# 24. CONVERT DETECTION BOX TO RAW IMAGE
# ============================================================

x1 = (
    float(box_x1.detach().cpu())
    - left
) / r

y1 = (
    float(box_y1.detach().cpu())
    - top
) / r

x2 = (
    float(box_x2.detach().cpu())
    - left
) / r

y2 = (
    float(box_y2.detach().cpu())
    - top
) / r


x1 = int(
    np.clip(
        x1,
        0,
        original_w - 1
    )
)

x2 = int(
    np.clip(
        x2,
        0,
        original_w - 1
    )
)

y1 = int(
    np.clip(
        y1,
        0,
        original_h - 1
    )
)

y2 = int(
    np.clip(
        y2,
        0,
        original_h - 1
    )
)


# ============================================================
# 25. DISPLAY
# ============================================================

plt.figure(
    figsize=(18, 6)
)


# ------------------------------------------------------------
# Original
# ------------------------------------------------------------

plt.subplot(
    1,
    3,
    1
)

plt.imshow(image)

plt.title(
    "Original Steel Image"
)

plt.axis("off")


# ------------------------------------------------------------
# Pure Grad-CAM++ heatmap
# ------------------------------------------------------------

plt.subplot(
    1,
    3,
    2
)

plt.imshow(
    heatmap
)

plt.title(
    "Grad-CAM++ Heatmap"
)

plt.axis("off")


# ------------------------------------------------------------
# Heatmap over original
# ------------------------------------------------------------

plt.subplot(
    1,
    3,
    3
)

plt.imshow(
    result
)

plt.title(
    "Grad-CAM++ Overlay - "
    + class_names[target_class]
)

plt.axis("off")


plt.tight_layout()

plt.show()
