# ============================================================
# STEEL SURFACE DEFECT DETECTION
# Exploratory Data Analysis (EDA)
# Dataset: Severstal Steel Defect Detection
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# 2. DATASET PATH
# ============================================================

# Change this path when running the script locally.
DATA_PATH = "PATH_TO_SEVERSTAL_DATASET"

TRAIN_CSV = os.path.join(DATA_PATH, "train.csv")
TRAIN_IMAGES = os.path.join(DATA_PATH, "train_images")


# ============================================================
# 3. LOAD TRAINING CSV
# ============================================================

train_df = pd.read_csv(TRAIN_CSV)

print("Dataset shape:", train_df.shape)

print("\nFirst five rows:")
print(train_df.head())

print("\nColumns:")
print(train_df.columns.tolist())


# ============================================================
# 4. DATASET INFORMATION
# ============================================================

print("\nDataset information:")
train_df.info()


# ============================================================
# 5. CHECK MISSING VALUES
# ============================================================

print("\nMissing values:")
print(train_df.isnull().sum())


# ============================================================
# 6. CLASS DISTRIBUTION
# ============================================================

print("\nClass distribution:")

class_distribution = (
    train_df["ClassId"]
    .value_counts()
    .sort_index()
)

print(class_distribution)


# Class mapping:
# 1 - Pitted Surface
# 2 - Crazing
# 3 - Scratches
# 4 - Patches

class_names = {
    1: "Pitted Surface",
    2: "Crazing",
    3: "Scratches",
    4: "Patches"
}


# Display class names with counts
for class_id, count in class_distribution.items():
    print(
        f"{class_id} - "
        f"{class_names[class_id]}: "
        f"{count}"
    )


# ============================================================
# 7. PLOT CLASS DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

class_distribution.plot(
    kind="bar"
)

plt.title(
    "Steel Defect Class Distribution"
)

plt.xlabel("Class ID")
plt.ylabel("Number of Annotations")

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.show()


# ============================================================
# 8. UNIQUE ANNOTATED IMAGES
# ============================================================

unique_annotated_images = (
    train_df["ImageId"]
    .nunique()
)

print(
    "\nUnique annotated images:",
    unique_annotated_images
)


# ============================================================
# 9. ANNOTATIONS PER IMAGE
# ============================================================

annotations_per_image = (
    train_df
    .groupby("ImageId")
    .size()
)

print(
    "\nAnnotations per image:"
)

print(
    annotations_per_image
    .value_counts()
    .sort_index()
)


# ============================================================
# 10. TOTAL TRAINING IMAGES
# ============================================================

image_files = [
    file
    for file in os.listdir(TRAIN_IMAGES)
    if file.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
]

total_images = len(image_files)

print(
    "\nTotal training images:",
    total_images
)


# ============================================================
# 11. DEFECT-FREE IMAGES
# ============================================================

annotated_images = set(
    train_df["ImageId"]
)

all_images = set(
    image_files
)

defect_free_images = (
    all_images - annotated_images
)

print(
    "Annotated images:",
    len(annotated_images)
)

print(
    "Defect-free images:",
    len(defect_free_images)
)


# ============================================================
# 12. ANNOTATED VS DEFECT-FREE
# ============================================================

image_status = [
    len(annotated_images),
    len(defect_free_images)
]

labels = [
    "Annotated",
    "Defect-free"
]

plt.figure(figsize=(7, 5))

plt.bar(
    labels,
    image_status
)

plt.title(
    "Annotated vs Defect-Free Images"
)

plt.ylabel("Number of Images")

plt.tight_layout()

plt.show()


# ============================================================
# 13. DISPLAY A SAMPLE IMAGE
# ============================================================

sample_image_id = train_df.iloc[0]["ImageId"]

sample_image_path = os.path.join(
    TRAIN_IMAGES,
    sample_image_id
)

sample_image = Image.open(
    sample_image_path
)

print(
    "\nSample image:",
    sample_image_id
)

print(
    "Image size:",
    sample_image.size
)


plt.figure(figsize=(16, 4))

plt.imshow(sample_image)

plt.title(
    f"Sample Steel Image: {sample_image_id}"
)

plt.axis("off")

plt.show()


# ============================================================
# 14. RLE DECODING
# ============================================================

def rle_decode(
    mask_rle,
    shape=(256, 1600)
):
    """
    Decode Severstal Run-Length Encoding
    into a binary mask.
    """

    if not mask_rle:
        return np.zeros(
            shape,
            dtype=np.uint8
        )

    values = mask_rle.split()

    starts = (
        np.asarray(
            values[0::2],
            dtype=int
        ) - 1
    )

    lengths = np.asarray(
        values[1::2],
        dtype=int
    )

    mask = np.zeros(
        shape[0] * shape[1],
        dtype=np.uint8
    )

    for start, length in zip(
        starts,
        lengths
    ):
        mask[
            start:start + length
        ] = 1

    # Severstal uses column-major ordering
    mask = mask.reshape(
        shape,
        order="F"
    )

    return mask


# ============================================================
# 15. DISPLAY SAMPLE DEFECT MASK
# ============================================================

sample_row = train_df.iloc[0]

sample_mask = rle_decode(
    sample_row["EncodedPixels"]
)

print(
    "\nMask shape:",
    sample_mask.shape
)

print(
    "Class ID:",
    sample_row["ClassId"]
)

print(
    "Mask pixels:",
    sample_mask.sum()
)


plt.figure(figsize=(16, 4))

plt.imshow(
    sample_mask,
    cmap="gray"
)

plt.title(
    "Sample Defect Mask"
)

plt.axis("off")

plt.show()


# ============================================================
# END OF EDA
# ============================================================
