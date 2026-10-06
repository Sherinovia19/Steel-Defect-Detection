# ============================================================
# STEEL SURFACE DEFECT DETECTION USING YOLO
# Dataset: Severstal Steel Defect Detection
# Model: YOLO11m Baseline
# ============================================================


# ============================================================
# 1. GOOGLE DRIVE SETUP
# ============================================================

from google.colab import drive

drive.mount("/content/drive")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


# Path to the Severstal dataset
data_path = (
    "/content/drive/MyDrive/"
    "Steel detection & segmentation/"
    "severstal-steel-defect-detection"
)

train_csv_path = os.path.join(
    data_path,
    "train.csv"
)

train_images_path = os.path.join(
    data_path,
    "train_images"
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

train_df = pd.read_csv(
    train_csv_path
)

print("Train data loaded")
print("Shape:", train_df.shape)

print("\nFirst five rows:")
print(train_df.head())


# ============================================================
# 3. BASIC EDA
# ============================================================

print("\nColumns:")
print(train_df.columns.tolist())


print("\nDataset information:")
train_df.info()


print("\nMissing values:")
print(train_df.isnull().sum())


# Class distribution
print("\nClass distribution:")
print(
    train_df["ClassId"]
    .value_counts()
    .sort_index()
)


# Number of unique annotated images
print("\nUnique annotated images:")
print(
    train_df["ImageId"].nunique()
)


# Number of annotations per image
print("\nAnnotations per image:")
print(
    train_df
    .groupby("ImageId")
    .size()
    .value_counts()
    .sort_index()
)


# ============================================================
# 4. IMAGE COUNT AND DEFECT-FREE IMAGES
# ============================================================

image_files = [
    f
    for f in os.listdir(train_images_path)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
]

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
    "\nTotal training images:",
    len(image_files)
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
# 5. DISPLAY SAMPLE IMAGE
# ============================================================

image_id = train_df.iloc[0]["ImageId"]

image_path = os.path.join(
    train_images_path,
    image_id
)

image = Image.open(
    image_path
)

plt.figure(
    figsize=(16, 4)
)

plt.imshow(image)

plt.title(
    f"Sample Image: {image_id}"
)

plt.axis("off")

plt.show()


# ============================================================
# 6. RLE MASK DECODING
# ============================================================

def rle_decode(
    mask_rle,
    shape=(256, 1600)
):

    # Return an empty mask when no annotation exists
    if not mask_rle:
        return np.zeros(
            shape,
            dtype=np.uint8
        )

    # Split the RLE string
    s = mask_rle.split()

    # Starting positions
    starts = (
        np.asarray(
            s[0::2],
            dtype=int
        )
        - 1
    )

    # Lengths of runs
    lengths = np.asarray(
        s[1::2],
        dtype=int
    )

    # Create flattened mask
    mask = np.zeros(
        shape[0] * shape[1],
        dtype=np.uint8
    )

    # Fill the mask
    for start, length in zip(
        starts,
        lengths
    ):

        mask[
            start:start + length
        ] = 1

    # Severstal RLE uses Fortran ordering
    return mask.reshape(
        shape,
        order="F"
    )


# Test RLE decoding
image_id = train_df.iloc[0]["ImageId"]

row = train_df[
    train_df["ImageId"] == image_id
].iloc[0]

mask = rle_decode(
    row["EncodedPixels"]
)

print("\nRLE mask information:")
print("Image:", image_id)
print("Class:", row["ClassId"])
print("Mask shape:", mask.shape)


# ============================================================
# 7. BUILD FOUR-CLASS MASK
# ============================================================

def build_mask(
    rle_list,
    shape=(256, 1600)
):

    if not rle_list:

        return np.zeros(
            (*shape, 0),
            dtype=np.uint8
        )

    return np.stack(
        [
            rle_decode(rle)
            for rle in rle_list
        ],
        axis=-1
    )


# ============================================================
# 8. IMAGE TILING
# ============================================================

# Width of each tile
TILE_WIDTH = 608

# Starting positions of the three tiles
TILE_POSITIONS = [
    0,
    495,
    991
]


def tiling(
    image,
    mask
):

    tiles = []

    for start in TILE_POSITIONS:

        end = (
            start + TILE_WIDTH
        )

        image_tile = image[
            :,
            start:end
        ]

        mask_tile = mask[
            :,
            start:end
        ]

        tiles.append(
            (
                image_tile,
                mask_tile
            )
        )

    return tiles


# ============================================================
# 9. BOUNDING BOX EXTRACTION
# ============================================================

import cv2


# Padding added around detected regions
BBOX_PAD = 3


# Minimum connected-component area
# for each of the four classes
MIN_AREA = [
    175,
    340,
    1250,
    1900
]


def extract_boxes(
    mask_tile
):

    H, W, C = mask_tile.shape

    bboxes = []

    # Process each class separately
    for class_id in range(C):

        binary = (
            mask_tile[
                :,
                :,
                class_id
            ]
            .astype(np.uint8)
        )

        # Find connected components
        num_labels, _, stats, _ = (
            cv2.connectedComponentsWithStats(
                binary,
                connectivity=8
            )
        )

        # Ignore background label 0
        for i in range(
            1,
            num_labels
        ):

            x, y, w, h, area = (
                stats[i]
            )

            # Remove very small regions
            if area < MIN_AREA[class_id]:
                continue

            # Add padding
            x1 = max(
                0,
                x - BBOX_PAD
            )

            y1 = max(
                0,
                y - BBOX_PAD
            )

            x2 = min(
                W,
                x + w + BBOX_PAD
            )

            y2 = min(
                H,
                y + h + BBOX_PAD
            )

            w_pad = (
                x2 - x1
            )

            h_pad = (
                y2 - y1
            )

            if (
                w_pad <= 0
                or h_pad <= 0
            ):
                continue

            # Convert bounding box
            # to YOLO normalized format
            bboxes.append(
                (
                    class_id,

                    (x1 + x2)
                    / 2
                    / W,

                    (y1 + y2)
                    / 2
                    / H,

                    w_pad / W,

                    h_pad / H
                )
            )

    return bboxes


# ============================================================
# 10. VERIFY BOUNDING BOXES ON SAMPLE IMAGE
# ============================================================

image_id = "0002cc93b.jpg"

# Four class RLE list
rle_list = [None] * 4

rows = train_df[
    train_df["ImageId"] == image_id
]

for _, row in rows.iterrows():

    class_id = (
        int(row["ClassId"]) - 1
    )

    rle_list[class_id] = (
        row["EncodedPixels"]
    )


# Build multi-class mask
mask4 = build_mask(
    rle_list
)


# Read original image
image_path = os.path.join(
    train_images_path,
    image_id
)

image = cv2.imread(
    image_path
)


# Create tiles
tiles = tiling(
    image,
    mask4
)


# Extract boxes from first tile
boxes = extract_boxes(
    tiles[0][1]
)


print("\nBounding box verification:")
print("Mask shape:", mask4.shape)
print("Number of tiles:", len(tiles))
print(
    "Image tile shape:",
    tiles[0][0].shape
)
print(
    "Mask tile shape:",
    tiles[0][1].shape
)
print("Bounding boxes:", boxes)


# ============================================================
# 11. VISUALIZE BOUNDING BOXES
# ============================================================

import matplotlib.patches as patches


tile_image = tiles[0][0]

fig, ax = plt.subplots(
    figsize=(12, 5)
)

# Convert BGR to RGB for matplotlib
ax.imshow(
    cv2.cvtColor(
        tile_image,
        cv2.COLOR_BGR2RGB
    )
)

H, W = (
    tile_image.shape[:2]
)


for (
    cls,
    x_center,
    y_center,
    width,
    height
) in boxes:

    # Convert normalized coordinates
    # back to pixel coordinates

    x_center *= W
    y_center *= H

    width *= W
    height *= H

    x = (
        x_center
        - width / 2
    )

    y = (
        y_center
        - height / 2
    )

    rect = patches.Rectangle(
        (x, y),
        width,
        height,
        linewidth=2,
        edgecolor="red",
        facecolor="none"
    )

    ax.add_patch(rect)

    ax.text(
        x,
        y,
        f"Class {cls}",
        color="red"
    )


ax.axis("off")

plt.show()


# ============================================================
# 12. CREATE YOLO DATASET FOLDERS
# ============================================================

output_path = os.path.join(
    data_path,
    "yolo_dataset"
)


for split in [
    "train",
    "val"
]:

    os.makedirs(
        os.path.join(
            output_path,
            "images",
            split
        ),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(
            output_path,
            "labels",
            split
        ),
        exist_ok=True
    )


print(
    "\nYOLO dataset folders created."
)


# ============================================================
# 13. TRAIN / VALIDATION SPLIT
# ============================================================

from sklearn.model_selection import (
    train_test_split
)

all_image_ids = sorted([
    file
    for file in os.listdir(
        train_images_path
    )
    if file.lower().endswith(".jpg")
])


# 80% training
# 20% validation
train_ids, val_ids = (
    train_test_split(
        all_image_ids,
        test_size=0.20,
        random_state=42
    )
)


print(
    "\nDataset split:"
)

print(
    "Total:",
    len(all_image_ids)
)

print(
    "Train:",
    len(train_ids)
)

print(
    "Validation:",
    len(val_ids)
)

print(
    "Overlap:",
    len(
        set(train_ids)
        & set(val_ids)
    )
)


# ============================================================
# 14. SAVE YOLO LABELS
# ============================================================

def save_yolo_labels(
    bboxes,
    file_path
):

    with open(
        file_path,
        "w"
    ) as f:

        for (
            cls,
            x,
            y,
            w,
            h
        ) in bboxes:

            f.write(
                f"{cls} "
                f"{x:.6f} "
                f"{y:.6f} "
                f"{w:.6f} "
                f"{h:.6f}\n"
            )


# ============================================================
# 15. PROCESS DATASET
# ============================================================

from tqdm import tqdm


def process_split(
    image_ids,
    split
):

    image_folder = os.path.join(
        output_path,
        "images",
        split
    )

    label_folder = os.path.join(
        output_path,
        "labels",
        split
    )


    for image_id in tqdm(
        image_ids,
        desc=f"Processing {split}"
    ):

        image_name = os.path.splitext(
            image_id
        )[0]


        # ----------------------------------------------------
        # Check whether all three tiles already exist
        # ----------------------------------------------------

        already_done = True

        for tile_index in range(3):

            tile_name = (
                f"{image_name}_tile{tile_index}"
            )

            image_save_path = os.path.join(
                image_folder,
                tile_name + ".jpg"
            )

            label_save_path = os.path.join(
                label_folder,
                tile_name + ".txt"
            )


            if not (
                os.path.exists(
                    image_save_path
                )
                and
                os.path.exists(
                    label_save_path
                )
            ):

                already_done = False

                break


        if already_done:
            continue


        # ----------------------------------------------------
        # Read original image
        # ----------------------------------------------------

        image_path = os.path.join(
            train_images_path,
            image_id
        )

        image = cv2.imread(
            image_path
        )


        if image is None:
            continue


        # ----------------------------------------------------
        # Collect annotations for all four classes
        # ----------------------------------------------------

        rle_list = [None] * 4

        rows = train_df[
            train_df["ImageId"]
            == image_id
        ]


        for _, row in rows.iterrows():

            class_id = (
                int(row["ClassId"])
                - 1
            )

            rle_list[class_id] = (
                row["EncodedPixels"]
            )


        # ----------------------------------------------------
        # Build mask and create tiles
        # ----------------------------------------------------

        mask = build_mask(
            rle_list
        )

        tiles = tiling(
            image,
            mask
        )


        # ----------------------------------------------------
        # Save tiles and YOLO labels
        # ----------------------------------------------------

        for (
            tile_index,
            (
                image_tile,
                mask_tile
            )
        ) in enumerate(tiles):

            tile_name = (
                f"{image_name}_tile{tile_index}"
            )


            image_save_path = os.path.join(
                image_folder,
                tile_name + ".jpg"
            )

            label_save_path = os.path.join(
                label_folder,
                tile_name + ".txt"
            )


            # Save tile image
            cv2.imwrite(
                image_save_path,
                image_tile
            )


            # Extract bounding boxes
            boxes = extract_boxes(
                mask_tile
            )


            # Save YOLO labels
            save_yolo_labels(
                boxes,
                label_save_path
            )


# ============================================================
# 16. GENERATE DATASET
# ============================================================

# The dataset has already been generated.
# Uncomment these lines only when rebuilding
# the YOLO dataset from the original Severstal data.

# process_split(
#     train_ids,
#     "train"
# )

# process_split(
#     val_ids,
#     "val"
# )


# ============================================================
# 17. VERIFY GENERATED DATASET
# ============================================================

for split in [
    "train",
    "val"
]:

    image_dir = os.path.join(
        output_path,
        "images",
        split
    )

    label_dir = os.path.join(
        output_path,
        "labels",
        split
    )


    images = [
        f
        for f in os.listdir(
            image_dir
        )
        if f.endswith(".jpg")
    ]


    labels = [
        f
        for f in os.listdir(
            label_dir
        )
        if f.endswith(".txt")
    ]


    print(
        f"\n{split}"
    )

    print(
        "Images:",
        len(images)
    )

    print(
        "Labels:",
        len(labels)
    )


# Expected:
#
# train
# Images: 30165
# Labels: 30165
#
# val
# Images: 7542
# Labels: 7542


# ============================================================
# 18. CREATE YOLO DATA.YAML
# ============================================================

yaml_content = f"""
path: {output_path}

train: images/train
val: images/val

nc: 4

names:
  0: pitted_surface
  1: crazing
  2: scratches
  3: patches
"""


yaml_path = os.path.join(
    output_path,
    "data.yaml"
)


with open(
    yaml_path,
    "w"
) as f:

    f.write(
        yaml_content
    )


print(
    "\nCreated:",
    yaml_path
)

print(
    open(yaml_path).read()
)


# ============================================================
# 19. YOLO11 BASELINE
# ============================================================

# Check GPU
!nvidia-smi


# Install Ultralytics
!pip install -U ultralytics


# ============================================================
# 20. IMPORT YOLO11
# ============================================================

from ultralytics import YOLO
import ultralytics


print(
    "Ultralytics version:",
    ultralytics.__version__
)


# Dataset YAML path
data_yaml = os.path.join(
    output_path,
    "data.yaml"
)


print(
    "Dataset YAML:",
    data_yaml
)


# ============================================================
# 21. LOAD YOLO11m MODEL
# ============================================================

model = YOLO(
    "yolo11m.pt"
)


# Training configuration
epochs = 25
imgsz = 800
batch = 8


print(
    "Epochs:",
    epochs
)

print(
    "Image size:",
    imgsz
)

print(
    "Batch size:",
    batch
)


# ============================================================
# 22. TRAIN YOLO11m
# ============================================================

results = model.train(

    data=data_yaml,

    epochs=epochs,

    imgsz=imgsz,

    batch=batch,

    # Save model checkpoints
    save=True,

    # Save checkpoint every epoch
    save_period=1,

    # Stop early if validation
    # performance does not improve
    patience=10
)


# ============================================================
# 23. VERIFY VALIDATION IMAGE
# ============================================================

test_image_path = os.path.join(
    output_path,
    "images",
    "val",
    "3aaaf3e64_tile1.jpg"
)


print(
    "\nValidation image exists:",
    os.path.exists(
        test_image_path
    )
)


try:

    img = Image.open(
        test_image_path
    )

    img.verify()

    print(
        "Image is readable and valid."
    )


except Exception as e:

    print(
        "Image has a problem:"
    )

    print(e)


# ============================================================
# 24. SAVE YOLO11 RESULTS TO GOOGLE DRIVE
# ============================================================

import shutil


# Training results created by Ultralytics
src = (
    "/content/runs/detect/train"
)


# Permanent Google Drive location
dst = (
    "/content/drive/MyDrive/"
    "Steel detection & segmentation/"
    "yolo11m_baseline"
)


shutil.copytree(
    src,
    dst,
    dirs_exist_ok=True
)


print(
    "\nYOLO11m baseline saved to:"
)

print(dst)


# ============================================================
# 25. VERIFY MODEL CHECKPOINTS
# ============================================================

weights_path = os.path.join(
    dst,
    "weights"
)


print(
    "\nbest.pt:",
    os.path.exists(
        os.path.join(
            weights_path,
            "best.pt"
        )
    )
)


print(
    "last.pt:",
    os.path.exists(
        os.path.join(
            weights_path,
            "last.pt"
        )
    )
)


# ============================================================
# END OF YOLO11 BASELINE
# ============================================================
