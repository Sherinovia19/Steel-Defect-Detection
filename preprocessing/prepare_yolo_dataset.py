import os
import cv2
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

data_path = "/content/drive/MyDrive/Steel detection & segmentation/severstal-steel-defect-detection"

train_image_dir = os.path.join(data_path, "train_images")
train_csv_path = os.path.join(data_path, "train.csv")

output_path = os.path.join(data_path, "yolo_dataset")


# ============================================================
# LOAD ANNOTATIONS
# ============================================================

import pandas as pd

train_df = pd.read_csv(train_csv_path)

print("Train data loaded")
print("Shape:", train_df.shape)


# ============================================================
# RLE DECODING
# ============================================================

def rle_decode(mask_rle, shape=(256, 1600)):

    if not mask_rle:
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()

    starts = np.asarray(s[0::2], dtype=int) - 1
    lengths = np.asarray(s[1::2], dtype=int)

    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for start, length in zip(starts, lengths):
        mask[start:start + length] = 1

    return mask.reshape(shape, order="F")


def build_mask(rle_list, shape=(256, 1600)):

    if not rle_list:
        return np.zeros((*shape, 0), dtype=np.uint8)

    return np.stack(
        [rle_decode(rle) for rle in rle_list],
        axis=-1
    )


# ============================================================
# TILING
# ============================================================

TILE_WIDTH = 608
TILE_POSITIONS = [0, 495, 991]


def tiling(image, mask):

    tiles = []

    for start in TILE_POSITIONS:

        end = start + TILE_WIDTH

        image_tile = image[:, start:end]
        mask_tile = mask[:, start:end]

        tiles.append((image_tile, mask_tile))

    return tiles


# ============================================================
# BOUNDING BOX EXTRACTION
# ============================================================

BBOX_PAD = 3

MIN_AREA = [175, 340, 1250, 1900]


def extract_boxes(mask_tile):

    H, W, C = mask_tile.shape

    bboxes = []

    for class_id in range(C):

        binary = mask_tile[:, :, class_id].astype(np.uint8)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )

        for i in range(1, num_labels):

            x, y, w, h, area = stats[i]

            if area < MIN_AREA[class_id]:
                continue

            x1 = max(0, x - BBOX_PAD)
            y1 = max(0, y - BBOX_PAD)

            x2 = min(W, x + w + BBOX_PAD)
            y2 = min(H, y + h + BBOX_PAD)

            w_pad = x2 - x1
            h_pad = y2 - y1

            if w_pad <= 0 or h_pad <= 0:
                continue

            bboxes.append(
                (
                    class_id,
                    (x1 + x2) / 2 / W,
                    (y1 + y2) / 2 / H,
                    w_pad / W,
                    h_pad / H,
                )
            )

    return bboxes


# ============================================================
# YOLO LABEL SAVING
# ============================================================

def save_yolo_labels(bboxes, file_path):

    with open(file_path, "w") as f:

        for cls, x, y, w, h in bboxes:

            f.write(
                f"{cls} {x:.6f} {y:.6f} "
                f"{w:.6f} {h:.6f}\n"
            )


# ============================================================
# CREATE YOLO DATASET FOLDERS
# ============================================================

for split in ["train", "val"]:

    os.makedirs(
        os.path.join(output_path, "images", split),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(output_path, "labels", split),
        exist_ok=True
    )


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

all_image_ids = sorted(
    [
        file
        for file in os.listdir(train_image_dir)
        if file.lower().endswith(".jpg")
    ]
)

train_ids, val_ids = train_test_split(
    all_image_ids,
    test_size=0.20,
    random_state=42
)

print("Total images:", len(all_image_ids))
print("Train images:", len(train_ids))
print("Validation images:", len(val_ids))
print(
    "Overlap:",
    len(set(train_ids) & set(val_ids))
)


# ============================================================
# PROCESS DATASET
# ============================================================

def process_split(image_ids, split):

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

        image_path = os.path.join(
            train_image_dir,
            image_id
        )

        image = cv2.imread(image_path)

        if image is None:
            continue

        # Collect annotations for all four classes
        rle_list = [None] * 4

        rows = train_df[
            train_df["ImageId"] == image_id
        ]

        for _, row in rows.iterrows():

            class_id = int(row["ClassId"]) - 1

            rle_list[class_id] = row["EncodedPixels"]

        # Build four-channel mask
        mask = build_mask(rle_list)

        # Create three tiles
        tiles = tiling(image, mask)

        for tile_index, (image_tile, mask_tile) in enumerate(tiles):

            image_name = os.path.splitext(image_id)[0]

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

            # Skip already generated tiles
            if (
                os.path.exists(image_save_path)
                and os.path.exists(label_save_path)
            ):
                continue

            # Save tiled image
            cv2.imwrite(
                image_save_path,
                image_tile
            )

            # Extract bounding boxes
            boxes = extract_boxes(mask_tile)

            # Save YOLO labels
            save_yolo_labels(
                boxes,
                label_save_path
            )


# ============================================================
# GENERATE TRAINING DATA
# ============================================================

process_split(train_ids, "train")

# ============================================================
# GENERATE VALIDATION DATA
# ============================================================

process_split(val_ids, "val")

print("Dataset generation complete.")


# ============================================================
# VERIFY GENERATED DATASET
# ============================================================

for split in ["train", "val"]:

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
        f for f in os.listdir(image_dir)
        if f.endswith(".jpg")
    ]

    labels = [
        f for f in os.listdir(label_dir)
        if f.endswith(".txt")
    ]

    print(split)
    print("Images:", len(images))
    print("Labels:", len(labels))


# ============================================================
# CREATE YOLO DATA.YAML
# ============================================================

yaml_content = f"""
path: {output_path}

train: images/train
val: images/val

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

with open(yaml_path, "w") as f:
    f.write(yaml_content)

print("Created:", yaml_path)
print(open(yaml_path).read())
