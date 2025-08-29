#!/usr/bin/env python3
import os
import glob
import shutil
import random
import argparse
from pathlib import Path


def find_image_label_pairs(images_dir: str, labels_dir: str):
    image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for pattern in image_patterns:
        image_paths.extend(glob.glob(os.path.join(images_dir, pattern)))

    pairs = []
    missing_labels = []
    for img in sorted(image_paths):
        stem = os.path.splitext(os.path.basename(img))[0]
        label_path = os.path.join(labels_dir, f"{stem}.txt")
        if os.path.exists(label_path):
            pairs.append((img, label_path))
        else:
            missing_labels.append(img)
    return pairs, missing_labels


def split_pairs(pairs, train_ratio: float, seed: int):
    random.seed(seed)
    shuffled = pairs[:]
    random.shuffle(shuffled)
    split_index = int(len(shuffled) * train_ratio)
    train_pairs = shuffled[:split_index]
    val_pairs = shuffled[split_index:]
    return train_pairs, val_pairs


def ensure_dirs(base_out: str):
    for sub in [
        'images/train', 'images/val',
        'labels/train', 'labels/val',
    ]:
        Path(os.path.join(base_out, sub)).mkdir(parents=True, exist_ok=True)


def copy_pairs(pairs, out_dir: str, split_name: str):
    img_dst_dir = os.path.join(out_dir, 'images', split_name)
    lbl_dst_dir = os.path.join(out_dir, 'labels', split_name)
    copied = 0
    for img, lbl in pairs:
        img_name = os.path.basename(img)
        lbl_name = os.path.basename(lbl)
        shutil.copy2(img, os.path.join(img_dst_dir, img_name))
        shutil.copy2(lbl, os.path.join(lbl_dst_dir, lbl_name))
        copied += 1
    return copied


def detect_num_classes(label_files):
    class_ids = set()
    for lf in label_files:
        try:
            with open(lf) as f:
                for ln in f:
                    parts = ln.strip().split()
                    if not parts:
                        continue
                    cid = parts[0]
                    if cid.isdigit():
                        class_ids.add(int(cid))
        except Exception:
            pass
    if not class_ids:
        return 0, []
    max_id = max(class_ids)
    num_classes = max_id + 1
    # Default names as strings of ids
    names = [str(i) for i in range(num_classes)]
    return num_classes, names


def write_data_yaml(out_dir: str, names):
    yaml_path = os.path.join(out_dir, 'data.yaml')
    lines = []
    lines.append(f"path: {out_dir}")
    lines.append("train: images/train")
    lines.append("val: images/val")
    lines.append("names:")
    for i, n in enumerate(names):
        lines.append(f"  {i}: {n}")
    content = "\n".join(lines) + "\n"
    with open(yaml_path, 'w') as f:
        f.write(content)
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description='Prepare YOLOv8 dataset with 80/20 split')
    parser.add_argument('--src-images', default='ToothNumber_TaskDataset/images', help='Source images directory')
    parser.add_argument('--src-labels', default='ToothNumber_TaskDataset/labels', help='Source labels directory')
    parser.add_argument('--out', default='dataset', help='Output dataset root')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Train split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for split')
    args = parser.parse_args()

    ensure_dirs(args.out)

    pairs, missing = find_image_label_pairs(args.src_images, args.src_labels)
    print(f"Found {len(pairs)} image-label pairs. Missing labels for {len(missing)} images.")

    train_pairs, val_pairs = split_pairs(pairs, args.train_ratio, args.seed)
    print(f"Split into train={len(train_pairs)} and val={len(val_pairs)}")

    ct_train = copy_pairs(train_pairs, args.out, 'train')
    ct_val = copy_pairs(val_pairs, args.out, 'val')
    print(f"Copied train={ct_train} pairs, val={ct_val} pairs")

    # Detect classes from all labels
    all_label_files = [p[1] for p in pairs]
    num_classes, names = detect_num_classes(all_label_files)
    if num_classes == 0:
        print("Warning: No classes detected in labels. You may need to set names manually in data.yaml")
        names = []
    yaml_path = write_data_yaml(args.out, names)
    print(f"Wrote {yaml_path} with {len(names)} classes")


if __name__ == '__main__':
    main()


