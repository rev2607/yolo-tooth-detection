#!/usr/bin/env python3
import os, glob, sys

root = sys.argv[1] if len(sys.argv) > 1 else '.'

img_patterns = ['*.jpg','*.jpeg','*.png','*.bmp']
imgs = []
for p in img_patterns:
    imgs.extend(glob.glob(os.path.join(root, '**', p), recursive=True))
print(f"Found {len(imgs)} images (sample 5):")
for s in imgs[:5]:
    print("  ", s)

txts = glob.glob(os.path.join(root, '**', '*.txt'), recursive=True)
print(f"\nFound {len(txts)} .txt files (possible YOLO labels). Sample 5:")
for s in txts[:5]:
    print("  ", s)

jsons = glob.glob(os.path.join(root, '**', '*.json'), recursive=True)
print(f"\nFound {len(jsons)} .json files (COCO?):")
for s in jsons[:10]:
    print("  ", s)

xmls = glob.glob(os.path.join(root, '**', '*.xml'), recursive=True)
print(f"\nFound {len(xmls)} .xml files (VOC?):")
for s in xmls[:10]:
    print("  ", s)

# show a sample image <-> label relationship
if imgs:
    sample = imgs[0]
    print("\nSample image:", sample)
    candidate = os.path.splitext(sample)[0] + '.txt'
    if os.path.exists(candidate):
        print("Corresponding .txt found:", candidate)
        print("First lines of that .txt:")
        with open(candidate) as f:
            for i, line in enumerate(f):
                if i >= 10: break
                print("  ", line.strip())
    else:
        print("No corresponding .txt for the sample image.")

# summarize class ids in .txt labels (if any)
class_counts = {}
for t in txts:
    try:
        with open(t) as f:
            for ln in f:
                p = ln.strip().split()
                if not p: continue
                cls = p[0]
                class_counts[cls] = class_counts.get(cls, 0) + 1
    except Exception as e:
        print("Error reading", t, e)

print("\nUnique class ids found in .txt labels (id:count). Up to 50 shown:")
for k, v in sorted(class_counts.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0])[:50]:
    print("  ", k, ":", v)

print("\nTop-level files/dirs in dataset root (first 50):")
print(sorted(os.listdir(root))[:50])


