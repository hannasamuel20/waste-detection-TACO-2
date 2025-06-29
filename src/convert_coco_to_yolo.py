from pycocotools.coco import COCO
import os
import shutil
from pathlib import Path
import random
import yaml

# --- Set up paths ---
project_dir = Path.home() / "course_project"
coco_json = project_dir / "data/TACO/data/annotations.json"
img_dir = project_dir / "data/TACO/data"
output_dir = project_dir / "data/TACO/yolo_dataset"

# --- Make sure annotation file exists ---
if not coco_json.exists():
    raise FileNotFoundError(f"Could not find {coco_json}")

# --- Create output directories ---
for split in ['train', 'val', 'test']:
    (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

# --- Load COCO annotations ---
coco = COCO(str(coco_json))

# --- Target classes to include (subset of TACO) ---
target_classes = [
    'Cigarette', 'Unlabeled litter', 'Plastic film', 'Clear plastic bottle',
    'Other plastic', 'Other plastic wrapper', 'Drink can', 'Plastic bottle cap',
    'Plastic straw', 'Broken glass', 'Styrofoam piece', 'Disposable plastic cup',
    'Glass bottle', 'Pop tab', 'Other carton'
]

# --- Build class map (COCO ID ➝ YOLO ID) ---
target_cats = [cat for cat in coco.loadCats(coco.getCatIds()) if cat['name'] in target_classes]
class_map = {cat['id']: i for i, cat in enumerate(target_cats)}
class_names = {i: cat['name'] for i, cat in enumerate(target_cats)}

print("🗂 Selected classes:")
for i, name in class_names.items():
    print(f"  {i}: {name}")

# --- Filter images containing at least one target object ---
img_ids = [img_id for img_id in coco.getImgIds()
           if any(ann['category_id'] in class_map for ann in coco.loadAnns(coco.getAnnIds(img_id)))]

random.seed(42)
random.shuffle(img_ids)
n = len(img_ids)
train_ids = img_ids[:int(0.8 * n)]
val_ids = img_ids[int(0.8 * n):int(0.9 * n)]
test_ids = img_ids[int(0.9 * n):]

# --- Helper: Convert and write YOLO annotations ---
def convert_to_yolo(img_ids, split):
    for img_id in img_ids:
        img = coco.loadImgs(img_id)[0]
        img_path = img_dir / img['file_name']
        if not img_path.exists():
            continue

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = [ann for ann in coco.loadAnns(ann_ids) if ann['category_id'] in class_map]
        if not anns:
            continue

        # Save image
        dest_img_name = f"{img_id}_{img['file_name'].replace('/', '_')}"
        shutil.copy(img_path, output_dir / split / "images" / dest_img_name)

        # Save YOLO annotations
        label_path = output_dir / split / "labels" / f"{Path(dest_img_name).stem}.txt"
        with open(label_path, 'w') as f:
            for ann in anns:
                x, y, w, h = ann['bbox']
                x_center = (x + w / 2) / img['width']
                y_center = (y + h / 2) / img['height']
                w_norm = w / img['width']
                h_norm = h / img['height']
                class_id = class_map[ann['category_id']]
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

# --- Perform conversion ---
convert_to_yolo(train_ids, 'train')
convert_to_yolo(val_ids, 'val')
convert_to_yolo(test_ids, 'test')

# --- Write taco.yaml for YOLOv8 ---
yaml_path = output_dir / "taco.yaml"
with open(yaml_path, 'w') as f:
    yaml.dump({
        'train': str(output_dir / 'train/images'),
        'val': str(output_dir / 'val/images'),
        'test': str(output_dir / 'test/images'),
        'nc': len(class_names),
        'names': [class_names[i] for i in range(len(class_names))]
    }, f)

print(f"\n✅ YOLO dataset created at: {output_dir}")
print(f"📝 YAML file saved to: {yaml_path}")
