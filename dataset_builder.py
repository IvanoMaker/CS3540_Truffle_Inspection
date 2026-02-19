from pathlib import Path
from PIL import Image

# INPUT IMAGE FOLDER STRUCTURAL REQUIRMENTS
# - each group member classified their own video of the production line
# - each group member therefor should have a directory with two subfolders,
#   one for satisfactory truffles and another for defective truffles
# - Each folder should be put in the 'satisfactory_defect_images' folder for this to run

img_count = 0
bad_size = 0

PARENT_DIR = "satisfactory_defect_images"
CHILD_DIR = [fld for fld in Path(PARENT_DIR).iterdir() if fld.is_dir()]
GRANDCHILD_DIR = {}
IMGS = {}

for child in CHILD_DIR:
    for fld in Path(child).iterdir(): 
        if fld.is_dir():
            GRANDCHILD_DIR[child] = fld
            IMGS[fld] = [img for img in Path(fld).iterdir()]
            img_count += len(IMGS[fld])

print("~ Dataset Construction Script ~")
print(f"searching through folders in {PARENT_DIR}")
print(f"video folders loaded : {len(CHILD_DIR)}")
print(f"       images loaded : {img_count}")

for folder, paths in IMGS.items():
    for img_path in paths:
        image = Image.open(img_path)
        width, height = image.size
        if (width != 128) or (height != 128):
            bad_size += 1

assert bad_size == 0, "bad sized image"