import random
import string
import shutil
from pathlib import Path

strings = string.ascii_letters + string.digits
img_count = 0

def random_string(length):
    return ''.join(random.choice(strings) for _ in range(length))

PARENT_DIR = "satisfactory_defect_images"
CHILD_DIR = [fld for fld in Path(PARENT_DIR).iterdir() if fld.is_dir()]
GRANDCHILD_DIR = {}
IMGS = {}

DB_DIR = "dataset"
BD_CHILD_DIR = [fld for fld in Path(DB_DIR).iterdir() if fld.is_dir()]
      
for child in CHILD_DIR:
    for fld in Path(child).iterdir(): 
        if fld.is_dir():
            GRANDCHILD_DIR[child] = fld
            t_type = (str(fld).split("\\")[-1])
            for img in Path(fld).iterdir():
                if img.is_file():
                    img_count += 1
                    new_name = random_string(10) + ".png"
                    if t_type == "satisfactory":
                        new_path = Path(DB_DIR) / "Satisfactory" / new_name
                    elif t_type == "defective":
                        new_path = Path(DB_DIR) / "Defective" / new_name
                    shutil.copy(img, new_path)