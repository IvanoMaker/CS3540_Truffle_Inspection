from PIL import Image
import numpy as np
from pathlib import Path

data = []
Y = []

PARENT_DIR = "dataset"
CHILD_DIR = [fld for fld in Path(PARENT_DIR).iterdir() if fld.is_dir()]

for child in CHILD_DIR:
    for img in Path(child).iterdir():
        if img.is_file():
            img_array = np.array(Image.open(img))
            data.append(img_array.flatten())
            if str(child).split("\\")[-1] == "Satisfactory":
                Y.append(0)
            elif str(child).split("\\")[-1] == "Defective":
                Y.append(1)

np.save("data.npy", np.array(data))
np.save("labels.npy", np.array(Y))