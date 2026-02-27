from PIL import Image
import numpy as np
from pathlib import Path

data = []  # image data
Y = []     # labels (0 for Satisfactory, 1 for Defective)

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

# To reshape the data, use the following
# data = np.array(data).reshape(-1, 128, 128, 3)
# this makes each image 128x128 pixels with 3 color channels (RGB)
np.save("x_data_flattened.npy", np.array(data))
np.save("y_labels.npy", np.array(Y))