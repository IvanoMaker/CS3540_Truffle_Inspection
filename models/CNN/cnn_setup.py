from pathlib import Path

# NOTE: This file is intended to be executed in VSCode with the working directory set to 
# the root of the project. If you run it from a different location or from the terminal in the CNN folder, 
# you may need to adjust the DATASET_PATH to "../../dataset".  -Evan

DATASET_PATH = "dataset"
print(list(Path(DATASET_PATH).glob("*")))