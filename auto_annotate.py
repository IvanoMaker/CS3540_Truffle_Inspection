import os

def create_annotations(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith('.png'):
            txt_file = os.path.splitext(file)[0] + '.txt'
            txt_path = os.path.join(folder_path, txt_file)
            with open(txt_path, 'w') as f:
                f.write('0 0.5 0.5 1.0 1.0\n')

# Annotate train
create_annotations('dataset/train/truffles')

# Annotate val
create_annotations('dataset/val/truffles')

print("Annotations created.")