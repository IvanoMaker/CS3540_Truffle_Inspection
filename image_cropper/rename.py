# Rename all files in the folder with a specific naming convention
import os

def rename_files(directory):
    # for each file in the specified directory
    for i, filename in enumerate(os.listdir(directory)):
        if 'txt' in filename:
            continue  # skip text files
        old_file_path = os.path.join(directory, filename)       # get old file name
        if 'satisfactory' in directory:
            new_file_name = f"t_satisfactory_{i:03d}.png"              # new satisfactory file name with zero-padded index
        else:
            new_file_name = f"t_defective_{i:03d}.png"                 # new defective file name with zero-padded index
        new_file_path = os.path.join(directory, new_file_name)

        # Check if it's a file and not a directory to avoid errors
        if os.path.isfile(old_file_path):
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {filename} -> {new_file_name}")

# Specify the folder containing the files to be renamed
defective_folder_path = 'image_cropper\defective_output'
satisfactory_folder_path = 'image_cropper\satisfactory_output'

# Call the function to perform the renaming
rename_files(defective_folder_path)
rename_files(satisfactory_folder_path)