import os
import shutil

# Path to your test folder
test_folder = "data/asl_alphabet_dataset/test"

# Loop through all files in the test folder
for filename in os.listdir(test_folder):
    # Skip non-image files
    if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff")):
        continue

    # Get the base name without extension
    name_without_ext = os.path.splitext(filename)[0]
    class_name = name_without_ext.removesuffix('_test')

    # Create a subfolder named after the image (without extension)
    subfolder_path = os.path.join(test_folder, class_name)
    os.makedirs(subfolder_path, exist_ok=True)

    # Move the image into its subfolder
    src_path = os.path.join(test_folder, filename)
    dst_path = os.path.join(subfolder_path, filename)
    shutil.move(src_path, dst_path)

    print(f"Moved '{filename}' → '{subfolder_path}/'")

print("All images organized into subfolders.")
