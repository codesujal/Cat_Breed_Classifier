import os
import shutil

# PATHS
SOURCE_DIR = "oxford-iiit-pet/images"
DEST_DIR = "samples"

os.makedirs(DEST_DIR, exist_ok=True)

# PROCESS FILES
count = 0

for file in os.listdir(SOURCE_DIR):

    # only images
    if not file.endswith(".jpg"):
        continue

    # RULE 1: first letter must be CAPITAL (CAT)
    if not file[0].isupper():
        continue

    # RULE 2: extract breed name before first underscore
    breed_name = file.split("_")[0]

    # COPY FILE
    src_path = os.path.join(SOURCE_DIR, file)

    # rename nicely for GUI usage
    dest_path = os.path.join(DEST_DIR, f"{breed_name}.jpg")

    # avoid overwrite duplicates
    if not os.path.exists(dest_path):
        shutil.copy(src_path, dest_path)
        count += 1

print(f"Done Copied {count} sample images to '{DEST_DIR}'")