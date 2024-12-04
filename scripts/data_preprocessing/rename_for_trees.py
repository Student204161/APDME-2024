
import os

# Define the directory where the images are located
folder_path = 'data/distorted_images/JPEGImages/lego'


#!/bin/bash
#BSUB -q gpua100
#BSUB -J mk_mesh
#BSUB -W 0:30
#BSUB -n 1
#BSUB -R "rusage[mem=40GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#### mode=exclusive_process"
#BSUB -B
#BSUB -N
#BSUB -u s204161@student.dtu.dk
#BSUB -o /work3/s204161/batch.out
#BSUB -e /work3/s204161/batch.out

# Iterate through all files in the directory
for filename in os.listdir(folder_path):
    # Full path of the original file
    old_file = os.path.join(folder_path, filename)

    # Check if the file name starts with 'dji_' and rename to strip the prefix
    if filename.startswith('DSCF'):
        # Find the underscore and get everything after it
        new_name = filename.split('DSCF', 1)[-1]  # Split and keep the part after the underscore

        # Construct the new file path
        new_file = os.path.join(folder_path, new_name)
        os.rename(old_file, new_file)
        print(f'Renamed: {old_file} -> {new_file}')
        # Update filename after renaming
        filename = new_name
        old_file = new_file  # Update old_file to the new path for the next check

    # Check if the file has a .JPG extension and rename it to .jpg
    if filename.endswith('.JPG'):
        # Rename only if the extension is uppercase .JPG
        new_name = filename[:-4] + '.jpg'  # Change the extension to lowercase
        new_file = os.path.join(folder_path, new_name)
        os.rename(old_file, new_file)
        print(f'Renamed extension: {old_file} -> {new_file}')
