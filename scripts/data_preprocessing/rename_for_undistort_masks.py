#change file name from jpg to png in order to undistort masks.
import sys
import os

path_to_txt = sys.argv[1]

with open(path_to_txt, 'r') as input_file:
    # Read the content of the input file
    file_content = input_file.read()

# Replace ".jpg" with ".png"
updated_content = file_content.replace(".jpg", ".png")

# Open the output file in write mode and write the updated content
with open(path_to_txt, 'w') as output_file:
    output_file.write(updated_content)

print(f'Replaced ".jpg" with ".png" in {path_to_txt}.')
