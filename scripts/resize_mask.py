# resize_masks.py
import os
import sys
from PIL import Image

def resize_masks(input_folder, output_folder, target_width, target_height):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                img = Image.open(file_path)
                resized_img = img.resize((target_width, target_height))
                resized_img.save(os.path.join(output_folder, filename))
                #print(f"Resized and saved {filename} to {output_folder}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

if __name__ == "__main__":
    # Check for proper arguments
    if len(sys.argv) != 5:
        print("Usage: python resize_masks.py <input_folder> <output_folder> <target_width> <target_height>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    target_width = int(sys.argv[3])
    target_height = int(sys.argv[4])

    resize_masks(input_folder, output_folder, target_width, target_height)
