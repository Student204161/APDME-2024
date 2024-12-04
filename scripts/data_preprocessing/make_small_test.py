import os
from PIL import Image

# Folder path containing images
folder_path = 'data/distorted_images/JPEGImages/plant'

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):  # add other formats if needed
        image_path = os.path.join(folder_path, filename)

        # Open the image
        with Image.open(image_path) as img:
            # Get the current size
            width, height = img.size

            # Calculate the new size (1/8th of the current size)
            new_size = (width // 8, height // 8)

            # Resize the image
            resized_img = img.resize(new_size, Image.ANTIALIAS)

            # Save the resized image, you can overwrite or save with a new name
            resized_img.save(image_path)  # Overwrite the original image
            # Alternatively, save with a new name: resized_img.save(f'{folder_path}/resized_{filename}')

        print(f'Resized {filename} to {new_size}')

print("Resizing of all images is complete.")
