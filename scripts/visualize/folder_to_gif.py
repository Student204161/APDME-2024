from PIL import Image
import os

def create_gif(input_folder, output_file, duration=500):
    """
    Creates a GIF from all images in the specified folder, resizing them to 1920x1080.

    :param input_folder: Path to the folder containing images.
    :param output_file: Output GIF file path.
    :param duration: Duration of each frame in milliseconds.
    """
    # Get all image files from the folder, sorted alphabetically
    image_files = sorted(
        [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]
    )
    
    # Ensure the folder contains images
    if not image_files:
        print("No images found in the folder.")
        return

    # Open and resize images to 1920x1080
    images = [
        Image.open(os.path.join(input_folder, img)).resize((1920, 1080), Image.Resampling.LANCZOS)
        for img in image_files
    ]

    # Save the first image as a GIF, appending the rest as frames
    images[0].save(
        output_file,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )

    print(f"GIF created successfully: {output_file}")


# Example usage
# input_folder = "data/GS_models/full_scene_MVG-XMEM_INT_0.6_RMVXYZ_0.05_RMV3STD/test_tree/1"  # Replace with your folder path
# output_file = "/work3/s204161/gifz/5.gif"  # Replace with your desired output file path
# create_gif(input_folder, output_file)

# List of input folder paths
input_folders = [
    # "data/GS_models/full_scene_MVG-XMEM_INT_0.95_RMVXYZ_10.0_RMV3.0STD_white/test_bonsai/1",
    # "data/GS_models/full_scene_MVG-XMEM_INT_0.95_RMVXYZ_10.0_white/test_bonsai/1",
    # "data/GS_models/full_scene_MVG-XMEM_INT_0.95_white/test_bonsai/1",

    # "data/GS_models/full_scene_MVG-XMEM_0.5_RMVXYZ_0.05_RMV3.0STD_white/test_tree/1",
    # "data/GS_models/full_scene_MVG-XMEM_0.5_RMVXYZ_0.05_white/test_tree/1",
    # "data/GS_models/full_scene_MVG-XMEM_0.5_white/test_tree/1",

    # "data/undistorted_images/full_scene_MVG-XMEM_INT_0.95_RMVXYZ_10.0_RMV3.0STD_white/test_bonsai/1",
    # "data/undistorted_images/full_scene_MVG-XMEM_INT_0.95_RMVXYZ_10.0_white/test_bonsai/1",
    # "data/undistorted_images/full_scene_MVG-XMEM_INT_0.95_white/test_bonsai/1",

    #'data/undistorted_images/full_scene_MVG-XMEM_INT_0.6/tree/1',
    #'data/undistorted_images/full_scene_MVG-XMEM_INT_0.6_RMVXYZ_0.05/tree/1',
    'data/undistorted_images/full_scene_MVG-XMEM_INT_0.6_RMVXYZ_0.05_RMV3STD/tree/1'

    # "data/undistorted_images/full_scene_MVG-XMEM_0.5_RMVXYZ_0.05_RMV3.0STD_white/tree/1",
    # "data/undistorted_images/full_scene_MVG-XMEM_0.5_RMVXYZ_0.05_white/tree/1",
    # "data/undistorted_images/full_scene_MVG-XMEM_0.5_white/tree/1"

]
from tqdm import tqdm

# Iterate through input folders to process and generate output file names
for folder in tqdm(input_folders):
    # Extract unique identifiers from the folder path
    parts = folder.split("/")
    identifier = parts[2]  # Adjust index based on your folder structure
    obj = parts[3]
    output_file = f"/work3/s204161/gifz/{obj}_{identifier}.gif"
    
    # Call the create_gif function
    create_gif(folder, output_file)
