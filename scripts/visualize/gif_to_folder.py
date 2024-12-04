import os
from PIL import Image

def convert_gif_to_png(folder_path):
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    # Get all GIF files in the folder that start with 'test_tree'
    gif_files = [f for f in os.listdir(folder_path) if f.lower().startswith('test_tree') and f.lower().endswith('.gif')]
    
    if not gif_files:
        print("No matching GIF files found in the folder.")
        return

    for gif_file in gif_files:
        gif_path = os.path.join(folder_path, gif_file)
        gif_name = os.path.splitext(gif_file)[0]
        
        # Create a subfolder for PNG frames
        output_folder = os.path.join(folder_path, gif_name)
        os.makedirs(output_folder, exist_ok=True)
        
        # Open the GIF
        with Image.open(gif_path) as gif:
            frame_number = 0
            while True:
                # Save each frame as a PNG
                frame_path = os.path.join(output_folder, f"frame_{frame_number + 1}.png")
                gif.seek(frame_number)
                gif.save(frame_path, format="PNG")
                print(f"Saved frame {frame_number + 1} of '{gif_file}' as PNG.")

                frame_number += 1
                try:
                    gif.seek(frame_number)  # Move to the next frame
                except EOFError:
                    break  # End of frames

    print("GIF to PNG conversion complete.")

# Specify the folder containing GIFs
folder_path = "/work3/s204161/gifz"  # Replace with your folder path
convert_gif_to_png(folder_path)

