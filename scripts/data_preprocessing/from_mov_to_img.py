
import os, sys
from PIL import Image, ImageOps
import json
import shutil
from utils import mk_meta_json, mov_to_frames

root_dir = sys.argv[1]
wish_frames = int(sys.argv[2])

mov_dataset_path = os.path.join(root_dir, 'data/movs')
images_dataset_path = os.path.join(root_dir,  'data/distorted_images/JPEGImages')
to_be_annotated_path = os.path.join(root_dir,  'data/distorted_images/ToBeAnnotated')
mask_path = os.path.join(root_dir,  'data/distorted_images/Annotations/XMEM')
# if os.path.exists(images_dataset_path):
#     print(f'delete previous images? \nif found in path:{images_dataset_path} \nand in path:{to_be_annotated_path}')
#     delete_previous_in_images_dataset_path = input("[for yes, write: yes]")
# else:
delete_previous_in_images_dataset_path = "no"

if delete_previous_in_images_dataset_path == "yes":
    shutil.rmtree(to_be_annotated_path)
    shutil.rmtree(images_dataset_path)


for movie in os.listdir(mov_dataset_path):
    mov = movie.split('.')[0]
    if not os.path.exists(os.path.join(images_dataset_path,mov)):
        print(f'doing {os.path.join(images_dataset_path,mov)}')
        mov_to_frames(rf'{mov_dataset_path}/{movie}',rf'{images_dataset_path}/{movie}',rf'{to_be_annotated_path}/{movie}', wish_frames)

# Directory containing frames for different items
dir_path = images_dataset_path

#data = mk_meta_json(dir_path,mask_path)

# Write the JSON data structure to a file
# with open(fr'{os.path.join(root_dir, "/data/distorted_images")}/meta.json', 'w') as f:
#     json.dump(data, f, indent=4)


