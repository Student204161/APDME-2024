from utils import mk_nerf_loss_json, mk_test_image_folder
import json
import sys
import os
import shutil
#get the object argument from the terminal command
object_name = sys.argv[1]
SEG_FOLDER_NAME = sys.argv[2]
IS_REPROJ_FOLDER = int(sys.argv[3]) #1 or 0

if SEG_FOLDER_NAME == 'full_scene':
    full_scene=True
else:
    full_scene=False

#this script is for splitting the images into train & test for NeRF, in mip style format (every 8th image is test image, starting from first)
# In GS, a flag can be enabled for that, so only needed for NeRF.

transforms_json_path = f'data/nerf_cameras/{SEG_FOLDER_NAME}/{object_name}/transforms.json'
root_dir = os.getcwd()
mk_nerf_loss_json(transforms_json_path, root_dir,mip_style=True, reproj_folder=IS_REPROJ_FOLDER,obj=object_name,full_scene=full_scene)

