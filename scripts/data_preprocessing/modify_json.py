import json
import sys
import os
import shutil
#get the object argument from the terminal command
object_name = sys.argv[1]
SEG_FOLDER_NAME = sys.argv[2]
situation = sys.argv[3]
UNDISTORTED = sys.argv[4]
root_dir = sys.argv[5]

#object_name='flask'
#SEG_FOLDER_NAME='RPCMVOS_30k'
#situation='1'
#images_dir="/work3/s204161/BachelorData/bachelor_data/images/JPEGImages/"


if not os.path.exists(f'data/nerf_cameras/{SEG_FOLDER_NAME}'):
    os.makedirs(f'data/nerf_cameras/{SEG_FOLDER_NAME}')

if UNDISTORTED == "1":
    if situation == "1":
        print(f'First time training for {object_name}. Copying json file to create camera json for {SEG_FOLDER_NAME} / {object_name}')

        # load the JSON file
        with open(fr'data/nerf_cameras/full_scene/{object_name}/transforms.json') as f:
            data = json.load(f)
        with open(fr'data/nerf_cameras/full_scene/{object_name}/transforms.json') as f:
            data_obj = json.load(f)

        # loop through all the "frames" in the JSON data
        for frame in data['frames']:
            # get the current "file_path" value
            current_path = frame['file_path']
            # replace "/data/object/{object_name}/" with an empty string to remove it from the path
            new_path = f'{root_dir}/data/undistorted_images/full_scene/{object_name}/' + current_path.split('/')[-1]  #current_path.replace(f'data/{object_name}/', '')
            # update the "file_path" value with the new path
            frame['file_path'] = new_path

        # do same but for modified path to segmented data
        for frame in data_obj['frames']:
            # get the current "file_path" value
            current_path = frame['file_path']
            # replace "/data/object/{object_name}/" with an empty string to remove it from the path
            new_path =  f'{root_dir}/data/undistorted_images/{SEG_FOLDER_NAME}/{object_name}/'+ current_path.split('/')[-1]  #current_path.replace(f'data/{object_name}/', '')
            # update the "file_path" value with the new path
            frame['file_path'] = new_path


        # save the updated JSON data
        if not os.path.exists(f'data/nerf_cameras/full_scene/{object_name}'):
            os.mkdir(f'data/nerf_cameras/full_scene/{object_name}')

        if not os.path.exists(f'data/nerf_cameras/{SEG_FOLDER_NAME}/{object_name}'):
            os.mkdir(f'data/nerf_cameras/{SEG_FOLDER_NAME}/{object_name}')



        with open(f'data/nerf_cameras/full_scene/{object_name}/transforms.json', 'w') as f:
            json.dump(data, f, indent=2)
        with open(f'data/nerf_cameras/{SEG_FOLDER_NAME}/{object_name}/transforms.json', 'w') as f:
            json.dump(data_obj, f, indent=2)
    elif situation=='2':
        print(f'json file found already for full scene. Copying json file to create camera json for {SEG_FOLDER_NAME} / {object_name}')
        if not os.path.exists(f'data/nerf_cameras/{SEG_FOLDER_NAME}/{object_name}'):
            with open(f'data/nerf_cameras/full_scene/{object_name}/transforms.json') as f:
                data_obj = json.load(f)

            for frame in data_obj['frames']:
                # get the current "file_path" value
                current_path = frame['file_path']
                # replace "/data/object/{object_name}/" with an empty string to remove it from the path
                new_path = f'{root_dir}/data/undistorted_images/{SEG_FOLDER_NAME}/{object_name}/' + current_path.split('/')[-1]   #current_path.replace('data', SEG_FOLDER_NAME)  #current_path.replace(f'data/{object_name}/', '')
                # update the "file_path" value with the new path
                frame['file_path'] = new_path

            os.mkdir(f'data/nerf_cameras/{SEG_FOLDER_NAME}/{object_name}')

            with open(f'data/nerf_cameras/{SEG_FOLDER_NAME}/{object_name}/transforms.json', 'w') as f:
                json.dump(data_obj, f, indent=2)
        



