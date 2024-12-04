import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

#Variables:
object_name=sys.argv[1]
#SEG_FOLDER_NAME = sys.argv[2]
GAUSSIAN_FILTER = int(sys.argv[2])
MODEL_NAME = sys.argv[3]
#IS_REPROJ_FOLDER = int(sys.argv[3]) #1 or 0
#MODEL_FOLDER = '/work3/s204161/BachelorData/bachelor_data/masks/SAM_GT' #'/work3/s204161/BachelorData/vos_data/datasets/result/resnet101_rpcm_ytb/eval/custom_test/custom_test_resnet101_rpcm_ytb/_ckpt_30000_mem_5_unc_1.0_res_1040.0_wRPA'

input_parent_folder = f"data/distorted_images/JPEGImages/" #'/work3/s204161/BachelorData/bachelor_data/images/JPEGImages'
mask_parent_folder =  f"data/distorted_images/Annotations/{MODEL_NAME}/" #f"/work3/s204161/BachelorData/bachelor_data/masks/MESH_GS_XMEM_MVG_THC0.95_R{str(rou)}_no_norm/Annotations/plant" #fr'{MODEL_FOLDER}/Annotations'
output_parent_folder = f"data/distorted_images/masked_JPEGImages/{MODEL_NAME}/" #f'/work3/s204161/BachelorData/bachelor_data/masks/MESH_GS_XMEM_MVG_THC0.95_R{str(rou)}_no_norm/JPEGImages/' #f'/work3/s204161/BachelorData/bachelor_data/masks/{SEG_FOLDER_NAME}/JPEGImages'
#Mask images using masks placed in masks subdirectory under bachelor_data.
#Variables:

if GAUSSIAN_FILTER:
    DILATION_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE, SIGMAX, SIGMAY =15,15,5,5

    DILATION_KERNEL = (DILATION_KERNEL_SIZE,DILATION_KERNEL_SIZE)

    GAUSSIAN_KERNEL = (GAUSSIAN_KERNEL_SIZE,GAUSSIAN_KERNEL_SIZE)


if GAUSSIAN_FILTER:
    out_dir = f'data/distorted_images/DIL{str(DILATION_KERNEL_SIZE)}_GK{str(GAUSSIAN_KERNEL_SIZE)}_SX{str(SIGMAX)}SY{str(SIGMAY)}'
    output_parent_folder = f'{out_dir}/JPEGImages'
    gaussian_mask_parent_folder = f'{out_dir}/Annotations'

subdirectories = sorted(os.listdir(input_parent_folder))
# Iterate over each subdirectory
for subdir in subdirectories:
    # Set the subdirectory paths for the input, mask, and output folders
    input_folder = os.path.join(input_parent_folder, subdir)
    mask_folder = os.path.join(mask_parent_folder, subdir)
    if not os.path.exists(mask_folder):
        print(f'masks for {subdir} does not exist for XMEM. Skipping')
        continue
    output_folder = os.path.join(output_parent_folder, subdir)

    if GAUSSIAN_FILTER:
        gaussian_mask_output_folder = os.path.join(gaussian_mask_parent_folder, subdir)
        # Create the output folder if it doesn't exist
        if not os.path.exists(gaussian_mask_output_folder):
            os.makedirs(gaussian_mask_output_folder)
        else:
            print('modified mask_output folder already exists. skippping', subdir)
            continue

        
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        print('output_folder already exists. skippping', subdir)
        continue
    
    # Get a list of all the image files in the input folder
    image_files = sorted(os.listdir(input_folder))
    print('doing', subdir)
    # Iterate over each image file in the subdirectory
    count = 0
    for filename in image_files:
        # Load the image and the corresponding mask
        img_path = os.path.join(input_folder, filename)
        mask_path = os.path.join(mask_folder, filename.split('.')[0]+".png")
        img = cv2.imread(img_path) #cv2.COLOR_BGR2RGB)
        #cv2.imwrite(img, scene_data_folder + subdir+ filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        x = img.shape[0]
        y = img.shape[1]

        if count == 0:        
            try:
                mask = cv2.resize(mask, (y,x),interpolation = cv2.INTER_AREA)
                first_y = True
            except: 
                mask = cv2.resize(mask, (x,y),interpolation = cv2.INTER_AREA)
                first_y = False
        else:
            if first_y:
                mask = cv2.resize(mask, (y,x),interpolation = cv2.INTER_AREA)
            else:
                mask = cv2.resize(mask, (x,y),interpolation = cv2.INTER_AREA)

        if not GAUSSIAN_FILTER:
            if mask.dtype != 'uint8':
                mask = mask.astype('uint8')
            result = cv2.bitwise_and(img, img, mask=mask)
        else:

            mask = cv2.dilate(mask, DILATION_KERNEL, iterations=1)
            mask = cv2.GaussianBlur(mask, GAUSSIAN_KERNEL,sigmaX=SIGMAX,sigmaY=SIGMAY)
            gaussian_mask_output_path = os.path.join(gaussian_mask_output_folder, filename.replace('.jpg','.png'))
            cv2.imwrite(gaussian_mask_output_path,mask)
            img = img.astype(np.float64) / 255.
            mask = mask.astype(np.float64) / 255.
            for chann in range(3):
                img[:,:,chann] = img[:,:,chann]*mask

            result = (img * 255).astype(np.uint8)


        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, result)

        count += 1
print('done')