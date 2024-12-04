
#before running shell script, make sure to be in h100sh queue for support of updated glibc
#Also run the line: module load colmap/3.8-cuda-11.8-avx512 outside of the shellfile in case of error loading module

working_dir=$BLACKHOLE/camproj
cd $working_dir

module load python3/3.10.12
source $working_dir/env_works/bin/activate
module load cuda/12.1


obj=tree
MODEL_NAME=full_scene #always get full_scene first for a object for doing colmap on it. - do full_scene or XMEM
reproj_masks="XMEM_INT"  # "XMEM_INT" if using interactive masks for reprojection.use "NONE" if not using masks for reprojecting gaussians


#from video to sampled images
#python scripts/data_preprocessing/from_mov_to_img.py $working_dir 400

#SAM for first segmentation mask
scripts/VOS_util.sh $obj $working_dir
if [ ! -d $working_dir/data/distorted_images/ToBeAnnotated/$obj ]; then
    echo sam not ran correctly for $obj
    exit
fi

##XMEM
#copy obj of interest folder to folder where xmem expects
#### cp $SAM_output_obj_of_interest $working_dir/data/distorted_images/Annotations -r --- Don't actually uncomment this line but make sure that distorted_images/SAM_data/object_of_interest is not empty
#make sure obj of interest folder contains the image you want then copy that image to .../distorted_images/Annotations

# cd XMem
# python interactive_demo.py --images ../data/distorted_images/JPEGImages/$obj
# cd $working_dir

if [ -d "$working_dir/XMem/workspace/$obj/masks" ] && [ ! -d "data/distorted_images/Annotations/$reproj_masks/$obj" ] && [ $reproj_masks == "XMEM_INT" ]; then
    if [ $obj == "tree" ]; then
        echo resizing masks from interactive segmentation $obj
        ./scripts/run_resize_masks.sh XMem/workspace/$obj/masks data/distorted_images/Annotations/$reproj_masks/$obj 4000 2250
    else
        echo resizing masks from interactive segmentation $obj
        ./scripts/run_resize_masks.sh XMem/workspace/$obj/masks data/distorted_images/Annotations/$reproj_masks/$obj 1920 1080
    fi
    python scripts/data_preprocessing/mask_out.py $obj "0" $reproj_masks

elif [ $reproj_masks == "XMEM" ]; then
    # Specify the folder you want to check
    folder_path=$working_dir/data/distorted_images/Annotations/XMEM/$obj
    file_count=$(find "$folder_path" -type f | wc -l)

    if [[ -d "$folder_path" ]]; then
        # Count the number of files in the folder
        file_count=$(find "$folder_path" -type f | wc -l)

        # Check if there are at least 2 files
        if [[ $file_count -ge 2 ]]; then
            echo "Skipping XMEM, already ran on $obj"
        else
            echo "Running XMEM on dataset, since not ran for $obj yet"
            cd XMem
            python eval.py --output ../data/distorted_images/Annotations/XMEM --generic_path ../data/distorted_images --dataset G --size 1500
            cd ..
        fi
    else
        echo "Directory does not exist: $folder_path"
        # Optionally, create the directory if you want:
        # mkdir -p "$folder_path"
    fi
    GAUSS_DIL="0"
    python scripts/data_preprocessing/mask_out.py $obj $GAUSS_DIL $reproj_masks
fi

#------------------------------------------------------------------------------------------------------

#mask out using XMEM masks

#do colmap on full scene
#module unload cuda/12.1
#module load colmap/3.8-cuda-11.8-avx512 --- this module was deleted or smfin from hpc
#module load colmap/3.8-cuda-11.7.1-haswell-avx2-python-3.10.7-test1 #use this instead - This was also deleted
#run colmap
if [ $MODEL_NAME == "full_scene" ]; then
    module load colmap/3.8-cuda-11.8-avx512 # this is the only module left 
    chmod +x scripts/COLMAP_util.sh
    scripts/COLMAP_util.sh $obj $working_dir $MODEL_NAME
    # scripts/data_preprocessing/undistort_masks.sh $obj $working_dir $MODEL_NAME
    fi

##undistort masked images and masks

if [ $MODEL_NAME != "full_scene" ] || [ $reproj_masks != "NONE" ]; then
    if [ ! -d "data/undistorted_images/$reproj_masks/Annotations/$obj" ] && [ ! -d "data/undistorted_images/$reproj_masks/JPEGImages/$obj" ]; then
    ./scripts/data_preprocessing/undistort_masks.sh $obj $working_dir $reproj_masks
        fi
    fi

# #module unload colmap/3.8-cuda-11.7.1-haswell-avx2-python-3.10.7-test1
# module unload colmap/3.8-cuda-11.8-avx512

# module load python3/3.10.12
# module load cuda/12.1

# ### only do colmap +  gauss splatt/NeRF + MVg for 1 object at a time bcs they take very long to do.


#instant-ngp or 3dgs+sugar for mesh
cd $working_dir
tot_iterations=60000
num_rounds=1
obj=shoe
thresh_coef=0.9
chmod +x scripts/sugar3dgs.sh
#scripts/3dgs.sh $obj $working_dir $MODEL_NAME $tot_iterations $num_rounds

# obj=plant
# scripts/sugar3dgs.sh $obj $working_dir $MODEL_NAME $tot_iterations $num_rounds

# #run mvg on gaussians and render gaussians
# cd SuGaR/gaussian_splatting
# python mvg.py $obj $thresh_coef $num_rounds
# cd $working_dir

# obj=chess_set
# scripts/sugar3dgs.sh $obj $working_dir $MODEL_NAME $tot_iterations $num_rounds




#mvg on mesh to get new masks
# thresh_holds=("0.5")
# obj=tree
# dilation_size=25
# erosion_size=20
# method="GS"
# chmod +x scripts/MVG.sh
# scripts/MVG.sh $method $obj $working_dir $MODEL_NAME $thresh_holds $num_rounds $dilation_size $erosion_size

if [ $reproj_masks != "NONE" ]; then
    thresh_holds=("0.6")
    obj=tree
    dilation_size=25
    erosion_size=20
    method="GS"
    chmod +x scripts/MVG.sh
    scripts/MVG.sh $method $obj $working_dir $MODEL_NAME $thresh_holds $num_rounds $dilation_size $erosion_size $reproj_masks $tot_iterations
    fi




# obj=shoe
# dilation_size=25
# erosion_size=20
# method="GS"
# chmod +x scripts/MVG.sh
# scripts/MVG.sh $method $obj $working_dir $MODEL_NAME $thresh_holds $num_rounds $dilation_size $erosion_size

# obj=chess_set
# dilation_size=25
# erosion_size=20
# method="GS"
# chmod +x scripts/MVG.sh
# scripts/MVG.sh $method $obj $working_dir $MODEL_NAME $thresh_holds $num_rounds $dilation_size $erosion_size



#render again using instant-ngp or 3dgs



