working_dir=$BLACKHOLE/camproj
cd $working_dir
#module load colmap/3.8-cuda-11.8-avx512
module load python3/3.10.12
source $working_dir/env_gl/bin/activate
module load cuda/12.1
#python scripts/data_preprocessing/from_mov_to_img.py $working_dir 300
#json_object_name=$(python ./scripts/pipeline_gui.py)
#obj_name=$(echo "$json_object_name" | jq -r '.obj_name')
obj_name="person"
# filename="/dtu/blackhole/07/155527/camproj/data/movs/dishwash_soap.MOV"
# obj_name=$(echo "$filename" | awk -F'/' '{split($NF, a, "."); print a[1]}')

reproj_masks=XMEM_INT
MODEL_NAME="full_scene"
num_rounds=1
tot_iterations=30000
model_path=$working_dir/data/GS_models/full_scene/$obj_name/$tot_iterations"_round_"$num_rounds

if [ $MODEL_NAME == "full_scene" ]; then
    if [ ! -d $working_dir/data/colmap_info/$obj_name ]; then
        #module load colmap/3.8-cuda-11.8-avx512 # this is the only module left 
        chmod +x scripts/COLMAP_util.sh
        scripts/COLMAP_util.sh $obj_name $working_dir $MODEL_NAME
    else
        echo "skipping colmap as it has already ran"
    fi
    # scripts/data_preprocessing/undistort_masks.sh $obj $working_dir $MODEL_NAME
    fi
if [ ! -d $model_path ]; then
    echo $model_path
    cd $BLACKHOLE/camproj
    chmod +x ./scripts/sugar3dgs.sh 
    scripts/3dgs.sh $obj_name $working_dir $MODEL_NAME $tot_iterations $num_rounds
else 
    echo "skipping model training as it has already ran"
fi

#default vals
thresh_holds=("0.8")
rmv_sz=0.5
rmv_dist=3
dilation_size=25 #not important 
erosion_size=20 #not important
method="GS"
segment=1
white_background=1
# Infinite loop
while true; do

    if [ "$segment" == 3 ] && [ "$reproj_masks" == "XMEM_INT" ]; then
        cd XMem
        python interactive_demo.py --images ../data/distorted_images/JPEGImages/$obj_name
        cd $working_dir
        first_jpg=$(ls "../data/distorted_images/JPEGImages/$obj_name"/*.jpg 2>/dev/null | head -n 1)
        dimensions=$(identify -format "%wx%h" $first_jpg)
        width=$(echo $dimensions | cut -d'x' -f1)
        height=$(echo $dimensions | cut -d'x' -f2)

        ./scripts/run_resize_masks.sh XMem/workspace/$obj_name/masks data/distorted_images/Annotations/$reproj_masks/$obj_name 1080 1920 #$width $height #3118 2078 #4946 3286 #4000 2250 #1920 1080 
        python scripts/data_preprocessing/mask_out.py $obj_name "0" $reproj_masks
        ./scripts/data_preprocessing/undistort_masks_int.sh $obj_name $working_dir $reproj_masks

        #delete reprojection hashmaps/dicts in cache if interactive viewer has been opened to change masks.
        # rm -r data/cache/$reproj_masks/$obj_name/$tot_iterations/*
    elif [ "$segment" == 3 ] && [ "$reproj_masks" == "XMEM" ]; then
        # Specify the folder you want to check
        folder_path=$working_dir/data/distorted_images/Annotations/XMEM/$obj_name
        file_count=$(find "$folder_path" -type f | wc -l)

        if [[ -d "$folder_path" ]]; then
            # Count the number of files in the folder
            file_count=$(find "$folder_path" -type f | wc -l)

            # Check if there are at least 2 files
            if [[ $file_count -ge 2 ]]; then
                echo "Skipping XMEM, already ran on $obj_name"
            else
                echo "Running XMEM on dataset, since not ran for $obj_name yet"
                cd XMem
                python eval.py --output ../data/distorted_images/Annotations/XMEM --generic_path ../data/distorted_images --dataset G --size 1500
                cd ..
                GAUSS_DIL="0"   
                python scripts/data_preprocessing/mask_out.py $obj_name $GAUSS_DIL $reproj_masks
                ./scripts/data_preprocessing/undistort_masks.sh $obj_name $working_dir $reproj_masks
            fi
        fi
    fi
    chmod +x scripts/MVG.sh
    #scripts/MVG.sh $method $obj_name $working_dir $MODEL_NAME $thresh_holds $num_rounds $dilation_size $erosion_size $reproj_masks $tot_iterations $rmv_sz $rmv_dist $white_background
    json_output=$(python scripts/pipeline_gui_2.py $obj_name $thresh_holds $rmv_sz $rmv_dist $reproj_masks $white_background)

    obj_name=$(echo "$json_output" | jq -r '.obj_name')
    thresh_holds=$(echo "$json_output" | jq -r '.cam_thresh')
    rmv_sz=$(echo "$json_output" | jq -r '.scale_thresh')
    rmv_dist=$(echo "$json_output" | jq -r '.rmv_dist')
    current_image_index=$(echo "$json_output" | jq -r '.current_image_index')
    breakout=$(echo "$json_output" | jq -r '.breakout')
    segment=$(echo "$json_output" | jq -r '.segment')
    white_background=$(echo "$json_output" | jq -r '.white_background')
    # Check if the output from script2 is "STOP"
    if [ "$breakout" == 1 ]; then
        echo "Received breakout signal. Exiting loop."
        break
    fi
    break

done


