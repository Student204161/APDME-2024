
#before running shell script, make sure to be in h100sh queue for support of updated glibc
obj="$1"
working_dir="$2"


#SAM for first segmentation mask
images_dir=$working_dir/data/distorted_images/ToBeAnnotated/
SAM_output_folder=$working_dir/data/distorted_images/SAM_data
SAM_output_all_objs=$SAM_output_folder/all_objs
SAM_output_obj_of_interest=$SAM_output_folder/"obj_of_interest"

if [ ! -d $SAM_output_obj_of_interest ]; then
    mkdir $SAM_output_folder
    mkdir $SAM_output_obj_of_interest
fi

for dir in "$images_dir"/*/; do
    if [ -d "$dir" ]; then
        # execute shell command only if directory and if there isn't already a transforms.json folder at train_nerf folder:
        obj_name=$(basename "$dir")
        obj_out_path="$SAM_output_all_objs/$obj_name" 
        if [ ! -d $obj_out_path ]; then
            mkdir -p $obj_out_path
            echo "SAM segmenting on object name: $obj_name"
            #echo "with full output path as $obj_out_path"
            python segment-anything/scripts/amg.py --checkpoint $working_dir/segment-anything/sam_vit_h_4b8939.pth --model-type vit_h --input $images_dir/$obj_name --output $SAM_output_all_objs/$obj_name --min-mask-region-area 5000

            read -p "Choose which object to keep track of. path to mask containing obj of interest e.g. (0000/3.png or 0002/1.png) or manually insert from /data/SAM_data/all_objs/ to ./data/SAM_DATA/obj_of_interest. Only choose from next 2 frames, if SAM really can't find obj of interest: " obj_inter
            mkdir $SAM_output_obj_of_interest/$obj_name
            cp $obj_out_path/$obj_inter $SAM_output_obj_of_interest/$obj_name/0000.png
        else
        echo "Skipping segmentation on" $obj_name "as it already exists"
        fi
    fi
done

#XMEM
# copy obj of interest folder to folder where xmem expects
# cp $SAM_output_obj_of_interest $working_dir/data/distorted_images/Annotations -r
# cd XMem
# python eval.py --output ../data/distorted_images/Annotations --generic_path ../data/distorted_images --dataset G --size -1
# cd ..
#------------------------------------------------------------------------------------------------------
