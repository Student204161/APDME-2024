


obj="$1"
working_dir="$2"
MODEL_NAME="$3"
IS_REPROJ_FOLDER="0"
UNDISTORTED="1"
if [ ! -d $working_dir/data/colmap_info ]; then
    mkdir $working_dir/data/colmap_info
fi

if [ ! -d $working_dir/data/undistorted_images/ ]; then
    mkdir $working_dir/data/undistorted_images/
fi
if [ ! -d $working_dir/data/undistorted_images/full_scene ]; then
    mkdir $working_dir/data/undistorted_images/full_scene
fi
if [ ! -d $working_dir/data/undistorted_images/full_scene/JPEGImages ]; then
    mkdir $working_dir/data/undistorted_images/full_scene/JPEGImages
fi

if [ ! -d $working_dir/data/nerf_cameras/full_scene ]; then
    mkdir $working_dir/data/nerf_cameras/full_scene
fi

if [ ! -d $working_dir/data/colmap_info/$obj ]; then
    zenity --info --text="Beware that since this is the first time running this scene, camera parameter estimation will colmap has to be ran. \nThis can take anything from minutes to hours depending on the scene..." --title="object extraction of 3D gaussian splatting model"
    mkdir $working_dir/data/colmap_info/$obj
    cp data/distorted_images/JPEGImages/$obj data/colmap_info/$obj/input -r
    cd SuGaR/gaussian_splatting
    python convert.py -s ../../data/colmap_info/$obj
    cd $working_dir
    cp data/colmap_info/$obj/images data/undistorted_images/full_scene/JPEGImages/$obj -r
    colmap model_converter \
    --input_path $working_dir/data/colmap_info/$obj/sparse/0 \
    --output_path $working_dir/data/colmap_info/$obj/sparse/0 \
    --output_type TXT

    # if [ ! -d $working_dir/data/undistorted_images/full_scene/JPEGImages/$obj ]; then
    #     mkdir $working_dir/data/undistorted_images/full_scene/JPEGImages/$obj
    #     cp $working_dir/data/colmap_info/$obj/images $working_dir/data/undistorted_images/full_scene/JPEGImages/$obj
    # fi
      
    python /zhome/a7/0/155527/Desktop/Bachelor-Project/instant-ngp/scripts/colmap2nerf.py --aabb_scale 16 --overwrite --images "data/undistorted_images/full_scene/JPEGImages/"$obj --out "data/nerf_cameras/full_scene/$obj/transforms.json" --text "data/colmap_info/"$obj"/sparse/0/"
    python scripts/data_preprocessing/modify_json.py $obj $MODEL_NAME "1" $UNDISTORTED $working_dir
    python scripts/data_preprocessing/make_val_json.py $obj "full_scene" $IS_REPROJ_FOLDER
    python scripts/data_preprocessing/make_val_json.py $obj $MODEL_NAME $IS_REPROJ_FOLDER

else
    echo "skipping colmap on $obj"
    if  [ ! -d $working_dir/data/nerf_cameras/full_scene/$obj ]; then
        echo "nerf camera folder doesnt exist"
        python /zhome/a7/0/155527/Desktop/Bachelor-Project/instant-ngp/scripts/colmap2nerf.py --aabb_scale 16 --overwrite --images "data/undistorted_images/full_scene/JPEGImages/"$obj --out "data/nerf_cameras/full_scene/$obj/transforms.json" --text "data/colmap_info/"$obj"/sparse/0/"
        python scripts/data_preprocessing/modify_json.py $obj $MODEL_NAME "1" $UNDISTORTED $working_dir
        python scripts/data_preprocessing/make_val_json.py $obj "full_scene" $IS_REPROJ_FOLDER
        python scripts/data_preprocessing/make_val_json.py $obj $MODEL_NAME $IS_REPROJ_FOLDER
    else
    python scripts/data_preprocessing/modify_json.py $obj $MODEL_NAME "2" $UNDISTORTED $working_dir
    python scripts/data_preprocessing/make_val_json.py $obj "full_scene" $IS_REPROJ_FOLDER
    python scripts/data_preprocessing/make_val_json.py $obj $MODEL_NAME $IS_REPROJ_FOLDER
    fi
fi

