#!/bin/sh


obj="$1"
working_dir="$2"
MODEL_NAME="$3"
#where to output the undistorted ground truth masks and imgs:
if [ -d $working_dir/data/undistorted_images/$MODEL_NAME/JPEGImages/$obj ]; then
    rm -r $working_dir/data/undistorted_images/$MODEL_NAME/JPEGImages/$obj
fi
if [ -d $working_dir/data/undistorted_images/$MODEL_NAME/Annotations/$obj ]; then
    rm -r $working_dir/data/undistorted_images/$MODEL_NAME/Annotations/$obj
fi


colmap_info_folder="data/colmap_info"/$obj

#dont end with "/"
distorted_masked_imgs_loc="data/distorted_images/masked_JPEGImages/$MODEL_NAME/$obj"
distorted_masks_loc="data/distorted_images/Annotations/$MODEL_NAME/$obj"
temp_loc=$working_dir"/"$colmap_info_folder"_temp"

cp -r $colmap_info_folder $temp_loc
echo "copied from $colmap_info_folder to $temp_loc"
rm -r $temp_loc"/input"
rm -r $temp_loc"/images"

cp -r $distorted_masked_imgs_loc $temp_loc"/input"
echo "replaced images with masked images..."

cd SuGaR/gaussian_splatting/
python convert.py -s $temp_loc --skip_matching
cd $working_dir
if [ ! -d $working_dir/data/undistorted_images/$MODEL_NAME ]; then
    mkdir $working_dir/data/undistorted_images/$MODEL_NAME
fi
if [ ! -d $working_dir/data/undistorted_images/$MODEL_NAME/JPEGImages ]; then
    mkdir $working_dir/data/undistorted_images/$MODEL_NAME/JPEGImages
fi
if [ ! -d $working_dir/data/undistorted_images/$MODEL_NAME/Annotations ]; then
    mkdir $working_dir/data/undistorted_images/$MODEL_NAME/Annotations
fi

cp -r $temp_loc"/images" $working_dir/data/undistorted_images/$MODEL_NAME/JPEGImages/$obj

cd $working_dir
rm -r $temp_loc"/input"
rm -r $temp_loc"/images"

cp -r $distorted_masks_loc $temp_loc"/input"
echo "replaced masked images with masks..."

colmap model_converter \
--input_path $temp_loc/distorted/sparse/0 \
--output_path $temp_loc/distorted/sparse/0 \
--output_type TXT

rm $temp_loc"/distorted/sparse/0/cameras.bin"
rm $temp_loc"/distorted/sparse/0/images.bin"
rm $temp_loc"/distorted/sparse/0/points3D.bin"


python scripts/data_preprocessing/rename_for_undistort_masks.py $temp_loc"/distorted/sparse/0/images.txt"

colmap model_converter \
--input_path $temp_loc/distorted/sparse/0 \
--output_path $temp_loc/distorted/sparse/0 \
--output_type BIN


cd SuGaR/gaussian_splatting/
python convert.py -s $temp_loc --skip_matching

echo $working_dir/data/undistorted_images/$MODEL_NAME/Annotations/$obj
echo $working_dir/data/undistorted_images/$MODEL_NAME/Annotations/$obj
echo $working_dir/data/undistorted_images/$MODEL_NAME/Annotations/$obj
echo $working_dir/data/undistorted_images/$MODEL_NAME/Annotations/$obj

cp -r $temp_loc"/images" $working_dir/data/undistorted_images/$MODEL_NAME/Annotations/$obj


rm -r $temp_loc

echo "masks have been undistorted"