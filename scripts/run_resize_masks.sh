#run this after using the interactive viewer in case masks were downsized from 4k

#!/bin/bash

# run_resize.sh
# Usage: ./run_resize.sh <input_folder> <output_folder> <width> <height>

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <input_folder> <output_folder> <width> <height>"
    exit 1
fi

input_folder=$1
output_folder=$2
width=$3
height=$4

python3 ./scripts/resize_mask.py "$input_folder" "$output_folder" "$width" "$height"
