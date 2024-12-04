#!/bin/sh

obj="$1"
working_dir="$2"
MODEL_NAME="$3"
tot_iterations="$4"
num_rounds="$5"

#source $working_dir/env_works/bin/activate

cd SuGaR

elapsed_times=()

for (( i = 1; i <= $num_rounds; i++ )); do

    start_time=$(date +%s.%N)
    python gaussian_splatting/train.py -s $working_dir"/data/colmap_info/"$obj -r 1 -m $working_dir"/data/GS_models"/$MODEL_NAME/$obj/$tot_iterations"_round_"$i --eval --iterations $tot_iterations --test_iterations $tot_iterations --save_iterations $tot_iterations --checkpoint_iterations $tot_iterations --images $working_dir"/data/undistorted_images/"$MODEL_NAME"/JPEGImages/"$obj
    #python train.py -s $working_dir"/data/colmap_info/"$obj -c $working_dir"/data/GS_models"/$MODEL_NAME/$obj/$tot_iterations"_round_"$i"/" -r "sdf" --eval True --export_ply True --i $tot_iterations
    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    elapsed_times+=("$elapsed_time")
    save_file=$working_dir"/data/GS_models"/$MODEL_NAME/$obj/$tot_iterations"_round_"$i"/"time.txt
    echo "Time spent in Python script: $elapsed_time seconds" >> "$save_file"

    python gaussian_splatting/render.py -m $working_dir"/data/GS_models"/$MODEL_NAME/$obj/$tot_iterations"_round_"$i"/" --eval --skip_train
    # mkdir $working_dir"/data/GS_models"/$MODEL_NAME/$obj/$tot_iterations"_round_"$i"/"
    mv $working_dir/SuGaR/output $working_dir"/data/GS_models"/$MODEL_NAME/$obj/$tot_iterations"_round_"$i"/"
done
save_all=$working_dir"/data/GS_models"/$MODEL_NAME/$obj/all_time.txt
echo "Current date and time: $(date)" >> "$log_file"
for indiv_time in $elapsed_times; do
    echo "$indiv_time" >> "$save_all"
done
