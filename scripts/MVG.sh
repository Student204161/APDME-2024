#!/bin/sh


elapsed_times=()
method="$1"
obj="$2"
working_dir="$3"
MODEL_NAME="$4"
thresh_holds="$5"
num_rounds="$6"
dilation_size="$7"
erosion_size="$8"
reproj_masks="$9"
tot_iterations="${10}"
rmv_sz="${11}"
rmv_dist="${12}"
white_background="${13}"

cd $working_dir

list_thresh_holds=$(IFS=, ; echo "${thresh_holds[*]}")

for (( i = 1; i <= $num_rounds; i++ )); do
    start_time=$(date +%s.%N)
    if [ $method == "GS" ]; then
        echo "doing reprojection with Gaussian splatting"
        cd SuGaR/gaussian_splatting
        python mvg.py $obj ${thresh_holds[0]} $i $dilation_size $erosion_size $MODEL_NAME $reproj_masks $tot_iterations $rmv_sz $rmv_dist $white_background
        cd $working_dir
        #python scripts/MVG_fin.py $method $obj $list_thresh_holds $dilation_size $erosion_size $i $working_dir
    else
        python scripts/MVG_fin.py $method $obj $list_thresh_holds $dilation_size $erosion_size $i $working_dir
    fi
    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    elapsed_times+=("$elapsed_time")
    echo "Time spent in Python script: $elapsed_time seconds"
done

# Calculate mean and standard deviation
total_time=0
for time in "${elapsed_times[@]}"; do
    total_time=$(echo "$total_time + $time" | bc)
done
mean=$(echo "scale=4; $total_time / $iterations" | bc)
sum_squared_diff=0
for time in "${elapsed_times[@]}"; do
    diff=$(echo "$time - $mean" | bc)
    squared_diff=$(echo "$diff * $diff" | bc)
    sum_squared_diff=$(echo "$sum_squared_diff + $squared_diff" | bc)
done
variance=$(echo "scale=4; $sum_squared_diff / ($iterations - 1)" | bc)
standard_deviation=$(echo "scale=4; sqrt($variance)" | bc)

# Calculate confidence interval (95% confidence)
confidence_interval=$(echo "scale=4; 1.96 * $standard_deviation / sqrt($iterations)" | bc)
lower_bound=$(echo "$mean - $confidence_interval" | bc)
upper_bound=$(echo "$mean + $confidence_interval" | bc)

save_file=$working_dir/data/$method"_models"/$MODEL_NAME/$obj/time_MVG.txt
printf "%s\n" "${elapsed_times[@]}" > $save_file
echo "Mean time: $mean seconds" >> $save_file
echo "Standard deviation: $standard_deviation" >> $save_file
echo "95% Confidence Interval: ($lower_bound, $upper_bound)" >> $save_file

