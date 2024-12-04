#!/bin/bash

# Number of iterations
iterations=7

# Array to store elapsed times
elapsed_times=()

# Run the Python script multiple times
for ((i=1; i<=$iterations; i++)); do
    echo "Iteration $i"
    start_time=$(date +%s.%N)
    python dummy.py
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

printf "%s\n" "${elapsed_times[@]}" > results.txt
echo "Mean time: $mean seconds" >> results.txt
echo "Standard deviation: $standard_deviation" >> results.txt
echo "95% Confidence Interval: ($lower_bound, $upper_bound)" >> results.txt