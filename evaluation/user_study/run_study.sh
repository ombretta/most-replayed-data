# python mergesort_indices.py

clips_dir="./clips/"
# Generate study_id as a UUID (universally unique identifier)
study_id=$(uuidgen)
mkdir -p "./studies/$study_id"
echo "Working directory:"
echo "./studies/$study_id"

python pick_random_videos.py --out "./studies/$study_id/random_videos.csv" -n 1 # NB needs splits json

# Read video_ids from random_videos.csv
video_ids=$(cat ./studies/$study_id/random_videos.csv)

variables_csv="./studies/$study_id/variables.csv"
# truncate -s0 ${variables_csv}
header="video_id"
# num_shuffles=15
# for i in $(seq 0 ${num_shuffles}); do
#   header+="mindex${i}_left,mindex${i}_right"
#   if ((i < num_shuffles)); then
#     header+=","
#   fi
# done
echo $header > ${variables_csv}

# Process each video_id
for video_id in $video_ids; do
    video_id=${video_id%%[[:space:]]}  # Remove trailing whitespace
    echo $video_id
    python generate_clips.py --out_dir "${clips_dir}" --write_speedrun --write_clips --write_control_clip --video_id="${video_id}"
    echo -e "${video_id}" >> ${variables_csv}

    # Loop from 0 to 15, pad $i to 2 digits
    # for i in $(seq -w 0 ${num_shuffles}); do
    #     python shuffle_indices.py --out "./studies/$study_id/shuffled_${video_id}_$i.csv"
    #     python append_variables.py --shuffled_indices "./studies/$study_id/shuffled_${video_id}_$i.csv" --video_id "${video_id}" --shuffle_id $i --out ${variables_csv}
    # done
done