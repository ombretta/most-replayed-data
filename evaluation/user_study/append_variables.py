import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--shuffled_indices")
parser.add_argument("--video_id", type=str)
parser.add_argument("--shuffle_id", type=int)
# parser.add_argument("--clips_dir", required=True, help="Clips directory")
parser.add_argument("--out")
args = parser.parse_args()

# clips_dir = Path(args.clips_dir)

with open(args.out, "a", newline='') as out_f:
    with open(args.shuffled_indices) as f:
        rows = csv.reader(f)
        out = csv.writer(out_f)
        id_to_clip = lambda i: args.video_id + "_clip_" + "{:02d}".format(int(i)) + ".mp4"
        # out_row = [x for x in (a,b) for (a,b) in 
        #     [(id_to_clip(i), id_to_clip(j)) for (i,j) in rows]
        # ]
        out_row = [args.video_id, args.shuffle_id] + [x for (a,b) in [(id_to_clip(i), id_to_clip(j)) for (i, j) in rows] for x in (a,b)]
        out.writerow(out_row)