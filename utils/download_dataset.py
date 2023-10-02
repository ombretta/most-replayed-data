import csv
import argparse
import re
import subprocess
from more_itertools import batched

parser = argparse.ArgumentParser()
parser.add_argument("csv", help="dataset .csv")
parser.add_argument("--out_csv", help="output .csv for videos that were successfully downloaded")
args = parser.parse_args()

def process_row(row, csv_writer):
        video_url = row[0]
        # match = re.search(".*\/(.*)", video_url)
        match = re.search('(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})', video_url)
        video_id = match[1]
        child = subprocess.Popen(["python", "show_most_replayed.py", "--", video_id])
        return child


with open(args.csv) as f:
    with open(args.out_csv, 'w') as out:
        csv_reader = csv.reader(f)
        csv_writer = csv.writer(out)
        for rows in batched(csv_reader, 100):
            children = [ (process_row(row, csv_writer), row) for row in rows ]
            for (child, row) in children:
                rc = child.wait()
                if rc == 0:
                    csv_writer.writerow(row)
                