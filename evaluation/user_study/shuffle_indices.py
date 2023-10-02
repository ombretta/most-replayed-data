import csv
import random
import argparse

comparisons = []

parser = argparse.ArgumentParser(description='Output unique shuffle of mergesort_indices to a CSV file.')
parser.add_argument('--indices', type=str, default='mergesort_indices.csv', help='the CSV file to read')
parser.add_argument('-o', '--out', type=str, default='shuffled_indices.csv', help='output CSV file')
args = parser.parse_args()

with open(args.indices, mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
        comparisons.append([int(i) for i in row])

print(f'Number of comparisons: {len(comparisons)}')

# Shuffle the rows of comparisons
random.shuffle(comparisons)

# Shuffle each entry in the rows
for i in range(len(comparisons)):
    random.shuffle(comparisons[i])

# Write shuffled comparisons to a CSV file
with open(args.out, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(comparisons)