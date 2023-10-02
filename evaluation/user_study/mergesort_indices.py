from sorting_techniques import pysort
import csv

comparisons = []

class EqLogger:
    def __init__(self, value):
        self.value = value

    def __lt__(self, other):
        comparisons.append([self.value, other.value])
        

sortObj = pysort.Sorting()
values = list(range(10))
l = [ EqLogger(i) for i in values ]
l_sorted = sortObj.mergeSort(l)

print(comparisons)

# Write comparisons to a CSV file
with open('mergesort_indices.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(comparisons)

print(f'Number of comparisons: {len(comparisons)}') 