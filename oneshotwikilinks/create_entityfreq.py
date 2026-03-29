import gzip
from collections import Counter

input_file = "shuffled_dedup_entities.tsv"
output_file = "entityfreq.gz"

counter = Counter()

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 3:
            entity = parts[2]
            counter[entity] += 1

with gzip.open(output_file, "wt", encoding="utf-8") as f:
    for entity, count in counter.items():
        # NOTE: count first, then entity
        f.write(f"{count}\t{entity}\n")

print("entityfreq.gz rebuilt correctly.")