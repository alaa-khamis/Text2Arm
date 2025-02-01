import numpy as np
import csv
import random
from collections import defaultdict
from itertools import product

ITEMS_IN_SCENE = {
    'sugar_box' : ['sugar', 'sugar box', 'box of sugar', 'box'],
    'large_clamp': ['clamp', 'large clamp'],
    'tuna_fish_can' : ['tuna fish can', 'tuna can', 'can of tuna', 'tuna', 'little can', 'small can'],
    'master_chef_can' : ['master chef can', 'big can', 'chef can', 'master chef']
}

LOCATIONS = {
    'redBin': ['red bin', 'red trashcan', 'red trash'],
    'blueBin': ['blue bin', 'blue trashcan', 'blue trash'], 
    'yellowBin': ['yellow bin', 'yellow trashcan', 'yellow trash']
}

verbs = ['Move', 'put', 'transfer', 'throw', 'place']
adverbs = ['inside', 'in', 'to']
connectors = ['and', 'then', 'after that']

sample_num = 500
np.random.seed(42)
random.seed(42)

def generate_command():
    item = random.choice(scene_items)
    location = random.choice(scene_locations)
    return (
        f"{random.choice(verbs)} the {random.choice(ITEMS_IN_SCENE[item])} "
        f"{random.choice(adverbs)} the {random.choice(LOCATIONS[location])}",
        (item, location)
    )

scene_items = sorted(list(ITEMS_IN_SCENE))
scene_locations = sorted(list(LOCATIONS.keys()))

dataset = []
counts = defaultdict(int)
used_inputs = set()

def add_to_dataset(cmd, outputs):
    if cmd not in used_inputs:
        used_inputs.add(cmd)
        dataset.append({"input": cmd, "output": outputs})
        for out in outputs:
            counts[out] += 1
        return True
    return False

single_samples = int(sample_num * 0.6)
double_samples = int(sample_num * 0.3)
triple_samples = sample_num - single_samples - double_samples

# Generate single commands (60%)
while len([d for d in dataset if len(d["output"]) == 1]) < single_samples:
    cmd, output = generate_command()
    add_to_dataset(cmd, [output])

# Generate double commands (30%)
while len([d for d in dataset if len(d["output"]) == 2]) < double_samples:
    cmd1, out1 = generate_command()
    cmd2, out2 = generate_command()
    while out2 == out1:
        cmd2, out2 = generate_command()
    
    combined_cmd = f"{cmd1} {random.choice(connectors)} {cmd2}"
    add_to_dataset(combined_cmd, [out1, out2])

# Generate triple commands (10%)
while len([d for d in dataset if len(d["output"]) == 3]) < triple_samples:
    cmd1, out1 = generate_command()
    cmd2, out2 = generate_command()
    cmd3, out3 = generate_command()
    while out2 == out1 or out3 == out1 or out3 == out2:
        cmd2, out2 = generate_command()
        cmd3, out3 = generate_command()
    
    combined_cmd = f"{cmd1} {random.choice(connectors)} {cmd2} {random.choice(connectors)} {cmd3}"
    add_to_dataset(combined_cmd, [out1, out2, out3])

random.shuffle(dataset)

file_path = "finetune_dataset.csv"
with open(file_path, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["input", "output"])
    for data in dataset:
        writer.writerow([data["input"], data["output"]])

# Print distribution stats
print("\nDistribution of outputs:")
total = sum(counts.values())
for item, location in product(scene_items, scene_locations):
    key = (item, location)
    percentage = (counts[key] / total) * 100
    print(f"{key}: {counts[key]} ({percentage:.1f}%)")

# Print command type distribution
singles = len([d for d in dataset if len(d["output"]) == 1])
doubles = len([d for d in dataset if len(d["output"]) == 2])
triples = len([d for d in dataset if len(d["output"]) == 3])
print(f"\nCommand distribution:")
print(f"Singles: {singles} ({singles/len(dataset)*100:.1f}%)")
print(f"Doubles: {doubles} ({doubles/len(dataset)*100:.1f}%)")
print(f"Triples: {triples} ({triples/len(dataset)*100:.1f}%)")

print(f"\nTotal unique commands: {len(dataset)}")