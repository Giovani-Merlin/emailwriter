import json
from collections import defaultdict

file_path = "data/annotated_enron.jsonl"

grouped_dataset = defaultdict(list)

for line in open(file_path):
    data = json.loads(line)
    text = data["text"]  # data["meta"]["main_content"]
    for label in data["labels"]:
        # if label[2] in ["paragraph","salutation","closing","quo"]
        if label[1] != -1:
            segment = text[label[0] : label[1]]
            print(label[2], segment)
            grouped_dataset[label[2]].append(data)
