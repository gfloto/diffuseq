import pdb
import json

inputs = []
with open('/ibex/scratch/tangz0a/workspace/DiffuSeq/datasets/detox/test_toxic_parallel.txt') as f:
    for line in f:
        inputs.append(line.strip())

outputs = []
with open('/ibex/scratch/tangz0a/workspace/DiffuSeq/datasets/detox/test_toxic_parallel_refs.txt') as f:
    for line in f:
        outputs.append(line.strip())

assert len(inputs) == len(outputs)

with open('/ibex/scratch/tangz0a/workspace/DiffuSeq/datasets/detox/test.jsonl', 'w') as f:
    for i in range(len(inputs)):
        line_dict = {"trg": outputs[i], "src": inputs[i]}
        json.dump(line_dict, f)
        f.write('\n')

train_list = []
with open('/ibex/scratch/tangz0a/workspace/DiffuSeq/datasets/detox/train.tsv') as f:
    counter  = 0
    for line in f:
        counter += 1
        if counter > 1:
            line = line.strip().split('\t')
            if len(line) == 2:
                train_list.append([line[1], line[0]])

with open('/ibex/scratch/tangz0a/workspace/DiffuSeq/datasets/detox/train.jsonl', 'w') as f:
    for i in range(len(train_list) - 1000):
        line_dict = {"trg": train_list[i][0], "src": train_list[i][1]}
        json.dump(line_dict, f)
        f.write('\n')

with open('/ibex/scratch/tangz0a/workspace/DiffuSeq/datasets/detox/valid.jsonl', 'w') as f:
    for i in range(len(train_list) - 1000, len(train_list)):
        line_dict = {"trg": train_list[i][0], "src": train_list[i][1]}
        json.dump(line_dict, f)
        f.write('\n')

