import csv
from random import shuffle

data = []
with open('模板.csv', 'r', encoding='utf8') as f:
    for i, row in enumerate(csv.reader(f)):
        if i != 0:
            data.append(row[0] + '\t' + str([int(w) if w else 0 for w in row[1:]]))

shuffle(data)
shuffle(data)
shuffle(data)

LEN = len(data)
print(LEN)

trains = data[int(LEN / 5):]
valids = data[:int(LEN / 5)]

with open('./multilabel/train.txt', 'w', encoding='utf8') as f:
    for train in trains:
        f.write(train + '\n')

with open('./multilabel/dev.txt', 'w', encoding='utf8') as f:
    for valid in valids:
        f.write(valid + '\n')

with open('./multilabel/test.txt', 'w', encoding='utf8') as f:
    for valid in valids:
        f.write(valid + '\n')
