import os

train_labels = os.listdir('dataset/train')
print(len(train_labels))
val_labels = os.listdir('dataset/validation')
print(len(train_labels))

import shutil

for val_label in val_labels:
    if val_label not in train_labels:
        shutil.rmtree(os.path.join('dataset/validation', val_label))

val_labels = os.listdir('dataset/validation')
print(len(val_labels))

for train_label in train_labels:
    if train_label not in val_labels:
        print(train_label)
        os.makedirs(os.path.join('dataset/validation', train_label), exist_ok=True)

val_labels = os.listdir('dataset/validation')
print(len(val_labels))