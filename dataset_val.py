import os
import random

train_dataset_path = "dataset/images/train"
val_dataset_path = "dataset/images/val"

train_label_path = "dataset/labels/train"
train_val_path = "dataset/labels/val"

if __name__ == '__main__':
    filenames = os.listdir(train_label_path)

    val_filenames = random.sample(filenames, int(len(filenames) / 5))
    val_filenames = [x.replace(".txt", "") for x in val_filenames]

    for filename in val_filenames:
        os.rename(os.path.join(train_dataset_path, filename + ".jpeg"), os.path.join(val_dataset_path, filename + ".jpeg"))
        os.rename(os.path.join(train_dataset_path, filename + ".txt"), os.path.join(val_dataset_path, filename + ".txt"))