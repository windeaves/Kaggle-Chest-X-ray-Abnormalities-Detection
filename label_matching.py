import os
import shutil
import tqdm

label_path = "dataset/labels/train"
image_path = "dataset/images/train"

save_path = "dataset/images/trains"

file_names = os.listdir(label_path)

for filename in tqdm.tqdm(file_names):
    shutil.copy(os.path.join(image_path, filename.replace(".txt", ".jpeg")),
                os.path.join(save_path, filename.replace(".txt", ".jpeg")))
