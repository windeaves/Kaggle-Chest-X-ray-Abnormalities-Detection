import csv
import os


file_path = r"C:\Users\Alan Li\Documents\GitHub\Kaggle-Chest-X-ray-Abnormalities-Detection\yolov5\data\train_set.csv"
# image_size_path = "IMAGE_SIZE"
save_folder = r"C:\Users\Alan Li\Documents\GitHub\Kaggle-Chest-X-ray-Abnormalities-Detection\yolov5\data\label"

# img_size = open(image_size_path, mode='r')

# read csv file
with open(file_path, mode='r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
            continue
        elif row[0] == "":
            break
        else:
            # create txt
            save_file = open(r"C:\Users\Alan Li\Documents\GitHub\Kaggle-Chest-X-ray-Abnormalities-Detection\yolov5\data\label\\" + row[0] + ".txt", "a")

            save_file.write(row[1] + "\t" + row[6] + "\t" + row[7] + "\t" + row[8] + "\t" + row[9] + "\n")
            save_file.close()
            print(row[0] + " saved")

