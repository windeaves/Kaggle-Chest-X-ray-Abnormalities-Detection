import csv
import os


file_path = r"C:\Users\Alan Li\Documents\GitHub\Kaggle-Chest-X-ray-Abnormalities-Detection\yolov5\data\train_set.csv"
# image_size_path = "IMAGE_SIZE"
save_folder = r"C:\Users\Alan Li\Documents\GitHub\Kaggle-Chest-X-ray-Abnormalities-Detection\yolov5\data\label"
img_size_file = r"C:\Users\Alan Li\Documents\GitHub\Kaggle-Chest-X-ray-Abnormalities-Detection\yolov5\data\train_image_size.csv"

# img_size = open(image_size_path, mode='r')

size_dict = {}
with open(img_size_file, mode='r') as img_size:
    size_reader = csv.reader(img_size, delimiter=',')
    line_count = 0
    for row in size_reader:
        if line_count == 0:
            line_count += 1
            continue
        elif row[0] == "":
            break
        else:
            size_dict[row[0]] = (row[2], row[1]) # (x, y)

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

            if row[1] == "14":
                save_file = open(
                    r"C:\Users\Alan Li\Documents\GitHub\Kaggle-Chest-X-ray-Abnormalities-Detection\yolov5\data\label\\" +
                    row[0] + ".txt", "a")
                save_file.close()
                continue

            save_file.write(row[1] + "\t" + str((int(row[2]) + int(row[4])) / int(size_dict[row[0]][0])) + "\t" + str((int(row[3]) + int(row[5])) / int(size_dict[row[0]][1])) + "\t" + str((int(row[4]) - int(row[2])) / int(size_dict[row[0]][0])) + "\t" + str((int(row[5]) + int(row[3])) / int(size_dict[row[0]][1])) + "\n")
            save_file.close()
            print(row[0] + " saved")

