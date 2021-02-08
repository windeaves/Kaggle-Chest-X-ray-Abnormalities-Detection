import csv


file_path = "READ_PATH"
image_size_path = "IMAGE_SIZE"
save_folder = "SAVE_PATH"

img_size = open(image_size_path, mode='r')

# read csv file
with open(file_path, mode='r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
            continue
        else:
            # create txt
            save_file = open(save_folder + row[0] + ".txt", "a")

            # load img size
            for img_row in img_size:
                if img_row[0] == row[0]:
                    img_height = img_row[1]
                    img_width = img_row[2]
                else:
                    raise Exception('No img size found')

            # process labels
            x_center = (int(row[6]) + int(row[4])) / 2 / img_width
            y_center = (int(row[7]) - int(row[5])) / 2 / img_height
            height = (int(row[7]) - int(row[5])) / img_height
            width = (int(row[6]) - int(row[4])) / img_width
            save_file.write(row[2] + "\t" + str(x_center) + "\t" + str(y_center) + "\t" + str(width) + "\t" + str(height))
            save_file.close()

img_size.close()
