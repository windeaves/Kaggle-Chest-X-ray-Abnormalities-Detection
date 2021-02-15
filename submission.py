import os

from data import read_meta

from tqdm import tqdm

dataset_detect_path = "../dataset/test"

label_path = "yolov5/runs/detect/exp11/labels"

result_save_path = "submission/"

if __name__ == '__main__':
    detect_files = os.listdir(dataset_detect_path)

    with open(os.path.join(result_save_path, "submission4.txt"), 'a') as f:
        f.write('image_id,PredictionString\n')
        for file in tqdm(detect_files):
            try:
                meta = read_meta(os.path.join(dataset_detect_path, file))
                x_size = int(meta["(0028, 0010)"]) # Rows
                y_size = int(meta["(0028, 0011)"]) # Columns
            except Exception(e):
                print(e)
            file = file.replace(".dicom", "")
            result_list = []
            if os.path.exists(os.path.join(label_path, file+".txt")):
                with open(os.path.join(label_path, file+".txt")) as ff:
                    result_list = ff.readlines()
                result_list = [[float(y) for y in x.split(' ')] for x in result_list]
                result_list = [[x[0], x[1] - x[3] / 2, x[2] - x[4] / 2, x[1] + x[3] / 2,  x[2] + x[4]/2, x[5]] for x in result_list]
                result_list = [[x[0], x[5], x[1] * x_size, x[2] * y_size, x[3] * x_size, x[4] * y_size] for x in result_list]
                result_lists_temp = [[int(y) for y in x] for x in result_list]
                for i in range(len(result_lists_temp)):
                    result_lists_temp[i][1] = result_list[i][1]
                result_list = result_lists_temp

                f.write(file)
                f.write(',')
                for result in result_list:
                    for x in result:
                        f.write(str(x))
                        f.write(' ')
                f.write('\n')
            else:
                f.write(file+',14 1 0 0 1 1\n')

