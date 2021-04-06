import json


file_name = r"submission_alan.txt"
save_name = r"submit.json"

with open(file_name) as output_file:
    data = {}
    for line in output_file:
        res_list = line.split()
        if len(res_list) <= 6:
            continue
        img_name = res_list[0]
        first_class = img_name[-1]
        res_list.insert(1, first_class)
        img_name = img_name[:-2]
        img_data = []

        for i in range((len(res_list)-1)//6):
            class_id = res_list[i*6+1]
            conf = res_list[i*6+2]
            x_min = res_list[i*6+3]
            y_min = res_list[i*6+4]
            x_max = res_list[i*6+5]
            y_max = res_list[i*6+6]

            img_data.append({'cls': int(class_id) + 1, 'conf': conf, 'bbox': [x_min, x_max, y_min, y_max]})

        data[img_name] = img_data

with open(save_name, 'w') as save_file:
    json.dump(data, save_file)
