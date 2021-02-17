import os
from typing import Any, Union

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.io.parsers import TextFileReader

from tqdm import tqdm
import multiprocessing as mp
import json

EXP_PATH = "./example_data"

box = lambda x: (x.x_min, x.y_min, x.x_max, x.y_max)

def sub_train_csv():
    training_data = pd.read_csv(os.path.join(EXP_PATH, "train.csv"))
    training_data = training_data[0:40]
    training_data.to_csv(os.path.join(EXP_PATH, "train_sub.csv"), index=False)


def read_train_csv():
    training_data = pd.read_csv(os.path.join(EXP_PATH, "train.csv"))
    return training_data


def extract_image_size():
    from data import read_meta_from_csv
    train_meta, _ = read_meta_from_csv()
    train_size = train_meta[['image_id', 'Rows', 'Columns']]
    train_size.columns = train_size.columns.map({
        "image_id": "image_id",
        "Columns": "x",
        "Rows": "y"
    })
    train_size.to_csv(os.path.join(os.curdir, "train_image_size.csv"))
    return train_size


def calc_IoU(box_a, box_b):
    ixmin = max(box_a[0], box_b[0])
    iymin = max(box_a[1], box_b[1])
    ixmax = min(box_a[2], box_b[2])
    iymax = min(box_a[3], box_b[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)

    # -----1----- intersection
    inters = iw * ih

    # -----2----- union, uni = S1 + S2 - inters
    uni = ((box_a[2] - box_a[0] + 1.) * (box_a[3] - box_a[1] + 1.) +
           (box_b[2] - box_b[0] + 1.) * (box_b[3] - box_b[1] + 1.) -
           inters)

    # -----3----- iou
    overlaps = inters / uni

    return overlaps


def nms_sub(args):
    (rest_data_id_list, rest_data, offset, batch) = args
    comp_data = pd.DataFrame()
    print(len(rest_data_id_list))
    rest_data_id_list = [rest_data_id_list[i] for i in range(offset, len(rest_data_id_list), batch)]
    print(len(rest_data_id_list))
    for nid in tqdm(rest_data_id_list):
        kp = rest_data[rest_data.nid == nid].reset_index(drop = True)

        # comp_data = comp_data.append(kp, ignore_index=True)
        # continue

        nkp = kp.image_id.size
        if nkp == 1:
            kp.pop('nid')
            comp_data = comp_data.append(kp)
            continue
        contain_list = [True] * nkp
        for i in range(nkp - 1):
            if contain_list[i]:
                box_i = box(kp.loc[i])
                for j in range(i + 1, nkp):
                    if contain_list[j]:
                        if calc_IoU(box_i, box(kp.loc[j])) > 0.3:
                            contain_list[j] = False

        kp["exist"] = pd.Series(contain_list)
        kp = kp[kp.exist == True]
        kp.pop("exist")
        kp.pop('nid')
        comp_data = comp_data.append(kp, ignore_index=True)
    return comp_data


if __name__ == "__main__":
    train_data = pd.read_csv(os.path.join(EXP_PATH, "train.csv"))
    train_data = train_data.sort_values("image_id")
    train_data.to_csv(os.path.join(EXP_PATH, "train_sorted.csv"), index=False)

    train_data.pop('class_name')
    train_data.pop('rad_id')

    comp_data = train_data[train_data.class_id == 14]
    rest_data = train_data[train_data.class_id != 14]

    print(comp_data.image_id.unique().size + rest_data.image_id.unique().size)

    rest_data = rest_data.assign(nid=lambda x: x.image_id + ":" + x.class_id.astype("str"))
    rest_data_id_list = rest_data["nid"].unique()

    train_size = extract_image_size()
    train_size = json.loads(train_size.to_json(orient="records"))
    train_size_table = {}
    for one_train_size in train_size:
        train_size_table[one_train_size['image_id']] = {'x':one_train_size['x'], 'y': one_train_size['y']}

    # remove repeating class 14 result
    comp_data = comp_data.reset_index(drop = True)
    comp_data = comp_data.drop_duplicates()

    # carring on nms
    pool = mp.Pool(mp.cpu_count())
    nmsed_datas = pool.map(nms_sub, [(rest_data_id_list, rest_data, offset, mp.cpu_count()) for offset in range(mp.cpu_count())])

    # pool = mp.Pool(1)
    # nmsed_datas = pool.map(nms_sub, [(rest_data_id_list, rest_data, offset, 1) for offset in range(1)])
    # nmsed_datas = [nms_sub((rest_data_id_list, rest_data, 0, 1))]

    cnt = comp_data.image_id.unique().size

    for nmsed_data in nmsed_datas:
        # nmsed_data.apply()
        # cnt += nmsed_data.image_id.unique().size
        # print(nmsed_data.image_id.unique().size)
        comp_data = comp_data.append(nmsed_data, ignore_index=True)

    print(comp_data.image_id.unique().size)


    comp_data.to_csv(os.path.join(EXP_PATH, "train_set.csv"))
    print("complete")