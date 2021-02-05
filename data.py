import os

import numpy as np
import pandas as pd

from tqdm import tqdm
import multiprocessing as mp

from pydicom import dcmread

# dataset path
PATH = "../dataset"

mapper_dict = {'image_id': 'image_id',
               '(0002, 0000)': "File Meta Information Group Length",
               '(0002, 0001)': "File Meta Information Version",
               '(0002, 0002)': "Media Storage SOP Class UID",
               '(0002, 0003)': "Media Storage SOP Instance UID",
               '(0002, 0010)': "Transfer Syntax UID",
               '(0002, 0012)': "Implementation Class UID",
               '(0002, 0013)': "Implementation Version Name",
               '(0002, 0016)': "Source Application Entity Title",
               '(0010, 0040)': "Patient's Sex",
               '(0010, 1010)': "Patient's Age",
               '(0010, 1020)': "Patient's Size",
               '(0010, 1030)': "Patient's Weight",
               '(0028, 0002)': "Samples per Pixel",
               '(0028, 0004)': "Photometric Interpretation",
               '(0028, 0008)': "Number of Frames",
               '(0028, 0010)': "Rows",
               '(0028, 0011)': "Columns",
               '(0028, 0030)': "Pixel Spacing",
               '(0028, 0034)': "Pixel Aspect Ratio",
               '(0028, 0100)': "Bits Allocated",
               '(0028, 0101)': "Bits Stored",
               '(0028, 0102)': "High Bit",
               '(0028, 0103)': "Pixel Representation",
               '(0028, 0106)': "Smallest Image Pixel Value",
               '(0028, 0107)': "Largest Image Pixel Value",
               '(0028, 1050)': "Window Center",
               '(0028, 1051)': "Window Width",
               '(0028, 1052)': "Rescale Intercept",
               '(0028, 1053)': "Rescale Slope",
               '(0028, 2110)': "Lossy Image Compression",
               '(0028, 2112)': "Lossy Image Compression Ratio",
               '(0028, 2114)': "Lossy Image Compression Method"
               }


def read_meta(file_path):
    ds = dcmread(file_path)
    image_id = os.path.basename(file_path)

    obs_dict = {}
    obs_dict['image_id'] = image_id.split(sep='.')[0]

    file_meta_keys = list(ds.file_meta.keys())
    other_meta_keys = list(ds.keys())

    for key in file_meta_keys:
        obs_dict[str(key)] = str(ds.file_meta[key].value)

    for key in other_meta_keys:
        if key != (0x7fe0, 0x0010):
            obs_dict[str(key)] = str(ds[key].value)

    return obs_dict


def extract_meta_information(args):
    (folder, offset, batch) = args
    folder_filename = os.listdir(os.path.join(PATH, folder))
    folder_filename = [folder_filename[i] for i in range(offset, len(folder_filename), batch)]
    one_obs = read_meta(os.path.join(PATH, folder, folder_filename[0]))
    metadata = pd.DataFrame(columns=one_obs.keys())

    for filename in tqdm(folder_filename, desc=folder + " " + str(offset+100)[1:]):
    # for filename in folder_filename:
        try:
            one_obs = read_meta(os.path.join(PATH, folder, filename))
            metadata = metadata.append(one_obs, ignore_index=True)
        except Exception as e:
            print(e)
    return metadata


def multi_extract_meta_information():
    return (extract_train_meta(),
    extract_test_meta())
    

def extract_train_meta():
    pool = mp.Pool(mp.cpu_count())
    train_meta_data = pool.map(extract_meta_information, [("train", x, mp.cpu_count()) for x in range(mp.cpu_count())])
    train_metadata = pd.DataFrame(columns=train_meta_data[0].columns)
    for meta_data in train_meta_data:
        train_metadata = train_metadata.append(meta_data, ignore_index=True)
    train_metadata.columns = train_metadata.columns.map(mapper_dict)
    train_metadata.to_csv(f"./train_dicom_metadata.csv", index=False)
    pool.close()
    return train_meta_data

def extract_test_meta():
    pool = mp.Pool(mp.cpu_count())
    test_meta_data = pool.map(extract_meta_information, [("test", x, mp.cpu_count()) for x in range(mp.cpu_count())])
    test_metadata = pd.DataFrame(columns=test_meta_data[0].columns)
    for meta_data in test_meta_data:
        test_metadata = test_metadata.append(meta_data, ignore_index=True)
    test_metadata.columns = test_metadata.columns.map(mapper_dict)
    test_metadata.to_csv(f"./test_dicom_metadata.csv", index=False)
    pool.close()
    return test_metadata


def read_meta_from_csv():
    train_metadata = pd.read_csv(os.path.join(os.curdir, "train_dicom_metadata.csv"))
    test_metadata = pd.read_csv(os.path.join(os.curdir, "test_dicom_metadata.csv"))
    return train_metadata, test_metadata


def read_dicom(file_path):
    ds = dcmread(file_path)
    image_id = os.path.basename(file_path).split(sep='.')[0]
    return ds


def read_dicom_pixel(file_path):
    return read_dicom(file_path).pixel_array


if __name__ == "__main__":
    multi_extract_meta_information()
