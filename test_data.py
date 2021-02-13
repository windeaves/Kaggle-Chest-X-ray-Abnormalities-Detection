import os
from tqdm import tqdm

import numpy as np

import multiprocessing as mp

from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_voi_lut

from PIL import Image


def read_dicom(file_path):
    ds = dcmread(file_path)
    return ds


def read_dicom_pixel(file_path):
    return read_dicom(file_path).pixel_array


def dicom_to_np(file_path, voi_lut=True, fix_monochrome=True):
    dicom = read_dicom(file_path)

    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    return data


PATH = "../dataset"
dataset_path = "dataset/images/test"


def save_png(args):
    (folder, offset, batch) = args
    folder_filename = os.listdir(os.path.join(PATH, folder))
    folder_filename = [folder_filename[i] for i in range(offset, len(folder_filename), batch)]

    for filename in tqdm(folder_filename, desc=folder + " " + str(offset + 100)[1:]):
        # for filename in folder_filename:
        try:
            one_obs = dicom_to_np(os.path.join(PATH, folder, filename))
            im = Image.fromarray(one_obs)
            im.save(os.path.join(dataset_path, filename.replace(".dicom", ".jpeg")))
        except Exception as e:
            print(e)


def multi_save_png():
    pool = mp.Pool(mp.cpu_count())
    pool.map(save_png, [("test", x, mp.cpu_count()) for x in range(mp.cpu_count())])


if __name__ == '__main__':
    multi_save_png()
