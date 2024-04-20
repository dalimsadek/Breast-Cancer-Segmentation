import os
import pprint
import numpy as np
import pandas as pd
import cv2
import shutil
import configuration as cfg


def findMultiTumour(csv_path, abnormality_col):
    try:
        df = pd.read_csv(csv_path, header=0)

        multi_df = df.loc[df[abnormality_col] > 1]

        multi_tumour_list = []
        for row in multi_df.itertuples():

            patient_id = row.patient_id
            lr = row.left_or_right_breast
            img_view = row.image_view

            identifier = "_".join([patient_id, lr, img_view])
            multi_tumour_list.append(identifier)

        multi_tumour_set = set(multi_tumour_list)

    except Exception as e:
        print((f"Unable to findMultiTumour!\n{e}"))

    return multi_tumour_set

def masksToSum(img_path, multi_tumour_set, extension):

    try:
        images = [
            f
            for f in os.listdir(img_path)
            if (not f.startswith(".") and f.endswith(extension))
        ]

        masks_to_sum = [
            m
            for m in images
            if ("MASK" in m and any(multi in m for multi in multi_tumour_set))
        ]

        masks_to_sum_dict = {patient_id: [] for patient_id in multi_tumour_set}

        for k, _ in masks_to_sum_dict.items():
            v = [os.path.join(img_path, m) for m in masks_to_sum if k in m]
            masks_to_sum_dict[k] = sorted(v)

        to_pop = [k for k, v in masks_to_sum_dict.items() if len(v) == 1]

        for k in to_pop:
            masks_to_sum_dict.pop(k)

    except Exception as e:
        print((f"Unable to get findMultiTumour!\n{e}"))

    return masks_to_sum_dict


def sumMasks(mask_list):

    summed_mask = np.zeros(mask_list[0].shape)

    for arr in mask_list:
        summed_mask = np.add(summed_mask, arr)

    _, summed_mask_bw = cv2.threshold(
        src=summed_mask, thresh=1, maxval=255, type=cv2.THRESH_BINARY
    )

    return summed_mask_bw

img_path = cfg.config_mmt["paths"]["images"]
csv_path = cfg.config_mmt["paths"]["csv"]
abnormality_col = cfg.config_mmt["abnormality_col"]
extension = cfg.config_mmt["extension"]
output_path = cfg.config_mmt["paths"]["output"]
save = cfg.config_mmt["save"]

multi_tumour_set = findMultiTumour(
        csv_path=csv_path, abnormality_col=abnormality_col
    )


masks_to_sum_dict = masksToSum(
        img_path=img_path, multi_tumour_set=multi_tumour_set, extension=extension
    )

sum_list = []
for k, v in masks_to_sum_dict.items():

        mask_list = [cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) for mask_path in v]
        
        print("summing", k, "masks")
        
        sum_list.append(k)
        
        reference_shape = mask_list[0].shape
        for i in range(len(mask_list)):
            mask = mask_list[i]
            reshaped_mask = cv2.resize(mask, reference_shape[::-1], interpolation=cv2.INTER_NEAREST)
            mask_list[i] = reshaped_mask
            
        summed_mask = sumMasks(mask_list=mask_list)

        if save:
            if "train" in img_path.lower():
                save_path = os.path.join(
                    output_path, ("_".join(["Calc-Training", k, "MASK___PRE.png"]))
                )
            elif "test" in img_path.lower():
                save_path = os.path.join(
                    output_path, ("_".join(["Calc-Test", k, "MASK___PRE.png"]))
                )
            cv2.imwrite(save_path, summed_mask)

destination_folder = "data\\Calc\\Test\\ALL MASKS"
files = os.listdir("data\\Calc\\Test\\MASK")

for file in files:

    file_name = os.path.basename(file)
    split_parts = file_name.split("_")
    s = split_parts[1]+"_"+split_parts[2]+"_"+split_parts[3]+"_"+split_parts[4]
    if s not in sum_list:
        source_file = "data\\Calc\\Test\\MASK\\"+file_name
        shutil.copy(source_file, destination_folder)
        print(file_name)