import os
import pprint
import numpy as np
import pandas as pd
from pathlib import Path
import configuration as cfg


def updateDcmPath(og_df, dcm_folder):

    try:

        # Creat new columns in og_df.
        og_df["full_path"] = np.nan
        og_df["crop_path"] = np.nan
        og_df["mask_path"] = np.nan

        # Get list of .dcm paths.
        dcm_paths_list = []
        for _, _, files in os.walk(dcm_folder):
            for f in files:
                if f.endswith(".dcm"):
                    dcm_paths_list.append(os.path.join(dcm_folder, f))

        for row in og_df.itertuples():

            row_id = row.Index

            # Get identification details.
            patient_id = row.patient_id
            img_view = row.image_view
            lr = row.left_or_right_breast
            abnormality_id = row.abnormality_id

            # Use this list to match DF row with .dcm path.
            info_list = [patient_id, img_view, lr]

            crop_suffix = "CROP_" + str(abnormality_id)
            mask_suffix = "MASK_" + str(abnormality_id)

            # Get list of relevant paths to this patient.
            full_paths = [
                path
                for path in dcm_paths_list
                if all(info in path for info in info_list + ["FULL"])
            ]

            crop_paths = [
                path
                for path in dcm_paths_list
                if all(info in path for info in info_list + [crop_suffix])
            ]

            mask_paths = [
                path
                for path in dcm_paths_list
                if all(info in path for info in info_list + [mask_suffix])
            ]

            # full_paths_str = ",".join(full_paths)
            # crop_paths_str = ",".join(crop_paths)
            # mask_paths_str = ",".join(mask_paths)

            # Update paths.
            if len(full_paths) > 0:
                og_df.loc[row_id, "full_path"] = full_paths
            if len(crop_paths) > 0:
                og_df.loc[row_id, "crop_path"] = crop_paths
            if len(mask_paths) > 0:
                og_df.loc[row_id, "mask_path"] = mask_paths

                    
    except Exception as e:
        print((f"Unable to get updateDcmPath!\n{e}"))
    
    

    return og_df

train = pd.read_csv(cfg.Calc_Train_Description)
test = pd.read_csv(cfg.Calc_Test_Description)

train_path = cfg.train_path
test_path = cfg.test_path

new_cols = [col.replace(" ", "_") for col in train.columns]
train.columns = new_cols   
new_cols = [col.replace(" ", "_") for col in test.columns]
test.columns = new_cols   

updated_train = updateDcmPath(train,train_path)
updated_test = updateDcmPath(test,test_path)

for i in range(1,1546) :
        if type(updated_train['full_path'][i]) == list :
            updated_train['full_path'][i] = updated_train['full_path'][i][0]
        if type(updated_train['crop_path'][i]) == list :
            updated_train['crop_path'][i] = updated_train['crop_path'][i][0]
        if type(updated_train['mask_path'][i]) == list :
            updated_train['mask_path'][i] = updated_train['mask_path'][i][0]   

updated_test['full_path'] = updated_test['full_path'].str.replace('\\', '/')
updated_test['crop_path'] = updated_test['crop_path'].str.replace('\\', '/')
updated_test['mask_path'] = updated_test['mask_path'].str.replace('\\', '/')

updated_train.to_csv('data/Calc/Updated_Calc_Train.csv', index=False)
updated_test.to_csv('data/Calc/Updated_Calc_Test.csv', index=False)