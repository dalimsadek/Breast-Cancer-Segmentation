import os
import shutil
import pydicom
import pandas as pd
from pathlib import Path

import numpy as np
import random

top = Path("CBIS-DDSM")


def new_name_dcm(dcm_path):


    try:
        ds = pydicom.dcmread(dcm_path)

        patient_id = ds.PatientID
        patient_id = patient_id.replace(".dcm", "")

        try:
            img_type = ds.SeriesDescription

            if "full" in img_type:
                new_name = patient_id + "_FULL" + ".dcm"
                print(f"FULL --- {new_name}")
                return new_name

            elif "crop" in img_type:

                suffix = patient_id.split("_")[-1]

                if suffix.isdigit():
                    new_patient_id = patient_id.split("_" + suffix)[0]
                    new_name = new_patient_id + "_CROP" + "_" + suffix + ".dcm"
                    print(f"CROP --- {new_name}")
                    return new_name

                elif not suffix.isdigit():
                    print(f"CROP ERROR, {patient_id}")
                    return False

            elif "mask" in img_type:

                suffix = patient_id.split("_")[-1]

                if suffix.isdigit():
                    new_patient_id = patient_id.split("_" + suffix)[0]
                    new_name = new_patient_id + "_MASK" + "_" + suffix + ".dcm"
                    print(f"MASK --- {new_name}")
                    return new_name

                elif not suffix.isdigit():
                    print(f"MASK ERROR, {patient_id}")
                    return False

        except:
            if "full" in dcm_path:
                new_name = patient_id + "_FULL" + ".dcm"
                return new_name

            else:
                arr = ds.pixel_array
                unique = np.unique(arr).tolist()

                if len(unique) != 2:

                    suffix = patient_id.split("_")[-1]

                    if suffix.isdigit():
                        new_patient_id = patient_id.split("_" + suffix)[0]
                        new_name = new_patient_id + "_CROP" + "_" + suffix + ".dcm"
                        print(f"CROP --- {new_name}")
                        return new_name

                    elif not suffix.isdigit():
                        print(f"CROP ERROR, {patient_id}")
                        return False

                elif len(unique) == 2:

                    suffix = patient_id.split("_")[-1]

                    if suffix.isdigit():
                        new_patient_id = patient_id.split("_" + suffix)[0]
                        new_name = new_patient_id + "_MASK" + "_" + suffix + ".dcm"
                        print(f"MASK --- {new_name}")
                        return new_name

                    elif not suffix.isdigit():
                        print(f"MASK ERROR, {patient_id}")
                        return False
                

    except Exception as e:
        print((f"Unable to new_name_dcm!\n{e}"))

def move_dcm_up(dest_dir, source_dir, dcm_filename):

    try:
        dest_dir_with_new_name = os.path.join(dest_dir, dcm_filename)

        if not os.path.exists(dest_dir_with_new_name):
            shutil.move(source_dir, dest_dir)

        elif os.path.exists(dest_dir_with_new_name):
            new_name_2 = dcm_filename.strip(".dcm") + "___a.dcm"
            shutil.move(source_dir, os.path.join(dest_dir, new_name_2))

    except Exception as e:
        print((f"Unable to move_dcm_up!\n{e}"))


def delete_empty_folders(top, error_dir):
    
    try:
        curdir_list = []
        files_list = []

        for (curdir, dirs, files) in os.walk(top=top, topdown=False):

            if curdir != str(top):

                dirs.sort()
                files.sort()

                print(f"WE ARE AT: {curdir}")
                print("=" * 10)

                print("List dir:")

                directories_list = [
                    f for f in os.listdir(curdir) if not f.startswith(".")
                ]
                print(directories_list)

                if len(directories_list) == 0:
                    print("DELETE")
                    shutil.rmtree(curdir, ignore_errors=True)

                elif len(directories_list) > 0:
                    print("DON'T DELETE")
                    curdir_list.append(curdir)
                    files_list.append(directories_list)

                print()
                print("Moving one folder up...")
                print("-" * 40)
                print()

        if len(curdir_list) > 0:
            not_empty_df = pd.DataFrame(
                list(zip(curdir_list, files_list)), columns=["curdir", "files"]
            )
            to_save_path = os.path.join(error_dir, "not-empty-folders.csv")
            not_empty_df.to_csv(to_save_path, index=False)

    except Exception as e:
        print((f"Unable to delete_empty_folders!\n{e}"))

def count_dcm(top):

    try:
        count = 0

        for _, _, files in os.walk(top):
            for f in files:
                if f.endswith(".dcm"):
                    count += 1

    except Exception as e:
        print((f"Unable to count_dcm!\n{e}"))

    return count

before = count_dcm(top=top)

for (curdir, dirs, files) in os.walk(top=top, topdown=False):
    dirs.sort()
    files.sort()
    for f in files:
        if f.endswith(".dcm"):
            old_name_path = os.path.join(curdir, f)
            new_name = new_name_dcm(dcm_path=old_name_path)
            if new_name:
                new_name_path = os.path.join(curdir, new_name)
                os.rename(old_name_path, new_name_path)
                move_dcm_up(
                        dest_dir=top, source_dir=new_name_path, dcm_filename=new_name
                    )

delete_empty_folders(top=top, error_dir=top)

after = count_dcm(top=top)

print(f"BEFORE --> Number of .dcm files: {before}")
print(f"AFTER --> Number of .dcm files: {after}")
print()
print("Getting out of extractDicom.")
print("-" * 30)


# This function was used because some of the calc_test files did not get named correctly

def rename_files(directory_path):
    # Get the list of files in the directory
    files = os.listdir(directory_path)

    for filename in files:
        # Get the full path of the file
        old_file_path = os.path.join(directory_path, filename)
        
        # Extract the file extension
        file_extension = os.path.splitext(filename)[1]
        
        # Construct the new file name
        new_filename = "Calc-Test_" + filename
        
        # Create the new file path
        new_file_path = os.path.join(directory_path, new_filename)
        
        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f"Renamed {filename} to {new_filename}")

directory_path = "CBIS-DDSM/Calc/New folder"
rename_files(directory_path)

after = count_dcm(top=top)