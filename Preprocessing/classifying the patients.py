import pandas as pd
import os
import shutil

df = pd.read_csv("DATA++\\Calc\\Updated_Calc_Test.csv")


folder1 = "DATA++\\Calc\\Test++\\BENIGN"
folder2 = "DATA++\\Calc\\Test++\\MALIGNANT"
folder3 = "DATA++\\Calc\\Test++\\BENIGN_WITHOUT_CALLBACK"


files = os.listdir("DATA++\\Calc\\Test")

for file in files:

    file_name = os.path.basename(file)
    split_parts = file_name.split("_")
    s = split_parts[1]+"_"+split_parts[2]
    if df.loc[df['patient_id'] == s, 'pathology'].values[0] == "BENIGN" :
        source_file = "DATA++\\Calc\\Test\\"+file_name
        shutil.copy(source_file, folder1)
        print(file_name,"is BENIGN")
        
    elif df.loc[df['patient_id'] == s, 'pathology'].values[0] == "MALIGNANT" :
        source_file = "DATA++\\Calc\\Test\\"+file_name
        shutil.copy(source_file, folder2)
        print(file_name,"is MALIGNANT")
        
    elif df.loc[df['patient_id'] == s, 'pathology'].values[0] == "BENIGN_WITHOUT_CALLBACK" :
        source_file = "DATA++\\Calc\\Test\\"+file_name
        shutil.copy(source_file, folder3)
        print(file_name,"is BENIGN_WITHOUT_CALLBACK")