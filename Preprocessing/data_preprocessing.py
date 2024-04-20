import os
import pprint
import numpy as np
import cv2
import pydicom
from pathlib import Path
import configuration as cfg

def cropBorders(img, l=0.01, r=0.01, u=0.04, d=0.04):
    
    nrows, ncols = img.shape
    
    l_crop = int(ncols * l)
    r_crop = int(ncols * (1 - r))
    u_crop = int(nrows * u)
    d_crop = int(nrows * (1 - d))
    
    cropped_img = img[u_crop:d_crop, l_crop:r_crop]
    
    return cropped_img

def minMaxNormalise(img):
    
    norm_img = (img - img.min()) / (img.max() - img.min())
    
    return norm_img


def globalBinarise(img, thresh, maxval) :
    
    binarised_img = np.zeros(img.shape, np.uint8)
    binarised_img[img >= thresh] = maxval
    
    return binarised_img

def editMask(mask, ksize=(23, 23), operation="open"):
    
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize = ksize)
    
    if operation == "open":
        edited_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    elif operation == "close":
        edited_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    edited_mask = cv2.morphologyEx(edited_mask, cv2.MORPH_DILATE, kernel)

    return edited_mask

def sortContoursByArea(contours, reverse=True):

    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=reverse)

    bounding_boxes = [cv2.boundingRect(c) for c in sorted_contours]

    return sorted_contours, bounding_boxes


def xLargestBlobs(mask, top_x=None, reverse=True):
    
    contours, hierarchy = cv2.findContours(
        image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE
    )
    
    n_contours = len(contours)
    
    if n_contours > 0:
        
        if n_contours < top_x or top_x == None:
            top_x = n_contours

        sorted_contours, bounding_boxes = sortContoursByArea(
            contours=contours, reverse=reverse
        )

        X_largest_contours = sorted_contours[0:top_x]

        to_draw_on = np.zeros(mask.shape, np.uint8)

        X_largest_blobs = cv2.drawContours(
            image=to_draw_on,  
            contours=X_largest_contours,  
            contourIdx=-1,
            color=1,  
            thickness=-1, 
        )

    return n_contours, X_largest_blobs


def applyMask(img, mask):

    masked_img = img.copy()
    masked_img[mask == 0] = 0

    return masked_img

def checkLRFlip(mask):

    nrows, ncols = mask.shape
    x_center = ncols // 2
    y_center = nrows // 2

    col_sum = mask.sum(axis=0)
    row_sum = mask.sum(axis=1)

    left_sum = sum(col_sum[0:x_center])
    right_sum = sum(col_sum[x_center:-1])

    if left_sum < right_sum:
        LR_flip = True
    else:
        LR_flip = False

    return LR_flip

def makeLRFlip(img):

    flipped_img = np.fliplr(img)

    return flipped_img

def clahe(img, clip=2.0, tile=(8, 8)):

    img = cv2.normalize(
        img,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )
    img_uint8 = img.astype("uint8")

    clahe_create = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    clahe_img = clahe_create.apply(img_uint8)

    return clahe_img

def pad(img):

    nrows, ncols = img.shape

    if nrows != ncols:

        if ncols < nrows:
            
            target_shape = (nrows, nrows)
        
        elif nrows < ncols:
            
            target_shape = (ncols, ncols)

        padded_img = np.zeros(shape=target_shape)
        padded_img[:nrows, :ncols] = img

    elif nrows == ncols:

        padded_img = img

    return padded_img

def fullMammoPreprocess(
    img,
    l,
    r,
    d,
    u,
    thresh,
    maxval,
    ksize,
    operation,
    reverse,
    top_x,
    clip,
    tile,
):

    # Step 1: Initial crop.
    cropped_img = cropBorders(img=img, l=l, r=r, d=d, u=u)

    # Step 2: Min-max normalise.
    norm_img = minMaxNormalise(img=cropped_img)

    # Step 3: Remove artefacts.
    binarised_img = globalBinarise(img=norm_img, thresh=thresh, maxval=maxval)
    edited_mask = editMask(
        mask=binarised_img, ksize=(ksize, ksize), operation=operation
    )
    _, xlargest_mask = xLargestBlobs(mask=edited_mask, top_x=top_x, reverse=reverse)

    masked_img = applyMask(img=norm_img, mask=xlargest_mask)

    # Step 4: Horizontal flip.
    lr_flip = checkLRFlip(mask=xlargest_mask)
    if lr_flip:
        flipped_img = makeLRFlip(img=masked_img)
    elif not lr_flip:
        flipped_img = masked_img

    # Step 5: CLAHE enhancement.
    clahe_img = clahe(img=flipped_img, clip=clip, tile=(tile, tile))

    # Step 6: pad.
    padded_img = pad(img=clahe_img)
    padded_img = cv2.normalize(
        padded_img,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )

    # Step 7: Min-max normalise.
    img_pre = minMaxNormalise(img=padded_img)

    return img_pre, lr_flip

def maskPreprocess(mask, lr_flip):

    # Step 1: Initial crop.
    mask = cropBorders(img=mask)

    # Step 2: Horizontal flip.
    if lr_flip:
        mask = makeLRFlip(img=mask)

    # Step 3: Pad.
    mask_pre = pad(img=mask)

    return mask_pre


def sumMasks(mask_list):

    summed_mask = np.zeros(mask_list[0].shape)

    for arr in mask_list:
        summed_mask = np.add(summed_mask, arr)

    _, summed_mask_bw = cv2.threshold(
        src=summed_mask, thresh=1, maxval=255, type=cv2.THRESH_BINARY
    )

    return summed_mask_bw


input_path = cfg.config_massTrainPre["paths"]["input"]
output_full_path = cfg.config_massTrainPre["paths"]["output_full"]
output_mask_path = cfg.config_massTrainPre["paths"]["output_mask"]

output_format = cfg.config_massTrainPre["output_format"]

dcm_paths = []
for curdir, dirs, files in os.walk(input_path):
    files.sort()
    for f in files:
        if f.endswith(".dcm"):
            dcm_paths.append(os.path.join(curdir, f))

fullmamm_paths = [f for f in dcm_paths if ("FULL" in f)]
mask_paths = [f for f in dcm_paths if ("MASK" in f)]

count = 0
for fullmamm_path in fullmamm_paths:
    ds = pydicom.dcmread(fullmamm_path)
    
    patient_id = ds.PatientID
    
    patient_id = patient_id.replace(".dcm", "")
    patient_id = patient_id.replace("Mass-Training_", "")

    fullmamm = ds.pixel_array

    l = cfg.config_massTrainPre["cropBorders"]["l"]
    r = cfg.config_massTrainPre["cropBorders"]["r"]
    u = cfg.config_massTrainPre["cropBorders"]["u"]
    d = cfg.config_massTrainPre["cropBorders"]["d"]
    thresh = cfg.config_massTrainPre["globalBinarise"]["thresh"]
    maxval = cfg.config_massTrainPre["globalBinarise"]["maxval"]
    ksize = cfg.config_massTrainPre["editMask"]["ksize"]
    operation = cfg.config_massTrainPre["editMask"]["operation"]
    reverse = cfg.config_massTrainPre["sortContourByArea"]["reverse"]
    top_x = cfg.config_massTrainPre["xLargestBlobs"]["top_x"]
    clip = cfg.config_massTrainPre["clahe"]["clip"]
    tile = cfg.config_massTrainPre["clahe"]["tile"]

    fullmamm_pre, lr_flip = fullMammoPreprocess(
            img=fullmamm,
            l=l,
            r=r,
            u=u,
            d=d,
            thresh=thresh,
            maxval=maxval,
            ksize=ksize,
            operation=operation,
            reverse=reverse,
            top_x=top_x,
            clip=clip,
            tile=tile,
        )

    fullmamm_pre_norm = cv2.normalize(
            fullmamm_pre,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )

    save_filename = (
            os.path.basename(fullmamm_path).replace(".dcm", "")
            + "_PRE"
            + output_format
        )
    save_path = os.path.join(output_full_path, save_filename)
    cv2.imwrite(save_path, fullmamm_pre_norm)
    print(f"DONE FULL: {fullmamm_path}")

    mask_path = [mp for mp in mask_paths if patient_id in mp]

    for mp in mask_path:
        
        mask_ds = pydicom.dcmread(mp)
        mask = mask_ds.pixel_array
        
        mask_pre = maskPreprocess(mask=mask, lr_flip=lr_flip)

        save_filename = (
                os.path.basename(mp).replace(".dcm", "") + "___PRE" + output_format
            )
        save_path = os.path.join(output_mask_path, save_filename)
        cv2.imwrite(save_path, mask_pre)

        print(f"DONE MASK: {mp}")

    count += 1


input_path = cfg.config_massTestPre["paths"]["input"]
output_full_path = cfg.config_massTestPre["paths"]["output_full"]
output_mask_path = cfg.config_massTestPre["paths"]["output_mask"]

output_format = cfg.config_massTestPre["output_format"]

dcm_paths = []
for curdir, dirs, files in os.walk(input_path):
    files.sort()
    for f in files:
        if f.endswith(".dcm"):
            dcm_paths.append(os.path.join(curdir, f))
            
fullmamm_paths = [f for f in dcm_paths if ("FULL" in f)]
mask_paths = [f for f in dcm_paths if ("MASK" in f)]

count = 0
for fullmamm_path in fullmamm_paths:
    ds = pydicom.dcmread(fullmamm_path)
    
    patient_id = ds.PatientID
    
    patient_id = patient_id.replace(".dcm", "")
    patient_id = patient_id.replace("Mass-Test_", "")

    fullmamm = ds.pixel_array

    l = cfg.config_massTestPre["cropBorders"]["l"]
    r = cfg.config_massTestPre["cropBorders"]["r"]
    u = cfg.config_massTestPre["cropBorders"]["u"]
    d = cfg.config_massTestPre["cropBorders"]["d"]
    thresh = cfg.config_massTestPre["globalBinarise"]["thresh"]
    maxval = cfg.config_massTestPre["globalBinarise"]["maxval"]
    ksize = cfg.config_massTestPre["editMask"]["ksize"]
    operation = cfg.config_massTestPre["editMask"]["operation"]
    reverse = cfg.config_massTestPre["sortContourByArea"]["reverse"]
    top_x = cfg.config_massTestPre["xLargestBlobs"]["top_x"]
    clip = cfg.config_massTestPre["clahe"]["clip"]
    tile = cfg.config_massTestPre["clahe"]["tile"]

    fullmamm_pre, lr_flip = fullMammoPreprocess(
            img=fullmamm,
            l=l,
            r=r,
            u=u,
            d=d,
            thresh=thresh,
            maxval=maxval,
            ksize=ksize,
            operation=operation,
            reverse=reverse,
            top_x=top_x,
            clip=clip,
            tile=tile,
        )

    fullmamm_pre_norm = cv2.normalize(
            fullmamm_pre,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )

    save_filename = (
            os.path.basename(fullmamm_path).replace(".dcm", "")
            + "_PRE"
            + output_format
        )
    save_path = os.path.join(output_full_path, save_filename)
    cv2.imwrite(save_path, fullmamm_pre_norm)
    print(f"DONE FULL: {fullmamm_path}")

    mask_path = [mp for mp in mask_paths if patient_id in mp]

    for mp in mask_path:
        
        mask_ds = pydicom.dcmread(mp)
        mask = mask_ds.pixel_array
        
        mask_pre = maskPreprocess(mask=mask, lr_flip=lr_flip)

        save_filename = (
                os.path.basename(mp).replace(".dcm", "") + "___PRE" + output_format
            )
        save_path = os.path.join(output_mask_path, save_filename)
        cv2.imwrite(save_path, mask_pre)

        print(f"DONE MASK: {mp}")

    count += 1


input_path = cfg.config_calcTrainPre["paths"]["input"]
output_full_path = cfg.config_calcTrainPre["paths"]["output_full"]
output_mask_path = cfg.config_calcTrainPre["paths"]["output_mask"]

output_format = cfg.config_calcTrainPre["output_format"]

dcm_paths = []
for curdir, dirs, files in os.walk(input_path):
    files.sort()
    for f in files:
        if f.endswith(".dcm"):
            dcm_paths.append(os.path.join(curdir, f))
            
fullmamm_paths = [f for f in dcm_paths if ("FULL" in f)]
mask_paths = [f for f in dcm_paths if ("MASK" in f)]

count = 0
for fullmamm_path in fullmamm_paths:
    ds = pydicom.dcmread(fullmamm_path)
    
    patient_id = ds.PatientID
    
    patient_id = patient_id.replace(".dcm", "")
    patient_id = patient_id.replace("Calc-Training_", "")

    fullmamm = ds.pixel_array

    l = cfg.config_calcTrainPre["cropBorders"]["l"]
    r = cfg.config_calcTrainPre["cropBorders"]["r"]
    u = cfg.config_calcTrainPre["cropBorders"]["u"]
    d = cfg.config_calcTrainPre["cropBorders"]["d"]
    thresh = cfg.config_calcTrainPre["globalBinarise"]["thresh"]
    maxval = cfg.config_calcTrainPre["globalBinarise"]["maxval"]
    ksize = cfg.config_calcTrainPre["editMask"]["ksize"]
    operation = cfg.config_calcTrainPre["editMask"]["operation"]
    reverse = cfg.config_calcTrainPre["sortContourByArea"]["reverse"]
    top_x = cfg.config_calcTrainPre["xLargestBlobs"]["top_x"]
    clip = cfg.config_calcTrainPre["clahe"]["clip"]
    tile = cfg.config_calcTrainPre["clahe"]["tile"]

    fullmamm_pre, lr_flip = fullMammoPreprocess(
            img=fullmamm,
            l=l,
            r=r,
            u=u,
            d=d,
            thresh=thresh,
            maxval=maxval,
            ksize=ksize,
            operation=operation,
            reverse=reverse,
            top_x=top_x,
            clip=clip,
            tile=tile,
        )

    fullmamm_pre_norm = cv2.normalize(
            fullmamm_pre,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )

    save_filename = (
            os.path.basename(fullmamm_path).replace(".dcm", "")
            + "_PRE"
            + output_format
        )
    save_path = os.path.join(output_full_path, save_filename)
    cv2.imwrite(save_path, fullmamm_pre_norm)
    print(f"DONE FULL: {fullmamm_path}")

    mask_path = [mp for mp in mask_paths if patient_id in mp]

    for mp in mask_path:
        
        mask_ds = pydicom.dcmread(mp)
        mask = mask_ds.pixel_array
        
        mask_pre = maskPreprocess(mask=mask, lr_flip=lr_flip)

        save_filename = (
                os.path.basename(mp).replace(".dcm", "") + "___PRE" + output_format
            )
        save_path = os.path.join(output_mask_path, save_filename)
        cv2.imwrite(save_path, mask_pre)

        print(f"DONE MASK: {mp}")

    count += 1


input_path = cfg.config_calcTestPre["paths"]["input"]
output_full_path = cfg.config_calcTestPre["paths"]["output_full"]
output_mask_path = cfg.config_calcTestPre["paths"]["output_mask"]

output_format = cfg.config_calcTestPre["output_format"]

dcm_paths = []
for curdir, dirs, files in os.walk(input_path):
    files.sort()
    for f in files:
        if f.endswith(".dcm"):
            dcm_paths.append(os.path.join(curdir, f))
            
fullmamm_paths = [f for f in dcm_paths if ("FULL" in f)]
mask_paths = [f for f in dcm_paths if ("MASK" in f)]

count = 0
for fullmamm_path in fullmamm_paths:
    ds = pydicom.dcmread(fullmamm_path)
    
    patient_id = ds.PatientID
    
    patient_id = patient_id.replace(".dcm", "")
    patient_id = patient_id.replace("Calc-Test_", "")

    fullmamm = ds.pixel_array

    l = cfg.config_calcTestPre["cropBorders"]["l"]
    r = cfg.config_calcTestPre["cropBorders"]["r"]
    u = cfg.config_calcTestPre["cropBorders"]["u"]
    d = cfg.config_calcTestPre["cropBorders"]["d"]
    thresh = cfg.config_calcTestPre["globalBinarise"]["thresh"]
    maxval = cfg.config_calcTestPre["globalBinarise"]["maxval"]
    ksize = cfg.config_calcTestPre["editMask"]["ksize"]
    operation = cfg.config_calcTestPre["editMask"]["operation"]
    reverse = cfg.config_calcTestPre["sortContourByArea"]["reverse"]
    top_x = cfg.config_calcTestPre["xLargestBlobs"]["top_x"]
    clip = cfg.config_calcTestPre["clahe"]["clip"]
    tile = cfg.config_calcTestPre["clahe"]["tile"]

    fullmamm_pre, lr_flip = fullMammoPreprocess(
            img=fullmamm,
            l=l,
            r=r,
            u=u,
            d=d,
            thresh=thresh,
            maxval=maxval,
            ksize=ksize,
            operation=operation,
            reverse=reverse,
            top_x=top_x,
            clip=clip,
            tile=tile,
        )

    fullmamm_pre_norm = cv2.normalize(
            fullmamm_pre,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )

    save_filename = (
            os.path.basename(fullmamm_path).replace(".dcm", "")
            + "_PRE"
            + output_format
        )
    save_path = os.path.join(output_full_path, save_filename)
    cv2.imwrite(save_path, fullmamm_pre_norm)
    print(f"DONE FULL: {fullmamm_path}")

    mask_path = [mp for mp in mask_paths if patient_id in mp]

    for mp in mask_path:
        
        mask_ds = pydicom.dcmread(mp)
        mask = mask_ds.pixel_array
        
        mask_pre = maskPreprocess(mask=mask, lr_flip=lr_flip)

        save_filename = (
                os.path.basename(mp).replace(".dcm", "") + "___PRE" + output_format
            )
        save_path = os.path.join(output_mask_path, save_filename)
        cv2.imwrite(save_path, mask_pre)

        print(f"DONE MASK: {mp}")

    count += 1