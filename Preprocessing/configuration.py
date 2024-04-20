
Calc_Train_Description = "..\Csv_files\calc_case_description_train_set.csv"
Calc_Test_Description = "..\Csv_files\calc_case_description_test_set.csv"
train_path = "data\Calc\Train\ALL DCM"
test_path = "data\Calc\Test\ALL DCM"

config_massTrainPre = {
    "paths": {
        "input": "data\Mass\Train\ALL",
        "output_full": "data\Mass\Train\FULL",
        "output_mask": "data\Mass\Train\MASK"
    },
    "output_format": ".png",
    "cropBorders": {
        "l": 0.01,
        "r": 0.01,
        "u": 0.04,
        "d": 0.04
    },
    "globalBinarise": {
        "thresh": 0.1,
        "maxval": 1.0
    },
    "editMask": {
        "ksize": 23,
        "operation": "open"
    },
    "sortContourByArea": {
        "reverse": True
    },
    "xLargestBlobs": {
        "top_x": 1
    },
    "clahe": {
        "clip": 2.0,
        "tile": 8
    }
}


config_massTestPre = {
    "paths": {
        "input": "data\Mass\Test\ALL",
        "output_full": "data\Mass\Test\FULL",
        "output_mask": "data\Mass\Test\MASK"
    },
    "output_format": ".png",
    "cropBorders": {
        "l": 0.01,
        "r": 0.01,
        "u": 0.04,
        "d": 0.04
    },
    "globalBinarise": {
        "thresh": 0.1,
        "maxval": 1.0
    },
    "editMask": {
        "ksize": 23,
        "operation": "open"
    },
    "sortContourByArea": {
        "reverse": True
    },
    "xLargestBlobs": {
        "top_x": 1
    },
    "clahe": {
        "clip": 2.0,
        "tile": 8
    }
}


config_calcTrainPre = {
    "paths": {
        "input": "data\Calc\Train\ALL",
        "output_full": "data\Calc\Train\FULL",
        "output_mask": "data\Calc\Train\MASK"
    },
    "output_format": ".png",
    "cropBorders": {
        "l": 0.01,
        "r": 0.01,
        "u": 0.04,
        "d": 0.04
    },
    "globalBinarise": {
        "thresh": 0.1,
        "maxval": 1.0
    },
    "editMask": {
        "ksize": 23,
        "operation": "open"
    },
    "sortContourByArea": {
        "reverse": True
    },
    "xLargestBlobs": {
        "top_x": 1
    },
    "clahe": {
        "clip": 2.0,
        "tile": 8
    }
}

config_calcTestPre = {
    "paths": {
        "input": "data\Calc\Test\ALL",
        "output_full": "data\Calc\Test\FULL",
        "output_mask": "data\Calc\Test\MASK"
    },
    "output_format": ".png",
    "cropBorders": {
        "l": 0.01,
        "r": 0.01,
        "u": 0.04,
        "d": 0.04
    },
    "globalBinarise": {
        "thresh": 0.1,
        "maxval": 1.0
    },
    "editMask": {
        "ksize": 23,
        "operation": "open"
    },
    "sortContourByArea": {
        "reverse": True
    },
    "xLargestBlobs": {
        "top_x": 1
    },
    "clahe": {
        "clip": 2.0,
        "tile": 8
    }
}


config_mmt = {
    "paths": {
        "images": "data\\Calc\\Test\\ALL PNG",
        "csv": "data\\Calc\\Updated_Calc_Test.csv",
        "output": "data\\Calc\\Test\\Summed MASKS"
    },
    "abnormality_col": "abnormality_id",
    "extension": ".png",
    "save": True
}