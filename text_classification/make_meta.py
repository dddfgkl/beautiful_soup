import os
import json
import numpy as np
import pandas as pd

"""
    Author: macoong.ucaser@gmail.com  
    FistEdit: 2020.04.22 00:35  
    LastEdit:  
"""

def construct_meta(train_csv, valid_csv, meta_pos):

    df_train = pd.read_csv(train_csv, delimiter=",", encoding="utf-8")
    df_test = pd.read_csv(valid_csv, delimiter=",", encoding="utf-8")
    class_num = 2

    meta = {}
    meta["task"] = "classification"
    # meta["language"] = "ZH"
    meta["language"] = "EN"
    meta["train_num"] = len(df_train)
    meta["valid_num"] = len(df_test)
    meta["class_num"] = class_num

    print()
    print(meta)

    # write meta
    with open(os.path.join(meta_pos, "meta.json"), 'w') as outfile:
        outfile.write('{}\n'.format(json.dumps(meta, indent=4)))

    print("construct meta over")