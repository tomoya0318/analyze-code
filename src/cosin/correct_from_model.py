import pandas as pd
from utils import lookup_white_list
from utils import cos_sim
from constants import path

def lookup_from_model(csvfile):
    convention_dict = {}
    total_convention = 0
    fixed_convention = 0
    df = pd.read_csv(csvfile)
    warning_id = df["Warning ID"].tolist()
    convention_exist = df["Target"].tolist()

    for i in range(len(warning_id)):
        if convention_exist[i] == 1:
            fixed_convention += 1

        total_convention += 1
        convention_dict[warning_id[i]] = [fixed_convention, total_convention]

    # coding_convention_distの値をパーセントに変更(有効数字2桁)
    for key, value in convention_dict.items():
        if value[1] == 0:
            continue
        convention_dict[key] = round(value[0] / value[1], 2)

    return dict(sorted(convention_dict.items()))
