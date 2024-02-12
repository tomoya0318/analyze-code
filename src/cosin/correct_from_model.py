import pandas as pd
import numpy as np
from utils import lookup_white_list
from constants import path

def __insert_key_if_none(v1, v2):
    """比較対象先に特定のキーが含まれていない場合，値を0として追加し，更新された比較先の辞書の値だけをリストとして返す．

    Args:
        v1 (dict): 比較先の辞書
        v2 (dict): 比較元の辞書

    Returns:
        list: 比較先の辞書に含まれる値のリスト
    """
    for key in v2:
        if key not in v1:
            v1[key] = 0
    
    sorted_v1_keys = sorted(v1.keys())
    sorted_v1_values = [v1[key] for key in sorted_v1_keys]

    return sorted_v1_values
        
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

def calc_cosin(v1, v2):
    v1_out = __insert_key_if_none(v1, v2)
    v2_out = __insert_key_if_none(v2, v1)

    return np.dot(v1_out, v2_out) / (np.linalg.norm(v1_out) * np.linalg.norm(v2_out))
    