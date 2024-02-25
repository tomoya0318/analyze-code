import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import lookup_white_list
from constants import path
def main():
    project_list = lookup_white_list(path.WHITELIST)
    df = pd.DataFrame()

    for i, name1 in tqdm(enumerate(project_list), total=len(project_list)):
        for name2 in tqdm((project_list[i + 1:]), desc=f"comparing with {name1}", leave=False):
            name1_dict = lookup_from_model(f"{path.PRERESULT}/raw_model/{name1}_train.csv")
            name2_dict = lookup_from_model(f"{path.PRERESULT}/raw_model/{name2}_train.csv")

            cosin_sim = calc_cosin(name1_dict, name2_dict)
            nan_count = count_both_none(name1_dict, name2_dict, border=0.1)
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "Project 1": [name1],
                            "fix_count1": [sum(item[0] for item in name1_dict.values())],
                            "total_count1": [sum(item[1] for item in name1_dict.values())],
                            "average1": [(sum(item[0] for item in name1_dict.values()) / sum(item[1] for item in name1_dict.values()))],
                            "Project 2": [name2],
                            "fix_count2": [sum(item[0] for item in name2_dict.values())],
                            "total_count2": [sum(item[1] for item in name2_dict.values())],
                            "average2": [(sum(item[0] for item in name2_dict.values()) / sum(item[1] for item in name2_dict.values()))],
                            "cos_sim": [cosin_sim],
                            "not_nan": [nan_count],
                        }
                    ),
                ],
                ignore_index=True,
            )
    
    df.to_csv(f"{path.OUT}/all_cosin_similarity.csv", encoding='utf-8', index=False)

def __insert_key_if_none(v1, v2):
    """比較対象先に特定のキーが含まれていない場合，値をNoneとして追加し，更新された比較先の辞書の値だけをリストとして返す．

    Args:
        v1 (dict): 比較先の辞書
        v2 (dict): 比較元の辞書

    Returns:
        list: 比較先の辞書に含まれる値のリスト
    """
    for key in v2:
        if key not in v1:
            v1[key] = None
    
    return [v1[key] for key in sorted(v1)]


def lookup_from_model(csvfile):
    """csvファイルから規約の修正率を取得

    Args:
        csvfile (csv): 規約IDと修正可否が含まれているcsvファイル

    Returns:
        dict: {Warning ID, [fixed_convention, total_convention]}という辞書型で返す
    """
    df = pd.read_csv(csvfile)
    convention_dict = {}

    for warning_id in df["Warning ID"].unique():
        warning_data = df[df["Warning ID"] == warning_id]
        fixed_convention = warning_data["Target"].sum()
        total_convention = warning_data["Target"].count()
        convention_dict[warning_id] = [fixed_convention, total_convention]
    return dict(sorted(convention_dict.items()))

def calc_cosin(v1, v2):
    """2つのプロジェクトの規約修正率のコサイン類似度の測定

    Args:
        v1 (dict): 1つ目のプロジェクトの規約IDと修正数，発生数が含まれている辞書
        v2 (dict): 2つ目のプロジェクトの規約IDと修正数，発生数が含まれている辞書

    Returns:
        float: 2つのプロジェクトの規約修正率のコサイン類似度
    """

    v1 = {key: values[0] / values[1] if values[1] != 0 else 0 for key, values in v1.items()}
    v2 = {key: values[0] / values[1] if values[1] != 0 else 0 for key, values in v2.items()}
    
    v1_nan_0 = [0 if pd.isna(value) else value for value in __insert_key_if_none(v1, v2)]
    v2_nan_0 = [0 if pd.isna(value) else value for value in __insert_key_if_none(v2, v1)]

    cosin_sim = format(np.dot(v1_nan_0, v2_nan_0) / (np.linalg.norm(v1_nan_0) * np.linalg.norm(v2_nan_0)), ".2f")

    return cosin_sim
    
def count_values_exceeding_limit(v1, v2 = None, border = 0.2):
    """両方のプロジェクトに共通してボーダー以上の修正率の規約が存在する数をカウントするメソッド

    Args:
        v1 (dict): 1つ目のプロジェクトの規約IDと修正数，発生数が含まれている辞書
        v2 (dict, optional):　2つ目のプロジェクトの規約IDと修正数，発生数が含まれている辞書. Defaults to None.
        border (float, optional): 収集する修正率の閾値. Defaults to 0.2.

    Returns:
        int: 閾値以上の修正率の規約の数
    """
    nan_count = 0

    if v2 != None:
        v1 = {key: values[0] / values[1] if values[1] != 0 else 0 for key, values in v1.items()}
        v2 = {key: values[0] / values[1] if values[1] != 0 else 0 for key, values in v2.items()}

        v1_values_list = __insert_key_if_none(v1, v2)
        v2_values_list = __insert_key_if_none(v2, v1)

        for value1, value2 in zip(v1_values_list, v2_values_list):
            if pd.isna(value1) or pd.isna(value2):
                continue
            if value1 >= border and value2 >= border:
                nan_count += 1

    else:
        v1 = {key: values[0] / values[1] if values[1] != 0 else 0 for key, values in v1.items()}
        v1_values_list = [v1[key] for key in sorted(v1)]

        for value1 in v1_values_list:
            if value1 >= border:
                nan_count += 1

    return nan_count


if __name__ == "__main__":
        main()