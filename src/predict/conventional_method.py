import pandas as pd
import os
from predict.modules import predict_single
from constants import path
from predict.modules import lookup_white_list
from utils import create_dir

def main():
    # 宣言
    model_name = "RandomForest"  # Logistic, RandomForest, SVMの３種類から選ぶ
    # 単体で予測するプロジェクト名
    project_list = lookup_white_list(f"{path.DATA}/white_list.txt")
    calculate_individual_project(project_list, model_name)


def calculate_individual_project(project_list, model_name):
    """project_list内のプロジェクト名単体での予測精度の算出

    Args:
        project_list (list): 予測するプロジェクトのリスト
        model_name (str): 使用するモデル名
    """
    # 結果格納用
    create_dir(f"{path.PRERESULT}/single/{model_name}")

    # 単体でのモデル作成，予測
    for project_name in project_list:
        try:
            df_value = pd.read_csv(f"{path.ML}/{project_name}_value.csv")
            df_label = pd.read_csv(f"{path.ML}/{project_name}_label.csv", header=None)
        except pd.errors.EmptyDataError as e:
            print(project_name)
        tmp1, tmp2 = predict_single(df_value, df_label, project_name, model_name)

        # 全体の結果
        tmp1.to_csv(f"{path.PRERESULT}/single/{model_name}/all/{project_name}.csv")
        # 規約ごとの結果
        tmp2.to_csv(f"{path.PRERESULT}/single/{model_name}/violations/{project_name}.csv")


if __name__ == "__main__":
    main()
