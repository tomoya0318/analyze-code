import pandas as pd
import os
from predict.modules import predict_single
from constants import path
def main():
    # 宣言
    model_name = "RandomForest"  # Logistic, RandomForest, SVMの３種類から選ぶ
    #単体で予測するプロジェクト名
    project_list = ["python-sdk", "hickle", "GPflow"]
    calculate_individual_project(project_list, model_name)


def calculate_individual_project(project_list, model_name):
    """project_list内のプロジェクト名単体での予測精度の算出

    Args:
        project_list (list): 予測するプロジェクトのリスト
        model_name (str): 使用するモデル名
    """
    # 結果格納用
    dir = f'{path.PRERESULT}/single/{model_name}'
    if not os.path.exists(dir):
        os.makedirs(dir)

    # 単体でのモデル作成，予測
    for project_name in project_list:
        try:
            df_value = pd.read_csv(f"{path.ML}/{project_name}_value.csv")
            df_label = pd.read_csv(f"{path.ML}/{project_name}_label.csv", header=None)
        except pd.errors.EmptyDataError as e:
            print(project_name)
        _, tmp2 = predict_single(df_value, df_label, project_name, model_name)

        # 規約ごとの結果
        tmp2.to_csv(f"{path.PRERESULT}/single/{model_name}/{project_name}.csv")

if __name__ == '__main__':
    main()