import pandas as pd
import os
from modules import predict_single
from constants import path
def main():
    # 宣言
    model_name = "RandomForest"  # Logistic, RandomForest, SVMの３種類から選ぶ
    #単体で予測するプロジェクト名
    project_list = ["python-sdk", "hickle", "GPflow"]
    conventional_method(project_list, model_name)


def conventional_method(project_list, model_name):
    """project_list内のプロジェクト名単体での予測精度の算出

    Args:
        project_list (list): 予測するプロジェクトのリスト
        model_name (str): 使用するモデル名
    """
    # 結果格納用
    result_df = pd.DataFrame(columns=["precision", "recall", "f1_score", "accuracy"])
    dir = f'{path.PRERESULT}/{model_name}'
    if not os.path.exists(dir):
        os.makedirs(dir)

    # 単体でのモデル作成，予測
    for file_name in project_list:
        try:
            df_value = pd.read_csv(f"{path.ML}/{file_name}_value.csv")
            df_label = pd.read_csv(f"{path.ML}/{file_name}_label.csv", header=None)
        except pd.errors.EmptyDataError as e:
            print(file_name)
        _, tmp2 = predict_single(df_value, df_label, file_name, model_name)

        # 規約ごとの結果
        tmp2.to_csv(f"{path.PRERESULT}/single/{model_name}/{file_name}.csv")

if __name__ == '__main__':
    main()