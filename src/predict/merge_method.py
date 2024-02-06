import pandas as pd
import os
from sklearn.model_selection import train_test_split
from predict import create_merge_model
from constants import path
from predict.modules import lookup_white_list
from utils import create_dir
from predict.modules import calc_score
def main():
    # すべてのプロジェクト
    project_list = lookup_white_list(f"{path.DATA}/white_list.txt")
    # project_list = ["GPflow", "hickle"]
    model_name = "RandomForest"  # Logistic, RandomForest, SVMの３種類から選ぶ

    # マージするプロジェクトの選択
    for i in range(len(project_list)):
        for j in range(i, len(project_list)):
            merge_list = [project_list[i], project_list[j]]
            model_all, dummys = create_merge_model(project_list, model_name)
            create_result(merge_list, model_all, dummys, model_name)


def __fetch_test(project_list):
    """project_listに格納されているプロジェクト名のテストデータの取得をするメソッド

    Args:
        project_list (list): 対象にするプロジェクトのリスト

    Returns:
        dataframe: 目的変数のテストデータ
    """
    for project_name in project_list:
        df_value = pd.read_csv(f"{path.ML}/{project_name}_value.csv")
        df_label = pd.read_csv(f"{path.ML}/{project_name}_label.csv", header=None)
        _, X_test, _, Y_test = train_test_split(df_value, df_label, test_size=0.2, shuffle=False)
        Y_test = Y_test.values.ravel()
        X_test["real_TF"] = Y_test
        X_test = X_test.reset_index(drop=True)
    return X_test


def __compare_id_dummys(dummys, X_test):
    """
    ダミーデータとテストデータの規約IDを比較し、結果を辞書に格納するメソッド

    Args:
        dummys(list): 規約IDのダミーデータ
        X_test(dataframe): テストデータのDataFrame

    Returns:
        dict: ダミーデータと，テストデータに存在した規約の格納
    """
    id_dict = {}
    for i in list(dummys):
        id_dict[i] = []

    for i in list(dummys):
        id_dict[i] = []
    for wid in X_test["Warning ID"]:
        if wid in id_dict:
            id_dict[wid].append(1)
            for i in id_dict:
                id_dict[i].append(0)
            id_dict[wid].pop(-1)
        else:
            for i in id_dict:
                id_dict[i].append(0)

    id_df = pd.DataFrame(id_dict)
    test_df = pd.concat([id_df, X_test], axis=1)

    return test_df


def create_result(merge_list, model_all, dummys, model_name):
    """マージモデルでの結果の表示

    Args:
        merge_list (list): マージするプロジェクト
        model_all (dataframe): プロジェクトの予測モデル
        dummys (list): 規約IDのダミーデータ
        model_name (str): モデル名
    """
    
    create_dir(f"{path.PRERESULT}/merge/{model_name}")

    #マージモデルの予測結果の算出
    X_test = __fetch_test(merge_list)
    test_df = __compare_id_dummys(dummys, X_test)
    predict_result = model_all.predict(test_df.drop(["Warning ID", "Project_name", "real_TF"], axis=1))
    test_df["predict_TF"] = predict_result
    result_df = pd.concat([X_test["Warning ID"], test_df["real_TF"], test_df["predict_TF"]], axis=1)
    calc_score(result_df, f"{path.PRERESULT}/merge/{model_name}/{merge_list[0]}_merge_{merge_list[0]}_{merge_list[1]}.csv")
    