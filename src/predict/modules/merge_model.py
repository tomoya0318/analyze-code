import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from .model_selection import select_model
from constants import path
warnings.filterwarnings("always", category=UserWarning)

def create_merge_model(project_list, model_name):
    """結合したモデルの作成

    Args:
        project_list (list): 結合するプロジェクト名のリスト
        model_name (str): モデルの名前

    Returns:
        dataframe: マージして作成したモデル.1つ目の引数
        list: 規約違反をダミー変数化したもの.2つ目の引数
    """
    train_df = pd.DataFrame()

    #モデルの初期化
    model_all = select_model(model_name)

    #プロジェクト数分回す
    for project_name in project_list:
        df_value = pd.read_csv(f"{path.ML}/{project_name}_value.csv")
        df_label = pd.read_csv(f"{path.ML}/{project_name}_label.csv", header=None)

        # 説明変数，目的変数を学習用，テスト用に分割
        X_train, _, Y_train, _ = train_test_split(df_value, df_label, test_size=0.2, shuffle=False)
        Y_train = Y_train.values.ravel()
        X_train["real_TF"] = Y_train.copy()
        train_df = pd.concat([train_df, X_train], axis=0)

        # コーディング規約IDをダミー変数化
    df_marge = pd.concat([pd.get_dummies(train_df["Warning ID"]), train_df.drop(columns="Warning ID")], axis=1)
    dummys = list(pd.get_dummies(train_df["Warning ID"]))

    try:
        model_all.fit(df_marge.drop(["Project_name", "real_TF"], axis=1), df_marge["real_TF"])
    except ValueError as e:
        print(e)

    return model_all, dummys