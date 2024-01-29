import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from modules import select_model
warnings.filterwarnings("always", category=UserWarning)

def predict_single(explanatory_variable, label, project_name, model_name):  
    """予測結果の算出

    Args:
        explanatory_variable (dataframe): 説明変数
        label (dataframe): 目的変数
        project_name (str): プロジェクト名
        model_name (str): モデルの名前
        X_train (list): 学習データの説明変数
        X_test (list): テストデータの説明変数
        Y_train (list): 学習データの目的変数
        Y_test (list): テストデータの目的変数

    Returns:
        dataframe: プロジェクトごとの予測結果（適合率,再現率,F1値,正解率）.1つ目の引数
        dataframe: プロジェクトごとの全ての規約に対する予測と実際の答えに関するデータフレーム.2つ目の引数
    """

    # モデルの初期化
    model = select_model(model_name)

    # コーディング規約IDをダミー変数化
    df_marge = pd.concat(
        [pd.get_dummies(explanatory_variable["Warning ID"]), explanatory_variable.drop(columns="Warning ID")], axis=1
    )

    # 説明変数，目的変数を学習用，テスト用に分割（後半2割をテスト用に使用）
    X_train, X_test, Y_train, Y_test = train_test_split(df_marge, label, test_size=0.2, shuffle=False)
    Y_train = Y_train.values.ravel()
    Y_test = Y_test.values.ravel()

    #モデルの学習
    try:
        model.fit(X_train.drop(["Project_name"], axis=1), Y_train)
    except ValueError as e:
        print(project_name)
        result = {"precision": "NaN", "recall": "NaN", "f1_score": "NaN", "accuracy": "NaN"}
        return pd.DataFrame([result], index=[project_name]), 0

    # テストデータでの説明変数の予測
    predict_result = model.predict(X_test.drop(["Project_name"], axis=1))

    # 分析用DF
    return_df = X_test
    return_df["real_TF"] = Y_test
    return_df["predict_TF"] = predict_result

    # 全体の結果の格納
    result = {"precision": format(precision_score(Y_test, predict_result, zero_division=np.nan), ".2f")}
    result["recall"] = format(recall_score(Y_test, predict_result, zero_division=np.nan), ".2f")
    result["f1_score"] = format(f1_score(Y_test, predict_result, zero_division=np.nan), ".2f")
    result["accuracy"] = format(accuracy_score(Y_test, predict_result), ".2f")


    
    return pd.DataFrame([result], index=[project_name]), return_df.reset_index(drop=True)