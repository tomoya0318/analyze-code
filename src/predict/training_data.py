import pandas as pd
import os
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings("always", category=UserWarning)
from constants import path
from predict.modules import lookup_white_list

def main():
    #プロジェクトの選択
    project_list = lookup_white_list(f'{path.DATA}/white_list.txt')
    #すべてのプロジェクトで，学習データの取得
    for project_name in project_list:
        try:
            df_value = pd.read_csv(f"{path.ML}/{project_name}_value.csv")
            df_label = pd.read_csv(f"{path.ML}/{project_name}_label.csv", header=None)
            extract_training_data(df_value, df_label, project_name)
        except pd.errors.EmptyDataError as e:
            print(project_name)
    
def extract_training_data(explanatory_variable, label, project_name): 
    """目的変数の学習用データの抽出

    Args:
        explanatory_variable (dataframe): 説明変数
        project_name (str): プロジェクト名
        model_name (str): モデルの名前
    """

    #ディレクトリの確認
    dir = f'{path.PRERESULT}/raw_model'
    if not os.path.exists(dir):
        os.makedirs(dir)

    #学習用データの抽出
    X_train, _, Y_train, _ = train_test_split(explanatory_variable, label, test_size=0.2, shuffle=False)
    Y_train.columns = ['Target'] 
    train_df = pd.concat([X_train['Warning ID'], Y_train], axis=1)
    train_df.to_csv(f"/{dir}/{project_name}_train.csv", index=False)

if __name__ == '__main__':
    main()