import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from constants import path
warnings.filterwarnings("always", category=UserWarning)

def extract_training_data(explanatory_variable, project_name): 
    """説明変数の学習用データの抽出

    Args:
        explanatory_variable (dataframe): 説明変数
        project_name (str): プロジェクト名
        model_name (str): モデルの名前
    """
    #学習用データの抽出
    df_marge = pd.DataFrame(explanatory_variable)
    X_train, _ = train_test_split(df_marge, test_size=0.2, shuffle=False)
    X_train.to_csv(f"{path.PRERESULT}/row_model/{project_name}_train.csv")