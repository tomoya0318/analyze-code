import pandas as pd
from utils import get_file_name
from constants import path

THRESHOLD = 3
PATH_IN = f'{path.OUT}/tracking_all_convention.csv'
PROJECT_NAME_LIST = get_file_name(f'{path.PROCESSED}/csv')
PROJECT_NUM = len(PROJECT_NAME_LIST)

#None以外の数が閾値以下のプロジェクトを除外
df = pd.read_csv(PATH_IN, index_col=0)
# print(df.head)
for project_name in PROJECT_NAME_LIST:
    count = df[project_name].apply(lambda x: pd.notna(x)).sum()
    print(f'{project_name}:{count}')
    if count < THRESHOLD:
        df.drop(project_name, axis=1, inplace=True)

# 欠損値を'None'に変換
df = df.apply(lambda x: x.map(lambda x: 'None' if pd.isna(x) else x))

#csvファイルに変換
df.to_csv(f'{path.OUT}/cleaned_tracking_all_convention.csv')