import os
import pandas as pd


# 行の削除
def delete_rows(in_directory, out_directory, init, last):
    files = os.listdir(in_directory)
    for file in sorted(files, key=lambda x: x.lower()):
        print(file)
        full_path = os.path.join(in_directory, file)
        try:
            df = pd.read_csv(full_path, dtype=str)
            if last != 0:
                df = df.drop(df.index[init:last])
            df.to_csv(f"{out_directory}/{file}", header=False)
        except pd.errors.EmptyDataError:
            print(f"Warning: EmptyDataError for file {full_path}. Skipping...")
            continue


# 列の削除
def delete_columns(in_directory, out_directory, init, last):
    files = os.listdir(in_directory)
    for file in sorted(files, key=lambda x: x.lower()):
        full_path = os.path.join(in_directory, file)
        print(file)
        try:
            df = pd.read_csv(full_path, dtype=str)
            # 指定したキーワードを含む列を検索して削除
            keyword = "Line Number"
            columns_to_drop = [col for col in df.columns if keyword in col]
            df = df.drop(columns_to_drop, axis=1)
            # 指定した列番号の範囲を取得して削除
            df = df.drop(df.columns[init : last + 1], axis=1)

            df.to_csv(f"{out_directory}/{file}", index=False, header=False)
        except pd.errors.EmptyDataError:
            print(f"Warning: EmptyDataError for file {full_path}. Skipping...")
            continue
