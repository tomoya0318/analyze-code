import pandas as pd

#csvファイルのカラムのリストの出力
def get_project_name(csvfile):
    input_df = pd.read_csv(csvfile, index_col=0)
    project_name_list = input_df.columns
    return project_name_list

def correction_from_csv(csvfile):
    # csvファイルから規約ごとの修正率の収集
    project_correction_rates = [] 
    input_df = pd.read_csv(csvfile, index_col=0)
    project_name_list = input_df.columns

    #収集した規約のリスト化
    for project_name in project_name_list:
        project_correction_rates.append(input_df[project_name].tolist())

    return project_correction_rates