import pandas as pd
import sys
sys.path.append("../")
from utils import calculate_cosin_similarity
from utils import load_correction_data
from constants import path

PATH_IN = f'{path.OUT}/cleaned_tracking_all_convention.csv'
PATH_OUT = f'{path.OUT}/cos_sim.csv'
PATH_OUT_SORTED = f'{path.OUT}/test_sort_cosine.csv'

#test
# PATH_IN = f'{PATH}/out/tracking_test_convention.csv'
#PATH_OUT = f'{path.OUT}//test_all_cos_sim.csv'
# PROJECT_NAME_LIST = get_file_name(f'{PATH}/test')

#csvファイルからプロジェクト名とプロジェクト数，修正率の取得
PROJECT_NAME_LIST = load_correction_data.get_project_name(PATH_IN)
PROJECT_NUM = len(PROJECT_NAME_LIST)
project_correction_rates = load_correction_data.correction_from_csv(PATH_IN)

# 表の作成
def createFrame(output_list, path, project_names):
    df = pd.DataFrame(output_list)
    df.index = project_names
    df.columns = project_names
    df.to_csv(path)


#コサイン類似度の計算
result_list = [[] for _ in range(PROJECT_NUM)]   
for i in range(PROJECT_NUM):
    for j in range(PROJECT_NUM):
        result_cosin = calculate_cosin_similarity.cos_sim(project_correction_rates[i], project_correction_rates[j])
        result_list[i].append(round(result_cosin[0], 3))

# 表の作成
createFrame(result_list, PATH_OUT_SORTED, PROJECT_NAME_LIST)



