import pandas as pd
import sys
sys.path.append("../")
from utils import calculate_cosin_similarity
from utils import load_correction_data

PATH = f'{sys.path[-1]}/data/'
PATH_IN = f'{PATH}/out/cleaned_tracking_all_convention.csv'
PATH_OUT = f'{PATH}/out/test1_sort_cosine.csv'
PROJECT_NAME_LIST = load_correction_data.get_project_name(PATH_IN)
PROJECT_NUM = len(PROJECT_NAME_LIST)

df = pd.DataFrame(columns=['Project 1', 'Project 2', 'Similarity', 'Count'])

# 組み合わせの表示
project_correction_rates = load_correction_data.correction_from_csv(PATH_IN)

for i in range(PROJECT_NUM):
    for j in range(i, PROJECT_NUM):
        result_combinations = calculate_cosin_similarity.cos_sim(project_correction_rates[i], project_correction_rates[j])
        cosin_similarity = result_combinations[0]
        count = result_combinations[1]
        df = pd.concat([df, pd.DataFrame({'Project 1': [PROJECT_NAME_LIST[i]], 'Project 2': [PROJECT_NAME_LIST[j]], 'Similarity': [cosin_similarity], 'Count': [count]})], ignore_index=True)

# 結果の表示
df.to_csv(PATH_OUT, index = False)
