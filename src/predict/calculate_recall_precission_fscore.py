import pandas as pd
import sys
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score # 評価指標算出用
sys.path.append('../')
from constants import path


PATH = f'{path.DATA}/predict_model'
#'Logistic', 'RandomForest', 'SVM'から選択
model_name = 'RandomForest'
PATH_IN = f'{PATH}/{model_name}/'
project = 'merge_GPandps_python-sdk'
files = os.listdir(PATH_IN)

#データの取得
input_df = pd.read_csv(f'{PATH_IN}/{project}.csv', encoding='utf-8')
input_df = input_df.dropna(subset=['real_TF', 'predict_TF'])
coding_ID = input_df['coding_ID'].tolist()
real_TF = input_df['real_TF'].tolist()
predict_TF = input_df['predict_TF'].tolist()

# 辞書化
data_dict = {}
for i, id in enumerate(coding_ID):
    if id not in data_dict:
        data_dict[id] = {'real_TF': [real_TF[i]], 'predict_TF': [predict_TF[i]]}
        continue
    data_dict[id]['real_TF'].append(real_TF[i])
    data_dict[id]['predict_TF'].append(predict_TF[i])

#結果の算出
result_dict = {'Precision': {}, 'Recall': {}, 'F1 Score': {}, 'Accuracy': {}}

for id in data_dict.keys():
    # Precision
    try:
        precision = format(precision_score(data_dict[id]['real_TF'], data_dict[id]['predict_TF'], zero_division=np.nan), '.2f')
    except ZeroDivisionError as e:
        precision = 'Err'
    result_dict['Precision'][id] = precision

    # Recall
    try:
        recall = format(recall_score(data_dict[id]['real_TF'], data_dict[id]['predict_TF'], zero_division=np.nan), '.2f')
    except ZeroDivisionError as e:
        recall = 'Err'
    result_dict['Recall'][id] = recall

    # F1 Score
    try:
        f1score = format(f1_score(data_dict[id]['real_TF'], data_dict[id]['predict_TF'], zero_division=np.nan), '.2f')
    except ZeroDivisionError as e:
        f1score = 'Err'
    result_dict['F1 Score'][id] = f1score

    # Accuracy
    try:
        accuracy = format(accuracy_score(data_dict[id]['real_TF'], data_dict[id]['predict_TF']), '.2f')
    except ZeroDivisionError as e:
        accuracy = 'Err'
    result_dict['Accuracy'][id] = accuracy

# 結果のデータフレーム化
df_combined = pd.DataFrame(result_dict)

print(df_combined)
df_combined.to_csv(f'{path.OUT}/{model_name}_{project}.csv')