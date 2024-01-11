import pandas as pd
import sys
sys.path.append("../")
from name_list import NameList

Nl = NameList()
NUM_PROJECT = 75
PROJECT_NAME_LIST = Nl.getProjectName(NUM_PROJECT)
NAME_ID_DICT = Nl.getNameIdDict()

V_NAME_LIST = list(NAME_ID_DICT.keys())
V_KEY_LIST = list(NAME_ID_DICT.values())
PATH = sys.path[-1] + '/data'
PATH_OUT = f'{PATH}/out/to_csv_out.csv'
TAG_D = 'disable'
TAG_I = 'disable-msg'
TAG_flag = ''

# 集計用のリスト作成
count_id_dict = {}
for key in V_KEY_LIST:
    count_id_dict[key] = 0

# ファイルから規約取得
def extract_violations_from_file(path, file_list):
    flag = ''
    with open(path, 'r') as f:
        for line in f:
            if TAG_I in line:
                flag = TAG_I
                continue
            if TAG_D in line:
                flag = TAG_D
                continue
            file_list.extend(line.split(','))
    return flag

# 規約の比較
def compare_conventions(flag, input_list):
    output_dict = {}

    for key in V_KEY_LIST:
        output_dict[key] = 'FALSE'

    if flag == TAG_D:
        for Il in input_list:
            for i in range(len(NAME_ID_DICT)):
                if Il in V_NAME_LIST[i] or Il in V_KEY_LIST[i]:
                    output_dict[V_KEY_LIST[i]] = 'TRUE'
                    count_id_dict[V_KEY_LIST[i]] += 1
    elif flag == TAG_I:
        for Il in input_list:
            for i in range(len(NAME_ID_DICT)):
                if Il in V_KEY_LIST[i]:
                    output_dict[V_KEY_LIST[i]] = 'TRUE'
                    count_id_dict[V_KEY_LIST[i]] += 1

    return output_dict


# 表の作成
def createFrame(output_list):
    df = pd.DataFrame(output_list)
    df.index = PROJECT_NAME_LIST
    df.to_csv(PATH_OUT)


# 結果の取得
def getResultDict(num):
    exist_list = []
    judge_id_dict = {}
    path_in = f'{PATH}/processed/text/{PROJECT_NAME_LIST[num]}_pd.txt'
    TAG_flag = extract_violations_from_file(path_in, exist_list)
    judge_id_dict = compare_conventions(TAG_flag, exist_list)
    return judge_id_dict

# プロジェクト数分回す
result_list = []
for i in range(NUM_PROJECT):
    result_list.append(getResultDict(i))
createFrame(result_list)
print('end')
