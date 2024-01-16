import re
import sys
sys.path.append("../")
from name_list import NameList

# コメント、空白行を削除し、[MESSAGES CONTROL]の中身のみを出力
Nl = NameList()
PROJECT_NAME_LIST = Nl.getProjectName(75)
PATH = sys.path[-1] + '/data'
char_remove = '#'
title = '[MESSAGES CONTROL]'
TAG_N = 'disable='
TAG_I = 'disable-msg='

#pylintrcの[MESSAGES CONTROL]のみを取得
for projectName in PROJECT_NAME_LIST:
    PATH_IN = f'{PATH}/pylintrc/{projectName}.txt'
    PATH_OUTPUT = f'{PATH}/processed/text/{projectName}_pd.txt'
    with open(PATH_OUTPUT, 'w') as f:
        in_messages_control = False
        for line in open(PATH_IN, 'r'):
            if char_remove in line:
                continue
            if not line.strip():
                continue
            if title in line:
                in_messages_control = True
                continue
            if not in_messages_control:
                continue
            #title以外が来た時
            if  re.match(r'^\[\w', line):
                in_messages_control = False
                continue
            if TAG_I in line:
                f.write(TAG_I + '\n')
            if TAG_N in line:
                f.write(TAG_N + '\n')
            f.write(line.replace(TAG_N, '').replace(TAG_I, '').replace('\n', '').replace(' ', ''))

print('end')
