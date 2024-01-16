import sys
sys.path.append("../")
from utils import clean_csv

PATH = sys.path[-1] + '/data'
PATH_IN = f'{PATH}/minami_output'
PATH_OUT = f'{PATH}/processed/csv'

#csvファイルの
clean_csv.delete_columns(PATH_IN, PATH_OUT, 0, 1)
print('end')

