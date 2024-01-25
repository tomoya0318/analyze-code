from utils import clean_csv
from constants import path

PATH_IN = f'{path.DATA}/minami_output'
PATH_OUT = f'{path.DATA}/processed/csv'

#csvファイルの不要行の削除
clean_csv.delete_columns(PATH_IN, PATH_OUT, 0, 1)


