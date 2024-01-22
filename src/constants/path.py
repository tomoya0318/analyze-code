"""使用するPATHの一覧
"""
import os


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
DATA = f'{ROOT}/data'
SRC = f'{ROOT}/src'
OUT = f'{DATA}/out'
PROCESSED = f'{DATA}/processed'