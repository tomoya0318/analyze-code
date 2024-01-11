import os
import sys
sys.path.append("../")
from utils import extract_data
from utils import get_file_name


class NameList:
    def __init__(self):
        self.PROJECT_NAME_LIST = []
        self.NAME_ID_DICT = {}
        self.PATH = sys.path[-1]
        self.num = 75
        

    def getProjectName(self, num):
        path = self.PATH + 'data/pylintrc'
        self.PROJECT_NAME_LIST = get_file_name(path)
        return self.PROJECT_NAME_LIST[:num]


    def getNameIdDict(self):
        path = self.PATH + 'data/version/nameList.txt'
        with open(path, 'r') as f:
            for line in f:
                key, value = extract_data(line)
                self.NAME_ID_DICT[key] = value
        return self.NAME_ID_DICT
