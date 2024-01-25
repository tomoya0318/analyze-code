from utils import extract_data
from utils import get_file_name
from constants import path

class NameList:
    def __init__(self):
        self.PROJECT_NAME_LIST = []
        self.NAME_ID_DICT = {}
        self.PATH = path.ROOT
        self.num = 75
        
    #プロジェクト名の取得
    def getProjectName(self, num):
        path = self.PATH + 'data/pylintrc'
        self.PROJECT_NAME_LIST = get_file_name(path)
        return self.PROJECT_NAME_LIST[:num]

    #規約名と規約IDの取得
    def getNameIdDict(self):
        path = self.PATH + 'data/version/nameList.txt'
        with open(path, 'r') as f:
            for line in f:
                key, value = extract_data(line)
                self.NAME_ID_DICT[key] = value
        return self.NAME_ID_DICT
