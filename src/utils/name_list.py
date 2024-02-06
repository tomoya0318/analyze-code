from utils import extract_data
from utils import get_file_name
from constants import path


class NameList:
    """ディレクトリ内からプロジェクト名と,規約名,規約IDを取得"""

    def __init__(self):
        self.PROJECT_NAME_LIST = []
        self.NAME_ID_DICT = {}
        self.PATH = path.ROOT
        self.num = 75

    # プロジェクト名の取得
    def getProjectName(self, num):
        """プロジェクト名の取得

        Args:
            num (int): 取得するプロジェクト数の入力

        Returns:
            list: プロジェクト名
        """
        path = self.PATH + "data/pylintrc"
        self.PROJECT_NAME_LIST = get_file_name(path)
        return self.PROJECT_NAME_LIST[:num]

    # 規約名と規約IDの取得
    def getNameIdDict(self):
        """規約名と規約IDの取得

        Returns:
            dict: {規約名:規約ID}の形の辞書型
        """
        path = self.PATH + "data/version/nameList.txt"
        with open(path, "r") as f:
            for line in f:
                key, value = extract_data(line)
                self.NAME_ID_DICT[key] = value
        return self.NAME_ID_DICT
