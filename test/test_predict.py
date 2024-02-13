import unittest
import os
import pandas as pd
from src.constants import path
from src.predict.model_creater import ModelCreater

class TestPredict(unittest.TestCase):
    def setUp(self):
        self.project_name = "python-sdk"
        self.merge_list = ['python-sdk', 'GPflow']
        self.model_name = "RandomForest"

    def test_単体モデル作成のテスト(self):
        mc = ModelCreater(self.model_name)
        model, dummys = mc.fit_single_model(self.project_name)
        result_all, result_convention = mc.predict_single_model(self.project_name, model, dummys)
        all_dir = f"{path.PRERESULT}/single/{self.model_name}/all/{self.project_name}.csv"
        convention_dir = f"{path.PRERESULT}/single/{self.model_name}/violations/{self.project_name}.csv"
        result_all.to_csv(all_dir)
        result_convention.to_csv(convention_dir)
        self.assertTrue(os.path.exists(all_dir), f'all CSV file not found.')
        self.assertTrue(os.path.exists(convention_dir), f'convention CSV file not found.')

    def test_マージモデル作成のテスト(self):
        mc = ModelCreater(self.model_name)
        model, dummys = mc.fit_merge_model(self.merge_list)
        result_dict = mc.predict_merge_model(self.merge_list, model, dummys)
        print(result_dict.keys())
        dir = f'{path.PRERESULT}/merge/{self.model_name}/{self.merge_list[0]}_merge_{self.merge_list[0]}_{self.merge_list[1]}.csv'
        result_convention = result_dict[self.merge_list[0]][1]
        result_convention.to_csv(dir)
        self.assertTrue(os.path.exists(dir), f'CSV file not found.')

if __name__ == '__main__':
    unittest.main()