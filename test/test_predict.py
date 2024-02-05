import unittest
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.constants import path
from src.predict.modules import predict_single
from src.predict import calculate_individual_project
from src.predict import create_result
from src.predict.modules import create_merge_model

class PredictTest(unittest.TestCase):
    def setUp(self):
        self.project_name = "python-sdk"
        self.model_name = "RandomForest"

    def test_predict_single(self):
        df_value = pd.read_csv(f"{path.ML}/{self.project_name}_value.csv")
        df_label = pd.read_csv(f"{path.ML}/{self.project_name}_label.csv", header=None)

        result = predict_single(df_value, df_label, self.project_name, self.model_name)
        self.assertIsInstance(result[0], pd.DataFrame)
        self.assertIsInstance(result[1], pd.DataFrame)

    def test_calculate_individual_project(self): 
        calculate_individual_project([self.project_name], self.model_name)
        #CSV ファイルが存在するか確認
        dir = f'{path.PRERESULT}/single/{self.model_name}/{self.project_name}.csv'
        self.assertTrue(os.path.exists(dir), f'CSV file for project {self.project_name} not found.')
        os.remove(dir)

    def test_merge_model(self):
        merge_list = ['python-sdk', 'GPflow']
        model_all, _ = create_merge_model(merge_list, self.model_name)
        self.assertIsInstance(model_all, RandomForestClassifier)

    def test_merge_method(self):
        merge_list = ['python-sdk', 'GPflow']
        model_all, dummys = create_merge_model(merge_list, self.model_name)
        create_result(merge_list, model_all, dummys, self.model_name)
        #csvファイルが存在するかの確認
        dir = f'{path.PRERESULT}/merge/{self.model_name}/{merge_list[0]}merge_{merge_list[0]}_{merge_list[1]}.csv'
        self.assertTrue(os.path.exists(dir), f'CSV file not found.')
        os.remove(dir)

if __name__ == '__main__':
    unittest.main()


