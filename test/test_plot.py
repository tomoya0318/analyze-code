import unittest
import numpy as np
import os
from plot import plot_scatter
from src.predict import ModelCreater
from src.cosin import lookup_from_model, calc_cosin
from src.constants import path

class TestPredict(unittest.TestCase):
    def setUp(self):
        self.merge_list = ['python-sdk', 'GPflow']
        self.model_name = "RandomForest"

    def test_散布図作成のテスト(self):
        fix_rate_list = []
        f1_score_list = []
        name2_dict = lookup_from_model(f"{path.PRERESULT}/raw_model/{self.merge_list[1]}_train.csv")
        fix_rate = sum(item[0] for item in name2_dict.values()) / sum(item[1] for item in name2_dict.values())
        fix_rate_list.append(fix_rate)
        mc = ModelCreater(self.model_name)
        model, dummys = mc.fit_merge_model(self.merge_list)
        result, _ = mc.predict_merge_model(self.merge_list, model, dummys)
        f1_score_list.append(result[0]["f1_score"].iloc[0])

        fix = np.array(fix_rate_list)
        f1 = np.array(f1_score_list)
        dir = f"{path.OUT}/fix_f1.png"
        plot_scatter(fix, f1, "修正率", "F値", dir)
        self.assertTrue(os.path.exists(dir), 'file not found')


if __name__ == '__main__':
    unittest.main()
