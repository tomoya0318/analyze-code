import unittest
from constants import path
from cosin import lookup_from_model
from cosin import calc_cosin
class TestCosin(unittest.TestCase):
    def setUp(self):
        self.project_name = "python-sdk"
        self.input_path = f"{path.PRERESULT}/raw_model/{self.project_name}_train.csv"

    def test_学習データの規約修正率の収集(self):
        result_dict = lookup_from_model(self.input_path)
        first_item = list(result_dict.items())[0]
        self.assertTrue(first_item == ('C0103', 0.95), "The first item is not 'C0103': 0.95")
    
    def test_コサイン類似度測定(self):
        v1 = lookup_from_model(self.input_path)
        v2 = lookup_from_model(f"{path.PRERESULT}/raw_model/GPflow_train.csv")
        cos_sim, count = calc_cosin(v1, v2)
        self.assertTrue(cos_sim, int)

if __name__ == '__main__':
    unittest.main()