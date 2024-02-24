import unittest
from constants import path
from cosin import lookup_from_model
from cosin import calc_cosin, count_values_exceeding_limit
class TestCosin(unittest.TestCase):
    def setUp(self):
        self.v1 = lookup_from_model(f"{path.PRERESULT}/raw_model/python-sdk_train.csv")
        self.v2 = lookup_from_model(f"{path.PRERESULT}/raw_model/GPflow_train.csv")

    def test_学習データの規約修正率の収集(self):
        first_item = list(self.v1.items())[0]
        self.assertTrue(first_item == ('C0103', [19, 19]), "The first item is different from expectations")
    
    def test_コサイン類似度測定(self):
        cos_sim = calc_cosin(self.v1, self.v2)
        self.assertTrue(cos_sim, int)

    def test_規約存在数測定(self):
        non_count = count_values_exceeding_limit(self.v1, self.v2, 0.2)
        self.assertTrue(non_count, int)

    def test_単体での規約の存在数測定(self):
        convention_count = count_values_exceeding_limit(self.v1, border=0.2)
        self.assertTrue(convention_count, int)

if __name__ == '__main__':
    unittest.main()