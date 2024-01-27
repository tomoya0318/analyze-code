import pandas as pd
import numpy as np


# コサイン類似度の測定(NaNは0に変換)
def cos_sim(v1, v2):
    """コサイン類似度の測定

    Args:
        v1 (list): プロジェクトの規約ごとの修正率
        v2 (list): プロジェクトの規約ごとの修正率

    Returns:
        list: コサイン類似度とNaNでない要素の数
    """
    v1_out = [float(value) if not pd.isna(value) else 0 for value in v1]
    v2_out = [float(value) if not pd.isna(value) else 0 for value in v2]

    count = sum(1 for value1, value2 in zip(v1, v2) if not pd.isna(value1) and not pd.isna(value2))
    result = [np.dot(v1_out, v2_out) / (np.linalg.norm(v1_out) * np.linalg.norm(v2_out)), count]
    return result
