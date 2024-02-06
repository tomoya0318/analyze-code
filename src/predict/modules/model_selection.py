from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def select_model(model_name):
    """Logistic,RandomForest,SVMからモデルの選択

    Args:
        model_name (str): 3つのモデルのうち,1つを選択

    Returns:
        object: 3つのモデルに対する設定
    """
    match model_name:
        case "Logistic":
            model = LogisticRegression(
                penalty="l2",  # 正則化項(L1正則化 or L2正則化が選択可能)
                class_weight="balanced",  # クラスに付与された重み
                random_state=0,  # 乱数シード
                solver="lbfgs",  # ハイパーパラメータ探索アルゴリズム
                max_iter=10000,  # 最大イテレーション数
                multi_class="auto",  # クラスラベルの分類問題（2値問題の場合'auto'を指定）
                warm_start=False,  # Trueの場合、モデル学習の初期化に前の呼出情報を利用
                n_jobs=None,  # 学習時に並列して動かすスレッドの数
            )
            return model

        case "RandomForest":
            model = RandomForestClassifier(
                class_weight="balanced",
                random_state=0,
            )
            return model

        case "SVM":
            model = SVC(kernel="linear", class_weight="balanced", C=1.0, random_state=0)
            return model

        case _:
            print("It is out of pattern")
            return "error"
