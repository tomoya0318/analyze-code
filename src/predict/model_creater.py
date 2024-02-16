from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from constants import path
warnings.filterwarnings("always", category=UserWarning)

class ModelCreater:

    def __init__(self, model_name):
        self.model_name = model_name
        match model_name:
            case "Logistic":
                self.model = LogisticRegression(
                    penalty="l2",  # 正則化項(L1正則化 or L2正則化が選択可能)
                    class_weight="balanced",  # クラスに付与された重み
                    random_state=0,  # 乱数シード
                    solver="lbfgs",  # ハイパーパラメータ探索アルゴリズム
                    max_iter=10000,  # 最大イテレーション数
                    multi_class="auto",  # クラスラベルの分類問題（2値問題の場合'auto'を指定）
                    warm_start=False,  # Trueの場合、モデル学習の初期化に前の呼出情報を利用
                    n_jobs=None,  # 学習時に並列して動かすスレッドの数
                )
                
            case "RandomForest":
                self.model = RandomForestClassifier(
                    class_weight="balanced",
                    random_state=0,
                )
    
            case "SVM":
                self.model = SVC(
                    kernel="linear",
                    class_weight="balanced", 
                    C=1.0, 
                    random_state=0
                )

    def __fetch_train(self, project_list):
        """学習用モデルの作成

        Args:
            project_list (list): 学習モデルを作成するプロジェクト名

        Returns:
            tuple: (model, convention_dummys) の形式のタプル．\n
               modelは学習済みの機械学習モデルオブジェクト,
               convention_dummysはモデル学習に使用した特徴量（規約IDのダミー変数化された特徴量）のリスト

        """
        train_df = pd.DataFrame()
        for project_name in project_list:
            df_value = pd.read_csv(f"{path.ML}/{project_name}_value.csv")
            df_label = pd.read_csv(f"{path.ML}/{project_name}_label.csv", header=None)

            # 説明変数，目的変数の前半8割取得(学習用)
            X_train, _, Y_train, _ = train_test_split(df_value, df_label, test_size=0.2, shuffle=False)
            
            Y_train = Y_train.values.ravel()
            X_train["real_TF"] = Y_train.copy()
            train_df = pd.concat([train_df, X_train], axis=0)

        #コーディング規約IDをダミー変数化
        train_dummys_df = pd.concat([pd.get_dummies(train_df["Warning ID"]), train_df.drop(columns="Warning ID")], axis=1)
        convention_dummys = list(pd.get_dummies(train_df["Warning ID"]))

        #モデルの学習
        try:
            self.model.fit(train_dummys_df.drop(["Project_name", "real_TF"], axis=1), train_dummys_df["real_TF"])
        except ValueError as e:
            print(e)
        return self.model, convention_dummys
    
    def  __fetch_test(self, project_name, convention_dummys):
        """テスト用データの作成

        Args:
            project_name (str): プロジェクト名
            convention_dummys (list): モデルデータの規約違反IDをダミー変数に変換したもののリスト

        Returns:
            tuple: (X_test["Warning ID"], test_df)の形式のタプル．\n
                X_test["Warning ID"]はテストデータの規約違反名のデータフレーム,
                test_dfは規約違反名をダミー変数化したものをテストデータと繋げたもの
        """
        
        df_value = pd.read_csv(f"{path.ML}/{project_name}_value.csv")
        df_label = pd.read_csv(f"{path.ML}/{project_name}_label.csv", header=None)
        _, X_test, _, Y_test = train_test_split(df_value, df_label, test_size=0.2, shuffle=False)
        Y_test = Y_test.values.ravel()
        X_test["real_TF"] = Y_test.copy()
        X_test = X_test.reset_index(drop=True)
        
        id_dict = {}
        for dum_wid in list(convention_dummys):
            id_dict[dum_wid] = []

        for test_wid in X_test["Warning ID"]:
            #全てのキーに対して0を追加
            for i in id_dict.keys():
                id_dict[i].append(0)
            #test_widがあれば、その位置の0を1に更新
            if test_wid in id_dict:
                id_dict[test_wid][-1] = 1

        id_df = pd.DataFrame(id_dict)
        test_df = pd.concat([id_df, X_test], axis=1)
        return X_test["Warning ID"], test_df
    
    def __calc_score_by_convention(self, input_df):
        """コーディング規約ごとの再現率，適合率，F値，正解率の算出

        Args:
            input_df (dataframe): 規約違反名をダミー変数化したものをテストデータと繋げたもの

        Returns:
            dataframe: 規約ごとの結果をdataframeに格納したもの
        """
        data_dict = {}
        warnig_ID = input_df["Warning ID"].tolist()
        real_TF = input_df["real_TF"].tolist()
        predict_TF = input_df["predict_TF"].tolist()
        
        for i, id in enumerate(warnig_ID):
            if id not in data_dict:
                data_dict[id] = {"real_TF": [real_TF[i]], "predict_TF": [predict_TF[i]]}
                continue
            data_dict[id]["real_TF"].append(real_TF[i])
            data_dict[id]["predict_TF"].append(predict_TF[i])

        result = {"Precision": {}, "Recall": {}, "F1 Score": {}, "Accuracy": {}}

        for id in data_dict.keys():
            # Precision
            precision = format(precision_score(data_dict[id]["real_TF"], data_dict[id]["predict_TF"], zero_division=np.nan), ".2f")
            result["Precision"][id] = precision

            # Recall
            recall = format(recall_score(data_dict[id]["real_TF"], data_dict[id]["predict_TF"], zero_division=np.nan), ".2f")
            result["Recall"][id] = recall

            # F1 Score
            f1score = format(f1_score(data_dict[id]["real_TF"], data_dict[id]["predict_TF"], zero_division=np.nan), ".2f")
            result["F1 Score"][id] = f1score

            # Accuracy
            accuracy = format(accuracy_score(data_dict[id]["real_TF"], data_dict[id]["predict_TF"]), ".2f")
            result["Accuracy"][id] = accuracy

        # 結果のデータフレーム化
        return pd.DataFrame(result)
        

    def fit_single_model(self, project_name):
        """単体でのモデル作成

        Args:
            project_name (str): モデルを作成するプロジェクト名

        Returns:
            tuple: (model, convention_dummys) の形式のタプル．\n
               modelは学習済みの機械学習モデルオブジェクト,
               convention_dummysはモデル学習に使用した特徴量（規約IDのダミー変数化された特徴量）のリスト
    
        """
        return self.__fetch_train([project_name])
        
    def fit_merge_model(self, project_list):
        """統合バージョンでのモデル作成

        Args:
            project_list (list): 統合したモデルを作成するプロジェクト名のリスト

        Returns:
            tuple: (model, convention_dummys) の形式のタプル．\n
               modelは学習済みの機械学習モデルオブジェクト,
               convention_dummysはモデル学習に使用した特徴量（規約IDのダミー変数化された特徴量）のリスト

        """
        return self.__fetch_train(project_list)
    
    def predict_single_model(self, project_name, single_model, convention_dummys):
        """単体モデルでのプロジェクトの予測

        Args:
            project_name (str): 予測をするプロジェクト名
            single_model (model): 予測をするのに使用するモデル名
            convention_dummys (list): モデルデータの規約違反IDをダミー変数に変換したもののリスト

        Returns:
            tuple: (resul_all, result_convention)の形式のタプル．\n
                result_allは全体の結果
                result_conventionは規約ごとの結果
        """
        warning_id, test_df = self.__fetch_test(project_name, convention_dummys)
        #予測結果の表示
        predict_result = single_model.predict(test_df.drop(["Warning ID", "Project_name", "real_TF"], axis=1))
        test_df["predict_TF"] = predict_result
        
        #全体の結果の格納
        result_all = {
            "precision": format(precision_score(test_df["real_TF"], predict_result, zero_division=np.nan), ".2f"),
            "recall": format(recall_score(test_df["real_TF"], predict_result, zero_division=np.nan), ".2f"),
            "f1_score": format(f1_score(test_df["real_TF"], predict_result, zero_division=np.nan), ".2f"),
            "accuracy": format(accuracy_score(test_df["real_TF"], predict_result), ".2f")
        }
        result_df = pd.DataFrame([result_all], index = [project_name])
        #コーディング規約ごとの結果の格納
        evaluation_df = pd.concat([warning_id, test_df["real_TF"], test_df["predict_TF"]], axis=1)
        result_convention = self.__calc_score_by_convention(evaluation_df)
        
        return result_df, result_convention

    def predict_merge_model(self, project_list, merge_model, convention_dummys):
        """統合したモデルでの予測

        Args:
            project_list (list): 統合したプロジェクトのリスト
            merge_model (model): 予測に使用するモデル
            convention_dummys (list): モデルデータの規約違反IDをダミー変数に変換したもののリスト

        Returns:
            list: [project1, project2]という形式で返す.それぞれのプロジェクトは全体の結果と，規約ごとの結果をもつ

        """
        result_list = [[] for _ in range(2)]
        for i, project_name in enumerate(project_list):
            warnig_id, test_df = self.__fetch_test(project_name, convention_dummys)

            #予測結果の表示
            predict_result = merge_model.predict(test_df.drop(["Warning ID", "Project_name", "real_TF"], axis=1))
            test_df["predict_TF"] = predict_result

            #全体の結果の格納
            result_all = {
            "precision": format(precision_score(test_df["real_TF"], predict_result, zero_division=np.nan), ".2f"),
            "recall": format(recall_score(test_df["real_TF"], predict_result, zero_division=np.nan), ".2f"),
            "f1_score": format(f1_score(test_df["real_TF"], predict_result, zero_division=np.nan), ".2f"),
            "accuracy": format(accuracy_score(test_df["real_TF"], predict_result), ".2f")
            }
            other_project_name = project_list[1 - i]
            result_df = pd.DataFrame([result_all], index = [other_project_name])

            #コーディング規約ごとの結果の格納
            result_convention_df = pd.concat([warnig_id, test_df["real_TF"], test_df["predict_TF"]], axis=1)
            self.__calc_score_by_convention(result_convention_df)

            result_list[i] = [result_df, result_convention_df]
        return result_list[0], result_list[1]

