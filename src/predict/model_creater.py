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

        return train_dummys_df, convention_dummys
    
    def  __fetch_test(self, project_name):
        """project_listに格納されているプロジェクト名のテストデータの取得をするメソッド

            Args:
                project_list (list): 対象にするプロジェクトのリスト

            Returns:
                dataframe: 目的変数のテストデータ
        """
        
        df_value = pd.read_csv(f"{path.ML}/{project_name}_value.csv")
        df_label = pd.read_csv(f"{path.ML}/{project_name}_label.csv", header=None)
        _, X_test, _, Y_test = train_test_split(df_value, df_label, test_size=0.2, shuffle=False)
        Y_test = Y_test.values.ravel()
        X_test["real_TF"] = Y_test.copy()
        X_test = X_test.reset_index(drop=True)
        return X_test
    
    def __calc_score_by_convention(self, input_df, out_path):
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
        result_df = pd.DataFrame(result)
        result_df.to_csv(out_path)

    def fit_merge_model(self, project_list):
        train_dummys_df, convention_dummys = self.__fetch_train(project_list)
        try:
            self.model.fit(train_dummys_df.drop(["Project_name", "real_TF"], axis=1), train_dummys_df["real_TF"])
        except ValueError as e:
            print(e)

        return self.model, convention_dummys
    
    def predict_merge_model(self, project_list, merge_model, convention_dummys):
        for project_name in project_list:
            print(project_name)
            id_dict = {}
            for dum_wid in list(convention_dummys):
                id_dict[dum_wid] = []
            X_test = self.__fetch_test(project_name)

            for test_wid in X_test["Warning ID"]:
                #全てのキーに対して0を追加
                for i in id_dict.keys():
                    id_dict[i].append(0)
                #test_widがあれば、その位置の0を1に更新
                if test_wid in id_dict:
                    id_dict[test_wid][-1] = 1

            id_df = pd.DataFrame(id_dict)
            test_df = pd.concat([id_df, X_test], axis=1)

            #予測結果の表示
            predict_result = merge_model.predict(test_df.drop(["Warning ID", "Project_name", "real_TF"], axis=1))
            test_df["predict_TF"] = predict_result
            result_df = pd.concat([X_test["Warning ID"], test_df["real_TF"], test_df["predict_TF"]], axis=1)
            self.__calc_score_by_convention(result_df, f"{path.PRERESULT}/merge/{self.model_name}/{project_name}_merge_{project_list[0]}_{project_list[1]}.csv")