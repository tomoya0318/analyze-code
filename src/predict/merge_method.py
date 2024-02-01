import pandas as pd
from predict import create_merge_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# 宣言
id_dict = {}
model_name = "RandomForest"  # Logistic, RandomForest, SVMの３種類から選ぶ
model_all, dummys = create_merge_model(10, model_name)
result_df = pd.DataFrame(columns=["precision", "recall", "f1_score", "accuracy"])
for i in list(dummys):
    id_dict[i] = []

# project_list = ["GPflow", "python-sdk"]
project_list = ["GPflow", "hickle"]

path = "few_data"

for project_name in project_list:
    df_value = pd.read_csv(f"{path}/{project_name}_value.csv")
    df_label = pd.read_csv(f"{path}/{project_name}_label.csv", header=None)
    _, X_test, _, Y_test = train_test_split(df_value, df_label, test_size=0.2, shuffle=False)
    Y_test = Y_test.values.ravel()
    X_test["real_TF"] = Y_test
    X_test = X_test.reset_index(drop=True)

    id_dict.clear()
    for i in list(dummys):
        id_dict[i] = []
    for wid in X_test["Warning ID"]:
        if wid in id_dict:
            id_dict[wid].append(1)
            for i in id_dict:
                id_dict[i].append(0)
            id_dict[wid].pop(-1)
        else:
            for i in id_dict:
                id_dict[i].append(0)

    id_df = pd.DataFrame(id_dict)
    test_df = pd.concat([id_df, X_test], axis=1)
    # print(test_df)

    # predict_result = model_all.predict(test_df.drop(['Warning ID', 'Project_name', 'Cluster_num', "AnsTF"], axis=1))
    predict_result = model_all.predict(test_df.drop(["Warning ID", "Project_name", "real_TF"], axis=1))

    test_df["predict_TF"] = predict_result
    test_df.to_csv(f"results/merge_{model_name}_{project_name}Gh.csv")

