import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from predict import ModelCreater
from cosin import lookup_from_model, calc_cosin
from utils import lookup_white_list
from constants import path

def main():
    model_name = "RandomForest"
    project_list = lookup_white_list(path.WHITELIST)
    fix_rate_list = []
    cosin_sim_list = []
    f1_score_list = []
    accuracy_list = []
    df = pd.DataFrame()

    for name1 in tqdm(project_list, total=len(project_list)):
        mc = ModelCreater(model_name)
        for name2 in tqdm(project_list, desc=f"comparing with {name1}", leave=False):
            if name1 == name2:
                continue
            name1_dict = lookup_from_model(f"{path.PRERESULT}/raw_model/{name1}_train.csv")
            name2_dict = lookup_from_model(f"{path.PRERESULT}/raw_model/{name2}_train.csv")
            fix_rate = sum(item[0] for item in name2_dict.values()) / sum(item[1] for item in name2_dict.values())
            fix_rate_list.append(fix_rate)
            cosin_sim, nan_count = calc_cosin(name1_dict, name2_dict)
            cosin_sim_list.append(cosin_sim)

            #統合モデルの作成
            merge_list = [name1, name2]
            merge_model, merge_dummys = mc.fit_merge_model(merge_list)
            result_merge, _ = mc.predict_merge_model(merge_list, merge_model, merge_dummys)
            f1_score_list.append(result_merge[0]["f1_score"].iloc[0])
            accuracy_list.append(result_merge[0]["accuracy"].iloc[0])
            
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "Project 1": [name1],
                            "Project 2": [name2],
                            "fix_count": [sum(item[0] for item in name2_dict.values())],
                            "total_count": [sum(item[1] for item in name2_dict.values())],
                            "fix_rate": [fix_rate],
                            "cos_sim": [cosin_sim],
                            "not_nan": [nan_count],
                            "f1_score":[result_merge[0]["f1_score"].iloc[0]],
                            "accuracy":[result_merge[0]["accuracy"].iloc[0]]
                        }
                    ),
                ],
                ignore_index=True,
            )

    #データ準備
    fix = np.array(fix_rate_list)
    cos = np.array(cosin_sim_list)
    f1 = np.array(f1_score_list)
    accuracy = np.array(accuracy_list)

    plot_scatter(fix, f1, "修正率", "F値", f"{path.OUT}/fix_f1.png")
    plot_scatter(cos, f1, "コサイン類似度", "F値", f"{path.OUT}/cos_f1.png")
    plot_scatter(fix, accuracy, "修正率", "正解率", f"{path.OUT}/fix_accuracy.png")
    plot_scatter(cos, accuracy, "コサイン類似度", "正解率", f"{path.OUT}/cos_accuracy.png")

    
    df.to_csv(f"{path.OUT}/f1_accuracy.csv", index=False)

def generate_comparison_results_csv():
    model_name = "RandomForest"
    project_list = lookup_white_list(path.WHITELIST)
    df = pd.DataFrame()

    for name1 in tqdm(project_list, total=len(project_list)):
        mc = ModelCreater(model_name)
        for name2 in tqdm(project_list, desc=f"comparing with {name1}", leave=False):
            if name1 == name2:
                continue
            name1_dict = lookup_from_model(f"{path.PRERESULT}/raw_model/{name1}_train.csv")
            name2_dict = lookup_from_model(f"{path.PRERESULT}/raw_model/{name2}_train.csv")
            fix_count = sum(item[0] for item in name1_dict.values()) + sum(item[0] for item in name2_dict.values())
            total_count = sum(item[1] for item in name1_dict.values()) + sum(item[1] for item in name2_dict.values())
            fix_rate = format(fix_count / total_count, ".2f")
            cosin_sim, nan_count = calc_cosin(name1_dict, name2_dict)

            #統合モデルの作成
            merge_list = [name1, name2]
            merge_model, merge_dummys = mc.fit_merge_model(merge_list)
            result_merge, _ = mc.predict_merge_model(merge_list, merge_model, merge_dummys)
            
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "Project 1": [name1],
                            "Project 2": [name2],
                            "fix_count": [fix_count],
                            "total_count": [total_count],
                            "fix_rate": [fix_rate],
                            "cos_sim": [cosin_sim],
                            "not_nan": [nan_count],
                            "f1_score":[result_merge[0]["f1_score"].iloc[0]],
                            "accuracy":[result_merge[0]["accuracy"].iloc[0]]
                        }
                    ),
                ],
                ignore_index=True,
            )
    df.to_csv(f"{path.OUT}/f1_accuracy.csv", index=False)

def plot_scatter(x, y, x_label, y_label, save_path):
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    plt.savefig(save_path)

if __name__ == "__main__":
    generate_comparison_results_csv()