import pandas as pd
from utils import calculate_cosin_similarity
from utils import load_correction_data as ld
from constants import path


def main():
    path_in = f"{path.OUT}/cleaned_tracking_all_convention.csv"
    # プロジェクトの選択
    project_list = []
    with open(f"{path.DATA}/white_list.txt", "r") as f:
        for line in f:
            project_list.append(line.strip())
    calculate_cosin_similarities(project_list, path_in)


def calculate_cosin_similarities(project_list, path_in):
    project_num = len(project_list)
    # データフレームの作成
    df = pd.DataFrame(columns=["Project 1", "Project 2", "Similarity", "Count"])

    # 組み合わせの表示
    project_correction_rates = ld.correction_from_csv(path_in)

    for i in range(project_num):
        for j in range(i, project_num):
            result_combinations = calculate_cosin_similarity.cos_sim(
                project_correction_rates[i], project_correction_rates[j]
            )
            cosin_similarity = result_combinations[0]
            count = result_combinations[1]
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "Project 1": [project_list[i]],
                            "Project 2": [project_list[j]],
                            "Similarity": [cosin_similarity],
                            "Count": [count],
                        }
                    ),
                ],
                ignore_index=True,
            )

    # 結果の表示
    df.to_csv(f"{path.OUT}/sort_cosine.csv", index=False)


if __name__ == "__main__":
    main()
