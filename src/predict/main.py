from tqdm import tqdm
import pandas as pd
from predict import ModelCreater
from constants import path
from utils import lookup_white_list

def single_convention():
    model_name = "RandomForest"
    project_list = lookup_white_list(path.WHITELIST)
    for project_name in tqdm(project_list):
        mc = ModelCreater(model_name)
        model, dummys = mc.fit_single_model(project_name)
        _, result_convention = mc.predict_single_model(project_name, model, dummys)
        result_convention.to_csv(f"{path.PRERESULT}/single/{model_name}/violations/{project_name}.csv")

def all():
    """全体の予測値を算出するメソッド
    """
    model_name = "RandomForest"
    project_list = lookup_white_list(path.WHITELIST)
    for project_name in tqdm(project_list, total=len(project_list)):
        mc = ModelCreater(model_name)
        single_model, single_dummys = mc.fit_single_model(project_name)
        combined_df, _ = mc.predict_single_model(project_name, single_model, single_dummys)

        for merge_name in tqdm((project_list), desc=f"Merging with {project_name}", leave=False):
            if project_name == merge_name:
                continue
            merge_list = [project_name, merge_name]
            merge_model, merge_dummys = mc.fit_merge_model(merge_list)
            result_merge, _ = mc.predict_merge_model(merge_list, merge_model, merge_dummys)
            combined_df = pd.concat([combined_df, result_merge[0]], axis=0)

        combined_df.to_csv(f"{path.PRERESULT}/{model_name}/all/{project_name}.csv", index=True)

        
if __name__ == "__main__":
    all()