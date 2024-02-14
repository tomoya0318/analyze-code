from tqdm import tqdm
import pandas as pd
from predict import ModelCreater
from constants import path
from utils import lookup_white_list
def single_all():
    model_name = "RandomForest"
    project_list = lookup_white_list(f"{path.DATA}/white_list.txt")
    for project_name in tqdm(project_list):
        mc = ModelCreater(model_name)
        model, dummys = mc.fit_single_model(project_name)
        result_all, _ = mc.predict_single_model(project_name, model, dummys)
        result_all.to_csv(f"{path.PRERESULT}/single/{model_name}/all/{project_name}.csv")

def single_convention():
    model_name = "RandomForest"
    project_list = lookup_white_list(f"{path.DATA}/white_list.txt")
    for project_name in tqdm(project_list):
        mc = ModelCreater(model_name)
        model, dummys = mc.fit_single_model(project_name)
        _, result_convention = mc.predict_single_model(project_name, model, dummys)
        result_convention.to_csv(f"{path.PRERESULT}/single/{model_name}/violations/{project_name}.csv")


if __name__ == "__main__":
    