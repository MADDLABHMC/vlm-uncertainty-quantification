import numpy as np 
import pandas as pd
from pathlib import Path

def get_csv_details(csv_path):

    classes_df = pd.read_csv(csv_path)
    values = classes_df["name"].values

    print(type(values))

    for val in values:
        print(f"{val} {type(val)}")

    

if __name__ == "__main__":
    dataset_path = Path("/home/ubuntu/spring2026maddlabsg/datasets/1/semantic_drone")
    csv_path = dataset_path / "classes.csv"

    get_csv_details(csv_path)