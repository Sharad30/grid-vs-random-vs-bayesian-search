from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from pydantic import BaseModel
import openml
from datalists import openml_df


class Dataset(BaseModel):
    name: str = ""
    x: pd.DataFrame = pd.DataFrame()
    y: np.ndarray = np.ndarray(shape=(0,))
    openml: pd.DataFrame = openml_df
    numerical_ix: list = []
    categorical_ix: list = []
    X_train: np.ndarray = np.ndarray(shape=(0,))
    X_test: np.ndarray = np.ndarray(shape=(0,))
    y_train: np.ndarray = np.ndarray(shape=(0,))
    y_test: np.ndarray = np.ndarray(shape=(0,))

    class Config:
        arbitrary_types_allowed = True

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int) -> Tuple:
        return (self.x[i], self.y[i])

    def describe(self):
        return self.x.describe()

    def fetch_dataset(self, name: str = 0, task_type: str = "classification"):
        self.name = name
        dataset_id = int(
            self.openml[self.openml.name == name].openml_dataset_id.values[0]
        )
        dataset = openml.datasets.get_dataset(dataset_id)
        print(f"Dataset {name} download starting.")
        self.x, self.y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute
        )
        print(f"Dataset {name} download complete.")
        self.numerical_ix = self.x.select_dtypes(
            include=[np.number, "int64", "float64"]
        ).columns
        self.categorical_ix = self.x.select_dtypes(
            include=["object", "bool", "category"]
        ).columns
        if task_type == "regression":
            stratify = None
        else:
            stratify = self.y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=0.2, random_state=2022, stratify=stratify
        )
        return self
