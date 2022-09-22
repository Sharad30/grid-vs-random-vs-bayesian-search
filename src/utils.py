from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    LabelEncoder,
)  # type:ignore
from openml.datasets import get_dataset
from openml.tasks import TaskType
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline  # type:ignore
from sklearn.compose import ColumnTransformer
from dataset import Dataset

def fetch_data(name: str, task_type: str):
    d = Dataset()
    print(f"Starting to train dataset: {name}")
    d = d.fetch_dataset(name=name, task_type=task_type)
    print(f"Downloaded dataset has {d.x.shape[0]} rows and {d.x.shape[1]} columns")
    print(f"Length of the target variable: {len(d.y)}")
    print(f"Column types: {d.x.dtypes}")
    return d


def get_pipeline(model_config : dict, task_type: str, dataset: Dataset):
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, dataset.numerical_ix),
            ("cat", categorical_transformer, dataset.categorical_ix),
        ]
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model_config[task_type]["model"]),
        ]
    )
    return pipeline