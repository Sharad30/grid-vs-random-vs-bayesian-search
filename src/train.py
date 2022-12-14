import time

import pandas as pd  # type:ignore
from dataset import Dataset  # type:ignore
from sklearn.compose import ColumnTransformer  # type:ignore
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)  # type:ignore
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV  # type:ignore
from sklearn.pipeline import Pipeline  # type:ignore
from sklearn.preprocessing import LabelEncoder  # type:ignore
import openml
from openml.datasets import get_dataset
from openml.tasks import TaskType
from sklearn.impute import SimpleImputer
from utils import fetch_data, get_pipeline

df_classification = openml.tasks.list_tasks(
    task_type=TaskType.SUPERVISED_CLASSIFICATION,
    output_format="dataframe",
)
df_classification = df_classification[df_classification.NumberOfClasses != 0]
df_regression = openml.tasks.list_tasks(
    task_type=TaskType.SUPERVISED_REGRESSION,
    output_format="dataframe",
)


rf_config = {
    "model_name": "rf",
    "classification": {"model": RandomForestClassifier(), "metric": "accuracy"},
    "regression": {
        "model": RandomForestRegressor(),
        "metric": "neg_root_mean_squared_error",
    },
}

xgb_config = {
    "model_name": "xgb",
    "classification": {"model": xgb.XGBClassifier(), "metric": "accuracy"},
    "regression": {
        "model": xgb.XGBRegressor(),
        "metric": "neg_root_mean_squared_error",
    },
}

def train(
    task_type: str = "classification",
    model_config: dict = rf_config,
):
    best_models = []
    all_models = pd.DataFrame()
    for name in d.openml.name.values:
        start = time.time()
        d = fetch_data(name=name, task_type=task_type)
        pipeline = get_pipeline(model_config=model_config, task_type=task_type, dataset=d)

        random_searcher = RandomizedSearchCV(
            pipeline,
            n_iter=1,
            param_distributions={
                "model__n_jobs": [-1],
                "model__random_state": [2022],
            },
            n_jobs=-1,
            cv=5,
            random_state=2022,
            scoring=model_config[task_type]["metric"],
        )
        le = LabelEncoder()
        d.y_train = le.fit_transform(d.y_train)
        d.y_test = le.transform(d.y_test)
        random_searcher.fit(d.X_train, d.y_train)
        end = time.time()

        best_models += [
            {
                "name": name,
                "time_elapsed": end - start,
                "best_score": random_searcher.best_score_,
                "test_score": random_searcher.score(d.X_test, d.y_test),
                **random_searcher.best_params_,
            }
        ]
        best_models_pd = pd.DataFrame.from_records(best_models).drop(
            ["model__random_state", "model__n_jobs"], axis=1
        )
        best_models_pd.to_csv(
            f"{model_config['model_name']}_{task_type}_best_models.csv"
        )

        new_model = pd.DataFrame(random_searcher.cv_results_)
        new_model["name"] = name
        all_models = all_models.append(new_model)
        all_models.drop(
            ["params", "param_model__random_state", "param_model__n_jobs"], axis=1
        ).to_csv(f"{model_config['model_name']}_{task_type}_all_models.csv")
        print(f"Completed {name} training")


train(task_type="classification", model_config=rf_config)
# train(task_type="regression", df_dataset=df_regression, model_config=rf_config)

train(task_type="classification", model_config=xgb_config)
# train(task_type="regression", df_dataset=df_regression, model_config=xgb_config)
