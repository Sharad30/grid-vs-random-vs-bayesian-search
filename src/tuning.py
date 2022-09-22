import time

import pandas as pd  # type:ignore
from dataset import Dataset  # type:ignore
from hyperparameters import rf_hp_dict, xgb_hp_dict
from sklearn.compose import ColumnTransformer  # type:ignore
from sklearn.ensemble import RandomForestClassifier  # type:ignore
from sklearn.model_selection import RandomizedSearchCV  # type:ignore
from sklearn.pipeline import Pipeline  # type:ignore
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type:ignore
from utils import fetch_data, get_pipeline
import xgboost as xgb
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from typing import Union

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
    output_path: Union[str, Path] = Path("output"),
    hyperparams_dict: dict = dict(),
):
    best_models = []
    all_models = pd.DataFrame()

    d = Dataset()

    for name in d.openml.name.values:
        start = time.time()
        d = fetch_data(name=name, task_type=task_type)
        pipeline = get_pipeline(model_config=model_config, task_type=task_type, dataset=d)

        random_searcher = RandomizedSearchCV(
            pipeline,
            param_distributions=hyperparams_dict,
            n_iter=1,
            n_jobs=-1,
            cv=5,
            random_state=2022,
            scoring="accuracy",
        )
        le = LabelEncoder()
        d.y_train = le.fit_transform(d.y_train)
        d.y_test = le.transform(d.y_test)
        random_searcher.fit(d.X_train, d.y_train)
        end = time.time()

        best_models += [
            {
                "name": d.name,
                "time_elapsed": end - start,
                "best_score": random_searcher.best_score_,
                "test_score": random_searcher.score(d.X_test, d.y_test),
                **random_searcher.best_params_,
            }
        ]
        best_models_pd = pd.DataFrame.from_records(best_models)
        best_models_pd.to_csv(
            output_path / "tuned" / f"{model_config['model_name']}_{task_type}_best_models.csv"
        )

        new_model = pd.DataFrame(random_searcher.cv_results_)
        new_model["name"] = d.name
        all_models = all_models.append(new_model)
        all_models.drop(
            ["params"], axis=1
        ).to_csv(output_path / "tuned" / f"{model_config['model_name']}_{task_type}_all_models.csv")
        print(f"Completed {name} training")

output_path = Path("output")
# train(task_type="classification", model_config=rf_config, output_path=output_path, hyperparams_dict=rf_hp_dict)
train(task_type="classification", model_config=xgb_config, output_path=output_path, hyperparams_dict=xgb_hp_dict)