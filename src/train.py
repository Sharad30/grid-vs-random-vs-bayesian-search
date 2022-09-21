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
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    LabelEncoder,
)  # type:ignore
import openml
from openml.datasets import get_dataset
from openml.tasks import TaskType
from sklearn.impute import SimpleImputer

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
    df_dataset: pd.DataFrame = None,
    model_config: dict = rf_config,
):
    # dataset_ids = list(df_dataset.did.head(20).values)
    d = Dataset()
    best_models = []
    all_models = pd.DataFrame()
    for name in d.openml.name.values:
        start = time.time()
        # dataset_name = df_dataset[df_dataset.did == dataset_id].name.values[0]
        print(f"Starting to train dataset: {name}")
        # try:
        d = d.fetch_dataset(name=name, task_type=task_type)
        # except:
        # print(f"Dataset download failed for {dataset_name}")
        # continue
        print(f"Downloaded dataset has {d.x.shape[0]} rows and {d.x.shape[1]} columns")
        print(f"Length of the target variable: {len(d.y)}")
        print(f"Column types: {d.x.dtypes}")
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, d.numerical_ix),
                ("cat", categorical_transformer, d.categorical_ix),
            ]
        )
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model_config[task_type]["model"]),
            ]
        )

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
        # try:
        le = LabelEncoder()
        d.y_train = le.fit_transform(d.y_train)
        d.y_test = le.transform(d.y_test)
        random_searcher.fit(d.X_train, d.y_train)
        # except:
        #     print(f"Model training failed for {dataset_name}")
        #     continue
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


# train(task_type="classification", df_dataset=df_classification, model_config=rf_config)
# train(task_type="regression", df_dataset=df_regression, model_config=rf_config)

train(task_type="classification", df_dataset=df_classification, model_config=xgb_config)
# train(task_type="regression", df_dataset=df_regression, model_config=xgb_config)
