import scipy as sp  # type:ignore
from sklearn.model_selection import ParameterSampler  # type:ignore

rf_hp_dict = {
    "model__max_features": sp.stats.uniform(0, 1),
    "model__max_samples": sp.stats.uniform(0, 1),
    "model__n_estimators": [
        50,
        100,
        150,
        200,
        250,
        300,
        350,
        400,
        450,
        500,
        550,
        600,
        650,
        700,
        750,
        800,
    ],
    "model__max_depth": [
        3,
        6,
        9,
        12,
        15,
        18,
        21,
        24,
        27,
        30,
        33,
        36,
        39,
        42,
        45,
        48,
        51,
        54,
        57,
        60,
        63,
        66,
    ],
    "model__min_samples_leaf": [
        1,
        2,
        3,
        5,
        7,
        9,
        11,
        14,
        17,
        20,
        23,
        26,
        29,
        32,
        35,
        38,
        41,
        44,
    ],
    "model__n_jobs": [-1],
    "model__random_state": [2022],
}

xgb_hp_dict = {"model__tree_method": ["gpu_hist"],  # this parameter means using the GPU when training our model to speedup the training process
"model__lambda": sp.stats.loguniform(1e-3, 10.0),
"model__alpha": sp.stats.loguniform(1e-3, 10.0),
"model__colsample_bytree": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
"model__subsample": [0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
"model__learning_rate": sp.stats.loguniform(0.01, 0.1),
"model__n_estimators": [
        50,
        100,
        150,
        200,
        250,
        300,
        350,
        400,
        450,
        500,
        550,
        600,
        650,
        700,
        750,
        800,
    ],
"model__max_depth": [
        3,
        6,
        9,
        12,
        15,
        18,
        21,
        24,
        27,
        30,
    ],
"model__min_child_weight": sp.stats.uniform(1, 300),
"model__seed": [2022]
}

hp_list = list(
    ParameterSampler(
        rf_hp_dict,
        n_iter=40,
        random_state=2022,
    )
)
hp = [dict((k, round(v, 4)) for (k, v) in d.items()) for d in hp_list]

if __name__ == "__main__":
    import pprint

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(hp_list)