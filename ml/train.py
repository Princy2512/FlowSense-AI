import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
try:
    import geohash  # type: ignore
except Exception:  # pragma: no cover
    import geohash2 as geohash  # type: ignore
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "ml" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def _parse_timestamp(ts: str) -> tuple[int, int]:
    # Dataset uses formats like "0:0", "2:15", etc.
    parts = str(ts).strip().split(":")
    if len(parts) != 2:
        return 0, 0
    try:
        h = int(parts[0])
    except Exception:
        h = 0
    try:
        m = int(parts[1])
    except Exception:
        m = 0
    h = max(0, min(23, h))
    m = max(0, min(59, m))
    return h, m


def _decode_geohash(gh: str) -> tuple[float, float]:
    try:
        lat, lon = geohash.decode(str(gh))
        return float(lat), float(lon)
    except Exception:
        return float("nan"), float("nan")


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # time features
    hm = df["timestamp"].apply(_parse_timestamp)
    df["hour"] = hm.apply(lambda x: x[0])
    df["minute"] = hm.apply(lambda x: x[1])
    df["total_minutes"] = df["hour"] * 60 + df["minute"]

    # spatial features
    ll = df["geohash"].apply(_decode_geohash)
    df["latitude"] = ll.apply(lambda x: x[0])
    df["longitude"] = ll.apply(lambda x: x[1])

    # Impute missing values (match your notebook logic)
    if "RoadType" in df.columns:
        df["RoadType"] = df["RoadType"].fillna(df["RoadType"].mode(dropna=True)[0])
    if "Weather" in df.columns:
        df["Weather"] = df["Weather"].fillna(df["Weather"].mode(dropna=True)[0])
    if "Temperature" in df.columns:
        df["Temperature"] = df["Temperature"].fillna(df["Temperature"].median())

    return df


def add_oof_target_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    group_cols: list,
    n_folds: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Adds OOF target mean encodings to train and full-mean encodings to test.
    Returns (train, test, mappings) where mappings are full-mean dicts for inference.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    global_mean = float(train[target_col].mean())
    mappings: dict[str, dict] = {"global_mean": global_mean, "encodings": {}}

    for col in group_cols:
        col_key = "_".join(col) if isinstance(col, list) else str(col)
        new_col_name = f"target_enc_{col_key}"
        train[new_col_name] = np.nan

        for tr_idx, va_idx in kf.split(train):
            df_train = train.iloc[tr_idx]
            means = df_train.groupby(col)[target_col].mean().reset_index()
            means.columns = (col if isinstance(col, list) else [col]) + [new_col_name]

            df_val = train.iloc[va_idx].drop(columns=[new_col_name])
            df_val = df_val.merge(means, on=col, how="left")
            train.loc[train.index[va_idx], new_col_name] = df_val[new_col_name].values

        # full means for test + inference
        full_means = train.groupby(col)[target_col].mean().reset_index()
        full_means.columns = (col if isinstance(col, list) else [col]) + [new_col_name]
        test = test.merge(full_means, on=col, how="left")

        train[new_col_name] = train[new_col_name].fillna(global_mean)
        test[new_col_name] = test[new_col_name].fillna(global_mean)

        mappings["encodings"][new_col_name] = {
            "group_cols": col if isinstance(col, list) else [col],
            "table": full_means.to_dict(orient="list"),
        }

    return train, test, mappings


def build_metadata(train_raw: pd.DataFrame) -> dict:
    # The UI needs dropdown values (keep geohash list small).
    road_types = (
        train_raw["RoadType"].dropna().astype(str).value_counts().index.tolist()
    )
    weathers = train_raw["Weather"].dropna().astype(str).value_counts().index.tolist()
    large_vehicles = (
        train_raw["LargeVehicles"].dropna().astype(str).value_counts().index.tolist()
    )
    landmarks = (
        train_raw["Landmarks"].dropna().astype(str).value_counts().index.tolist()
    )

    # Most frequent geohashes for dropdown; allow custom input in UI too.
    geohash_top = (
        train_raw["geohash"].dropna().astype(str).value_counts().head(250).index.tolist()
    )

    day_min = int(pd.to_numeric(train_raw["day"], errors="coerce").min())
    day_max = int(pd.to_numeric(train_raw["day"], errors="coerce").max())

    lanes = (
        pd.to_numeric(train_raw["NumberofLanes"], errors="coerce")
        .dropna()
        .astype(int)
        .value_counts()
        .index.sort_values()
        .tolist()
    )
    if not lanes:
        lanes = [1, 2, 3, 4]

    # Use observed timestamp values for dropdown
    timestamp_values = (
        train_raw["timestamp"].dropna().astype(str).value_counts().index.tolist()
    )

    temp_series = pd.to_numeric(train_raw["Temperature"], errors="coerce")
    temp_min = float(np.nanmin(temp_series.values))
    temp_max = float(np.nanmax(temp_series.values))

    return {
        "categorical": {
            "RoadType": road_types,
            "Weather": weathers,
            "LargeVehicles": large_vehicles,
            "Landmarks": landmarks,
        },
        "geohashTop": geohash_top,
        "dayRange": {"min": day_min, "max": day_max},
        "lanesOptions": lanes,
        "timestampOptions": timestamp_values,
        "temperatureRange": {"min": temp_min, "max": temp_max},
    }


def main() -> None:
    train_path = ROOT / "train.csv"
    test_path = ROOT / "test.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing {test_path}")

    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)

    train = preprocess_df(train_raw)
    test = preprocess_df(test_raw)

    # Encode categoricals with per-column LabelEncoders fit separately on train and test
    # to mimic the original pipeline behavior.
    categorical_cols = ["RoadType", "LargeVehicles", "Landmarks", "Weather"]
    encoders: dict[str, dict] = {}
    for col in categorical_cols:
        le_train = LabelEncoder()
        le_test = LabelEncoder()

        le_train.fit(train[col].astype(str))
        train[col] = le_train.transform(train[col].astype(str))

        le_test.fit(test[col].astype(str))
        test[col] = le_test.transform(test[col].astype(str))

        # For inference, we replicate the test-side encoding.
        encoders[col] = {cls: int(i) for i, cls in enumerate(le_test.classes_)}

    # Add target encoding features (your exact columns)
    train, test, te_mappings = add_oof_target_features(
        train=train,
        test=test,
        target_col="demand",
        group_cols=["geohash", "timestamp", ["geohash", "hour"]],
        n_folds=5,
        random_state=42,
    )

    features = [
        "day",
        "total_minutes",
        "latitude",
        "longitude",
        "RoadType",
        "NumberofLanes",
        "LargeVehicles",
        "Landmarks",
        "Temperature",
        "Weather",
        "target_enc_geohash",
        "target_enc_timestamp",
        "target_enc_geohash_hour",
    ]

    X = train[features]
    y = train["demand"].astype(float)

    lgb_params = {
        "objective": "regression",
        "learning_rate": 0.03,
        "n_estimators": 3000,
        "random_state": 42,
    }
    xgb_params = {
        "objective": "reg:squarederror",
        "learning_rate": 0.03,
        "n_estimators": 3000,
        "max_depth": 10,
        "random_state": 42,
    }
    cat_params = {
        "loss_function": "RMSE",
        "learning_rate": 0.03,
        "iterations": 3000,
        "depth": 8,
        "random_seed": 42,
        "verbose": False,
    }

    # CV score for sanity (your ensemble weights)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    for tr_idx, va_idx in kf.split(X, y):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[va_idx]

        m_lgb = lgb.LGBMRegressor(**lgb_params)
        m_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100)])

        m_xgb = xgb.XGBRegressor(**xgb_params)
        m_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        m_cat = CatBoostRegressor(**cat_params)
        m_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val))

        oof[va_idx] = (
            (m_lgb.predict(X_val) * 0.4)
            + (m_xgb.predict(X_val) * 0.3)
            + (m_cat.predict(X_val) * 0.3)
        )
    cv_r2 = float(r2_score(y, oof))

    # Train final models on full data
    final_lgb = lgb.LGBMRegressor(**lgb_params).fit(X, y)
    final_xgb = xgb.XGBRegressor(**xgb_params).fit(X, y, verbose=False)
    final_cat = CatBoostRegressor(**cat_params).fit(X, y)

    joblib.dump(
        {
            "models": {"lgb": final_lgb, "xgb": final_xgb, "cat": final_cat},
            "weights": {"lgb": 0.4, "xgb": 0.3, "cat": 0.3},
            "features": features,
            "label_maps": encoders,
            "target_encoding": te_mappings,
        },
        ARTIFACTS_DIR / "model.joblib",
    )

    metadata = build_metadata(train_raw)
    metadata["cvR2"] = float(cv_r2)
    (ARTIFACTS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Saved: {ARTIFACTS_DIR / 'model.joblib'}")
    print(f"Saved: {ARTIFACTS_DIR / 'metadata.json'}")
    print(f"CV R2 (sanity): {cv_r2:.4f}")


if __name__ == "__main__":
    main()

