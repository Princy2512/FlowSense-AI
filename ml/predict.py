import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
try:
    import geohash  # type: ignore
except Exception:  # pragma: no cover
    import geohash2 as geohash  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "ml" / "artifacts" / "model.joblib"


def _parse_timestamp(ts: str) -> tuple[int, int]:
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


def _safe_label(label_maps: dict, col: str, value: object) -> int:
    mp = label_maps.get(col, {})
    s = str(value) if value is not None else ""
    if s in mp:
        return int(mp[s])
    # unseen category -> map to 0 (same behavior as LabelEncoder unknown fallback would not exist)
    return 0


def _target_enc_value(te_bundle: dict, enc_name: str, keys: list[object]) -> float:
    global_mean = float(te_bundle.get("global_mean", 0.0))
    enc = te_bundle.get("encodings", {}).get(enc_name)
    if not enc:
        return global_mean

    cols = enc["group_cols"]
    table = enc["table"]  # dict of lists
    # build lookup dict for fast access
    # keys are tuples aligned with cols
    lookup = {}
    n = len(table[enc_name])
    for i in range(n):
        k = tuple(table[c][i] for c in cols)
        lookup[k] = float(table[enc_name][i])
    return float(lookup.get(tuple(keys), global_mean))


def preprocess_one(payload: dict) -> pd.DataFrame:
    gh = payload.get("geohash", "")
    ts = payload.get("timestamp", "0:0")
    hour, minute = _parse_timestamp(ts)
    total_minutes = hour * 60 + minute
    lat, lon = _decode_geohash(gh)

    return pd.DataFrame(
        [
            {
                "geohash": str(gh),
                "day": payload.get("day", np.nan),
                "timestamp": str(ts),
                "hour": hour,
                "minute": minute,
                "total_minutes": total_minutes,
                "latitude": lat,
                "longitude": lon,
                "RoadType": payload.get("RoadType", None),
                "NumberofLanes": payload.get("NumberofLanes", np.nan),
                "LargeVehicles": payload.get("LargeVehicles", None),
                "Landmarks": payload.get("Landmarks", None),
                "Temperature": payload.get("Temperature", np.nan),
                "Weather": payload.get("Weather", None),
            }
        ]
    )


def predict(payload: dict) -> float:
    bundle = joblib.load(ARTIFACTS)
    X = preprocess_one(payload)

    label_maps = bundle.get("label_maps", {})
    te_bundle = bundle.get("target_encoding", {})

    # label encode categoricals
    for col in ["RoadType", "LargeVehicles", "Landmarks", "Weather"]:
        X[col] = X[col].apply(lambda v: _safe_label(label_maps, col, v)).astype(int)

    # target encoding features
    X["target_enc_geohash"] = X.apply(
        lambda r: _target_enc_value(te_bundle, "target_enc_geohash", [r["geohash"]]), axis=1
    )
    X["target_enc_timestamp"] = X.apply(
        lambda r: _target_enc_value(te_bundle, "target_enc_timestamp", [r["timestamp"]]), axis=1
    )
    X["target_enc_geohash_hour"] = X.apply(
        lambda r: _target_enc_value(te_bundle, "target_enc_geohash_hour", [r["geohash"], int(r["hour"])]),
        axis=1,
    )

    features = bundle["features"]
    Xf = X[features]
    pred = float(bundle["model"].predict(Xf)[0])
    return float(pred)


def main() -> None:
    if not ARTIFACTS.exists():
        raise FileNotFoundError(
            f"Missing model artifacts at {ARTIFACTS}. Run: python ml/train.py"
        )

    raw = sys.stdin.read().strip()
    if not raw:
        print(json.dumps({"error": "Empty input"}))
        return
    payload = json.loads(raw)

    try:
        pred = predict(payload)
        print(json.dumps({"demand": pred}))
    except Exception as e:
        print(json.dumps({"error": str(e)}))


if __name__ == "__main__":
    main()

