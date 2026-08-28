"""Train and persist the single WebShield Random Forest model."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from feature_definitions import FEATURE_NAMES, HTML_FEATURES, extract_url_features


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "PhiUSIIL_Phishing_URL_Dataset.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "web_risk_model.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
RANDOM_SEED = 42
MODEL_VERSION = "1.0.0"


def load_training_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series, dict]:
    frame = pd.read_csv(path)
    required = {"URL", "label", *HTML_FEATURES}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    original_records = len(frame)
    duplicate_count = int(frame.duplicated(subset=["URL"], keep="first").sum())
    frame = frame.drop_duplicates(subset=["URL"], keep="first").copy()

    url_features = pd.DataFrame(frame["URL"].map(extract_url_features).tolist(), index=frame.index)
    model_frame = pd.concat([url_features, frame[HTML_FEATURES]], axis=1)[FEATURE_NAMES]
    model_frame = model_frame.apply(pd.to_numeric, errors="coerce")

    # UCI uses 1=legitimate, 0=phishing. WebShield explicitly reverses this.
    target = (1 - pd.to_numeric(frame["label"], errors="coerce")).rename("target")
    usable = model_frame.notna().all(axis=1) & target.isin([0, 1])
    removed_unusable = int((~usable).sum())
    model_frame = model_frame.loc[usable].astype(float)
    target = target.loc[usable].astype(int)

    audit = {
        "dataset_records": original_records,
        "exact_duplicate_urls_removed": duplicate_count,
        "unusable_records_removed": removed_unusable,
        "records_used": len(model_frame),
    }
    return model_frame, target, audit


def evaluate_model(model, features: pd.DataFrame, target: pd.Series) -> dict:
    predictions = model.predict(features)
    phishing_probability = model.predict_proba(features)[:, list(model.classes_).index(1)]
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision_phishing": float(precision_score(target, predictions, pos_label=1)),
        "recall_phishing": float(recall_score(target, predictions, pos_label=1)),
        "f1_phishing": float(f1_score(target, predictions, pos_label=1)),
        "roc_auc": float(roc_auc_score(target, phishing_probability)),
        "confusion_matrix": confusion_matrix(target, predictions, labels=[0, 1]).tolist(),
    }

def train() -> dict:
    X, y, audit = load_training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    model = RandomForestClassifier(
        n_estimators=160,
        max_depth=22,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)
    training_metrics = evaluate_model(model, X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    feature_importance = sorted(
        ({"feature": name, "importance": float(value)} for name, value in zip(FEATURE_NAMES, model.feature_importances_)),
        key=lambda item: item["importance"],
        reverse=True,
    )
    class_distribution = {
        "Legitimate": int((y == 0).sum()),
        "Phishing": int((y == 1).sum()),
    }
    train_hashes = pd.util.hash_pandas_object(X_train, index=False)
    test_hashes = pd.util.hash_pandas_object(X_test, index=False)
    repeated_test_rows = int(test_hashes.isin(set(train_hashes)).sum())
    feature_labels = X.copy()
    feature_labels["_target"] = y
    conflicting_groups = int(
        (feature_labels.groupby(FEATURE_NAMES)["_target"].nunique() > 1).sum()
    )
    overfitting_checks = {
        "training_accuracy": training_metrics["accuracy"],
        "testing_accuracy": metrics["accuracy"],
        "accuracy_gap": training_metrics["accuracy"] - metrics["accuracy"],
        "test_rows_matching_a_training_feature_vector": repeated_test_rows,
        "matching_test_row_fraction": repeated_test_rows / len(X_test),
        "identical_feature_groups_with_conflicting_labels": conflicting_groups,
    }
    metadata = {
        "model_version": MODEL_VERSION,
        "algorithm": "Random Forest Classifier",
        "task": "Binary Classification",
        "classes": ["Legitimate", "Phishing"],
        "label_mapping": {
            "source_uci": {"0": "Phishing", "1": "Legitimate"},
            "webshield_normalized": {"0": "Legitimate", "1": "Phishing"},
        },
        "dataset": "PhiUSIIL Phishing URL Website Dataset",
        "dataset_source": "UCI Machine Learning Repository",
        "dataset_id": 967,
        "dataset_license": "CC BY 4.0",
        "random_seed": RANDOM_SEED,
        "selected_features": FEATURE_NAMES,
        "selected_feature_count": len(FEATURE_NAMES),
        "url_feature_policy": "URL-derived columns are recomputed from raw URL with the shared inference function.",
        "training_sample_count": len(X_train),
        "testing_sample_count": len(X_test),
        "class_distribution": class_distribution,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "training_metrics": training_metrics,
        "metrics": metrics,
        "overfitting_checks": overfitting_checks,
        "global_feature_importance": feature_importance,
        **audit,
        "random_forest_parameters": {
            "n_estimators": 160,
            "max_depth": 22,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
        },
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Records used: {audit['records_used']:,}")
    print(f"Train/test: {len(X_train):,}/{len(X_test):,}")
    print(f"Training accuracy: {training_metrics['accuracy']:.8f}")
    print(f"Testing accuracy: {metrics['accuracy']:.8f}")
    print(f"Accuracy gap: {overfitting_checks['accuracy_gap']:.8f}")
    for name, value in metrics.items():
        print(f"test_{name}: {value}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")
    return metadata


if __name__ == "__main__":
    train()

