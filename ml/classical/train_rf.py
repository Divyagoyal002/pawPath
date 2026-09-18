"""Classical ML track: feature engineering + SelectKBest + Random Forest.

Lightweight, interpretable, works with smaller datasets (docs/PRD.md
section 6.3). Reports both the 3-class (healthy/orthopedic/neurological)
and the binary healthy-vs-abnormal task, evaluated on dogs unseen during
training (FR-4).
"""

import argparse
import os
import sys

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ml.common.dataset import build_windowed_dataset
from signal_processing.features import extract_feature_matrix


def to_binary(labels: np.ndarray) -> np.ndarray:
    return np.where(labels == "healthy", "healthy", "abnormal")


def build_pipeline(k_features: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k=k_features)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "..", "data", "synthetic"))
    parser.add_argument("--k-features", type=int, default=30)
    parser.add_argument("--model-out", default=os.path.join(os.path.dirname(__file__), "rf_model.joblib"))
    args = parser.parse_args()

    print("Loading and windowing dataset...")
    data = build_windowed_dataset(args.data_dir)
    print(f"Train dogs: {data['train_dogs']}")
    print(f"Test dogs (held out): {data['test_dogs']}")
    print(f"Train windows: {data['X_train'].shape[0]}, Test windows: {data['X_test'].shape[0]}")

    print("Extracting features...")
    X_train_feats, feature_names = extract_feature_matrix(data["X_train"])
    X_test_feats, _ = extract_feature_matrix(data["X_test"])
    k = min(args.k_features, X_train_feats.shape[1])

    for task_name, y_train, y_test in [
        ("3-class (healthy/orthopedic/neurological)", data["y_train"], data["y_test"]),
        ("binary (healthy/abnormal)", to_binary(data["y_train"]), to_binary(data["y_test"])),
    ]:
        print(f"\n=== {task_name} ===")
        pipeline = build_pipeline(k)
        pipeline.fit(X_train_feats, y_train)
        y_pred = pipeline.predict(X_test_feats)

        print(classification_report(y_test, y_pred, zero_division=0))
        labels = sorted(set(y_test) | set(y_pred))
        print("Confusion matrix", labels)
        print(confusion_matrix(y_test, y_pred, labels=labels))

        if task_name.startswith("3-class"):
            joblib.dump({"pipeline": pipeline, "feature_names": feature_names}, args.model_out)
            print(f"Saved 3-class model to {args.model_out}")


if __name__ == "__main__":
    main()
