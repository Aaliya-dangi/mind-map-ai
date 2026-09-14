"""
MindMap AI - ML Model Training, Evaluation & Comparison
Builds preprocessing pipeline, trains Logistic Regression, Random Forest, and Gradient Boosting,
computes non-inflated genuine evaluation metrics, and persists artifacts via Joblib
per design.md §19-21 and RMD.md §9.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Add project root to path for imports
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from features.definitions import (
    RAW_NUMERICAL_FEATURES,
    DERIVED_NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    CATEGORICAL_VOCABULARIES,
)


def build_preprocessor() -> ColumnTransformer:
    """
    Creates a scikit-learn ColumnTransformer for numerical scaling and categorical one-hot encoding.
    """
    numerical_features = RAW_NUMERICAL_FEATURES + DERIVED_NUMERICAL_FEATURES
    categorical_features = CATEGORICAL_FEATURES

    # Ensure consistent categories across train and inference
    categories = [CATEGORICAL_VOCABULARIES[col] for col in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            (
                "cat",
                OneHotEncoder(categories=categories, handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names_out(preprocessor: ColumnTransformer) -> list:
    """
    Extracts human-readable feature names after one-hot encoding.
    """
    numerical_features = RAW_NUMERICAL_FEATURES + DERIVED_NUMERICAL_FEATURES
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    return numerical_features + cat_names


def train_and_evaluate_models(
    data_path: Path = config.PROCESSED_DATA_FILE,
    models_dir: Path = config.MODELS_DIR,
    metrics_file: Path = config.MODEL_METRICS_FILE,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Executes end-to-end ML training, evaluation, comparison, and persistence.
    """
    print(f"Loading cleaned dataset from {data_path}...")
    df = pd.read_csv(data_path)

    feature_cols = RAW_NUMERICAL_FEATURES + DERIVED_NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols]
    y = df["burnout_risk"]

    # Defined class order
    target_classes = ["Low", "Moderate", "High", "Very High"]

    # Stratified Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    print(f"Dataset split: Train = {len(X_train)} samples, Test = {len(X_test)} samples (stratified)")

    # Fit Preprocessing Pipeline
    preprocessor = build_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    feature_names = get_feature_names_out(preprocessor)

    # Initialize Candidate Models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=4,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=random_state,
        ),
    }

    results = {}
    trained_estimators = {}

    print("\n--- Training and Evaluating Models ---")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_trans, y_train)
        y_pred = model.predict(X_test_trans)
        trained_estimators[name] = model

        # Compute Genuine Metrics
        acc = float(accuracy_score(y_test, y_pred))
        prec_macro = float(precision_score(y_test, y_pred, labels=target_classes, average="macro", zero_division=0))
        rec_macro = float(recall_score(y_test, y_pred, labels=target_classes, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_test, y_pred, labels=target_classes, average="macro", zero_division=0))
        f1_weighted = float(f1_score(y_test, y_pred, labels=target_classes, average="weighted", zero_division=0))

        cm = confusion_matrix(y_test, y_pred, labels=target_classes)
        clf_report = classification_report(y_test, y_pred, labels=target_classes, output_dict=True, zero_division=0)

        # Feature Importances / Coefficients
        feature_importance_dict = {}
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_importance_dict = {
                feat: round(float(imp), 4)
                for feat, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
            }
        elif hasattr(model, "coef_"):
            # Mean absolute coefficient across classes
            mean_coefs = np.mean(np.abs(model.coef_), axis=0)
            feature_importance_dict = {
                feat: round(float(c), 4)
                for feat, c in sorted(zip(feature_names, mean_coefs), key=lambda x: x[1], reverse=True)
            }

        results[name] = {
            "accuracy": round(acc, 4),
            "precision_macro": round(prec_macro, 4),
            "recall_macro": round(rec_macro, 4),
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
            "confusion_matrix": cm.tolist(),
            "target_classes": target_classes,
            "classification_report": clf_report,
            "top_features": dict(list(feature_importance_dict.items())[:12]),
            "all_features": feature_importance_dict,
        }

        print(f"  {name:20s} | Accuracy: {acc:.4f} | F1-Macro: {f1_macro:.4f} | F1-Weighted: {f1_weighted:.4f}")

    # Determine Best Model based on F1-Macro (balanced across all risk classes)
    best_model_name = max(results, key=lambda k: results[k]["f1_macro"])
    best_estimator = trained_estimators[best_model_name]
    print(f"\n[OK] Selected Best Model: {best_model_name} (F1-Macro: {results[best_model_name]['f1_macro']:.4f})")

    # Persist Models & Preprocessor
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, config.PREPROCESSOR_FILE)
    joblib.dump(best_estimator, config.BEST_MODEL_FILE)

    # Also persist individual estimators for multi-model dashboard comparison
    joblib.dump(trained_estimators["Random Forest"], models_dir / "random_forest.joblib")
    joblib.dump(trained_estimators["Logistic Regression"], models_dir / "logistic_regression.joblib")
    joblib.dump(trained_estimators["Gradient Boosting"], models_dir / "gradient_boosting.joblib")

    print(f"[OK] Preprocessor saved to {config.PREPROCESSOR_FILE}")
    print(f"[OK] Best Model ({best_model_name}) saved to {config.BEST_MODEL_FILE}")

    # Compile Summary Payload
    summary_payload = {
        "best_model_name": best_model_name,
        "classes": target_classes,
        "feature_names": feature_names,
        "test_set_size": len(y_test),
        "test_distribution": pd.Series(y_test).value_counts().to_dict(),
        "models": results,
    }

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"[OK] Model evaluation metrics saved to {metrics_file}")

    return summary_payload


def main():
    train_and_evaluate_models()


if __name__ == "__main__":
    main()
