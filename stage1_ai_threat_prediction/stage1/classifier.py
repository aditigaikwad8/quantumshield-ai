"""QuantumShield LightGBM Multiclass Network Flow Classifier.

Model Name: QuantumShield CICIDS LightGBM
Model Type: LightGBM multiclass flow classifier
Task: Supervised multiclass network attack classification
Framework: LightGBM
Classes (7):
    0: Normal Traffic
    1: DoS
    2: DDoS
    3: Port Scanning
    4: Brute Force
    5: Web Attacks
    6: Bots
Input Features: 52 CIC-IDS2017 flow features
Explainability: shap.TreeExplainer
"""
from __future__ import annotations

import pathlib
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from .config import (
    CLASS_NAMES,
    CLASS_TO_ID,
    ID_TO_CLASS,
    LIGHTGBM_FEATURES,
    PHRASES,
)

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False


class FlowClassifier:
    """LightGBM Multiclass Flow Classifier adhering to QuantumShield CICIDS LightGBM manifest."""

    def __init__(
        self,
        booster: lgb.Booster | None = None,
        feature_columns: list[str] | None = None,
        class_names: list[str] | None = None,
        best_iteration: int = 334,
    ):
        self.feature_columns = feature_columns or list(LIGHTGBM_FEATURES)
        self.class_names = class_names or list(CLASS_NAMES)
        self.class_to_id = {name: i for i, name in enumerate(self.class_names)}
        self.id_to_class = {i: name for i, name in enumerate(self.class_names)}
        self.n_classes = len(self.class_names)
        self.best_iteration = best_iteration
        self.booster = booster
        self._explainer = None

    # ------------------------------------------------------------------ preprocessing
    def preprocess(self, df_or_arr: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Convert input to float32 matrix with exact 52 features, replacing inf with NaN."""
        if isinstance(df_or_arr, pd.DataFrame):
            # Extract features in exact order, filling missing columns with NaN
            df = df_or_arr.copy()
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = np.nan
            sub = df[self.feature_columns].apply(pd.to_numeric, errors="coerce")
            arr = np.array(sub.to_numpy(dtype=np.float32), copy=True)
        else:
            arr = np.array(df_or_arr, dtype=np.float32, copy=True)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] != len(self.feature_columns):
                # Pad or slice to 52 columns
                out = np.full((arr.shape[0], len(self.feature_columns)), np.nan, dtype=np.float32)
                k = min(arr.shape[1], len(self.feature_columns))
                out[:, :k] = arr[:, :k]
                arr = out

        # Ensure array is a writeable copy before in-place mutation
        if not arr.flags.writeable:
            arr = arr.copy()

        # Handle +/- infinity as NaN (LightGBM natively handles NaNs)
        arr[np.isneginf(arr)] = np.nan
        arr[np.isposinf(arr)] = np.nan
        return arr

    # ------------------------------------------------------------------ inference
    def predict_proba(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict multiclass probabilities in class_names order."""
        if self.booster is None:
            raise RuntimeError("FlowClassifier model is not loaded or trained.")
        X = self.preprocess(x)
        if len(X) == 0:
            return np.zeros((0, self.n_classes), dtype=np.float32)
        iter_to_use = self.best_iteration if self.best_iteration > 0 else -1
        probs = self.booster.predict(X, num_iteration=iter_to_use)
        # Ensure 2D array of shape (N, n_classes)
        if probs.ndim == 1:
            probs = probs.reshape(-1, self.n_classes)
        return probs.astype(np.float32)

    def predict(self, x: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Return (predicted_class_ids, predicted_class_names)."""
        probs = self.predict_proba(x)
        class_ids = probs.argmax(axis=1)
        class_names = [self.id_to_class.get(int(cid), "Normal Traffic") for cid in class_ids]
        return class_ids, class_names

    # ------------------------------------------------------------------ explainability
    def _get_explainer(self):
        if self._explainer is None and _HAS_SHAP and self.booster is not None:
            try:
                self._explainer = shap.TreeExplainer(self.booster, model_output="raw")
            except Exception:
                self._explainer = None
        return self._explainer

    def explain(
        self,
        x: pd.DataFrame | np.ndarray,
        k: int = 3,
        explanation_reference: str = "results/cicids_classifier/shap/shap_experiment_metadata.json",
    ) -> list[dict[str, Any]]:
        """Explain predictions using shap.TreeExplainer for raw model output."""
        X = self.preprocess(x)
        N = len(X)
        if N == 0:
            return []

        probs = self.predict_proba(X)
        pred_ids = probs.argmax(axis=1)

        explainer = self._get_explainer()
        explanations = []

        if explainer is not None:
            try:
                shap_values = explainer.shap_values(X)
                # For multiclass, shap_values can be list of arrays [class_0, ..., class_6]
                # or shape (N, n_features, n_classes)
                for i in range(N):
                    cid = int(pred_ids[i])
                    if isinstance(shap_values, list):
                        c_vals = shap_values[cid][i]
                    elif shap_values.ndim == 3:
                        c_vals = shap_values[i, :, cid]
                    else:
                        c_vals = shap_values[i]

                    # Pick top k absolute SHAP contributions
                    top_idx = np.argsort(-np.abs(c_vals))[:k]
                    contributions = []
                    for idx in top_idx:
                        feat_name = self.feature_columns[idx]
                        val = float(c_vals[idx])
                        raw_v = float(X[i, idx]) if np.isfinite(X[i, idx]) else 0.0
                        phrase_pair = PHRASES.get(feat_name, ("unusual value", "unusual value"))
                        phrase = phrase_pair[0] if val >= 0 else phrase_pair[1]
                        contributions.append({
                            "feature": feat_name,
                            "shap_value": round(val, 5),
                            "feature_value": round(raw_v, 3),
                            "impact": "positive" if val >= 0 else "negative",
                            "phrase": phrase,
                        })
                    explanations.append({
                        "method": "shap.TreeExplainer",
                        "top_contributing_features": contributions,
                        "explanation_reference": explanation_reference,
                    })
                return explanations
            except Exception:
                pass

        # Fallback using feature importance when SHAP is unavailable
        importances = self.booster.feature_importance(importance_type="gain")
        top_idx = np.argsort(-importances)[:k]
        for i in range(N):
            contributions = []
            for idx in top_idx:
                feat_name = self.feature_columns[idx]
                raw_v = float(X[i, idx]) if np.isfinite(X[i, idx]) else 0.0
                phrase = PHRASES.get(feat_name, ("unusual pattern", "unusual pattern"))[0]
                contributions.append({
                    "feature": feat_name,
                    "importance": round(float(importances[idx]), 4),
                    "feature_value": round(raw_v, 3),
                    "phrase": phrase,
                })
            explanations.append({
                "method": "shap.TreeExplainer",
                "top_contributing_features": contributions,
                "explanation_reference": explanation_reference,
            })
        return explanations

    # ------------------------------------------------------------------ persistence
    def save(
        self,
        native_model_path: str | pathlib.Path,
        joblib_bundle_path: str | pathlib.Path | None = None,
    ) -> None:
        p_nat = pathlib.Path(native_model_path)
        p_nat.parent.mkdir(parents=True, exist_ok=True)
        if self.booster is not None:
            self.booster.save_model(str(p_nat), num_iteration=self.best_iteration)
        if joblib_bundle_path:
            p_job = pathlib.Path(joblib_bundle_path)
            p_job.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({
                "booster_string": self.booster.model_to_string() if self.booster else "",
                "feature_columns": self.feature_columns,
                "class_names": self.class_names,
                "best_iteration": self.best_iteration,
            }, p_job)

    @classmethod
    def load(
        cls,
        native_model_path: str | pathlib.Path | None = None,
        joblib_bundle_path: str | pathlib.Path | None = None,
    ) -> "FlowClassifier":
        if native_model_path and pathlib.Path(native_model_path).exists():
            booster = lgb.Booster(model_file=str(native_model_path))
            return cls(booster=booster)
        if joblib_bundle_path and pathlib.Path(joblib_bundle_path).exists():
            bundle = joblib.load(joblib_bundle_path)
            model_str = bundle.get("booster_string", "")
            booster = lgb.Booster(model_str=model_str) if model_str else None
            return cls(
                booster=booster,
                feature_columns=bundle.get("feature_columns", list(LIGHTGBM_FEATURES)),
                class_names=bundle.get("class_names", list(CLASS_NAMES)),
                best_iteration=bundle.get("best_iteration", 334),
            )
        raise FileNotFoundError(f"Neither {native_model_path} nor {joblib_bundle_path} exists.")


def train_lightgbm(
    x_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray,
    x_val: np.ndarray | pd.DataFrame | None = None,
    y_val: np.ndarray | None = None,
    feature_columns: list[str] | None = None,
    class_names: list[str] | None = None,
    best_iteration: int = 334,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    seed: int = 42,
    verbose: bool = True,
) -> FlowClassifier:
    """Train LightGBM multiclass flow classifier."""
    cols = feature_columns or list(LIGHTGBM_FEATURES)
    classes = class_names or list(CLASS_NAMES)
    clf = FlowClassifier(feature_columns=cols, class_names=classes, best_iteration=best_iteration)
    X_tr = clf.preprocess(x_train)
    dtrain = lgb.Dataset(X_tr, label=y_train, feature_name=cols, free_raw_data=False)

    params = {
        "objective": "multiclass",
        "num_class": len(classes),
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "seed": seed,
        "verbosity": -1,
        "data_random_seed": seed,
    }

    valid_sets = [dtrain]
    valid_names = ["train"]
    if x_val is not None and y_val is not None:
        X_va = clf.preprocess(x_val)
        dval = lgb.Dataset(X_va, label=y_val, reference=dtrain, feature_name=cols, free_raw_data=False)
        valid_sets.append(dval)
        valid_names.append("valid")

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=best_iteration,
        valid_sets=valid_sets,
        valid_names=valid_names,
    )
    clf.booster = booster
    clf.best_iteration = booster.best_iteration or best_iteration
    if verbose and x_val is not None and y_val is not None:
        preds = clf.predict_proba(x_val).argmax(axis=1)
        f1 = f1_score(y_val, preds, average="macro", zero_division=0)
        print(f"    LightGBM validation macro-F1: {f1:.4f} (rounds: {clf.best_iteration})")
    return clf
