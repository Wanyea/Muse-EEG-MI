# mi/model.py
import os
import json
from typing import Dict, List

import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score

from .features import bandpower_features, feature_names

class MIModel:
    """
    Simple baseline MI model:
      - bandpower (theta/alpha/beta) features
      - StandardScaler + LDA
      - CV metrics + confusion matrix output for UI heatmap
    """
    def __init__(self, fs: int, channels: List[str]):
        self.fs = fs
        self.channels = channels
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis())
        ])
        self.is_fit = False
        self.classes_ = None

    def _load_exported_trials(self, session_dir: str, allowed_classes: List[str] | None = None):
        export_index_path = os.path.join(session_dir, "export_index.json")
        if not os.path.exists(export_index_path):
            return None, None, {"error": "export_index.json not found. Export Trials first."}

        with open(export_index_path, "r", encoding="utf-8") as f:
            exported = json.load(f).get("exported", [])

        X, y, files = [], [], []
        for item in exported:
            label = item.get("label")
            if not label:
                continue
            if allowed_classes and label not in allowed_classes:
                continue

            csv_path = os.path.join(session_dir, item.get("file", ""))
            if not os.path.exists(csv_path):
                continue

            df = pd.read_csv(csv_path)
            if not all(ch in df.columns for ch in self.channels):
                continue

            x = df[self.channels].to_numpy(dtype=float)
            X.append(bandpower_features(x, fs=self.fs))
            y.append(label)
            files.append(os.path.basename(csv_path))

        if len(X) < 4:
            return None, None, {"error": f"Not enough trials after filtering ({len(X)}). Collect more and Export."}

        return np.vstack(X), np.array(y), {"files": files}

    def train_from_session_dir(self, session_dir: str, allowed_classes: List[str] | None = None) -> Dict:
        X, y, meta = self._load_exported_trials(session_dir, allowed_classes)
        if X is None:
            return meta

        # stable label ordering
        present = sorted(list(set(y.tolist())))
        if allowed_classes:
            classes = [c for c in allowed_classes if c in present]
        else:
            classes = present

        # choose CV folds based on smallest class count
        codes = pd.Categorical(y, categories=classes).codes
        counts = np.bincount(codes[codes >= 0])
        min_count = int(np.min(counts)) if len(counts) else 0
        n_splits = max(2, min(5, min_count))  # 2..5

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        y_pred = cross_val_predict(self.pipeline, X, y, cv=cv)

        acc = float(accuracy_score(y, y_pred))
        bacc = float(balanced_accuracy_score(y, y_pred))
        cm = confusion_matrix(y, y_pred, labels=classes).astype(int)

        # Fit final model
        self.pipeline.fit(X, y)
        self.is_fit = True
        self.classes_ = classes

        per_class = {}
        for i, c in enumerate(classes):
            denom = int(cm[i, :].sum())
            per_class[c] = float(cm[i, i] / denom) if denom > 0 else 0.0

        return {
            "n_trials": int(len(y)),
            "classes": classes,
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "confusion_matrix": cm.tolist(),
            "per_class_accuracy": per_class,
            "features": feature_names(self.channels),
        }
