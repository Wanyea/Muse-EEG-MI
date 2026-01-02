import os
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .features import bandpower_features, feature_names

LABELS = ["UP", "DOWN", "LEFT", "RIGHT"]

class MIModel:
    def __init__(self, fs: int, channels: List[str]):
        self.fs = fs
        self.channels = channels
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis())
        ])
        self.is_fit = False

    def train_from_session_dir(self, session_dir: str) -> Dict:
        export_index_path = os.path.join(session_dir, "export_index.json")
        if not os.path.exists(export_index_path):
            return {"error": "export_index.json not found. Export trials first."}

        with open(export_index_path, "r") as f:
            exported = json.load(f).get("exported", [])

        X, y = [], []
        for item in exported:
            csv_path = os.path.join(session_dir, item["file"])
            df = pd.read_csv(csv_path)
            x = df[self.channels].to_numpy(dtype=float)
            X.append(bandpower_features(x, fs=self.fs))
            y.append(item["label"])

        if len(X) < 8:
            return {"error": f"Not enough trials to train ({len(X)}). Need at least ~8."}

        X = np.vstack(X)
        y = np.array(y)

        cv = StratifiedKFold(n_splits=min(5, np.min(np.bincount(pd.Categorical(y).codes))), shuffle=True, random_state=42)
        scores = cross_val_score(self.pipeline, X, y, cv=cv)
        self.pipeline.fit(X, y)
        self.is_fit = True

        model_path = os.path.join(session_dir, "trained_model.json")
        # Save minimal model info (sklearn object not JSON). We'll use joblib for real saving.
        # For v1, just keep it in memory and report metrics.
        return {
            "n_trials": int(len(y)),
            "cv_mean_acc": float(np.mean(scores)),
            "cv_std_acc": float(np.std(scores)),
            "features": feature_names(self.channels),
        }

    def predict_epoch(self, epoch_df: pd.DataFrame) -> Dict:
        if not self.is_fit:
            return {"error": "Model not trained yet."}
        x = epoch_df[self.channels].to_numpy(dtype=float)
        feats = bandpower_features(x, fs=self.fs).reshape(1, -1)
        pred = self.pipeline.predict(feats)[0]
        probs = None
        if hasattr(self.pipeline.named_steps["clf"], "predict_proba"):
            probs = self.pipeline.predict_proba(feats)[0].tolist()
        return {"pred": pred, "probs": probs, "labels": self.pipeline.named_steps["clf"].classes_.tolist()}