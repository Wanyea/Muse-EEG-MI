import os
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from collections import deque

import numpy as np
import pandas as pd

@dataclass
class Trial:
    trial_id: int
    label: str
    cue_on_ts: float
    imagery_on_ts: float
    imagery_off_ts: float
    relax_off_ts: float
    notes: str = ""
    valid: bool = True

class MuseRecorder:
    """
    Stores raw EEG samples in a ring buffer and writes labeled trials to disk.
    """

    def __init__(self, eeg_channels: List[str], fs: int = 256, buffer_seconds: int = 300):
        self.eeg_channels = eeg_channels
        self.fs = fs
        self.buffer = deque(maxlen=fs * buffer_seconds)  # (ts, [ch...])
        self.session_dir: Optional[str] = None
        self.session_meta: Dict = {}
        self.trials: List[Trial] = []
        self._trial_counter = 0

    # -------------------------
    # Session
    # -------------------------
    def start_session(self, base_dir: str, subject_id: str, montage: Dict, extra_meta: Optional[Dict] = None) -> str:
        ts = time.strftime("%Y%m%d_%H%M%S")
        session_id = f"{subject_id}_{ts}"
        self.session_dir = os.path.join(base_dir, session_id)
        os.makedirs(self.session_dir, exist_ok=True)

        self.session_meta = {
            "session_id": session_id,
            "subject_id": subject_id,
            "started_unix": time.time(),
            "fs": self.fs,
            "channels": self.eeg_channels,
            "montage": montage,
            "extra_meta": extra_meta or {},
        }
        self.trials = []
        self._trial_counter = 0

        with open(os.path.join(self.session_dir, "session_meta.json"), "w") as f:
            json.dump(self.session_meta, f, indent=2)

        return session_id

    def end_session(self):
        if not self.session_dir:
            return
        self.session_meta["ended_unix"] = time.time()
        with open(os.path.join(self.session_dir, "session_meta.json"), "w") as f:
            json.dump(self.session_meta, f, indent=2)

    # -------------------------
    # Raw sample ingest
    # -------------------------
    def add_eeg_samples(self, rows: np.ndarray, timestamps: np.ndarray):
        """
        rows shape: (N, C) float
        timestamps shape: (N,) float unix seconds
        """
        for ts, row in zip(timestamps, rows):
            self.buffer.append((float(ts), row.astype(float).tolist()))

    # -------------------------
    # Trials
    # -------------------------
    def start_trial(self, label: str, cue_on_ts: float, imagery_on_ts: float, imagery_off_ts: float, relax_off_ts: float) -> Trial:
        self._trial_counter += 1
        tr = Trial(
            trial_id=self._trial_counter,
            label=label,
            cue_on_ts=cue_on_ts,
            imagery_on_ts=imagery_on_ts,
            imagery_off_ts=imagery_off_ts,
            relax_off_ts=relax_off_ts
        )
        self.trials.append(tr)
        return tr

    def mark_trial_invalid(self, trial_id: int, notes: str):
        for tr in self.trials:
            if tr.trial_id == trial_id:
                tr.valid = False
                tr.notes = notes
                return

    # -------------------------
    # Export
    # -------------------------
    def export_trials(self, epoch_offset_s: float, epoch_len_s: float) -> Dict:
        """
        For each trial, slice EEG in [imagery_on + offset, imagery_on + offset + len]
        Write CSV per trial and a trial table CSV + JSON.
        """
        if not self.session_dir:
            raise RuntimeError("No active session_dir. Start a session first.")

        # convert buffer to arrays once for efficient slicing
        if len(self.buffer) < 10:
            return {"error": "Not enough EEG buffered to export."}

        buf_ts = np.array([x[0] for x in self.buffer], dtype=float)
        buf_x = np.array([x[1] for x in self.buffer], dtype=float)  # shape (M, C)

        exported = []
        for tr in self.trials:
            if not tr.valid:
                continue

            start = tr.imagery_on_ts + epoch_offset_s
            end = start + epoch_len_s

            idx = np.where((buf_ts >= start) & (buf_ts <= end))[0]
            if len(idx) < max(10, int(self.fs * 0.5)):
                tr.valid = False
                tr.notes = f"Too few samples in epoch window ({len(idx)})."
                continue

            ts_slice = buf_ts[idx]
            x_slice = buf_x[idx, :]

            df = pd.DataFrame(x_slice, columns=self.eeg_channels)
            df.insert(0, "timestamp", ts_slice)

            trial_fname = f"trial_{tr.trial_id:04d}_{tr.label}.csv"
            df.to_csv(os.path.join(self.session_dir, trial_fname), index=False)

            exported.append({
                "trial_id": tr.trial_id,
                "label": tr.label,
                "file": trial_fname,
                "n_samples": len(df),
                "epoch_start_ts": float(start),
                "epoch_end_ts": float(end),
            })

        # trial table
        trial_table = [asdict(t) for t in self.trials]
        pd.DataFrame(trial_table).to_csv(os.path.join(self.session_dir, "trials.csv"), index=False)

        with open(os.path.join(self.session_dir, "export_index.json"), "w") as f:
            json.dump({
                "exported": exported,
                "epoch_offset_s": epoch_offset_s,
                "epoch_len_s": epoch_len_s,
                "fs": self.fs,
                "channels": self.eeg_channels,
            }, f, indent=2)

        return {"exported": exported, "n_exported": len(exported), "session_dir": self.session_dir}

    # -------------------------
    # Load exported trials for training
    # -------------------------
    def list_exported_trials(self) -> List[Dict]:
        if not self.session_dir:
            return []
        path = os.path.join(self.session_dir, "export_index.json")
        if not os.path.exists(path):
            return []
        with open(path, "r") as f:
            return json.load(f).get("exported", [])