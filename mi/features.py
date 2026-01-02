from typing import List
import numpy as np
from scipy.signal import welch
from scipy.integrate import trapezoid

BANDS = {
    "theta": (4.0, 7.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
}

def bandpower_features(x: np.ndarray, fs: int, bands=BANDS) -> np.ndarray:
    """
    x: (N, C) trial epoch samples
    returns: (C * len(bands),) log bandpowers
    """
    feats = []
    for c in range(x.shape[1]):
        f, pxx = welch(x[:, c], fs=fs, nperseg=min(256, x.shape[0]))
        for _, (lo, hi) in bands.items():
            idx = (f >= lo) & (f <= hi)
            bp = trapezoid(pxx[idx], f[idx]) + 1e-12
            feats.append(np.log(bp))
    return np.array(feats, dtype=float)

def feature_names(channels: List[str], bands=BANDS) -> List[str]:
    names = []
    for ch in channels:
        for band in bands.keys():
            names.append(f"{ch}_{band}_logbp")
    return names