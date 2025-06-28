from pathlib import Path
from typing import Tuple
import numpy as np
import mne
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# 2. Helper functions
# ---------------------------------------------------------------------
def load_eeg(edf_path: Path) -> np.ndarray:
    """Return EEG data as np.ndarray (channels, time)."""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    return raw.get_data(picks="eeg")

def align_lengths(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Trim the longer array so a and b share the same #samples."""
    T = min(a.shape[1], b.shape[1])
    return a[:, :T], b[:, :T]

def minmax_per_channel(x: np.ndarray) -> np.ndarray:
    """Scale each channel to [0, 1] independently."""
    xmin = x.min(axis=1, keepdims=True)
    xmax = x.max(axis=1, keepdims=True)
    rng = np.where((xmax - xmin) == 0, 1, xmax - xmin)
    return (x - xmin) / rng

def save_npy(array: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, array)
    logger.info(f"Saved {out_path.name:45}  shape={array.shape}")

# ---------------------------------------------------------------------
# 3. Grid definition
# ---------------------------------------------------------------------
GRID = [
    ("zeropad_30", "spk9-lst10",
     ("individual/nt9_speak_zeropad_30_components_preprocessed.edf",
      "individual/nt10_listen_zeropad_30_components_preprocessed.edf")),
    ("zeropad_30", "lst9-spk10",
     ("individual/nt9_listen_zeropad_30_components_preprocessed.edf",
      "individual/nt10_speak_zeropad_30_components_preprocessed.edf")),
    ("zeropad_30", "stacked",
     ("stacked/nt9_zeropad_speak_listen_stacked.edf",
      "stacked/nt10_zeropad_listen_speak_stacked.edf")),
    ("cut_60", "spk9-lst10",
     ("individual/nt9_speak_cut_60_components_preprocessed.edf",
      "individual/nt10_listen_cut_60_components_preprocessed.edf")),
    ("cut_60", "lst9-spk10",
     ("individual/nt9_listen_cut_60_components_preprocessed.edf",
      "individual/nt10_speak_cut_60_components_preprocessed.edf")),
    ("cut_60", "stacked",
     ("stacked/nt9_cut_speak_listen_stacked.edf",
      "stacked/nt10_cut_listen_speak_stacked.edf")),
]

# ---------------------------------------------------------------------
# 4. Processing function
# ---------------------------------------------------------------------
def process_pair(clean: str, pairing: str, edf_a_rel: str, edf_b_rel: str) -> None:
    edf_a = RAW / clean / edf_a_rel
    edf_b = RAW / clean / edf_b_rel

    if not edf_a.exists() or not edf_b.exists():
        logger.warning(f"Missing file: {edf_a if not edf_a.exists() else edf_b} for pairing {pairing}")
        return

    try:
        A = load_eeg(edf_a)
        B = load_eeg(edf_b)
        A, B = align_lengths(A, B)
        combined = np.vstack([A, B])

        save_npy(combined, PROC / clean / "raw" / f"{pairing}.npy")
        save_npy(minmax_per_channel(combined), PROC / clean / "normalized" / f"{pairing}.npy")
    except Exception as e:
        logger.error(f"Error processing {pairing}: {e}")

# ---------------------------------------------------------------------
# 5. Main loop
# ---------------------------------------------------------------------
def main():
    for clean, pairing, (edf_a_rel, edf_b_rel) in GRID:
        process_pair(clean, pairing, edf_a_rel, edf_b_rel)
    logger.info("All available NumPy files generated.")

if __name__ == "__main__":
    main()
