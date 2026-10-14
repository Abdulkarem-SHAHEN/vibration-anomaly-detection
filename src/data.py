"""Step 1 - Loading the CWRU recordings.

Each CWRU file is a MATLAB ``.mat`` file holding one continuous vibration
recording. We only keep the drive-end (DE) accelerometer channel.

Quirk handled here: some CWRU files contain variables from more than one
recording (e.g. ``99.mat`` also stores ``X098_*`` arrays), so we select the
variable whose number matches the file number whenever possible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat


@dataclass
class Recording:
    """One continuous vibration recording."""

    name: str            # e.g. "normal_L0" or "IR007_L2"
    label: str           # "normal" or "fault"
    fault: str           # "none", "IR007", "B014", ...
    load: int            # motor load in HP (0-3)
    fs: int              # sampling rate in Hz
    signal: np.ndarray   # 1-D float array

    @property
    def fault_type(self) -> str:
        """IR / B / OR, or 'none' for normal data."""
        return "none" if self.fault == "none" else re.match(r"[A-Z]+", self.fault).group(0)

    @property
    def fault_size(self) -> str:
        """Fault diameter in mils ('007', '014', '021'), or 'none'."""
        return "none" if self.fault == "none" else self.fault[-3:]


def read_mat_signal(path: Path, file_number: int, channel: str = "DE") -> np.ndarray:
    """Return the requested accelerometer channel of one CWRU .mat file."""
    mat = loadmat(path)
    suffix = f"_{channel}_time"
    candidates = [k for k in mat if k.endswith(suffix)]
    if not candidates:
        raise KeyError(f"{path.name}: no variable ending with '{suffix}' (found {list(mat)})")

    preferred = [k for k in candidates if k.startswith(f"X{file_number:03d}")]
    key = preferred[0] if preferred else sorted(candidates)[0]
    return np.asarray(mat[key], dtype=np.float64).ravel()


def load_cwru(cfg: dict) -> list[Recording]:
    """Load every normal and fault recording listed in the config."""
    dcfg = cfg["data"]
    raw_dir = Path(dcfg["raw_dir"])
    fs, channel = int(dcfg["sampling_rate"]), dcfg["channel"]
    recordings: list[Recording] = []

    def _load(num: int, name: str, label: str, fault: str, load: int) -> None:
        path = raw_dir / f"{num}.mat"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run `python scripts/download_data.py` first "
                "(or download the file manually, see README)."
            )
        recordings.append(Recording(name, label, fault, load, fs, read_mat_signal(path, num, channel)))

    for load, num in dcfg["normal"].items():
        _load(int(num), f"normal_L{load}", "normal", "none", int(load))
    for fault, per_load in dcfg["faults"].items():
        for load, num in per_load.items():
            _load(int(num), f"{fault}_L{load}", "fault", fault, int(load))
    return recordings


def summarize(recordings: list[Recording]) -> str:
    """Human-readable inventory, printed before every experiment."""
    lines = [f"{'recording':<14}{'samples':>10}{'seconds':>9}{'std':>9}"]
    for r in recordings:
        lines.append(f"{r.name:<14}{len(r.signal):>10}{len(r.signal) / r.fs:>9.2f}{r.signal.std():>9.4f}")
    return "\n".join(lines)
