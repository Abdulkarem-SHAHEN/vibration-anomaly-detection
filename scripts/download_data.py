"""Download the CWRU files listed in config.yaml into data/raw/.

Usage (from the repository root):
    python scripts/download_data.py

Files already present are skipped. If the server refuses a download, the
script prints which files are missing so you can fetch them by hand from
https://engineering.case.edu/bearingdatacenter/download-data-file
and drop them into data/raw/ as <number>.mat.
"""

from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

import yaml
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]


def wanted_files(cfg: dict) -> list[int]:
    nums = [int(n) for n in cfg["data"]["normal"].values()]
    for per_load in cfg["data"]["faults"].values():
        nums += [int(n) for n in per_load.values()]
    return sorted(set(nums))


def download(url: str, dest: Path, retries: int = 3) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research download)"})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                dest.write_bytes(resp.read())
            loadmat(dest)                       # fail fast on HTML error pages
            return
        except Exception as exc:                # noqa: BLE001
            dest.unlink(missing_ok=True)
            if attempt == retries:
                raise RuntimeError(f"{url}: {exc}") from exc
            time.sleep(2 * attempt)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    raw = ROOT / cfg["data"]["raw_dir"]
    raw.mkdir(parents=True, exist_ok=True)
    base = cfg["data"]["base_url"].rstrip("/")
    missing = []
    for num in wanted_files(cfg):
        dest = raw / f"{num}.mat"
        if dest.exists():
            print(f"  [skip] {dest.name} already present")
            continue
        try:
            download(f"{base}/{num}.mat", dest)
            print(f"  [ok]   {dest.name}  ({dest.stat().st_size / 1e6:.1f} MB)")
        except RuntimeError as exc:
            print(f"  [fail] {exc}")
            missing.append(num)
    if missing:
        print(f"\n{len(missing)} file(s) missing: {missing}")
        print("Download them manually and place them in", raw)
        sys.exit(1)
    print("\nAll files ready.")


if __name__ == "__main__":
    main()
