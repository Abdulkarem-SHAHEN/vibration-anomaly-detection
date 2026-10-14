"""Run the full experiment: load -> split -> features -> models -> evaluation -> report.

Usage (from the repository root):
    python scripts/run_experiment.py                 # real CWRU data
    python scripts/run_experiment.py --synthetic     # smoke test, no download needed
    python scripts/run_experiment.py --quick         # one seed only (faster)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import plots  # noqa: E402
from src.data import load_cwru, summarize  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.features import handcrafted_features, log_spectrum  # noqa: E402
from src.models import build_models  # noqa: E402
from src.splits import build_folds  # noqa: E402
from src.synthetic import make_synthetic  # noqa: E402


def represent(fold, fs: int, fcfg: dict) -> dict:
    """Compute both representations for every split of a fold."""
    def both(w):
        return {"features": handcrafted_features(w, fs, fcfg["n_bands"]),
                "spectrum": log_spectrum(w, fcfg["spectrum_bins"])}
    return {
        "train": both(fold.train),
        "val": both(fold.val),
        "test_normal": both(fold.test_normal),
        "test_faults": {k: both(v) for k, v in fold.test_faults.items()},
    }


def fmt(mean: float, std: float) -> str:
    return f"{mean:.3f} ± {std:.3f}"


def write_report(df: pd.DataFrame, df_fault: pd.DataFrame, out: Path, synthetic: bool, target_fpr: float) -> str:
    lines = ["# Results", ""]
    if synthetic:
        lines += ["> **SYNTHETIC DATA - pipeline smoke test only. Do not report these numbers.**", ""]
    lines += [f"Threshold: {100 * (1 - target_fpr):.0f}th percentile of validation-normal scores "
              f"(target false-positive rate {target_fpr:.0%}). Mean ± std over folds and seeds.", ""]
    for protocol, g in df.groupby("protocol", sort=False):
        lines += [f"## Protocol: `{protocol}`", "",
                  "| Model | ROC-AUC | Test FPR (target %.2f) | Detection rate |" % target_fpr,
                  "|---|---|---|---|"]
        for model, gm in g.groupby("model", sort=False):
            lines.append(f"| {model} | {fmt(gm.roc_auc.mean(), gm.roc_auc.std(ddof=0))} | "
                         f"{fmt(gm.test_fpr.mean(), gm.test_fpr.std(ddof=0))} | "
                         f"{fmt(gm.detection_rate.mean(), gm.detection_rate.std(ddof=0))} |")
        lines.append("")
        pf = (df_fault[df_fault.protocol == protocol]
              .groupby(["model", "fault"], sort=False).detection_rate.mean().unstack())
        lines += ["Detection rate per fault (type + diameter in mils):", "", pf.round(3).to_markdown(), ""]
    if "cross_load" in df.protocol.values:
        lines += ["## Per held-out load (`cross_load`)", "",
                  df[df.protocol == "cross_load"].groupby(["model", "fold"], sort=False)[
                      ["roc_auc", "test_fpr", "detection_rate"]].mean().round(3).to_markdown(), ""]
    text = "\n".join(lines)
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--synthetic", action="store_true", help="use generated signals instead of CWRU")
    ap.add_argument("--quick", action="store_true", help="use only the first seed")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    w, ev = cfg["windowing"], cfg["evaluation"]
    seeds = ev["seeds"][:1] if args.quick else ev["seeds"]
    out = ROOT / (cfg["output"]["results_dir"] + ("_synthetic" if args.synthetic else ""))
    (out / "figures").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- Step 1: data -------------------------------------------------------
    print("Step 1/5  Loading recordings", "(SYNTHETIC)" if args.synthetic else "")
    if args.synthetic:
        recordings = make_synthetic(cfg)
    else:
        cfg["data"]["raw_dir"] = str(ROOT / cfg["data"]["raw_dir"])
        recordings = load_cwru(cfg)
    print(summarize(recordings))
    fs = recordings[0].fs
    plots.signals_and_spectra(recordings, out / "figures" / "signals_and_spectra.png",
                              window_size=w["window_size"])

    rows, fault_rows = [], []
    for protocol in ev["protocols"]:
        # ---- Step 2: leakage-aware splits -----------------------------------
        folds = build_folds(recordings, protocol, w)
        print(f"\nStep 2/5  Protocol '{protocol}': {len(folds)} fold(s)")
        for fold in folds:
            n_fault = sum(len(v) for v in fold.test_faults.values())
            print(f"  {fold.name:<14} train={len(fold.train):>5}  val={len(fold.val):>4}  "
                  f"test_normal={len(fold.test_normal):>4}  test_fault={n_fault:>5}")

            # ---- Step 3: features ------------------------------------------
            rep = represent(fold, fs, cfg["features"])

            for seed in seeds:
                # ---- Step 4: models (trained on NORMAL data only) -----------
                for model in build_models(cfg):
                    r = model.representation
                    model.fit(rep["train"][r], rep["val"][r], seed=seed)
                    val_s = model.score(rep["val"][r])
                    normal_s = model.score(rep["test_normal"][r])
                    fault_s = {k: model.score(v[r]) for k, v in rep["test_faults"].items()}

                    # ---- Step 5: evaluation ---------------------------------
                    summary, per_fault = evaluate(val_s, normal_s, fault_s, ev["target_fpr"])
                    rows.append({"protocol": protocol, "fold": fold.name, "model": model.name,
                                 "seed": seed, **summary})
                    fault_rows += [{"protocol": protocol, "fold": fold.name, "model": model.name,
                                    "seed": seed, "fault": k, "detection_rate": v}
                                   for k, v in per_fault.items()]
                    print(f"    seed={seed} {model.name:<20} AUC={summary['roc_auc']:.3f}  "
                          f"testFPR={summary['test_fpr']:.3f}  detection={summary['detection_rate']:.3f}")

                    # Figures: one representative fold, first seed.
                    if seed == seeds[0] and fold is folds[-1]:
                        fold.__dict__.setdefault("_plot", {})[model.name] = (
                            normal_s, np.concatenate(list(fault_s.values())), summary["threshold"])
            if hasattr(fold, "_plot"):
                tag = f"{protocol}_{fold.name}"
                plots.score_distributions(fold._plot, out / "figures" / f"scores_{tag}.png",
                                          f"Score distributions - {protocol} ({fold.name})")
                plots.roc_curves({k: v[:2] for k, v in fold._plot.items()},
                                 out / "figures" / f"roc_{tag}.png", f"ROC - {protocol}\n({fold.name})",
                                 ev["target_fpr"])

    df, df_fault = pd.DataFrame(rows), pd.DataFrame(fault_rows)
    df.to_csv(out / "metrics_all_runs.csv", index=False)
    df_fault.to_csv(out / "detection_per_fault.csv", index=False)
    print("\n" + write_report(df, df_fault, out, args.synthetic, ev["target_fpr"]))
    print(f"\nDone in {time.time() - t0:.1f}s. Results in {out.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
