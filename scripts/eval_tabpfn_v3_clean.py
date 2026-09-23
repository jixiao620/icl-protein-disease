"""
TabPFN v3 (Prior Labs, tabpfn==8.1.0) evaluated on the CLEAN patient-
disjoint pools produced by preprocess_clean.py.

Purpose: architectural baseline. If TabPFN v3 — which embodies the
"per-feature distribution embeddings + cross-feature CLS tokens +
strict train-only ICL keys" paradigm we discussed borrowing into v6 —
recovers meaningful AUROC on our clean patient-disjoint splits, then
that paradigm is worth porting into our own model. If TabPFN v3 also
fails, the ceiling is set by the small-positive rare-disease regime,
not by our specific architecture.

Setup (identical to eval_clean_2way.py, model replaced):
  * 70/15/15 patient split (patient_split.yaml), reused
  * For each test disease: merge ctx + query pools into a single 40%
    eval pool, then sample K context indices with prevalence-aware
    positive count (max(1, K*min(3*prev, 0.30))). Remaining indices
    become the query set (auto_eval convention).
  * K per block: match auto pipeline — I=128, C=64, G=128
  * 15 seeds, mean AUROC / AUPRC / Brier / ECE

Inputs:
  processed_data_clean_{letter}/{context,query}_data.pkl  (from
  preprocess_clean.py). No training needed; TabPFN v3 is pre-trained.

Output:
  analysis_results/tabpfn_v3_clean/results.json     per-disease metrics
  analysis_results/tabpfn_v3_clean/summary.csv      compact table
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from metrics import compute_all as compute_all_metrics


PROJECT = os.environ.get("ICL_PROJECT_ROOT",
                         str(Path(__file__).resolve().parent.parent))
OUT_DIR = os.path.join(PROJECT, "analysis_results/tabpfn_v3_clean")
os.makedirs(OUT_DIR, exist_ok=True)

K_DEFAULT   = 64
K_BY_LETTER = {"G": 128, "I": 128}
N_SEEDS     = 15
N_EVAL_MAX  = 10000
BATCH_QRY   = 512


def tabpfn_eval_2way(clf_ctor, X, y, ctx_size, n_seeds=N_SEEDS):
    X = np.asarray(X, np.float32); y = np.asarray(y, np.int32)
    N = len(y)
    if N > N_EVAL_MAX:
        rng_sub = np.random.RandomState(0)
        idx = rng_sub.choice(N, N_EVAL_MAX, replace=False); idx.sort()
        X, y = X[idx], y[idx]; N = len(y)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    empty = {"auroc": float("nan"), "auprc": float("nan"),
             "brier": float("nan"), "ece": float("nan"),
             "n_ctx_pos": 0, "n_seeds_valid": 0}
    if len(pos_idx) < 2 or len(neg_idx) < 2:
        return empty

    prev = len(pos_idx) / max(N, 1)
    n_ctx_pos = max(1, int(ctx_size * min(prev * 3, 0.3)))
    n_ctx_neg = max(1, ctx_size - n_ctx_pos)

    per_seed = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        cp = pos_idx[rng.choice(len(pos_idx), n_ctx_pos,
                                replace=(len(pos_idx) < n_ctx_pos))]
        cn = neg_idx[rng.choice(len(neg_idx), n_ctx_neg,
                                replace=(len(neg_idx) < n_ctx_neg))]
        ctx_idx = np.concatenate([cp, cn])
        ctx_set = set(ctx_idx.tolist())
        qry_idx = np.array([i for i in range(N) if i not in ctx_set],
                           dtype=np.int64)
        if len(qry_idx) == 0 or len(np.unique(y[qry_idx])) < 2:
            continue

        X_ctx, y_ctx = X[ctx_idx], y[ctx_idx]
        X_qry, y_qry = X[qry_idx], y[qry_idx]

        try:
            clf = clf_ctor(seed=seed)
            clf.fit(X_ctx, y_ctx)
            probs = []
            for start in range(0, len(qry_idx), BATCH_QRY):
                end = min(start + BATCH_QRY, len(qry_idx))
                p = clf.predict_proba(X_qry[start:end])
                pos_col = list(clf.classes_).index(1) if 1 in list(clf.classes_) else -1
                probs.append(p[:, pos_col])
            probs = np.concatenate(probs)
            per_seed.append(compute_all_metrics(y_qry, probs))
        except Exception as e:
            print(f"    seed {seed} failed: {e}", flush=True)

    def _mean(k):
        vals = [d[k] for d in per_seed if not np.isnan(d[k])]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "auroc": _mean("auroc"), "auprc": _mean("auprc"),
        "brier": _mean("brier"), "ece":   _mean("ece"),
        "n_ctx_pos": n_ctx_pos,
        "n_seeds_valid": sum(1 for d in per_seed if not np.isnan(d["auroc"])),
        "per_seed": per_seed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    args = parser.parse_args()

    from tabpfn import TabPFNClassifier
    import torch, tabpfn as _tp

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"tabpfn: {_tp.__version__}  torch: {torch.__version__}  device: {device}",
          flush=True)

    def make_clf(seed):
        return TabPFNClassifier(
            device=device, n_estimators=8, random_state=seed,
            ignore_pretraining_limits=True,
        )

    with open(args.selection) as f:
        sel = yaml.safe_load(f)

    results_path = os.path.join(OUT_DIR, "results.json")
    results = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)

    for letter, block_out in sel.get("blocks", {}).items():
        if block_out.get("skipped"): continue
        letter_lower = letter.lower()
        clean_dir = os.path.join(PROJECT, f"processed_data_clean_{letter_lower}")
        ctx_pkl = os.path.join(clean_dir, "context_data.pkl")
        qry_pkl = os.path.join(clean_dir, "query_data.pkl")
        if not (os.path.exists(ctx_pkl) and os.path.exists(qry_pkl)):
            print(f"\nBlock {letter}: missing pkl files — skip", flush=True); continue
        with open(ctx_pkl, "rb") as f: ctx_pool = pickle.load(f)
        with open(qry_pkl, "rb") as f: qry_pool = pickle.load(f)

        k_eval = K_BY_LETTER.get(letter, K_DEFAULT)
        print(f"\n{'='*70}\n  Block {letter} — TabPFN v3 CLEAN 2-way, "
              f"K={k_eval}\n{'='*70}", flush=True)

        for code in block_out["test"]:
            key = f"{letter}_{code}"
            if (key in results and
                    all(m in results[key] for m in ("auroc","auprc","brier","ece"))):
                print(f"  {code}: cached AUROC={results[key]['auroc']:.4f}",
                      flush=True); continue
            if code not in ctx_pool or code not in qry_pool:
                print(f"  {code}: missing in pools", flush=True); continue
            Xc, yc = ctx_pool[code]
            Xq, yq = qry_pool[code]
            X = np.concatenate([Xc, Xq], axis=0)
            y = np.concatenate([yc, yq], axis=0)
            print(f"  {code}: merged pool N={len(y)} N+={int(y.sum())}",
                  flush=True)
            m = tabpfn_eval_2way(make_clf, X, y, k_eval)
            results[key] = {
                "block": letter, "code": code,
                "auroc": m["auroc"], "auprc": m["auprc"],
                "brier": m["brier"], "ece": m["ece"],
                "n_pos_eval": int(y.sum()),
                "n_ctx_pos": m["n_ctx_pos"], "ctx_size": k_eval,
                "n_seeds_valid": m["n_seeds_valid"],
            }
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"    AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  "
                  f"Brier={m['brier']:.4f}  ECE={m['ece']:.4f}", flush=True)

    rows = []
    for letter, block_out in sel.get("blocks", {}).items():
        if block_out.get("skipped"): continue
        for code in block_out.get("test", []):
            key = f"{letter}_{code}"
            v = results.get(key, {})
            rows.append({
                "block": letter, "disease": code,
                "auroc": v.get("auroc", np.nan),
                "auprc": v.get("auprc", np.nan),
                "brier": v.get("brier", np.nan),
                "ece":   v.get("ece",   np.nan),
                "n_pos_eval": v.get("n_pos_eval", 0),
            })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "summary.csv")
    df.to_csv(csv_path, index=False, float_format="%.4f")

    print(f"\n{'='*70}\nSUMMARY (TabPFN v3 CLEAN 2-way)\n{'='*70}", flush=True)
    print(df.to_string(index=False), flush=True)
    for letter, grp in df.groupby("block"):
        line = f"  {letter}-block mean over {len(grp)}: "
        for m in ["auroc","auprc","brier","ece"]:
            v = grp[m].dropna()
            line += f"{m}={v.mean():.4f}  " if len(v) else f"{m}=nan  "
        print(line, flush=True)


if __name__ == "__main__":
    main()
