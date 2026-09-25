"""
CLEAN 3-way eval — v7 kitchen sink ENSEMBLE over 3 seeds (s42, s43, s44).

For each disease + each context seed, runs forward through all 3 ensemble
members and averages logits before sigmoid. Uses more seeds (30) than
standard eval (15) for lower variance.

Reads:
  processed_data_clean_i/{context,query}_data.pkl
  checkpoints_clean_v7_622_i_s{42,43,44}/best_model.pt

Writes:
  analysis_results/clean_v7_622_ensemble_eval/results.json + summary.csv
"""
import argparse, json, os, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from models.transformer_icl_v11_featattn_diseaseemb import ICLTransformerV11FeatAttnDiseaseEmb
from metrics import compute_all as compute_all_metrics


PROJECT   = os.environ.get("ICL_PROJECT_ROOT",
                           str(Path(__file__).resolve().parent.parent))
OUT_DIR   = os.path.join(PROJECT, "analysis_results/clean_v11_ensemble_eval_g_milton")
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS     = [42, 43, 44]
K_EVAL    = 128            # standard G-block eval K (matches A+B eval)
N_SEEDS   = 15             # aligned with clean/ntrain/K-sweep/A+B evals
BATCH_QRY = 16             # per-protein attn deeper: shrink batch
N_EVAL_MAX = 10000
BLOCK_LETTER = "G"
CKPT_DIR_TEMPLATE = "checkpoints_clean_v11_ntrain12_g_s{}"
DATA_DIR = "processed_data_clean_g_milton_g"


def load_v7(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"] if isinstance(ck, dict) and "config" in ck else {}
    mc = cfg.get("model", {})
    # v10 needs prefix_vocab.pkl beside best_model.pt (written by trainer)
    vocab_path = os.path.join(os.path.dirname(ckpt_path), "prefix_vocab.pkl")
    with open(vocab_path, "rb") as f:
        v = pickle.load(f)
    model = ICLTransformerV11FeatAttnDiseaseEmb(
        prefix_vocab=v["prefix_vocab"],
        code_to_prefix_ids=v["code_to_prefix_ids"],
        protein_dim=mc.get("protein_dim", 2941),
        hidden_dim=mc.get("hidden_dim", 768),
        n_layers=mc.get("n_layers", 12),
        n_heads=mc.get("n_heads", 12),
        dropout=mc.get("dropout", 0.2),
        dim_feedforward=mc.get("dim_feedforward", 3072),
        per_prot_hidden=mc.get("per_prot_hidden", 32),
        per_prot_heads=mc.get("per_prot_heads", 4),
        per_prot_layers=mc.get("per_prot_layers", 3),
        n_cls_tokens=mc.get("n_cls_tokens", 4),
        n_cls_feat=mc.get("n_cls_feat", 24),
        n_feat_layers=mc.get("n_feat_layers", 1),
        feat_heads=mc.get("feat_heads", 4),
    )
    sd = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    model.load_state_dict(sd)
    model.to(device).eval()

    # ---- Monkey-patch _lookup_disease_emb to handle unknown Milton codes ----
    # Milton disease codes were NOT in prefix_vocab at training time. For each
    # unknown code, compute its ancestor prefixes on the fly; use only the ones
    # that ARE in the trained vocab (rest are dropped). If nothing overlaps,
    # fall back to zero embedding.
    orig_prefix_vocab = model.prefix_vocab
    def _lookup_patched(self, codes, device):
        outs = []
        for c in codes:
            if c in self.code_to_row:
                row = self.code_to_row[c]
                idx = self.prefix_idx[row]
                mask = self.prefix_mask[row]
                emb = self.prefix_emb(idx)
                outs.append((emb * mask.unsqueeze(-1)).sum(dim=0))
            else:
                prefix_ids = []
                for i in range(1, len(c)+1):
                    p = c[:i]
                    if p in orig_prefix_vocab:
                        prefix_ids.append(orig_prefix_vocab[p])
                if prefix_ids:
                    idx_t = torch.tensor(prefix_ids, dtype=torch.long, device=device)
                    outs.append(self.prefix_emb(idx_t).sum(dim=0))
                else:
                    outs.append(torch.zeros(self.hidden_dim, device=device))
        return torch.stack(outs, dim=0)
    import types
    model._lookup_disease_emb = types.MethodType(_lookup_patched, model)
    return model


def ensemble_eval(models, X_ctx, y_ctx, X_qry, y_qry, device,
                  ctx_size, code=None, n_seeds=N_SEEDS):
    X_ctx = np.asarray(X_ctx, np.float32); y_ctx = np.asarray(y_ctx, np.int32)
    X_qry = np.asarray(X_qry, np.float32); y_qry = np.asarray(y_qry, np.int32)

    if len(X_qry) > N_EVAL_MAX:
        rng_sub = np.random.RandomState(0)
        idx = rng_sub.choice(len(X_qry), N_EVAL_MAX, replace=False); idx.sort()
        X_qry, y_qry = X_qry[idx], y_qry[idx]

    ctx_pos = np.where(y_ctx == 1)[0]
    ctx_neg = np.where(y_ctx == 0)[0]
    empty = {"auroc": float("nan"), "auprc": float("nan"),
             "brier": float("nan"), "ece": float("nan"),
             "n_ctx_pos": 0, "n_seeds_valid": 0}
    if len(ctx_pos) < 1 or len(ctx_neg) < 1 or len(np.unique(y_qry)) < 2:
        return empty

    prev_ctx = len(ctx_pos) / max(len(y_ctx), 1)
    n_ctx_pos = max(1, int(ctx_size * min(prev_ctx * 3, 0.3)))
    n_ctx_neg = max(1, ctx_size - n_ctx_pos)

    N_qry = len(y_qry)
    per_seed = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        pi = ctx_pos[rng.choice(len(ctx_pos), n_ctx_pos,
                                replace=(len(ctx_pos) < n_ctx_pos))]
        ni = ctx_neg[rng.choice(len(ctx_neg), n_ctx_neg,
                                replace=(len(ctx_neg) < n_ctx_neg))]
        ctx_idx = np.concatenate([pi, ni])

        ctx_X_t = torch.tensor(X_ctx[ctx_idx], device=device)
        ctx_y_t = torch.tensor(y_ctx[ctx_idx], device=device)
        X_qry_t = torch.tensor(X_qry, device=device)

        probs_all = []
        with torch.no_grad():
            for start in range(0, N_qry, BATCH_QRY):
                end = min(start + BATCH_QRY, N_qry); bsz = end - start
                batch_dict = {
                    "context_proteins": ctx_X_t.unsqueeze(0).expand(bsz, -1, -1).contiguous(),
                    "context_labels":   ctx_y_t.unsqueeze(0).expand(bsz, -1).long(),
                    "query_proteins":   X_qry_t[start:end],
                    "query_disease_codes": [code] * bsz,
                }
                # Average logits across ensemble members
                logits_sum = None
                for m in models:
                    l = m(batch_dict)
                    logits_sum = l if logits_sum is None else logits_sum + l
                logits_avg = (logits_sum / len(models)).cpu().numpy()
                probs_all.append(1.0 / (1.0 + np.exp(-logits_avg.reshape(-1))))
        probs_all = np.concatenate(probs_all)
        per_seed.append(compute_all_metrics(y_qry, probs_all))

    def _mean(k):
        vals = [d[k] for d in per_seed if not np.isnan(d[k])]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "auroc": _mean("auroc"), "auprc": _mean("auprc"),
        "brier": _mean("brier"), "ece":   _mean("ece"),
        "n_ctx_pos": n_ctx_pos,
        "n_seeds_valid": sum(1 for d in per_seed if not np.isnan(d["auroc"])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", default="selection.yaml")
    args = parser.parse_args()

    with open(args.selection) as f:
        sel = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    ckpts = [os.path.join(PROJECT, CKPT_DIR_TEMPLATE.format(s), "best_model.pt")
             for s in SEEDS]
    missing = [c for c in ckpts if not os.path.exists(c)]
    if missing:
        print(f"MISSING checkpoints: {missing}", flush=True); return

    print(f"Loading {len(ckpts)} ensemble members: seeds {SEEDS}", flush=True)
    models = [load_v7(c, device) for c in ckpts]

    letter = BLOCK_LETTER
    block_out = sel["blocks"][letter]
    clean_dir = os.path.join(PROJECT, DATA_DIR)
    ctx_pkl = os.path.join(clean_dir, "context_data.pkl")
    qry_pkl = os.path.join(clean_dir, "query_data.pkl")
    with open(ctx_pkl, "rb") as f: ctx_pool = pickle.load(f)
    with open(qry_pkl, "rb") as f: qry_pool = pickle.load(f)

    only = os.environ.get("ONLY_DISEASE", "").strip()
    per_disease_suffix = f"_{only}" if only else ""
    results_path = os.path.join(OUT_DIR, f"results{per_disease_suffix}.json")
    results = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
    # If ONLY_DISEASE, also honor the shared results.json cache to skip work
    if only:
        shared_path = os.path.join(OUT_DIR, "results.json")
        if os.path.exists(shared_path):
            with open(shared_path) as f:
                shared = json.load(f)
            key = f"{letter}_{only}"
            if key in shared and not np.isnan(shared[key].get("auroc", float("nan"))):
                print(f"  {only}: already in shared cache (AUROC={shared[key]['auroc']:.4f}) — skip", flush=True)
                return

    print(f"\n{'='*70}\n  Block {letter} — v11 featattn+diseaseemb ENSEMBLE, K={K_EVAL}, "
          f"n_seeds={N_SEEDS}, ensemble_size={len(SEEDS)}"
          + (f", ONLY_DISEASE={only}" if only else "")
          + f"\n{'='*70}", flush=True)
    for code in block_out["test"]:
        if only and code != only: continue
        key = f"{letter}_{code}"
        if key in results and not np.isnan(results[key].get("auroc", float("nan"))):
            print(f"  {code}: already done (AUROC={results[key]['auroc']:.4f}) — skip", flush=True); continue
        if code not in ctx_pool or code not in qry_pool:
            print(f"  {code}: not present — skip", flush=True); continue
        Xc, yc = ctx_pool[code]; Xq, yq = qry_pool[code]
        print(f"  {code}: ctx N={len(yc)} N+={int(yc.sum())}   "
              f"query N={len(yq)} N+={int(yq.sum())}", flush=True)
        m = ensemble_eval(models, Xc, yc, Xq, yq, device, K_EVAL, code=code)
        results[key] = {
            "block": letter, "code": code,
            "auroc": m["auroc"], "auprc": m["auprc"],
            "brier": m["brier"], "ece": m["ece"],
            "n_pos_ctx": int(yc.sum()), "n_pos_query": int(yq.sum()),
            "n_ctx_pos": m["n_ctx_pos"], "ctx_size": K_EVAL,
            "n_seeds_valid": m["n_seeds_valid"],
            "ensemble_size": len(SEEDS),
        }
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"    AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  "
              f"Brier={m['brier']:.4f}  ECE={m['ece']:.4f}", flush=True)

    rows = []
    for code in block_out.get("test", []):
        v = results.get(f"{letter}_{code}", {})
        rows.append({
            "block": letter, "disease": code,
            "auroc": v.get("auroc", np.nan),
            "auprc": v.get("auprc", np.nan),
            "brier": v.get("brier", np.nan),
            "ece":   v.get("ece",   np.nan),
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "summary.csv")
    df.to_csv(csv_path, index=False, float_format="%.4f")

    print(f"\n{'='*70}\nSUMMARY (v7 ensemble)\n{'='*70}", flush=True)
    print(df.to_string(index=False), flush=True)
    for metric in ["auroc","auprc","brier","ece"]:
        v = df[metric].dropna()
        if len(v): print(f"  {letter}-block mean {metric}: {v.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
