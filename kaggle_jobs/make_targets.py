"""Denoised training targets: mix the report labels (v2) with the 4-fold teacher's out-of-fold predictions.

Per finding the teacher score is mapped onto the report-label scale (isotonic fit of report label on teacher score,
or quantile matching; whichever agrees better with gold-58), then target = (1 - B) * report + B * mapped teacher.
Gold-58 never enters training: the check scores the same recipe on gold, using the teacher folds' gold predictions.

usage: python -P make_targets.py TEACHER_DIR OUT_PARQUET [B=0.5]
"""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
tdir, out = Path(sys.argv[1]), Path(sys.argv[2])
B = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
root = Path(__file__).resolve().parent.parent / "data"

tr = pd.read_csv(root / "rsna-knee-abnormality-detection/train.csv"); tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
gold = tr[tr[LAB].notna().all(axis=1)].set_index("StudyInstanceUID")[LAB].astype(int)
rep = pd.read_parquet(root / "labels_v2/labels_llm_soft.parquet").set_index("StudyInstanceUID")[LAB].astype(np.float64)
oof = pd.concat([pd.read_csv(f, dtype={"StudyInstanceUID": str}) for f in sorted(glob.glob(str(tdir / "oof_t*.csv")))]).set_index("StudyInstanceUID")[LAB]
train_ids = [u for u in rep.index if u not in gold.index]
assert oof.index.is_unique and not set(oof.index) & set(gold.index), "OOF must cover each training study once and no gold"
assert set(oof.index) == set(train_ids), f"OOF covers {len(set(oof.index) & set(train_ids))} of {len(train_ids)} training studies"
folds = [json.load(open(f))["gold_pred"] for f in sorted(glob.glob(str(tdir / "m_t*.json")))]
assert len(folds) == 4, f"{len(folds)} teacher folds"
tg = pd.DataFrame({u: np.mean([f[u] for f in folds], 0) for u in folds[0]}, index=LAB).T.loc[gold.index]   # fold-mean gold predictions
rep_t, oof = rep.loc[train_ids], oof.loc[train_ids]


def mapped(c, kind, q):
    """Teacher scores q (train OOF or gold) on the report-label scale, fitted on the training studies only."""
    x, y = oof[c].values, rep_t[c].values
    if kind == "iso":
        return IsotonicRegression(out_of_bounds="clip").fit(x, y).predict(q)
    pct = np.searchsorted(np.sort(x), q, side="right") / len(x)
    return np.quantile(y, np.clip(pct, 0, 1))


def gold_macro(kind):
    return np.mean([roc_auc_score(gold[c], (1 - B) * rep.loc[gold.index, c] + B * mapped(c, kind, tg[c].values)) for c in LAB])


print("gold-58 macro AUC | report labels %.4f | teacher (fold mean) %.4f" % (
    np.mean([roc_auc_score(gold[c], rep.loc[gold.index, c]) for c in LAB]), np.mean([roc_auc_score(gold[c], tg[c]) for c in LAB])))
scores = {k: gold_macro(k) for k in ("iso", "quantile")}
kind = max(scores, key=scores.get)
print(f"mix B={B}: " + " | ".join(f"{k} {v:.4f}" for k, v in scores.items()) + f" -> {kind}")
assert scores[kind] >= 0.92, "denoised targets do not beat 0.92 on gold-58; not writing"
tgt = pd.DataFrame({c: (1 - B) * rep_t[c].values + B * mapped(c, kind, oof[c].values) for c in LAB}, index=train_ids).clip(0, 1)
for c in LAB:
    print(f"  {c:17s} report mean {rep_t[c].mean():.3f} -> target mean {tgt[c].mean():.3f} | "
          f"gold AUC {roc_auc_score(gold[c], (1 - B) * rep.loc[gold.index, c] + B * mapped(c, kind, tg[c].values)):.3f}")
tgt.index.name = "StudyInstanceUID"
tgt.astype(np.float32).reset_index().to_parquet(out, index=False)
print("wrote", out, tgt.shape)
