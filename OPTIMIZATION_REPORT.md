# RSNA Knee V2: optimization report (2026-10-09)

Notebook: `rsna-knee-abnormality-detection-v2.ipynb`, edited in place.
The pre-change copy is `Past Version/rsna-knee-abnormality-detection-v2.before-optimization.ipynb`.
Edits were applied by an exact-match patch script, which fails loudly if any anchor text is missing.
Cell outputs were cleared because they no longer matched the new code; the backup keeps them.

---

## 1. Audit: where the time went and why

| # | Problem | Root cause | Cost per run |
|---|---|---|---|
| 1 | Every Kaggle session rebuilt the cache | The cache lives in `/kaggle/working`, which is wiped between sessions, and nothing reused an attached copy. `CACHE_DATASET` was honoured off-Kaggle only | ~47–80 min (your estimates; not re-measured here) |
| 2 | Cell 20B uploaded the whole 13 GB cache on every Kaggle run | The cell's only guard was `ON_KAGGLE and not CACHE_DATASET` | Upload time on every run |
| 3 | The cache was audited twice with full decompression | Cells 20 and 46 both `np.load` every volume just to read `.shape` | 1.3–1.9 min per pass, measured locally (17.5–26 ms/file); 2 passes |
| 4 | The manifest was useless | Cell 19 indexed `slice_counts["sagittal"]` while the keys are `"Sagittal"`, so all 4,405 rows were `read_error: KeyError('sagittal')` | It also decompressed every file |
| 5 | NLI "hypothesis testing" was recomputed every run | Scores (55,668 pairs: 2,784 in Cell 33 + 696 in Cell 34 + 52,188 in Cell 36) were never saved, and Cell 34 recomputed Cell 33's best hypotheses | ~55k fp32 DeBERTa passes per run |
| 6 | Diagnostics sat on the critical path | Sample-study tests, 20-study preprocessing benchmarks (Cells 16–18), report audits and HF network checks ran on every rerun, and off-Kaggle they downloaded DICOMs | Minutes, plus network |
| 7 | The notebook could not produce a submission | No multi-epoch loop, no checkpoint save, no test inference or `submission.csv`. Cells 61/63 called the undefined `RSNAKnee2p5DMILSafe`. Model construction downloads ImageNet weights, which fails in the offline rerun | — |
| 8 | Training-step overheads | The loss looped over 12 targets with `.cpu()` per target (~15 GPU syncs per step), plus per-step `.item()` statistics. The CNN micro-batch of 8 saved no memory under autograd (measured: 8.0 vs 8.1 GB) | Unmeasurable locally (see §6) |
| 9 | Dataset overhead | Each item made a float32 copy of a 4.7 MB uint8 volume just to call `isfinite` on data that cannot be non-finite | 44.8 ms/study |
| 10 | Two studies were missing from the local cache | Windows `rename` of a `.part` folder hit a transient `PermissionError` (antivirus or indexer lock), with no retry | 2/4,407 studies; downstream cells assert 4,407 |
| 11 | Validation was unreliable | It used only 11–12 gold studies per fold, and weak studies were never held out, so epochs were picked by noise | Model quality |

## 2. Plan (priority → change → expected effect → risk)

| P | Change | Expected | Risk / validation |
|---|---|---|---|
| 1 | Reuse an attached cache, NLI scores, folds and checkpoints from `/kaggle/input`; gate Cell 20B | Cold-start cost paid once instead of every session | Refuse caches whose `cache_config.json` differs |
| 1 | Header-only, incremental manifest, plus an optional deep audit that is remembered per file | Seconds instead of minutes | Deep audit (CRC, non-constant check) is available and was run once: 0 problems |
| 1 | Disk cache for NLI scoring calls | 0 model calls on a warm run | Must be bit-exact with the original scores (verified) |
| 2 | `%%stage train` / `%%stage diagnostic` cell tags plus `RUN_MODE` | One "Run All" path for cold and warm runs, and a separate submit path | Static dependency check for every mode |
| 3 | Vectorized loss, lighter Dataset, batched CNN call (`cnn_chunk`), measured `channels_last` | Fewer syncs, faster loading | Loss equivalence test, identical Dataset outputs, eval-mode model equivalence |
| 4 | Cross-validated engine: weak studies get folds too (metric A), cross-fitted gold-58 (metric B), checkpoint, resume, early stopping, experiment ids | Trustworthy epoch selection; nothing is retrained needlessly | Resume and reuse tests |
| 5 | Inference + `submission.csv` validated against `sample_submission`, with an offline-safe model build | A valid submission | Smoke run end-to-end |

---

## 3. Cache: what is reused, and what invalidates it

| Artifact | Where | Reused when | Rebuilt when |
|---|---|---|---|
| `studies/*.npz` | `working/mri_cache_v2`, or any attached `*/studies` | The file exists in any read directory | It is missing. A directory whose `cache_config.json` differs from the current preprocessing is refused |
| `cache_config.json` | next to `studies/` | Always | Written automatically for the existing (legacy) cache |
| `artifacts/cache_manifest.csv` | working dir | Size and mtime unchanged | A file is new or changed (only its `.npy` header is read) |
| `artifacts/nli_cache/*.parquet` | working dir, or copied from an input | Same model, hypothesis, batch size and ordered report list | Any of those change |
| `artifacts/folds.csv` | working dir | Same study list and gold folds | They change |
| `checkpoints/*_best.pt`, `*.json`; `oof/*.csv` | working dir, or copied from an input | Same experiment id | Config, supervision, folds or cache change |

**Why the NLI cache works per call and in the original batch layout:** this DeBERTa is deterministic run to run (Δ = 0), and invariant to row order (Δ = 0). But its output depends on padding: batch 8 vs batch 1 differs by up to **0.011**. The pseudo-label cutoffs in Cells 38/42 (0.99, 0.999, 0.01) were tuned on scores from the original layout. Length-sorted batching (1.4× faster) or fp16 (shifts up to 0.036) would move studies across those cutoffs, so neither is used. A cold run reproduces the original scores bit for bit (verified, Δ = 0.0).

---

## 4. Fast-rerun workflow

- **Cold** (new dataset, or nothing attached): Run All. Missing studies are built, NLI scores are computed once and saved, then the notebook trains and predicts.
- **Warm** (artifacts present or attached): Run All. Nothing is rebuilt; the notebook goes straight to training.
- **Submit**: `RUN_MODE = "submit"` (set automatically in Kaggle's rerun). 14 code cells run: config, data, preprocessing functions, model, inference.

On Kaggle, after a training run: **Save Version** → in the next version, **Add Input → Your Work → this notebook**. Cell 1 then finds `mri_cache_v2/studies`, `artifacts/`, `checkpoints/` and `oof/`. Alternatively, publish the cache once with `PUBLISH_CACHE = True`.

---

## 5. Training changes

- Engine (new Cell 64): 5 folds over **all** 4,407 studies. The 58 gold studies keep the Cell 52 assignment. Weak studies are multilabel-stratified (iterstrat if installed, otherwise StratifiedKFold on the number of pseudo-positives).
  - Fold *k* trains on the other folds and is validated on fold *k*.
  - **A** = macro AUC vs pseudo-labels on ~870 held-out weak studies. It covers only the 4 targets whose pseudo-labels have both classes (Lateral Meniscus, Synovitis, Baker's, Contusion). It has low variance but inherits NLI's errors.
  - **B** = gold macro AUC. Per fold it is noisy (~12 studies); cross-fitted over all folds it covers 58 studies, reported with a bootstrap CI.
  - The best epoch is chosen on A. B is reported, never tuned on.
- Studies with no label at all are skipped in training and validation. They carried zero loss; Cell 64 prints the count.
- Checkpoints use plain-python metadata and load with `weights_only=True`. `load_model_state` is strict and strips the `_orig_mod.` / `module.` prefixes, so a load never fails silently into ImageNet or random weights.
- Resume from `*_last.pt` after an interruption. Patience-3 early stopping on A. Experiment log in `experiments.csv`.
- Levels:
  - **A**: sanity checks (~1 min): shapes, finite loss, gradient coverage, loss decreasing on a fixed batch, exact checkpoint round trip.
  - **Step benchmark**: original chunking vs batched, and `channels_last` vs contiguous, measured on the actual GPU.
  - **B**: optional screening on fold 0 with 30 % of weak studies × 3 epochs: `cnn_chunk_8`, `augment`, `gold_x2`, `lr_1e-4`.
  - **C**: full folds.

**97 % training accuracy.** Treat it as a diagnostic only. The notebook reports masked training accuracy (known labels, threshold 0.5) next to validation A/B every epoch, so overfitting shows as a widening gap. The metric that counts is macro AUC.

With noisy 0.25-weight pseudo-labels, and gold-only supervision for 5 targets (ACL, the 3 OA targets, Effusion), 97 % accuracy would mostly mean memorizing label noise and 46 gold studies per fold. The biggest quality lever is still label coverage: those 5 targets have only ~46 training labels each per fold. See `IMPROVEMENT_PLAN.md` §1 (LLM labels).

---

## 6. Benchmarks

> Local GPU caveat: during this session the RTX 4060 Laptop was pinned at **210 MHz** with "SW Power Cap" active. Its power sensor reports a bogus 590 W, so a plain fp16 matmul ran at 0.32 TFLOPS, roughly 200× below spec. **No local GPU timing is representative.** Memory numbers are valid. The CPU-bound numbers (cache, dataset, pandas) are real.

Measured this session unless marked. "Original" is the pre-change notebook, timed from its saved outputs or by running its code on the same files.

| Metric | Original | Optimized | Improvement |
|---|---:|---:|---:|
| Cache initialization, cold (new Kaggle session) | ~47–80 min rebuild every session (your estimate) | 0 min when the previous output or the dataset is attached; otherwise one build | Paid once instead of every session |
| Cache initialization, warm (Cell 19, cache present) | Manifest inflated all 4,407 files: 1.3–1.9 min (17.5–26 ms/file, measured) | **1.25 s** (manifest reused); 3.9 s on first header read | ~60–90× |
| Missing studies rebuilt | 2 lost to a Windows lock, never retried | Rebuilt in 27 s; cache now **4,407/4,407** | — |
| Cache audit | 2 full-inflate passes (Cells 20 + 46): 2.6–3.9 min | **0.06 s** structural; deep audit (CRC + content) once in 26 s, then remembered | ~2,600× (structural) |
| Hypothesis testing (NLI) | ~55,668 fp32 pairs recomputed every run; T4 time not measured (≈10 h on this capped GPU) | Warm: **0 model calls**, ~29 ms per cached call (61 calls ≈ 2 s). Cold: identical scores, bit-exact | Repeat runs no longer pay it |
| Fast-rerun init (kernel start → audited cache) | ~6–7 min (Cell 16: 112 s, Cell 17: 12.7 s, Cell 18: 2 s, 2 × manifest/audit inflate), plus DICOM downloads off-Kaggle | **12.3 s** (Cells 0–25, measured) | ~30× |
| Label chain, warm (Cells 25–44, excluding NLI) | Cell 44 Python loop: 1.5 s | 0.18 s; the rest is unchanged pandas (seconds) | — |
| DataLoader `__getitem__` (CPU) | 44.8 ms → **22.3 studies/s per worker** | 33.1 ms → **30.2 studies/s per worker** (identical outputs) | 1.35× |
| Loss: host syncs per step | ~15 (`.cpu()` per target, boolean `.any()`) | 0 (vectorized; equal to the original on 200 random batches) | — |
| Training step time | Unavailable: no V2 epoch was ever completed, and this laptop's GPU is capped | Measured on the T4 by Cell 65's step benchmark (original chunk-8 vs batched, contiguous vs `channels_last`) | Kaggle run needed |
| `channels_last` (capped 4060, B=1) | — | 37.4 s/step vs **4.7 s/step** contiguous → contiguous on GPU, `channels_last` on CPU (1.55× faster there) | Avoided an 8× slowdown |
| Training time per epoch / full training | Unavailable | Unavailable (needs the T4); logged per epoch as `train_min` and `train_studies_per_s` | — |
| Peak GPU memory, train (B4 × 3 planes × 12 slices) | 8.02 GB (micro-batch 8) | 8.10 GB batched; **5.44 GB** with grad checkpoint (auto on < 12 GB GPUs) | Micro-batching never saved memory |
| Peak GPU memory, inference (1 study × 72 slices) | — | 0.59 GB | — |
| Peak CPU memory | Not measured | Logged per epoch (`cpu_rss_gb`) | — |
| Test inference (3 public studies, this capped laptop) | Not implemented | decode 0.93–1.94 s/study/thread (4 threads); model 1.97 s/study; one-time cuDNN autotune up to 124 s | — |
| Test inference, **CPU only** (4 threads, emulating a Kaggle CPU notebook) | Not implemented | Whole submit run 17 s for 3 studies; model 1.38 s/study (`channels_last`; contiguous 1.99 s) → ~29 min / 1,300 studies here | No GPU quota needed to submit |
| Best validation score | V2: none. V1: 0.579 (12 studies) | None yet: no real model trained (see §9) | — |
| Training accuracy | Unavailable | Logged per epoch (`train_masked_acc`) | — |

Correctness checks run:
- Model in eval mode vs the original class with identical weights: max |Δlogit| 1.5e-4.
- NLI cache vs the original scorer: Δ = 0.0.
- Cell 44 vectorized vs loop: identical matrices.
- Dataset old vs new: identical tensors.
- Static dependency check: every name each running cell reads is defined, in train (32 cells), submit (14) and diagnostics (74) modes. The one flag is a runtime-guarded `globals()` lookup in an old audit cell.
- Kaggle input discovery: tested on a simulated `/kaggle/input` (notebook-output and dataset layouts; competition data skipped; existing files never overwritten).
- End-to-end smoke run, train mode: real cache, real gold labels, synthetic weak labels, 16+16 studies, 2 epochs. It covered: Level A ✓, step benchmark ✓, checkpoint ✓, interrupted fold resumes ✓, finished fold reused ✓, same experiment id from a fresh directory ✓, inference ✓, submission validated + reloaded ✓.
- Submit mode: 14 code cells, 143–148 s, valid `submission.csv` ✓.

---

## 7. Model evaluation

**No real model has been trained yet.** The full label chain (Cells 33–36, ~55k DeBERTa passes) and the training both need a working GPU. On this laptop, during this session, the NLI alone would take ~10 h, so every training number above comes from a code-path smoke test with **synthetic** weak labels and is meaningless as a model score. A smoke-test checkpoint scoring "B ≈ 0.50" on 12 studies is noise.

On the first real Kaggle run, Cell 65 prints the following per epoch:
- train loss, masked train accuracy, validation loss, A and B, so overfitting shows as a widening train-acc / A gap;
- cross-fitted gold macro AUC with a 95 % bootstrap CI, and per-target AUC.

**Why the gold-only targets will be weak:** ACL, Medial / Lateral / PF OA and Effusion get **no** pseudo-labels (Cell 44 policy), so each fold trains them on ~46 gold studies.

Gold positives per target (of 58): ACL 24, MCL 9, Medial Meniscus 26, Lateral Meniscus 23, Medial OA 15, Lateral OA 11, PF OA 21, Effusion 35, Synovitis 27, Baker's 12, Contusion 19, Fracture 18.

## 8. Submission artifact

**Not yet produced from a real model.** The pipeline is verified end to end: it writes `/kaggle/working/submission.csv` and checks it against `sample_submission` (columns, row count, ids and order, finite values in [0, 1], unique ids, save + reload). The verification used a smoke checkpoint, deliberately kept out of `working/`.

Your Kaggle GPU quota is used up this week, so plan around that. Training needs a GPU; **the submission does not.** A CPU-only Kaggle notebook was verified here (submit mode, 14 cells, 17 s for 3 studies, 1.38 s/study/model on 4 threads → ~29 min per 1,300 studies on this laptop; Kaggle's CPUs are slower but far inside 9 h).

**Path 1: train on the laptop (no Kaggle GPU needed).**
1. Clear the GPU power cap: reboot, or reseat the charger; if it persists, update the NVIDIA driver.
2. Check it with `python -c "import torch,time;a=torch.randn(4096,4096,device='cuda',dtype=torch.half);a@a;torch.cuda.synchronize();t=time.time();[a@a for _ in range(50)];torch.cuda.synchronize();print(50*2*4096**3/(time.time()-t)/1e12,'TFLOPS')"`. It should print tens of TFLOPS, not 0.3.
3. Run All locally on `C:\Python314\python.exe` with `RUN_MODE="train"`, `FOLDS_TO_TRAIN=[0]`.
   - The cache is already complete and audited (12 s).
   - NLI runs once and is then cached.
   - Gradient checkpointing switches on automatically for 8 GB.
4. Create a private Kaggle dataset containing `working/checkpoints/` (one `*_best.pt` is ~35 MB).
5. On Kaggle, start a **CPU** notebook: attach that dataset and the competition, set `RUN_MODE="submit"`, Internet OFF → Save → Submit.

**Path 2: when the Kaggle quota resets (weekly).**
1. **Session 1** (T4, Internet ON, `RUN_MODE="train"`, `FOLDS_TO_TRAIN=[0]`): Run All → Save Version (~3–4 GPU-h estimated: cache build ~47–80 min unless attached, NLI once, fold 0, inference).
2. **Submit:** attach the output, `RUN_MODE="submit"`, Internet OFF (CPU or GPU).
3. **Session 2:** attach the output again, `RUN_SCREENING=True`, `FOLDS_TO_TRAIN=[0,1,2,3,4]`. Finished folds are reused; resubmit with the 5-fold ensemble.

The entry deadline (Oct 15) only requires accepting the rules / joining. The final submission is due Oct 22.

## 9. Remaining issues / not verified

- **Unverified on a real GPU:**
  - T4 step time, epoch time, full-training time, peak memory and data-wait %. The notebook logs all of them.
  - The cold-run time of the NLI cells on a T4.
  - The real supervision statistics (studies with ≥1 label, metric A coverage).
- **Laptop GPU is power-capped** at 210 MHz (driver reports 590 W draw). Reboot or reseat the charger; if it persists, update the NVIDIA driver. Until then, local GPU timing is meaningless.
- **Hidden-rerun variable:** `KAGGLE_IS_COMPETITION_RERUN` is the conventional rerun flag; I could not test it here. Set `RUN_MODE="submit"` explicitly in the submission version anyway.
- **Metric A covers only 4 targets** (two-sided pseudo-labels), and selection on A inherits NLI bias. B is unbiased but noisy (58 studies).
- **Grad checkpointing on < 12 GB GPUs** gives BatchNorm running stats a double update per step, and there `cnn_chunk` is superseded by `CNN_MICRO_BATCH`. Neither applies on the T4.
- **Missing planes at test time** are zero-filled; all 4,407 training studies had all three planes, so this case is out of distribution (counted and reported).
- **Folds are not site-grouped:** no site field is cached. `IMPROVEMENT_PLAN.md` estimates random folds overestimate the score by ~0.05.
- **`torch.compile`, EMA and AUC-loss fine-tuning** were not added. They would need T4 benchmarks first.
- **Biggest quality lever is unchanged:** label coverage for ACL / OA / Effusion (LLM labels, `IMPROVEMENT_PLAN.md` §1).
- **Stale files left untouched:**
  - `working/mri_cache_v2/manifest.csv` (old, 100 % `KeyError`; superseded by `working/artifacts/cache_manifest.csv`);
  - `failures/failures.csv` (lists the 2 studies that are now built).
- **Superseded cells removed:** old Cells 61–63 (they called the undefined `RSNAKnee2p5DMILSafe`). They remain in the backup.
