# RSNA Knee Abnormality Detection: Improvement Plan

This is a review of `arnav-rsna-knee-abnormality-detection.ipynb` (45 cells, run on a Kaggle GPU), followed by a step-by-step plan to make it competitive.
Written 2026-10-04. Competition facts are sourced at the end. Numbers quoted from other teams come from their public write-ups and have not been verified.

---

## 0. TL;DR

- **Where you are:** the best validation macro AUC is **0.579**, measured on **12 studies**. With that few studies, the number is mostly noise.
- **Public reference points:** **0.903** public LB (18th of 792, from a public write-up) and **0.939** (from a public GitHub repo).
- **Where the gap is:** the CNN is not the problem. The gap comes from **labels, slices and validation**.

| # | Change | Why | Impact |
|---|---|---|---|
| 1 | Replace the regex labels with **multilingual LLM labels** for all 4,407 reports | Regex AUC against gold is 0.42–0.58 on 4 targets. The patterns are English-only, but the reports are in about 12 languages | **Largest** |
| 2 | **Fix the slice-ordering bug**: sort along the slice normal, not z | Sagittal and coronal stacks are currently scrambled | Large (ACL, menisci) |
| 3 | 16–32 central slices at 256–288 px, fed to a **2.5D multiple-instance (MIL) model**: a per-slice CNN with attention pooling | 5 slices, including the edge slices, miss the ACL and menisci | Large |
| 4 | **5-fold grouped CV** plus a **cross-fitted gold-58** metric | A 12-study holdout cannot rank models | Lets you make correct decisions |
| 5 | Split the training and inference notebooks, and add a **5-fold ensemble** | The notebook currently re-runs about 2 h of preprocessing and training on every submission | Safety, and a reliable small gain |
| 6 | A real training recipe: batch size, LR, epochs, EMA, and no flipping of sagittal images | Currently 5 epochs at a backbone LR of 1e-5 | Medium |

**Deadlines:** the entry / team-merger deadline is **Oct 15, 2026** and the final submission deadline is **Oct 22, 2026**. If you have not accepted the rules yet, do it today.

---

## 1. Competition facts that shape every decision

- **Metric:** macro ROC AUC over 12 targets, all weighted equally.
  - AUC only looks at **rank**, so calibration does not matter. Do not spend time on it.
  - Every target counts the same, so hard and rare targets (MCL, Lateral OA) matter as much as Effusion.
- **Test set:** about 1,300 hidden studies, **images only, with no reports**. Reports are a training-time supervision source only.
- **Submission:** a Kaggle notebook with at most 9 h runtime and no internet. A separate **Efficiency prize** puts runtime into the score.
- **Data:**
  - 4,407 training studies and 24,371 series (about 5.5 per study).
  - **58 gold-labelled studies.**
  - Reports come from 16–19 sites and are written in about 12 languages (Spanish, Dutch, …).
- **Site shift:** one public repo reports that random folds overestimate the site-grouped CV score by about 0.05. Group your folds by site.
- **Public vs. private:** finding prevalence differs between the public and private test splits, so do not overfit to the public LB.

---

## 2. Diagnosis of the current notebook

Cells are counted **0-based** in notebook order.

### 2.1 The weak labels are close to random (cells 3–7, 10–11): critical

| Target | Regex AUC vs gold (N) | Parsed prevalence | Gold prevalence |
|---|---|---|---|
| ACL | 0.556 (18) | 0.42 | 0.41 |
| MCL | 0.583 (10) | 0.31 | 0.16 |
| Medial Meniscus | 0.764 (16) | 0.52 | 0.45 |
| Lateral Meniscus | 0.762 (24) | 0.35 | 0.40 |
| Medial OA | n/a (N<5) | 0.64 | 0.26 |
| Lateral OA | n/a (N<5) | 0.59 | 0.19 |
| PF OA | 0.542 (7) | 0.50 | 0.36 |
| Effusion | 0.685 (28) | **0.37** | **0.60** |
| Synovitis | **0.417** (9) | **0.87** | **0.47** |
| Baker's | 1.000 (9) | 0.24 | 0.21 |
| Contusion | **0.461** (19) | 0.45 | 0.33 |
| Fracture | 0.929 (14) | 0.17 | 0.31 |

Only **184 (Lateral OA) to 1,572 (Effusion)** of the 4,407 reports produce any label at all.

Root causes:
1. **The patterns are English-only, but the reports are multilingual.** Most non-English reports return nothing.
2. **Negation leaks across sentences.** `negated()` looks 120 characters back for `no`, `normal`, `intact` and similar words.
   - Example: "No fracture. Moderate joint effusion." gets its effusion negated.
   - This explains why Effusion comes out at 0.37 when gold says 0.60.
3. **The look-ahead window negates real findings.** In "ACL tear. PCL intact.", the word "intact" within 80 characters after the match negates the ACL tear.
4. **Some patterns are too broad.**
   - `medial.{0,100}degenerative` fires on "medial meniscus … degenerative signal" and produces a false Medial OA.
   - `marrow edema` is counted as Contusion even when the edema is degenerative.
5. **An unmentioned finding becomes NaN, so most studies end up with no labels.** The evidence is in the outputs:
   - Cell 29 prints `Valid labels in batch: 0`.
   - Cell 32 prints `Sanity-check loss: 0.0`.
6. **`LABEL_CONFIDENCE` is not tied to measured quality.** ACL and MCL get weight 1.0 even though their AUC is about 0.56–0.58.

### 2.2 Slice-ordering bug (cell 12): critical

`slices.sort(key=lambda x: float(x[0].ImagePositionPatient[2]))` sorts by z. z is the stacking axis **only for axial series**.
- Sagittal slices are spaced along x and coronal slices along y, so their z is roughly constant.
- The sort therefore falls back to filename order or to tiny oblique offsets, and neither is guaranteed to be anatomical.
- As a result, the "uniform sampling" picks arbitrary slices, and the 5-channel input has no spatial order.

### 2.3 The input throws away most of the pathology (cells 12–15)

- **5 slices per series**, taken with `linspace(0, N-1)`. This includes the first and last slices, which are often outside the joint.
  - The ACL shows on only a few mid-sagittal slices, and 5 samples often miss it.
  - This fits the ACL validation AUC staying below 0.5 in every epoch.
- **224 px with `INTER_AREA`**: meniscal tears are only a few pixels wide.
- **Per-slice percentile normalization**: intensity is inconsistent from one slice to the next within a series.
- **One series per plane**: the code takes the first fluid-sensitive series it finds.
  - `Fat_Suppression` is never used.
  - T1 sequences, which help with fracture lines and osteophytes, are never used.
- **Every DICOM file is decoded just to keep 5 slices.** That costs 1.5 s per study, or **107 min** for the training set.

### 2.4 Model (cells 17–19)

- **5-channel conv hack.** Channels 0–2 get the R, G and B filters, and channels 3–4 get the mean filter.
  - Each slice is processed by a different, arbitrary filter set, so slice order matters, and that order is scrambled (see 2.2).
- Missing planes are fed as zero images through the CNN instead of being masked out.
- Inputs are not ImageNet-normalized (minor).

### 2.5 Augmentation (cell 27)

`torch.flip(x, dims=[-1])` is applied to all three planes together.
- **On sagittal images this swaps anterior and posterior**, so the patella ends up behind the femur. That never happens at test time.
- On coronal and axial images, the flip turns a right knee into a mirrored left knee, which is plausible on its own. But the sagittal slice order is not reversed to match, so the geometry no longer agrees across planes.
- The 18th-place public pipeline never flips.

### 2.6 Validation (cells 26, 34–35): critical

- There are **12 gold studies** in the validation split.
  - Lateral OA and Fracture come out as **NaN** because the split contains only one class for them.
  - With 1–3 positives per target, a single study moves the AUC by 0.1–0.3. MCL scoring 0.07–0.22 means nothing.
  - The "best epoch" (epoch 3, AUC 0.579) is chosen by noise.
- `evaluate_model` ignores `mask`. That is harmless while validation is gold-only, but it becomes wrong as soon as weakly labelled studies enter validation.

### 2.7 Training (cells 28, 33, 35)

- **Batch size 4** means BatchNorm statistics come from 4 images per plane.
  - In the "frozen" first epoch, `model.train()` still updates the BN running statistics.
- **The backbone LR is 1e-5, with 5 epochs and `T_max=5`**, so the backbone barely moves.
- Most of the 4,395 training studies have **no label at all**, so those steps are wasted.
- `torch` is never seeded, so runs are not reproducible.
- An epoch takes 0.8 min, so you are using a small fraction of the compute you have.

### 2.8 Submission (cells 21–22, 37–44)

- **The submission notebook re-runs the 107-min training preprocessing and the training itself** on every submission, and again during the hidden-test rerun. That risks the 9 h limit and burns GPU quota.
- There is one model, no TTA and no fold ensemble.
- **There is no fallback** if a hidden-test DICOM fails to decode, so one bad study can crash the whole run.
- The notebook metadata says `accelerator: none`, although the outputs show `cuda`. Double-check that the GPU is enabled on the submission notebook.

### 2.9 Hygiene

- Dead code: `choose_series` / `selected_series` (cell 9) and `select_series_for_study` (cell 14).
- Duplicates: `load_study` (cells 15 and 37), `LABEL_CONFIDENCE` (cells 10 and 24), and cell 20 repeats cell 19.
- The label-weights DataFrame `weights` (cell 11) is overwritten by `models.ResNet18_Weights.DEFAULT` (cell 17). That is why cell 24 had to rebuild it.

---

## 3. The plan, step by step

### Step 1: Labels (P0, the biggest lever)

#### 1a. Quick win (day 1): public LLM labels

1. Attach the Kaggle dataset **`pilkwang/rsna-knee-llm-labels`**, which the 18th-place pipeline fused into its labels. Check its license.
2. Compute the per-target AUC of these labels against the 58 gold studies, using the same loop as cell 7.
3. If they beat the regex on every target, which is very likely, use them right away while 1b runs.

#### 1b. Your own multilingual LLM labeler

- **Where to run it:** a Kaggle notebook with **internet on** and 2×T4, or locally.
  - Only the *submission* notebook has to be offline.
  - Use **vLLM** for batched inference.
- **Models:** open-weight, multilingual instruct models. Candidates:
  - The Qwen3 family or newer. 14B and 32B fit on 2×T4 with AWQ 4-bit quantization.
  - Gemma 3 27B.
  - MedGemma 27B (text).
  - **Check the competition's external-model rules first.**
- **Recipe:**
  1. Send one prompt per report at **temperature 0** and ask for **JSON only**. Use vLLM's structured-output / guided-JSON option if your version supports it.
  2. For each finding, ask for one of: `positive | probable | uncertain | negative | not_mentioned`. Also ask for `language`, which you will reuse as a site proxy in Step 4.
  3. Map the answers to soft targets. This is a starting point; tune it on gold.

     | LLM answer | Target | Loss weight |
     |---|---|---|
     | positive | 1.0 | 1.0 |
     | probable | 0.8 | 1.0 |
     | uncertain | 0.5 | 0.5 |
     | negative | 0.0 | 1.0 |
     | not_mentioned | 0.05 | 0.5 |

  4. **Validate on gold-58** with per-target AUC and accuracy.
     - Read **every** disagreement and refine the definitions.
     - Keep about 20 gold studies **untouched** for a final honest check of the prompt, so you don't tune it to all 58.
  5. Run the labeler on all 4,407 reports.
     - Do 100 reports first and measure throughput.
     - Retry any output that fails to parse.
  6. **Ensemble at least 2 labelers** (yours plus the public one, or two different LLMs) by averaging their soft targets. Labeler noise is the ceiling in this competition: the 18th-place author estimates the report-vs-image ceiling at about 0.88–0.92.
  7. Gold studies always keep their gold labels.

Prompt skeleton. The definitions are **hypotheses to check against the gold disagreements**, not ground truth:

```text
You are a musculoskeletal radiologist. Read the knee MRI report (it may be in any language)
and classify 12 findings. Return JSON only, with exactly these keys:
{"language": "...", "ACL": "...", "MCL": "...", "Medial Meniscus": "...", "Lateral Meniscus": "...",
 "Medial OA": "...", "Lateral OA": "...", "PF OA": "...", "Effusion": "...", "Synovitis": "...",
 "Baker's": "...", "Contusion": "...", "Fracture": "..."}
Allowed values: positive | probable | uncertain | negative | not_mentioned

Definitions:
- ACL: partial or complete tear of the native ACL or of an ACL graft. Mucoid degeneration alone = negative.
- MCL: sprain or tear of any grade (I-III).
- Medial / Lateral Meniscus: tear of any type (horizontal, radial, root, complex, bucket-handle,
  degenerative tear). Intrasubstance signal not reaching the surface = negative.
- Medial / Lateral / PF OA: osteoarthritis, arthrosis or chondral loss in that compartment
  (cartilage thinning/defects, osteophytes, subchondral change).
- Effusion: joint effusion of any size. "Physiological amount of fluid" = negative.
- Synovitis: synovitis or synovial thickening; include Hoffa fat-pad edema/impingement and plica syndrome.
- Baker's: popliteal (Baker's) cyst, intact or ruptured.
- Contusion: bone bruise / traumatic bone-marrow edema. Purely degenerative subchondral edema = negative.
- Fracture: any fracture, incl. osteochondral, impaction, avulsion (e.g. Segond), stress/insufficiency.

Report:
<<<{report}>>>
```

vLLM driver:

```python
import json
from vllm import LLM, SamplingParams

llm = LLM(model=MODEL_DIR, tensor_parallel_size=2, max_model_len=8192)
params = SamplingParams(temperature=0.0, max_tokens=512)

conversations = [
    [{"role": "user", "content": PROMPT.replace("{report}", str(r))}]
    for r in train["Report"].fillna("")
]
# Qwen3: also pass chat_template_kwargs={"enable_thinking": False} (supported in recent vLLM)
outputs = llm.chat(conversations, params)

rows = []
for o in outputs:
    text = o.outputs[0].text
    try:
        rows.append(json.loads(text[text.find("{"): text.rfind("}") + 1]))
    except json.JSONDecodeError:
        rows.append(None)  # collect these and re-run with a stricter prompt
```

#### 1c. Using the labels at train time

- Use BCE with **soft targets** and per-label weights: gold 1.0 (try ×2), LLM labels per the table above.
- Every study now has all 12 targets, so batches are never empty and the `mask` logic becomes trivial.
- Delete the regex labeler.

---

### Step 2: Preprocessing v2 (P0)

Fixed slice ordering, header-only reads, a central slab, per-series normalization and uint8 output:

```python
import numpy as np
import pydicom
import cv2
from pathlib import Path


def slice_pos(ds):
    """Position along the axis the slices are stacked on, in patient coordinates."""
    try:
        o = np.asarray(ds.ImageOrientationPatient, dtype=float)
        normal = np.cross(o[:3], o[3:])
        return float(ds.ImagePositionPatient[int(np.argmax(np.abs(normal)))])
    except AttributeError:
        return float(getattr(ds, "InstanceNumber", 0))


def load_series(path, n_slices=24, keep=0.8, size=256):
    files = sorted(Path(path).glob("*.dcm"))
    if not files:
        return None
    # headers only: cheap, no pixel decoding
    pos = [slice_pos(pydicom.dcmread(f, stop_before_pixels=True)) for f in files]
    files = [f for _, f in sorted(zip(pos, files))]

    # central slab: drop the outer (1 - keep) of slices, which are often outside the joint
    n = len(files)
    lo = int(n * (1 - keep) / 2)
    hi = max(lo, n - 1 - lo)
    idx = np.linspace(lo, hi, n_slices).round().astype(int)

    vol = []
    for i in idx:
        ds = pydicom.dcmread(files[i])
        img = ds.pixel_array.astype(np.float32)
        img = img * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
        vol.append(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA))
    vol = np.stack(vol)

    # one window for the whole series, not per slice
    fg = vol[vol > 0]
    p1, p99 = np.percentile(fg, [1, 99]) if fg.size else (0.0, 1.0)
    vol = np.clip((vol - p1) / (p99 - p1 + 1e-6), 0, 1)
    return (vol * 255).astype(np.uint8)  # [n_slices, size, size]
```

- **Parallelize** with `joblib.Parallel(n_jobs=4)(delayed(process_study)(sid) for sid in ids)`. With header-only reads, this should bring the 107 min down a lot.
- **Choosing a series for each plane:**
  1. Prefer `Fluid_Sensitive == 1 & Fat_Suppression == 1`.
  2. Otherwise take a fluid-sensitive series.
  3. Otherwise take any series.
  - Optionally add a 4th series: a non-fluid-sensitive (T1-like) sagittal or coronal, for fractures and osteophytes.
  - Store a `present` mask for each series slot.
- **Cache:** save each study with `np.savez_compressed(path, sag=..., cor=..., ax=..., present=...)`.
  - The raw size is 3 × 24 × 256² uint8 ≈ 4.7 MB per study, or about 21 GB before compression.
  - MRI background compresses well, but measure it.
  - Kaggle notebook output is capped at about 20 GB. If you hit the cap, split the cache across 2 notebooks/datasets or store 16 slices.
- **Save the cache as a Kaggle Dataset.** The training notebooks attach it, and the inference notebook reuses `load_series`.
- **While preprocessing, also store the site-proxy DICOM tags** for Step 4: `Manufacturer`, `ManufacturerModelName`, `MagneticFieldStrength`, and `InstitutionName` if it is not blank.
- **Hidden-test robustness:**
  - Wrap each study in `try/except` and fall back to a constant prediction, such as the training prevalence.
  - Bring `pylibjpeg` / `gdcm` wheels offline in case of compressed transfer syntaxes.
- **Train on 12–16 slices, infer on all 24.** During training, take a random contiguous window of 12–16 of the 24 stored slices. This is augmentation and saves memory. At inference, use all 24, since attention pooling handles any number of slices.

---

### Step 3: Model v2, a 2.5D MIL model (P0)

Each slice plus its two neighbours forms a 3-channel image, which matches the pretrained RGB stem exactly. A 2D backbone encodes each slice, attention pools the slices within each series, and the series are fused into 12 logits:

```python
import numpy as np
import timm
import torch
import torch.nn as nn


def to_25d(vol):
    """[T, H, W] -> [T, 3, H, W]: each slice with its two neighbours."""
    p = np.pad(vol, ((1, 1), (0, 0), (0, 0)), mode="edge")
    return np.stack([p[:-2], p[1:-1], p[2:]], axis=1)


class KneeMIL(nn.Module):
    def __init__(self, name="tf_efficientnet_b3.ns_jft_in1k", n_series=3, n_targets=12):
        super().__init__()
        self.enc = timm.create_model(name, pretrained=True, in_chans=3, num_classes=0)
        d = self.enc.num_features
        self.series_emb = nn.Parameter(torch.zeros(n_series, d))
        self.attn = nn.Sequential(nn.Linear(d, 256), nn.Tanh(), nn.Linear(256, 1))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(d * n_series, n_targets))

    def forward(self, x, present):
        # x: [B, S, T, 3, H, W]   present: [B, S] bool
        B, S, T = x.shape[:3]
        f = self.enc(x.flatten(0, 2)).view(B, S, T, -1)    # per-slice features
        w = self.attn(f).softmax(dim=2)                     # attention over slices
        s = (w * f).sum(dim=2) + self.series_emb            # [B, S, d]
        s = s * present.unsqueeze(-1).float()               # mask missing series
        return self.head(s.flatten(1))
```

- **Backbones to try**, via timm. Upload the weights as a Kaggle Dataset or Model for offline use.
  - `tf_efficientnet_b3.ns_jft_in1k`
  - `tf_efficientnetv2_s.in21k_ft_in1k`
  - `convnext_tiny.fb_in22k_ft_in1k`
  - `vit_small_patch14_dinov2.lvd142m`. Set `img_size` to a multiple of 14, such as 252 or 280.
  - DINOv3 (Meta, 2025), if its license is allowed.
- **Memory:** 4 studies × 3 series × 12 slices is 144 images per step at 256 px, which fits a T4 in fp16 for EfficientNet-B3.
  - If it doesn't fit, call `self.enc.set_grad_checkpointing()`.
  - BatchNorm now sees 144 images per step, so the batch-of-4 BN problem disappears.
- **Normalization:** convert the uint8 input to float and normalize with the backbone's mean and std (`self.enc.pretrained_cfg`).
- **Upgrade (P2): per-target queries.** Replace the attention pooling and head with **12 learnable queries** that cross-attend over all slice tokens of all series (ML-Decoder / Query2Label style). This lets ACL look at mid-sagittal slices, MCL at coronal slices and Baker's at posterior axial slices.

---

### Step 4: Validation you can trust (P0)

1. **Site proxy:** combine `language` (from the LLM) with `Manufacturer` and `ManufacturerModelName` (from DICOM). If the tags are stripped, use language alone.
2. **Folds:** use `sklearn.model_selection.StratifiedGroupKFold(n_splits=5)` with `groups = site_proxy` and `y =` the number of positive LLM labels, clipped to 0–5.
   - Check that every fold gets about 11–12 gold studies, and fix it by hand if not.
   - Save the result to **`folds.csv` once and never change it.**
3. **Track two metrics for every experiment:**
   - **A, OOF macro AUC against the LLM labels** on the non-gold studies, with soft targets binarized at 0.5.
     - N ≈ 4,300, so variance is low, but label noise biases it.
   - **B, cross-fitted gold-58 macro AUC.** Each gold study is scored **only** by the fold model that did not train on it.
     - This is unbiased but noisy.
     - The 18th-place author first measured it with leakage and had to redo it. Don't repeat that mistake.
4. **Measure the noise before you trust a difference:**

```python
import numpy as np
from sklearn.metrics import roc_auc_score


def macro_auc(y, p):
    scores = [roc_auc_score(y[:, j], p[:, j]) for j in range(y.shape[1]) if 0 < y[:, j].sum() < len(y)]
    return float(np.mean(scores))


def bootstrap_diff(y, p_a, p_b, n=2000, seed=0):
    """95% CI of macro_auc(model A) - macro_auc(model B) on the same studies."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        diffs.append(macro_auc(y[i], p_a[i]) - macro_auc(y[i], p_b[i]))
    return np.percentile(diffs, [2.5, 50, 97.5])
```

**Decision rule:** keep a change if A improves **and** B does not drop beyond its bootstrap noise. Treat the public LB as a sanity check, not a selector.

---

### Step 5: Training recipe (P1)

- **Batch:** 4–8 studies, each with 3 series × 12–16 slices.
  - Use AMP **fp16** on T4/P100 (bf16 needs Ampere or newer) and `channels_last`.
- **Optimizer and schedule:**
  - **AdamW** with lr 2e-4 and weight decay 1e-2.
  - 1 epoch of linear warmup, then **cosine** decay, for **10–15 epochs**.
  - **No freeze phase.**
- **Weight averaging:** keep an **EMA** of the weights (decay about 0.999, via `timm.utils.ModelEmaV3` or `torch.optim.swa_utils.AveragedModel`) and evaluate the EMA model.
- Keep the gradient clipping at 1.0.
- **Augmentation:** apply the same transform to every slice in a series.
  - Rotation ±15°, scale 0.9–1.1, shift ±8%.
  - Brightness and contrast ±0.2, gamma 0.8–1.2, light Gaussian noise.
  - Slice-window jitter.
  - **Series dropout** with p = 0.1: zero out a series and set `present = 0`.
  - **No horizontal flip.**
- **Use both T4s.** The simplest way is to train two folds in parallel, one per GPU. Otherwise use DDP via `torchrun` or `accelerate`.
- **Seed everything:** `random`, `numpy`, `torch`, and the DataLoader `worker_init_fn`.
- **Budget the GPU quota.** Kaggle's weekly quota (typically about 30 h; check your settings) is the real constraint.
  - Screen ideas with **proxy runs**: 1 fold, 25% of the data, 3 epochs.
  - Run the full 5-fold training only for winners.

---

### Step 6: Inference notebook and ensembling (P1)

- **Use a new, inference-only notebook.**
  - Attach the checkpoint dataset and the preprocessing code.
  - Decode the hidden test inside a `DataLoader(num_workers=4)` so CPU decoding overlaps with the GPU.
- **Ensemble:** start with 5 fold models, then grow to 2–3 backbones × 5 folds.
  - Average the sigmoid probabilities.
  - When mixing architectures, **rank-average per target** instead, since AUC only looks at rank.
- **Runtime guard:**
  - Time the first 50 studies and project to about 1,300.
  - If the projection is over about 7 h, cut slices (24 → 16), then cut models.
  - Write `submission.csv` periodically so a partial run still produces a file.
- **Measure before submitting:** time the 3 public test studies, multiply by about 430, and keep a 2× safety margin.
- **Fallback:** give any study that fails a constant prediction, so the CSV is always complete.
- **TTA:** only if CV shows it helps, for example small scale or shift crops. Skip flips.

---

### Step 7: Second-round boosts (P2)

- **Pseudo-label refinement (noisy student):**
  1. Set the new target to 0.7 × LLM label + 0.3 × the **OOF** ensemble prediction. Never use in-fold predictions.
  2. Retrain and check metric B.
- Upweight gold studies ×2–3 in the final training run.
- **AUC maximization:** after BCE training, fine-tune for 1–2 epochs with LibAUC's `AUCMLoss` and the `PESG` optimizer.
  - It optimizes the metric directly; this is the method behind the top CheXpert entry.
  - LibAUC is only needed for training, not inference.
- **Asymmetric Loss (ASL)** for the rare targets (MCL, Lateral OA, Baker's): test it against BCE.
- **Model soups / SWA** over the last checkpoints of a run.
- **Efficiency prize:** distill the ensemble into one small student model (EfficientNet-B0, 16 slices, 224 px), trained on the ensemble's OOF soft predictions.

---

### Step 8: Hygiene

- Delete the dead and duplicate code listed in 2.9, and the regex labeler.
- Put each piece in one place: config at the top (`SEED`, `N_SLICES`, `IMG_SIZE`, `BACKBONE`), then one `load_series`, one dataset class, one model and one train loop.
- Make `evaluate_model` respect `mask`, or compute AUC from the saved OOF arrays instead.
- Don't name anything `weights` next to torchvision's `Weights` objects.

---

## 4. Current-algorithm toolbox: what to use, how and when

| Technique | Use it for | How to apply it here | Priority |
|---|---|---|---|
| LLM report labeling with vLLM and JSON output (Qwen3 family, Gemma 3, MedGemma) | Turning 4,349 multilingual reports into 12 soft labels | Step 1b, validated on gold-58 | **P0** |
| Labeler ensembling / self-consistency | Less label noise | Average the soft targets of 2 or more labelers | P1 |
| 2.5D MIL with gated attention pooling (Ilse et al., 2018) | Study-level labels from slice stacks | Step 3 | **P0** |
| Pretrained slice encoders via timm: EfficientNet-B3, EfficientNetV2-S, ConvNeXt-T, DINOv2 / DINOv3 ViT | The per-slice encoder | Change `name=` and ensemble 2–3 *different* families | **P0** / P1 |
| Per-target query decoder (ML-Decoder, 2023; Query2Label, 2021) | Letting each finding attend to its own plane and slices | Replace the pooling and head (Step 3 upgrade) | P2 |
| Soft-label BCE with per-label weights | Noisy, graded labels | Step 1c | **P0** |
| Asymmetric Loss (Ridnik et al., 2021) | Rare positives | Test against BCE on MCL, Lateral OA and Baker's | P2 |
| AUC maximization with LibAUC AUCM + PESG (Yuan et al., 2021) | Optimizing the metric directly | 1–2 epoch fine-tune after BCE | P2 |
| EMA / SWA / model soups (Wortsman et al., 2022) | Free accuracy and stability | Always keep an EMA; soup the last checkpoints | P1 |
| Noisy Student / pseudo-label refinement (Xie et al., 2020) | Fixing labeler errors using image evidence | Step 7 | P2 |
| Optuna TPE on proxy runs | Hyperparameter search within the GPU quota | Section 5 | P1 |
| Rank averaging and greedy ensemble selection (Caruana et al., 2004) | Combining models for AUC | Step 6 and section 5 | P1 |
| Knowledge distillation | The Efficiency prize | Step 7 | P2 |
| fp16 AMP, `channels_last`, gradient checkpointing, 2-GPU training | More slices and more speed | Step 5 | P1 |

---

## 5. How to optimize systematically: finding out what actually works

1. **Build baseline v2** from Steps 1–4, then record A, B (with its confidence interval), the LB score and the runtime.
2. **Change one thing per experiment.** Log every run to `experiments.csv` with these columns: `id, date, change, A, B, B_CI, LB, train_min, infer_min`.
3. **Screen on proxy runs** (1 fold, 25% of the data, 3 epochs). Promote the top ideas to full 5-fold training.
4. **Run a hyperparameter search with Optuna (TPE)** on proxy runs.
   - Search over: lr, number of slices T, image size, pooling (attention / max / mean), dropout, the `not_mentioned` target and weight, and augmentation strength.
   - Objective: `0.7*A + 0.3*B`. This is the blend the 18th-place author used.
5. **Work through this ablation queue in order:**
   1. Labels: regex → public LLM labels → your own LLM labels → fused.
   2. Slices: 5 → 12 → 24 for training; 24 → 32 for inference.
   3. Resolution: 224 → 256 → 288 → 320.
   4. Pooling: max vs. attention vs. per-target queries.
   5. Backbone: EfficientNet-B3 vs. ConvNeXt-T vs. EfficientNetV2-S vs. DINOv2-S.
   6. Series: 3 vs. 4 (add T1).
   7. `not_mentioned` handling: target 0 vs. 0.05 vs. ignored; weight 0.3 / 0.5 / 1.0.
   8. Augmentation strength. Optionally ablate a full "mirror knee" (flip coronal and axial in-plane, and reverse the sagittal slice order).
   9. EMA on or off; 8 / 12 / 16 epochs.
6. **Select the ensemble** by greedy forward selection on metric A, then confirm with B. Prefer **diverse backbones** over more seeds of the same one.
7. **Pick the final two submissions (Oct 21–22):**
   - Submission 1: the best on A + B.
   - Submission 2: the best public-LB entry among those within noise of Submission 1.

---

## 6. Roadmap (Oct 4 → Oct 22, 2026)

| Dates | Goal | Done when |
|---|---|---|
| **Oct 4 (today)** | Accept the rules or form a team (entry deadline Oct 15). Attach the public LLM labels and score them against gold | A per-target AUC table, regex vs. LLM |
| Oct 5–6 | Preprocessing v2 (sort fix, 24 slices, 256 px, uint8, series mask, site tags), saved as a Kaggle Dataset. Start your own LLM labeling run | Cache dataset and `folds.csv` exist |
| Oct 7–9 | 2.5D MIL model with 5-fold CV (metrics A and B). Inference notebook. **First v2 submission** | An LB score from v2 |
| Oct 10–14 | Ablations from section 5 on proxy runs. Fuse the labelers. Try the 4th series | `experiments.csv` has 10 or more rows |
| Oct 15–18 | Pseudo-label round. Train the final 2–3 backbones × 5 folds | OOF predictions exist for every model |
| Oct 19–20 | Ensemble selection and a full runtime projection for about 1,300 studies | Ensemble submitted |
| Oct 21–22 | Choose the 2 final submissions. **No new experiments** | Done |

---

## 7. Sources

- [Kaggle competition page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
- [RSNA Knee MRI AI Challenge (2026)](https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge)
- [18th-place pipeline write-up (Hugging Face blog)](https://huggingface.co/blog/bishnoiyash/rsna-competetion): test size, 9 h limit, LLM labels, MIL, gold-58 leak warning, no-flip policy
- [TranBaDat2607 solution repo](https://github.com/TranBaDat2607/RSNA-Knee-Abnormality-Detection): DINOv2 with per-diagnosis attention, 0.939 public LB
- [Sibusiso-K repo](https://github.com/Sibusiso-K/RSNA-Knee-Abnormality-Detection): deadlines, site-grouped CV note
- [AuntMinnie: RSNA launches 2026 knee challenge](https://www.auntminnie.com/imaging-informatics/artificial-intelligence/news/15831808/rsna-launches-2026-knee-abnormality-detection-ai-challenge)
