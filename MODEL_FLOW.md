# How the model works: a study guide

This guide follows the system from raw data to `submission.csv` and explains why each step exists.
The last sections are for interview or viva preparation: likely questions with answers, numbers to remember, and how to pitch the project.

Contents:
1. [The task in one page](#1-the-task-in-one-page)
2. [Medical background (just enough)](#2-medical-background-just-enough)
3. [End-to-end flow](#3-end-to-end-flow)
4. [Labels: from reports to soft targets](#4-labels-from-reports-to-soft-targets)
5. [Preprocessing: from DICOM to a 96-slice volume](#5-preprocessing-from-dicom-to-a-96-slice-volume)
6. [The model: 2.5D attention-MIL](#6-the-model-25d-attention-mil)
7. [Training recipe](#7-training-recipe)
8. [Label denoising: teacher → student](#8-label-denoising-teacher--student)
9. [Inference and the ensemble](#9-inference-and-the-ensemble)
10. [Running inside 9 hours](#10-running-inside-9-hours)
11. [How models are evaluated](#11-how-models-are-evaluated)
12. [Interview questions and answers](#12-interview-questions-and-answers)
13. [Numbers to remember](#13-numbers-to-remember)
14. [How to talk about the project](#14-how-to-talk-about-the-project)
15. [Glossary](#15-glossary)

---

## 1. The task in one page

- **Input:** one knee MRI *study*: a folder of DICOM files grouped into about 5.5 *series*. Each series is a stack of 2D slices in one plane (sagittal, coronal or axial) with one MRI sequence.
- **Output:** 12 probabilities, one per finding:

  | Group | Findings |
  |---|---|
  | Ligaments | ACL, MCL |
  | Menisci | Medial meniscus, Lateral meniscus |
  | Osteoarthritis | Medial OA, Lateral OA, Patellofemoral (PF) OA |
  | Joint | Effusion, Synovitis, Baker's cyst |
  | Bone | Contusion, Fracture |

- **Supervision:**
  - 4,349 training studies come with only a free-text radiology report, multilingual.
  - 58 studies have official gold labels.
  - The ~1,300 test studies are images only, with no reports.
- **Metric:** macro ROC-AUC, the mean of 12 per-finding AUCs.
  - AUC measures *ranking*: the probability that a random positive study scores higher than a random negative one.
  - Calibration is irrelevant, and every finding counts equally. A rare, hard finding like MCL matters as much as effusion.
- **Constraints:**
  - Inference runs in a Kaggle notebook: offline, T4 GPU, 9 hours maximum.
  - 5 submissions a day.
  - Training also runs on Kaggle, within a weekly GPU quota of about 30 h.

**The core difficulty** is weak supervision. Report text is an imperfect proxy for what is visible on the images, and the gold labels come from reading the images (Section 8).

---

## 2. Medical background (just enough)

| Plane | Cut | Best for |
|---|---|---|
| Sagittal | Side view: slices run left↔right | ACL, menisci, effusion (suprapatellar recess), patellofemoral joint |
| Coronal | Front view | MCL, menisci, medial/lateral compartments (OA) |
| Axial | Top-down view | Patellofemoral joint, Baker's cyst, effusion |

| Sequence | Appearance | Shows |
|---|---|---|
| Fluid-sensitive (PD / T2 fat-sat) | Fluid and oedema are bright | Effusion, bone contusion (marrow oedema), tears |
| Other (T1, PD without fat-sat) | Anatomy is clear | Fracture lines, osteophytes |

| Finding | What it is |
|---|---|
| ACL / MCL tear | Torn anterior cruciate / medial collateral ligament |
| Meniscal tear | Tear in the C-shaped cartilage pads between femur and tibia |
| OA (medial / lateral / PF) | Cartilage loss and osteophytes in one compartment of the knee |
| Effusion | Excess fluid in the joint |
| Synovitis | Inflamed joint lining; hard to see without contrast |
| Baker's cyst | A fluid-filled popliteal cyst behind the knee |
| Contusion | Bone bruise (marrow oedema) |
| Fracture | Break in the bone |

**Why this matters for the model:**
- Different findings live on different planes and slices. The model must see several series and pick the relevant slices itself, which is the job of MIL (Section 6).
- Left/right mirroring has to be done per plane (Section 7).

---

## 3. End-to-end flow

```mermaid
flowchart TD
    subgraph Labels
        R[Reports, multilingual] --> P[Public LLM label sets]
        P -->|score each on gold-58| V2[Labels v2 = mean of best two<br/>gold 0.903]
    end
    subgraph Images
        D[DICOM study] --> FR[fastread: choose series per plane/sequence,<br/>96 slices, 384 px, central 320 crop]
        FR --> NPZ[(Raptor-96 corpus<br/>uint8 volume + mask)]
    end
    V2 --> TEA[4-fold teacher<br/>ConvNeXt-small]
    NPZ --> TEA
    TEA --> OOF[Out-of-fold predictions<br/>for every training study]
    OOF --> MIX[Quantile-map to label scale,<br/>0.5 report + 0.5 teacher]
    V2 --> MIX
    MIX -->|assert gold ≥ 0.92| V3[Labels v3, gold 0.927]
    V3 --> STU[Students: ConvNeXt-small x2, ConvNeXt-base]
    NPZ --> STU
    subgraph Kaggle inference notebook, 9 h
        RA[Stage 1: Raptor CoAtNet 384 + 224<br/>public, ~15-20 min] --> BL
        R7[Stage 2: R7 ConvNeXt-small<br/>public, CPU decode, deadline 8.3 h] --> BL
        MEM[Stage 2b: my members / students<br/>deadline 8.75 h, failures caught] --> BL
        BL[Stage 3: per-finding percentile-rank blend]
    end
    STU --> MEM
    BL --> SUB[submission.csv]
```

---

## 4. Labels: from reports to soft targets

**Step 1: why not regex?** The original notebook parsed reports with English regexes. It scored 0.42–0.58 AUC against gold on several findings:
- most non-English reports produced nothing;
- negation leaked across sentences;
- and unmentioned findings became "missing".

**Step 2: LLM label sets.** Several public datasets used large language models to read every report and output a probability per finding. I scored each set against the 58 gold studies:

| Label set | Gold macro AUC |
|---|---|
| JEV hybrid | 0.903 |
| flight0234 v4hybrid | 0.899 |
| stevenleehans | 0.893 |
| pilkwang | 0.867 |

I excluded sets that contained the gold labels themselves: they would leak, and they can't be evaluated.

**Step 3: labels v2.** The element-wise mean of the two best sets, which gives soft targets in [0, 1] and scores 0.903 on gold.

**Soft targets** keep the extractor's uncertainty. A report saying "possible small tear" can be 0.5 instead of being forced to 0 or 1. BCE loss works directly with soft targets.

**The ceiling:** every report-derived set landed near 0.90. Section 8 explains why, and how I broke through it.

---

## 5. Preprocessing: from DICOM to a 96-slice volume

The same code runs at training and test time. That is deliberate: if training inputs differ from test inputs even slightly, accuracy drops in ways no validation set reveals.

1. **Read headers** of every DICOM file (fast, multithreaded). Classify each series by plane and by fluid sensitivity.
2. **Pick slices into 5 fixed slots** of a 96-slice volume:

   | Slot | Slices |
   |---|---|
   | Sagittal fluid-sensitive | 26 |
   | Sagittal other | 22 |
   | Coronal fluid-sensitive | 18 |
   | Coronal other | 12 |
   | Axial | 18 |

   Slices are ordered along the series' normal vector. Sorting by z would scramble sagittal and coronal stacks.
3. **Resample** to 384 px and take a **central 320 px crop** covering about 140 mm, which is where the knee joint sits.
4. **Store** as `uint8` (`vol`: 96 × 320 × 320) plus a `mask` marking real slices, since a study may lack a series. This is one `.npz` per study, the "Raptor-96 corpus".

The training corpus was built by 3 CPU Kaggle kernels in parallel, so it used no GPU quota.

---

## 6. The model: 2.5D attention-MIL

### 6.1 Windows: making 3D data look like RGB images
A **window** is 3 adjacent slices `[s-1, s, s+1]` stacked as the 3 channels of an image. That gives:
- a 2D ImageNet-pretrained backbone (expects 3 channels) with no modification;
- a little 3D context: the model sees thickness and continuity across neighbouring slices.

This is the "2.5D" in the name: between 2D (single slices) and 3D (volumetric convolutions, heavier and with weaker pretraining).

### 6.2 Architecture

```
study ──► K windows (K × 3 × 320 × 320)
             │ ConvNeXt backbone, shared, applied to each window
             ▼
          K feature vectors h_1..h_K   (768-d for ConvNeXt-small, 1024-d for base)
             │ LayerNorm
             ▼
   attention scores  e_k = W2 · tanh(W1 · h_k)        → shape K × 12 (one score per finding)
   attention weights a_{k,n} = softmax over k of e_{k,n}
   pooled feature    z_n = Σ_k a_{k,n} · h_k           → one 768-d vector per finding
   logit             y_n = w_n · z_n + b_n             → 12 logits → sigmoid → probabilities
```

**Multiple-instance learning (MIL):** the study is a *bag* of windows with one label per bag. We don't know which windows show the tear. Attention learns to weight the informative ones.

**Per-finding attention** is the key design choice. Each finding has its own attention distribution. ACL attention can focus on mid-sagittal windows while Baker's cyst attention focuses on posterior axial windows, all from the same features.

**Order invariance:** the pooling is a weighted sum, so window order doesn't matter. That makes random window sampling during training and mirroring safe.

The head design (`RaptorClassifier`) comes from the public Raptor pipeline. I kept the exact weight layout so my checkpoints load into the existing inference stage. Only the backbone changes:
- public Raptor: CoAtNet;
- mine: ConvNeXt-small / ConvNeXt-base, from `timm`, ImageNet-22k pretrained.

---

## 7. Training recipe

Source: `kaggle_jobs/members-a/train_member.py`.

| Item | Setting | Why |
|---|---|---|
| Windows per study | 12 random centres inside the masked range (train); 24 evenly spaced (per-epoch eval, OOF); 94 + mirror (final gold eval) | Random sampling is augmentation; more windows at test time cover the whole volume |
| Batch | 4 studies = 48 window images per step | Fits 16 GB T4 memory in fp16 |
| Augmentation | Per study: rotation ±10°, scale ±10%, small shift; brightness gain ±15%, bias ±0.05; left/right mirror with p = 0.5 | Same transform for all windows of a study, so the anatomy stays consistent |
| Mirror | Sagittal windows reverse their 3 slices; coronal/axial windows flip width | In a sagittal stack, left↔right runs *across* slices. Flipping a sagittal image would swap front and back, which never happens |
| Loss | `BCEWithLogitsLoss` on soft targets, `pos_weight = clip((1-p)/p, 1, 10)` per finding, p = mean label | Rare findings get more weight in the loss |
| Optimiser | AdamW, weight decay 0.05; LR 1e-4 backbone, 1e-3 head | The pretrained backbone needs gentle updates; the new head learns fast |
| Schedule | 300 warmup steps, then cosine down to 1% | Standard and stable for fine-tuning |
| Regularisation | Drop path 0.2–0.3, attention dropout 0.2, gradient clipping at 3.0 | |
| Precision | fp16 autocast + GradScaler | ~2× speed on T4 tensor cores |
| Speed | `channels_last`, `torch.compile(backbone)` (0.74 → 0.60 s/step) | |
| Time budget | Times 60 steps, then sets epochs = min(9, budget / epoch time) | Kaggle sessions and quota are limited |
| SWA | Average weights of the last 3 epochs | Smoother, more robust model; no need to pick a "best epoch" on 58 noisy studies |
| OOM | On the first out-of-memory error, turn on gradient checkpointing and continue | ConvNeXt-base is close to the memory limit |
| Validation | Gold-58 macro AUC every epoch | Monitoring only, never used to pick the model |

**Parallelism:** each Kaggle session has 2 T4s. I run two independent trainings, one per GPU, not `nn.DataParallel`. DataParallel leaked about 30 MB of host RAM per step on this setup, which killed the run, and it was 2× slower.

**Smoke test first:** every launch runs 80 studies × 2 epochs before the real run (`smoke && full`), so a bug costs minutes, not hours.

---

## 8. Label denoising: teacher → student

### 8.1 The discovery
Comparing report labels with gold on the 58 studies showed that **gold is an independent read of the images, not of the reports.**

| Finding | Gold prevalence | Report prevalence |
|---|---|---|
| Synovitis | 0.47 | 0.12 |
| Fracture | 0.31 | 0.07 |
| Lateral meniscus | 0.40 | 0.15 |

- Many disagreements are report-positive/gold-negative, for example 15 for effusion.
- Severity wording doesn't explain the gap.

**Conclusion:** better text extraction can't fix this. The information has to come from the images.

### 8.2 The evidence that mixing works
A 50/50 rank mix of the report labels and an image model's honest predictions agreed with gold at **0.928**. The labels alone scored 0.903 and the model alone 0.910. The 90% bootstrap CI of the gain was [+0.014, +0.039].

The report and the image model make *different* errors, so averaging cancels part of each.

### 8.3 The pipeline (`teacher-a/`, `make_targets.py`)
1. **Folds.** `fold = md5(StudyInstanceUID) mod 4`. The split is deterministic and gold is excluded from every fold.
2. **Teacher.** For each fold f, train a ConvNeXt-small on the other 3 folds (labels v2). Predict fold f with 24 windows + mirror. These are the **out-of-fold (OOF)** predictions. Every training study gets a prediction from a model that never saw it.
3. **Map to the label scale.** Raw sigmoid outputs and soft labels have different distributions. For each finding, *quantile matching* replaces a teacher score with the report-label value at the same percentile:
   `Q(o) = quantile(report labels, percentile of o among all OOF scores)`.
   Isotonic regression was the alternative. I kept the map with higher gold agreement: quantile, 0.9272 vs 0.9249.
4. **Mix.** `target = 0.5 · report + 0.5 · Q(teacher OOF)`. B = 0.5 was fixed in advance.
5. **Gate.** Apply the same recipe to the gold studies, using the 4 teacher folds' gold predictions. Refuse to write the file unless macro AUC ≥ 0.92. The result was **0.927**.
6. **Students.** Train new models on labels v3, with gold still held out.

**Why OOF matters:** a model's predictions on its own training data copy the training labels, noise included, so mixing those in changes nothing. Only held-out predictions carry independent information.

**Why the per-finding mean is unchanged:** quantile mapping preserves the label distribution. The loss's `pos_weight` and the prior stay the same, and only *which* studies are positive shifts.

**Per-finding effect (gold AUC, report → v3):**

| Finding | Report | v3 |
|---|---|---|
| Effusion | .880 | .964 |
| Fracture | .901 | .942 |
| Contusion | .868 | .920 |
| Baker's | .920 | .967 |
| Lateral meniscus | .871 | .898 |
| Lateral OA | .855 | .836 (the one finding that didn't improve) |

---

## 9. Inference and the ensemble

### 9.1 The models in the final notebook

| Model | Author | Input | Notes |
|---|---|---|---|
| Raptor CoAtNet-384 | public: nartaa / dreaddevelopment | Raptor-96 volume, 94 windows + mirror | 0.949 alone on LB |
| Raptor CoAtNet-224 | public: nartaa | 0.8 centre crop resized to 224 | 0.945 alone |
| R7 ConvNeXt-small | public: heliosli | Knee-centred local crops from native DICOM; 4 folds; extra stackers for effusion and Baker's | CPU-heavy decode, ~40 s/study |
| My ConvNeXt members / students | mine | Raptor-96 volume, 24 windows + mirror | 0.90–0.91 on gold, honest |

**Test-time augmentation (TTA):** each model predicts the original and the left/right-mirrored study, and the probabilities are averaged.

### 9.2 Rank blending
Each model's outputs are converted, per finding, to **percentile ranks** over the test studies (0 = lowest, 1 = highest), then averaged:

```
raptor_half = 0.65 · rank(Raptor-384) + 0.35 · rank(Raptor-224)
base        = 0.5 · raptor_half + 0.5 · rank(R7)
final       = (1 − W) · base + W · mean_rank(my models)      W = 0.15 (or 0.25), fixed before the run
```

**Why ranks:**
- AUC depends only on order, and ranks preserve order.
- Models are calibrated differently. One may output 0.3 where another outputs 0.8 for the same confidence. Ranks put every model on the same uniform scale, so the weights mean what they say.
- Ranks are robust to outliers.

**Failure handling:**
- A study a model failed on (all outputs 0.5) is excluded from that model's ranking and takes the other models' rank.
- If too many studies fail (>3% for Raptor, >50% for R7), the notebook stops rather than silently submitting a degraded blend.

---

## 10. Running inside 9 hours

| Stage | Runs as | Time | Guard |
|---|---|---|---|
| 1. Raptor-384 | subprocess | ~15–20 min | Must succeed (else stop) |
| 1b. Raptor-224 | subprocess | minutes | Must succeed |
| 2. R7 | subprocess, 2 CPU decode workers | up to 8.3 h from start | Deadline; unscored studies fall back to Raptor |
| 2b. My models | subprocess, after R7 | until 8.75 h from start | Per-study deadline, exceptions caught → base submission kept |
| 3. Blend | in notebook | seconds | Receipt JSON logs weights and how many studies each model scored |

**Each stage is a separate process,** so its GPU memory is fully released before the next one.

**Lesson (Decision D17 in DECISIONS.md):** when my model ran *before* R7, it ate R7's time budget. That may be part of why those submissions scored lower. Extra models now run *after* R7.

---

## 11. How models are evaluated

- **Gold-58, held out:**
  - It is the only clean labelled set. Every model is scored on it with the inference-time setting (94 windows + mirror).
  - It is honest because no gold study is trained on and nothing is selected on it beyond one documented choice (quantile vs isotonic).
- **Its limits:**
  - 58 studies, with 9–35 positives per finding.
  - Seed-to-seed noise is about ±0.006, so it can't separate models closer than about 0.01.
  - A member scoring 0.9075 on gold still lowered the LB.
- **Bootstrap confidence intervals:** resample the 58 studies with replacement many times and recompute the metric. Used to check that the denoising gain (+0.026) is real.
- **Public leaderboard:**
  - The final judge, but only 5 submissions a day and 3 decimals.
  - A change is kept only if it moves the LB by at least 0.001.
  - Weights are never tuned to it, because the public and private splits differ.
- **Diagnostic submissions:** for example "my models alone" (c3), to learn whether they are weak or merely redundant with the base.

---

## 12. Interview questions and answers

**Q: Why macro ROC-AUC, and what does it imply for modelling?**
AUC measures ranking quality and ignores calibration and thresholds, so I never calibrate. Macro averaging weights each finding equally, so rare findings matter as much as common ones. That is why I use `pos_weight` in the loss and why I care about MCL and lateral OA, not just overall accuracy.

**Q: Why not plain accuracy?**
With imbalanced findings, a model that always says "no" gets high accuracy and is useless. AUC is threshold-free and insensitive to prevalence.

**Q: What is multiple-instance learning, and why use it here?**
Labels are per study, but evidence lives in a few slices. MIL treats the study as a bag of windows and learns attention weights that pick the informative ones. Per-finding attention lets each abnormality look at different slices.

**Q: Why 2.5D instead of a 3D CNN?**
2D backbones have strong ImageNet pretraining and are cheaper. Slice spacing is anisotropic (thick slices), which suits 3D convolutions poorly. Three-slice windows add local 3D context at almost no cost.

**Q: How did you handle noisy labels?**
First I measured the noise against gold. That showed it is systematic: reports under-call findings visible on images. Then I used teacher → student denoising: out-of-fold predictions from a 4-fold teacher, quantile-mapped and mixed 50/50 with report labels. Gold agreement rose from 0.903 to 0.927 without any gold study in training.

**Q: Why out-of-fold predictions?**
In-sample predictions memorise the noisy labels, so mixing them in adds nothing. OOF predictions come from a model that never saw the study, so they carry independent evidence.

**Q: Isn't mixing in model predictions circular?**
It would be with in-sample predictions. With OOF predictions, the image model and the report make different errors, so the mix is closer to the truth. Gold confirmed this, with a bootstrap CI excluding zero. It is related to knowledge distillation and to "bootstrapping" losses for noisy labels.

**Q: What is SWA and why use it?**
Stochastic weight averaging: average the weights from the last few epochs. It finds a flatter minimum that generalises better, and it avoids choosing a best epoch on a tiny, noisy validation set.

**Q: Why rank blending instead of averaging probabilities?**
Models are calibrated differently, and AUC only cares about order. Percentile ranks put every model on a common scale and are robust to outliers.

**Q: Why did a model with good validation hurt the leaderboard?**
Two reasons. First, the gold set is small (±0.006 noise). Second, a model that is good but similar to the ensemble adds little diversity. Also, in that run, my model executed before R7 and took its time budget. The fix was to run it after R7, plus diagnostic submissions to separate the causes.

**Q: How did you prevent leakage?**
- Gold studies are never trained on.
- Folds are hash-based and asserted disjoint.
- OOF predictions are used for denoising.
- Label sets that copied gold were excluded.
- Blend weights were fixed before runs.

**Q: How did you speed up training?**
- fp16 autocast, `channels_last`, and `torch.compile`, measured by an A/B smoke test on two GPUs (0.74 → 0.60 s/step).
- One process per GPU instead of `DataParallel`, which leaked memory and was 2× slower.
- Epochs fitted to a time budget, and smoke tests to fail fast.

**Q: How do you mirror an MRI correctly?**
Per plane. In coronal and axial images, left/right is the image width, so flip the width. In a sagittal stack, left/right runs across slices, so reverse the slice order. Flipping a sagittal image would swap anterior and posterior.

**Q: Why preprocess training data with the inference reader?**
Train/test parity. Any difference in slice choice, crop or intensity mapping between training and test silently hurts accuracy. Using one code path removes that risk.

**Q: What would you do with more time or compute?**
- More students on denoised labels with different inputs: knee-centred compartment crops gave a public team +0.001 through diversity.
- A second denoising round, with students acting as the next teachers.
- Per-finding mix weights.
- Group folds by site, since site shift inflates random-fold CV.

**Q: What were the main limitations?**
- The gold set is tiny.
- The public LB has 3 decimals and 5 submissions a day.
- The weekly GPU quota limited the number of students.
- The strongest models in the ensemble are public; my gains are on top of them.

---

## 13. Numbers to remember

| Fact | Value |
|---|---|
| Teams | 5,957 |
| Training studies / gold / test | 4,407 (4,349 report-only + 58 gold) / ~1,300 |
| Findings | 12; metric macro ROC-AUC |
| Original notebook | 0.579 AUC on 12 studies |
| Labels v2 vs gold | 0.903 |
| Labels v3 (denoised) vs gold | 0.927 (+0.024) |
| Mixing evidence | 0.928, 90% CI of gain [0.014, 0.039] |
| My ConvNeXt-small / base (gold) | 0.9075 / 0.9090 |
| Teacher folds (gold) | 0.883–0.887 |
| Public LB (mine / top) | 0.956 / 0.966 |
| Volume | 96 slices × 320 × 320, 5 slots (26/22/18/12/18) |
| Windows | 12 train, 24 eval, 94 inference |
| Step time | 0.74 → 0.60 s/step with torch.compile |
| DataParallel leak | +30 MB host RAM per step |
| Blend | R7 0.5 + Raptor 0.5 (0.65 × 384 + 0.35 × 224), mine at 0.15 |
| Inference deadlines | R7 8.3 h, my models 8.75 h, Kaggle limit 9 h |

---

## 14. How to talk about the project

**30-second pitch:**
"I worked on knee MRI classification with 12 abnormalities, where almost all labels came from multilingual radiology reports. By scoring labels against a small gold set, I found that the reports systematically miss findings that are visible on the images. So I built a teacher–student pipeline that mixes report labels with out-of-fold image-model predictions. That raised label agreement with gold from 0.903 to 0.927 AUC. The models are 2.5D ConvNeXt attention-MIL networks trained on Kaggle GPUs and ensembled by rank blending in a 9-hour offline pipeline, reaching 0.956 macro AUC on the public leaderboard."

**Resume bullets** (update the last one with the final rank):
- Built a teacher–student label-denoising pipeline for weakly supervised multi-label knee MRI classification. Mixing LLM-extracted report labels with 4-fold out-of-fold predictions raised agreement with gold labels from 0.903 to 0.927 macro AUC.
- Trained 2.5D attention-MIL ConvNeXt models (PyTorch, timm) on 4,349 MRI studies with soft-label BCE, SWA and fp16. Cut step time 19% with torch.compile, and diagnosed a host-memory leak in `nn.DataParallel`, replacing it with one process per GPU.
- Engineered a 9-hour offline inference pipeline that ensembles 5+ models by per-finding rank blending, with deadline-aware stages and graceful fallbacks. Public leaderboard 0.956 macro AUC (RSNA Knee Abnormality Detection, Kaggle, 5,957 teams).

**Be ready to say honestly:** the strongest base models (Raptor, R7) are public. My contributions are the label analysis and denoising, my own trained members, the training and inference engineering, and the experiment discipline.

---

## 15. Glossary

| Term | Meaning |
|---|---|
| **AUC (ROC-AUC)** | Probability that a random positive is scored above a random negative; 0.5 = random, 1.0 = perfect |
| **Macro AUC** | Unweighted mean of per-class AUCs |
| **Weak / soft labels** | Noisy labels from an indirect source (reports), as probabilities in [0, 1] |
| **Gold labels** | The 58 officially labelled studies; the only clean ground truth |
| **MIL** | Multiple-instance learning: one label for a bag of instances (windows) |
| **Attention pooling** | A learned weighted average over instances |
| **2.5D** | 2D CNN on stacks of a few adjacent slices |
| **OOF** | Out-of-fold: predictions on data a model was not trained on |
| **Quantile matching** | Mapping values so their distribution matches a target distribution, rank for rank |
| **Isotonic regression** | Fitting a monotone non-decreasing function |
| **SWA** | Stochastic weight averaging |
| **TTA** | Test-time augmentation (here: left/right mirror) |
| **pos_weight** | Extra loss weight on positive examples of rare classes |
| **Rank blending** | Averaging percentile ranks instead of raw scores |
| **Drop path** | Randomly skipping residual blocks during training (regularisation) |
| **AMP / fp16** | Mixed-precision training: faster and less memory |
| **channels_last** | Memory layout (N, H, W, C) that is faster for convolutions on tensor cores |
| **torch.compile** | PyTorch's JIT compiler that fuses operations for speed |
| **DICOM** | The standard file format for medical images, with headers (position, orientation, sequence) |
| **Sagittal / coronal / axial** | Side / front / top-down image planes |
| **Fat-sat / fluid-sensitive** | MRI sequences where fluid and oedema appear bright |
