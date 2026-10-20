# Decision log

This file records the decisions made while building this project: what was decided, what else was considered, why, and what happened.
Entries are grouped by phase and roughly in date order (October 2026).
"Gold" means the 58 training studies with official labels. "LB" means the public leaderboard.

---

## Phase 1: auditing the original notebook (4 October)

### D1. Fix the labels before touching the model
- **Context:** the first notebook scored 0.579 AUC on 12 validation studies.
- **Finding:** its regex report parser scored 0.42–0.58 AUC against gold on several findings.
  - The patterns were English-only, but the reports are multilingual.
  - Negation leaked across sentences: "No fracture. Moderate effusion." negated the effusion.
  - Unmentioned findings became NaN, so most training steps had no labels at all.
- **Decision:** replace the regex labels with multilingual LLM-extracted labels, scored against gold.
- **Why:** a CNN can't learn from labels that are close to random. Labels were the biggest lever.

### D2. Sort slices along the slice normal, not by z
- **Context:** slices were sorted by `ImagePositionPatient[2]` (z).
- **Problem:** z is the stacking axis only for axial series. Sagittal and coronal stacks came out in arbitrary order, so "uniform slice sampling" picked random slices.
- **Decision:** order slices by their position along the series' normal vector.

### D3. A 2.5D multiple-instance model instead of a 5-channel "conv hack"
- **Context:** the old model fed 5 slices as 5 channels of one image, with fake filters for channels 4–5.
- **Decision:** encode each slice window separately with a pretrained 2D CNN and pool with attention (multiple-instance learning, MIL).
- **Why:**
  - Pathology like an ACL tear is visible on only a few slices. MIL lets the model find them.
  - A 2D ImageNet backbone keeps its pretrained weights intact.

### D4. Validate on a cross-fitted gold set, not a 12-study split
- **Problem:** with 1–3 positives per finding, one study moved the AUC by 0.1–0.3, and "best epoch" was noise.
- **Decision:** in the rebuilt notebook, cross-fit the 58 gold studies: each fold is validated on its share of them. From Phase 3 on, all 58 are held out of training entirely and form the validation set.

### D5. No horizontal flip on sagittal images
- **Problem:** flipping a sagittal image swaps anterior and posterior, which puts the patella behind the femur. That never occurs at test time.
- **Decision:** mirror left/right correctly per plane:
  - Sagittal windows reverse their slice order, because left/right runs *across* sagittal slices.
  - Coronal and axial windows flip their width.

---

## Phase 2: optimising my rebuilt notebook (9 October)

### D6. Cache everything expensive, and make reruns cheap
- **Problems:**
  - Each Kaggle session rebuilt a 13 GB preprocessing cache (~47–80 min).
  - It recomputed ~55k NLI model calls.
  - It audited the cache by decompressing every file twice.
- **Decision:**
  - Reuse an attached cache.
  - Store NLI scores on disk, verified bit-exact.
  - Make the manifest header-only.
- **Outcome:** a warm restart went from ~6–7 min to 12 s.

### D7. Never benchmark on the local GPU
- **Context:** the laptop's RTX 4060 was pinned at 210 MHz by a software power cap, about 200× below spec.
- **Decision:** all training and timing runs on Kaggle T4s. Local work is limited to CPU tasks: label analysis, target building, notebook generation.

### D8. Don't upload large files from local
- **Context:** the upload link was about 60 kB/s.
- **Decision:** build every large artefact (corpus, checkpoints) inside Kaggle and only upload small files. The labels parquet is 0.5 MB.

---

## Phase 3: building on the strongest public pipeline (14 October)

### D9. Pivot: keep the public 0.956 blend as the base and add my own models on top
- **Context:**
  - My rebuilt notebook reached 0.855 on 12 gold studies, far below the public 0.956 blend.
  - Eight days remained before the deadline.
- **Options:**
  - (a) Keep improving my model from scratch.
  - (b) Reproduce the best public blend, then add genuinely different models and better labels.
- **Decision:** (b). Reproduce first, so every later change is measured against a known score.
- **Outcome:** my reproduction scored 0.956 on the LB, matching the public forks.

### D10. Labels v2 = mean of the two best public label sets
- **What I did:** scored the public LLM label sets against gold:

  | Label set | Gold macro AUC |
  |---|---|
  | JEV hybrid | 0.903 |
  | flight0234 v4hybrid | 0.899 |
  | stevenleehans v4_blend | 0.893 |
  | pilkwang | 0.867 |

- **Excluded:** sets that copy the gold labels verbatim. They can't be evaluated honestly and would leak.
- **Decision:** labels v2 = the mean of v4hybrid and JEV, which scores 0.903.
- **Skipped:** my own Qwen relabelling of the reports. Every report-derived set capped near 0.90, which suggested that the reports themselves were the ceiling (confirmed in D23).

### D11. Preprocess the training set with exactly the inference reader
- **Decision:** run the public Raptor "fastread" preprocessing on all training studies:
  - 96 slices in 5 slots: sagittal fluid 26, sagittal other 22, coronal fluid 18, coronal other 12, axial 18.
  - 384 px, central 320 crop.
- **How:** as 3 CPU Kaggle kernels (`raptor96-corpus-s0/s1/s2`), so it costs no GPU quota.
- **Why:** train/test parity. My models then see exactly the tensors the inference stage produces, and they load as extra "arms" of that stage with no new inference code.

### D12. Fine-tuning the public Raptor checkpoint was a dead end
- **Tried:** fine-tune the public CoAtNet on labels v2 with a low backbone LR (1e-5), 3 epochs, SWA.
- **Result:** gold fell from 0.9186 to 0.911 (seed 7) and 0.908 (seed 11).
- **Explanation:** the starting 0.9186 was inflated, because the public checkpoint was selected on gold.
- **Decision:** not blended. Train new members from ImageNet weights instead, so they are honestly held out and more diverse.

### D13. One process per GPU; never `nn.DataParallel`
- **Diagnosis:** a dedicated leak-test kernel showed `DataParallel` on Kaggle T4×2 (torch 2.11):
  - leaked about +30 MB of host RAM per step;
  - was killed after ~1.5 epochs;
  - and was 2× slower than a single T4.
- **Ruled out:** pinned memory and data workers as the cause.
- **Decision:** run two independent trainings, one per GPU (`CUDA_VISIBLE_DEVICES`). Each session then trains two models in parallel.

### D14. Members from ImageNet, in the Raptor checkpoint format
- **Decision:** train new backbones (ConvNeXt-small/base, EfficientNetV2-S) with the same MIL head and checkpoint layout as Raptor, keeping gold held out.
- **Why:** they drop into the existing inference stage, and a different backbone adds diversity.
- **Outcome:**

  | Model | Gold |
  |---|---|
  | ConvNeXt-small | 0.9075 |
  | ConvNeXt-base | 0.9090 |
  | ConvNeXt-small, seed 7 | 0.9017 |
  | EfficientNetV2-S | 0.8972 |

- **Dropped:** EfficientNetV2-S. It was weaker and 0.935-correlated with ConvNeXt, so it added nothing.

### D15. Time-budgeted epochs and SWA
- **Decision:** the trainer times its first 60 steps, then fits the epoch count and cosine-schedule length to a `HOURS` budget. It saves the average of the last 3 epochs' weights (SWA).
- **Why:**
  - Kaggle sessions have hard limits and a weekly GPU quota.
  - SWA is a free, robust gain that removes the need to pick a "best epoch" on a noisy 58-study set.

### D16. Training kernels never assert at the end
- **Decision:** a failed run only prints its error.
- **Why:** an errored Kaggle kernel can't be attached as a source, so one failed model would make the other GPU's good model unusable.

---

## Phase 4: learning from the leaderboard (14–15 October)

### D17. My members inside the Raptor stage hurt the LB
- **Tried:** blending ConvNeXt-small into the Raptor half at 25% and at 33%.
- **Result:** LB 0.954 and 0.953, against 0.956 for the base.
- **Two explanations:**
  - The member is similar in kind to Raptor (same input), so it adds little diversity.
  - Running it inside the Raptor stage, before R7, took time from R7. R7 decodes about 40 s per study on CPU under an 8.3 h deadline, so studies it doesn't reach lose its score.
- **Lesson:** a model scoring 0.9075 on gold can still lower the LB. Gold can't gate members on its own.

### D18. Run extra models after R7, with their own deadline
- **Decision:** move my members to a separate stage that starts after R7 finishes.
  - It is capped at notebook start + 8.75 h, with a per-study deadline 3 minutes earlier.
  - Unscored studies keep the base blend's value.
  - Any exception is caught and logged.
- **Why:** my models can never cost R7 time, and a failure can never break the submission.
- **File:** `kaggle_jobs/patch_957.py`.

### D19. Patch the public notebook with exact-match substitutions
- **Decision:** generate each submission notebook with a script. Every replacement must match its anchor text exactly once, or the script fails. Every code cell is compile-checked before writing.
- **Why:** hand-editing 600-line cells invites silent mistakes. Any upstream change makes the patch fail loudly instead.

### D20. Blend weights are fixed before each run
- **Decision:** use pre-chosen weights (members at 0.15 or 0.25) and keep a change only if it moves the LB by at least 0.001.
- **Why:**
  - 5 submissions a day and 3-decimal scores make LB tuning noisy.
  - The public/private split has different prevalences, so tuning to the public LB risks overfitting.

### D21. Rank blending, per finding
- **Decision:** convert every model's scores to percentile ranks per finding before averaging.
- **Why:**
  - AUC depends only on ordering.
  - Models are calibrated differently, and ranks put them all on the same uniform scale.
  - A failed study is excluded from that model's ranking and takes the other model's rank.

### D22. Diagnostic submissions
- **Decision:** spend some submissions on measurements rather than candidates, for example "my 3 ConvNeXts alone" (c3).
- **Why:** it separates "my models are weak" from "my models are strong but redundant with the base". That decides whether to invest in more members or in different inputs.

---

## Phase 5: label denoising (15 October)

### D23. The gold labels are an independent image read
- **Evidence:**
  - Disagreements where the report is positive but gold is negative dominate: effusion 15, contusion 12, lateral OA 8.
  - Gold prevalence is far higher than report prevalence for some findings:

    | Finding | Gold | Reports |
    |---|---|---|
    | Synovitis | 0.47 | 0.12 |
    | Fracture | 0.31 | 0.07 |
    | Lateral meniscus | 0.40 | 0.15 |

- **Ruled out:**
  - **Severity wording:** "mild effusion" can be gold-positive and "moderate" gold-negative.
  - **Cross-finding rules:** gold co-occurrence is positive, for example fracture → contusion 0.56.
- **Consequence:** better text extraction can't fix the labels. The image model has to correct them.

### D24. Denoise by mixing report labels with model predictions
- **Evidence:** a 50/50 rank mix of the report labels and my honest 3-member predictions scored 0.928 on gold, against 0.903 for the labels and 0.910 for the models. The gain was +0.026, with a 90% bootstrap CI of [0.014, 0.039].
- **Decision:** a teacher → student pipeline. Out-of-fold teacher predictions replace half of every training label.

### D25. Four folds by md5 hash, gold excluded from every fold
- **Decision:** fold = `int(md5(StudyInstanceUID)) % 4`.
- **Checks:** folds of 1,048 / 1,148 / 1,038 / 1,115 studies, disjoint, union = all 4,349 training studies, no gold. The trainer asserts train ∩ held-out = ∅ and (train ∪ held-out) ∩ gold = ∅.
- **Why OOF:** a model's predictions on its own training data are overconfident and simply copy the noisy labels. Only held-out predictions carry independent information.

### D26. Map teacher scores onto the label scale before mixing
- **Options:** isotonic regression, or quantile matching, of report label on teacher score.
- **Decision:** compute both and keep the one with higher gold agreement. Quantile won, 0.9272 vs 0.9249.
- **Fixed in advance:** the mix weight B = 0.5, never tuned. Choosing between two maps is the only decision made on gold.
- **Why map:** raw sigmoid outputs and soft labels live on different scales. Mapping keeps each finding's mean label unchanged, which also keeps the loss's `pos_weight` unchanged.

### D27. A gate before writing the labels
- **Decision:** `make_targets.py` asserts gold agreement ≥ 0.92 and asserts full OOF coverage before writing the parquet.
- **Outcome:** labels v3 scored 0.927 and passed.

### D28. torch.compile chosen by measurement
- **Decision:** the teacher kernel first ran two 2-epoch smoke tests in parallel, compile off on GPU0 and on on GPU1, and kept the faster.
- **Outcome:** compile 0.60 s/step vs 0.74 s/step. Compile is now on for the students.
- **Detail:** `model.backbone.compile()` compiles in place, so checkpoint keys stay loadable by the inference stage.

### D29. Smoke test chained to every full run
- **Decision:** each run starts with 80 studies × 2 epochs and continues to the full run only if that succeeds (`smoke && full`).
- **Why:** a crash costs minutes instead of a session's GPU quota.

### D30. Spend the remaining quota on two students now, not a v2-vs-v3 A/B
- **Context:** about 3 GPU hours were left before the weekly reset.
- **Decision:** train two ConvNeXt-small students (seeds 42 and 7) on labels v3, capped at 1.9 h, leaving about 1 h for submission test runs.
- **Why:** the old-label baselines for the same architecture and seeds already existed (0.9075 and 0.9017). Two v3 students give a less noisy gold estimate, and they can be tested on the LB the same day.

### D31. Fracture ← contusion post-processing dropped
- **Idea:** fracture score += 0.5 × contusion score lifted gold fracture AUC from 0.868 to 0.899. It rested on only 18 positives and was picked after looking at the data.
- **Result:** LB 0.956, unchanged. Dropped.

---

### D32. Result: the students did not inherit the label gain
- **Result:** students on labels v3 scored 0.8946 (seed 42, 7 epochs) and 0.8899 (seed 7, 8 epochs) on gold.
  - Old-label teacher folds at 7 epochs on 75% of the data scored 0.883–0.887, so v3 is ahead by at most about +0.005, within seed noise.
  - Adding the students to the 3 old-label members lowered gold from 0.9102 to 0.9082.
- **Explanation:** self-distillation. Half of each v3 target is the teacher's prediction from the same images and the same architecture, which the student can already produce on its own. The +0.024 label gain comes from combining two independent sources (report + image). The student keeps only the report half of that independence, and at test time there are no reports.
- **Consequence:** the denoising lever is spent. The fresh GPU quota goes to whatever c4 and the public evidence support (input diversity), not to more v3 students.

### D33. Bug: double delete in the cleanup step marked the kernel ERROR
- **What happened:** the student launcher removed `glob("m*_smoke*") + glob("m*_last.pt")`. Both patterns match `m_s3b_smoke_last.pt`, so the second `os.remove` raised `FileNotFoundError` after both models had finished.
- **Cost:** errored kernels can't be attached as sources, so the students' weights can't reach a submission notebook.
- **Fix:** iterate over `set(...)`, with a local check that overlapping patterns are removed exactly once.
- **Lesson:** the "no assert at the end" rule (D16) covers asserts but not *any* exception in the tail of the launcher. The teacher launcher was safe only because its patterns didn't overlap.

### D34. Final picks: the two strongest public-base blends
- **Evidence:** our Raptor-input members lowered the base at every weight tried: 15% → 0.955, 25% → 0.954, 33% → 0.953. Alone they scored 0.936, against 0.949 for public Raptor.
- **Ruled out:** R7 running out of time. The commit run decoded 37 s/study over 4 parallel loaders, about 3.5–4.5 h for 1,300 studies against an 8.3 h deadline.
- **Decision:** final picks are 57202247 (R7 + Raptor 384 + 224) and 57170367 (R7 + Raptor 384). Both scored 0.956, and they are the most different pair among the best scores. The fracture tweak (also 0.956) was left out because it was picked after looking at the data.

### D35. Last bet: close-up ("zoom") members for input diversity
- **Why:** the only public evidence of a gain on top of the base is a team adding a knee-centred crop model (+0.001). A weaker model with the *same* input as the base hurts, so a new member has to see the knee differently.
- **Design:**
  - The same trainer and corpus, but every window is a fixed central close-up (zoom 0.7 ≈ 100 mm, R7's local-view size) resized to 320 px.
  - The corpus is 140 mm at 320 px (0.44 mm/px), about the acquired resolution, so cropping it loses no detail, while each structure covers more of the backbone's input.
  - Labels v2, so zoom is the only change.
- **Avoiding train/inference drift:** one `zoom()` function, used by `prep()` in training. `patch_957.py` copies its source verbatim into the inference stage and wraps zoomed arms. A local check asserts that the injected copy equals the trainer's output.
- **Gate:** smoke-test training and inference before the reset. Afterwards, submit the close-up members alone first. Blend them at 15% only if they clearly beat the 0.936 of the full-field members alone.
- **Smoke tests (2026-10-15), both passed:**
  - Training at zoom 0.7 ran at 0.64 s/step, the same as full-field.
  - The full submission notebook with that checkpoint as a zoomed arm scored 3 of 3 visible test studies (`MEMBER_STAGE_DONE` in 31 s, no fallback).

## Things deliberately not done
- **Hand-labelling test images, multiple accounts, private sharing:** against the competition rules. I used only public, equally available resources.
- **Checking for patient-ID overlap between train and test:** a grey area under the rules, and not pursued.
- **Severity-aware relabelling and cross-finding hierarchy rules:** ruled out by the gold analysis (D23).
- **Tuning blend weights to the public LB:** see D20.
