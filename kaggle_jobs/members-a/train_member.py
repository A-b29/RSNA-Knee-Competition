"""Train a new Raptor-pipeline member from ImageNet weights on labels v2 (gold-58 held out).

Inputs, windows, head and checkpoint format follow the public Raptor stage (dreaddevelopment / nartaa, CC0):
96-slice fastread volume, central 320 crop, 3-slice windows, per-finding attention MIL. So a member drops into
that stage as one more arm; only the backbone (ARCH) differs, for blend diversity. The epoch count is fitted
to HOURS after timing the first steps; the saved model is the SWA of the last SWA_LAST epochs.
FOLD>=0 holds out one md5 fold of the training studies and writes its out-of-fold predictions (teacher for label
denoising); TARGETS names the soft-label parquet to train on (default labels v2).
"""
import os, glob, json, math, time, random, hashlib
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import timm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
ARCH = os.environ["ARCH"]
TAG = os.environ.get("TAG", "")
EPOCHS_MAX = int(os.environ.get("EPOCHS_MAX", 9))   # gold curves plateau by epoch 8
SWA_LAST = int(os.environ.get("SWA_LAST", 3))
BS = int(os.environ.get("BS", 4))
K, K_EVAL, K_TTA = 12, 24, 94              # train windows, per-epoch gold windows, inference windows (Raptor stage)
BB_LR, HEAD_LR, WD = float(os.environ.get("BB_LR", 1e-4)), 1e-3, 0.05
DROP_PATH = float(os.environ.get("DROP_PATH", 0.2))
WARM = 300                                 # warmup steps
SEED = int(os.environ.get("SEED", 42))
LIMIT = int(os.environ.get("LIMIT", 0))    # >0: smoke run on this many studies
HOURS = float(os.environ.get("HOURS", 6.5))
WORKERS = int(os.environ.get("WORKERS", 2))
FOLD, NFOLDS = int(os.environ.get("FOLD", -1)), int(os.environ.get("NFOLDS", 4))
TARGETS = os.environ.get("TARGETS", "labels_llm_soft.parquet")
COMPILE = os.environ.get("COMPILE", "0") == "1"
OUT = "/kaggle/working"
SLOT_OF = np.repeat(np.arange(5), [26, 22, 18, 12, 18])   # corpus slice layout: sag fluid/other, cor fluid/other, axial

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True
dev = torch.device("cuda")
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 1, 3, 1, 1)


def find(name):
    for d in sorted(glob.glob("/kaggle/input/*/")):
        if "competition" in d or "abnormality-detection" in d:
            continue
        hits = glob.glob(os.path.join(d, "**", name), recursive=True)
        if hits:
            return hits[0]
    raise FileNotFoundError(name)


comp = next(r for r in ("/kaggle/input/competitions/rsna-knee-abnormality-detection",
                        "/kaggle/input/rsna-knee-abnormality-detection") if os.path.exists(r + "/train.csv"))
paths = {}
for d in sorted(glob.glob("/kaggle/input/*/")):
    if "competition" in d or "abnormality-detection" in d:
        continue
    for c in glob.glob(os.path.join(d, "**", "raptor96_c320"), recursive=True):
        for p in glob.glob(c + "/*.npz"):
            paths[os.path.basename(p)[:-4]] = p
print("corpus studies:", len(paths), flush=True)

tr = pd.read_csv(comp + "/train.csv"); tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
gold = tr[tr[LAB].notna().all(axis=1)].set_index("StudyInstanceUID")[LAB].astype(np.float32)
soft = pd.read_parquet(find(TARGETS)).set_index("StudyInstanceUID")[LAB].astype(np.float32)
gold_ids = [u for u in gold.index if u in paths]
train_ids = [u for u in soft.index if u in paths and u not in gold.index]
heldout = []
if FOLD >= 0:
    in_fold = [int(hashlib.md5(u.encode()).hexdigest(), 16) % NFOLDS == FOLD for u in train_ids]
    heldout = [u for u, f in zip(train_ids, in_fold) if f]
    train_ids = [u for u, f in zip(train_ids, in_fold) if not f]
if LIMIT:
    train_ids, gold_ids, heldout = train_ids[:LIMIT], gold_ids[:max(8, LIMIT // 4)], heldout[:LIMIT // 4]
assert not (set(train_ids) & set(heldout)) and not (set(train_ids + heldout) & set(gold_ids))
labels = {**{u: soft.loc[u].values for u in train_ids + heldout}, **{u: gold.loc[u].values for u in gold_ids}}
print(f"{ARCH} | {TARGETS} | train {len(train_ids)} | heldout fold {FOLD}: {len(heldout)} | gold-val {len(gold_ids)} | compile {COMPILE}", flush=True)


def centers(mask, k, train):
    D = len(mask)
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, D))
    lo, hi = int(valid.min()), int(valid.max())
    cs = list(range(lo + 1, hi)) or [max(1, min((lo + hi) // 2, D - 2))]
    if train:
        pool = cs * (k // len(cs) + 1); random.shuffle(pool)
        return np.array(pool[:k])
    return np.array([cs[i] for i in np.linspace(0, len(cs) - 1, k).round().astype(int)])


def mirror(x, cs):
    """L/R mirror as the Raptor stage's MIRROR_TTA: sagittal windows reverse their 3 slices, others flip width."""
    sag = SLOT_OF[cs] < 2
    return np.where(sag[:, None, None, None], x[:, ::-1], x[..., ::-1])


class Bags(Dataset):
    def __init__(self, ids, k, train, flip=False):
        self.ids, self.k, self.train, self.flip = ids, k, train, flip

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        uid = self.ids[i]
        with np.load(paths[uid]) as z:
            vol, mask = z["vol"], z["mask"]
        cs = np.clip(centers(mask, self.k, self.train), 1, len(mask) - 2)
        x = vol[np.stack([cs - 1, cs, cs + 1], 1)]                     # (k,3,320,320) uint8
        if self.flip or (self.train and random.random() < 0.5):
            x = mirror(x, cs)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(labels[uid])


def prep(x, train):
    """uint8 windows -> normalised float on the GPU; training adds a per-study affine and gain/bias jitter."""
    x = x.to(dev, non_blocking=True).float().div_(255)                # (B,k,3,H,W)
    if train:
        B, k = x.shape[:2]
        a = (torch.rand(B, device=dev) - 0.5) * math.radians(20)
        s = 1 + (torch.rand(B, device=dev) - 0.5) * 0.2
        t = (torch.rand(B, 2, device=dev) - 0.5) * 0.12
        theta = torch.stack([torch.stack([a.cos() / s, -a.sin() / s, t[:, 0]], 1),
                             torch.stack([a.sin() / s, a.cos() / s, t[:, 1]], 1)], 1).repeat_interleave(k, 0)
        flat = x.flatten(0, 1)
        flat = F.grid_sample(flat, F.affine_grid(theta, flat.shape, align_corners=False), align_corners=False)
        gain = 1 + (torch.rand(B, 1, 1, 1, 1, device=dev) - 0.5) * 0.3
        bias = (torch.rand(B, 1, 1, 1, 1, device=dev) - 0.5) * 0.1
        x = (flat.view_as(x) * gain + bias).clamp_(0, 1)
    return (x - MEAN) / STD


class RaptorClassifier(nn.Module):
    """From the public Raptor stage (same weights layout): per-finding attention-MIL over windows; channels_last input added."""
    def __init__(self, backbone, F_dim=768, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.n = n

    def encode(self, x):
        B, K = x.shape[:2]
        flat = x.flatten(0, 1)
        f = torch.cat([self.backbone(chunk.contiguous(memory_format=torch.channels_last)) for chunk in flat.split(16)], dim=0)
        return f.view(B, K, -1)

    def head(self, feats):
        h = self.norm(feats)
        a = torch.softmax(self.att(h), dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        return (pooled * self.clsW).sum(-1) + self.clsb

    def forward(self, x):
        return self.head(self.encode(x))


bb = timm.create_model(ARCH, pretrained=True, num_classes=0, in_chans=3, global_pool="avg", drop_path_rate=DROP_PATH)
model = RaptorClassifier(bb, F_dim=bb.num_features).to(dev, memory_format=torch.channels_last)
if COMPILE:
    model.backbone.compile()   # in place: state_dict keys stay loadable by the Raptor stage
ckpt_on = False

tl = DataLoader(Bags(train_ids, K, True), batch_size=BS, shuffle=True, num_workers=WORKERS,
                drop_last=True, pin_memory=True)
prev = np.clip(np.stack([labels[u] for u in train_ids]).mean(0), 0.03, 0.7)
lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(np.clip((1 - prev) / prev, 1, 10), dtype=torch.float32, device=dev))
head = [p for n_, p in model.named_parameters() if not n_.startswith("backbone.")]
opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": BB_LR}, {"params": head, "lr": HEAD_LR}], weight_decay=WD)
scaler = torch.amp.GradScaler("cuda")


def lr_scale(step, total):
    if step < WARM:
        return (step + 1) / WARM
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - WARM) / max(1, total - WARM))))


@torch.no_grad()
def predict(ids, k, flip=False):
    model.eval(); P = []
    for x, _ in DataLoader(Bags(ids, k, False, flip), batch_size=1, num_workers=2):
        with torch.autocast("cuda", dtype=torch.float16):
            P.append(torch.sigmoid(model(prep(x, False)).float()).cpu().numpy())
    return np.concatenate(P)


def gold_auc(P):
    Y = np.stack([labels[u] for u in gold_ids])
    aucs = {n: roc_auc_score(Y[:, j], P[:, j]) for j, n in enumerate(LAB) if len(set(Y[:, j])) > 1}
    return float(np.mean(list(aucs.values()))), aucs


def save(sd, name, **meta):
    torch.save({"model": sd, "arch": ARCH, "res": 320, "lab": LAB, **meta}, f"{OUT}/{name}")


t0 = time.time()
epochs, total = EPOCHS_MAX, EPOCHS_MAX * len(tl)
cal = min(60, len(tl) - 1); cal0 = min(10, cal // 2)
gstep, snapshots, hist = 0, [], []
for ep in range(1, EPOCHS_MAX + 1):
    if ep > epochs:
        break
    model.train(); tot, n, wait = 0.0, 0, 0.0; te = tw = time.time()
    for step, (x, y) in enumerate(tl):
        wait += time.time() - tw
        if step == cal0 and ep == 1:
            tc = time.time()
        if step == cal and ep == 1:   # fit the epoch count (and the cosine length) to the time budget
            epoch_s = (time.time() - tc) / (cal - cal0) * len(tl) + 60
            epochs = max(1, min(EPOCHS_MAX, int(0.97 * HOURS * 3600 / epoch_s)))
            total = epochs * len(tl)
            print(f"  {(time.time() - tc) / (cal - cal0):.2f} s/step -> {epochs} epochs of ~{epoch_s / 60:.0f} min", flush=True)
        for g, base in zip(opt.param_groups, (BB_LR, HEAD_LR)):
            g["lr"] = base * lr_scale(gstep, total)
        oom = False
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(prep(x, True))
            loss = lossf(logits.float(), y.to(dev))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(opt); scaler.update()
        except torch.OutOfMemoryError:
            if ckpt_on:
                raise
            oom = True
        if oom:   # fixed input shape, so one OOM means this backbone needs gradient checkpointing for the whole run
            logits = loss = None; opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            model.backbone.set_grad_checkpointing(True); ckpt_on = True
            print("  OOM -> gradient checkpointing on", flush=True)
            tw = time.time(); continue
        tot += loss.item(); n += 1; gstep += 1
        if step % 200 == 0:
            el = time.time() - te
            print(f"  ep{ep} step {step}/{len(tl)} loss {tot / n:.4f} | {el / max(step, 1):.2f} s/step | wait {100 * wait / max(el, 1e-9):.0f}% | "
                  f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB", flush=True)
        if time.time() - t0 > (HOURS + 0.5) * 3600:
            break
        tw = time.time()
    au, aucs = gold_auc(predict(gold_ids, K_EVAL))
    hist.append({"ep": ep, "loss": tot / max(n, 1), "gold_auc": au, "minutes": (time.time() - te) / 60})
    print(f"epoch {ep}/{epochs} loss {tot / max(n, 1):.4f} gold macro AUC {au:.4f} | {(time.time() - te) / 60:.1f} min | "
          + " ".join(f"{k[:6]}={v:.3f}" for k, v in aucs.items()), flush=True)
    save(model.state_dict(), f"m{TAG}_last.pt", epoch=ep)
    if ep > epochs - SWA_LAST:
        snapshots.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    if time.time() - t0 > (HOURS + 0.5) * 3600:
        print("time budget reached", flush=True)
        break

last_tta, _ = gold_auc(0.5 * (predict(gold_ids, K_TTA) + predict(gold_ids, K_TTA, flip=True)))
if snapshots:
    swa = {k: (sum(s[k].float() for s in snapshots) / len(snapshots)).to(snapshots[-1][k].dtype) for k in snapshots[-1]}
    model.load_state_dict(swa)
    save(swa, f"m{TAG}_swa.pt", swa_over=[h["ep"] for h in hist][-len(snapshots):])
P = 0.5 * (predict(gold_ids, K_TTA) + predict(gold_ids, K_TTA, flip=True))   # as the Raptor stage infers: 94 windows + mirror
au_tta, aucs_tta = gold_auc(P)
json.dump({"arch": ARCH, "swa_gold_tta": au_tta, "last_gold_tta": last_tta, "swa_aucs_tta": aucs_tta, "hist": hist,
           "epochs": epochs, "swa_over": len(snapshots), "bb_lr": BB_LR, "drop_path": DROP_PATH, "grad_ckpt": ckpt_on,
           "n_train": len(train_ids), "fold": FOLD, "targets": TARGETS, "compile": COMPILE, "gold_pred": {u: p.tolist() for u, p in zip(gold_ids, P)}},
          open(f"{OUT}/m{TAG}.json", "w"), indent=1)
if heldout:   # teacher: out-of-fold predictions of the saved (SWA) model, 24 windows + mirror
    Q = 0.5 * (predict(heldout, K_EVAL) + predict(heldout, K_EVAL, flip=True))
    oof = pd.DataFrame(Q.astype(np.float32), columns=LAB); oof.insert(0, "StudyInstanceUID", heldout)
    oof.to_csv(f"{OUT}/oof{TAG}.csv", index=False)
    print(f"out-of-fold predictions: {len(heldout)} studies -> oof{TAG}.csv", flush=True)
print(f"SWA over {len(snapshots)} epochs: gold macro AUC (94 windows + mirror) {au_tta:.4f} | last epoch {last_tta:.4f} | "
      f"total {(time.time() - t0) / 60:.0f} min", flush=True)
