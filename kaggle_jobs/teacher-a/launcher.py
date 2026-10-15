import os, re, glob, subprocess, sys
# 4-fold ConvNeXt-small teacher: out-of-fold predictions for label denoising. One process per GPU (DataParallel leaks on Kaggle).
base = dict(os.environ, ARCH='convnext_small.fb_in22k_ft_in1k_384', BB_LR='1e-4', DROP_PATH='0.2', NFOLDS='4', HOURS='1.2', WORKERS='2')
PY = f"{sys.executable} -u /tmp/train_member.py"
W = "/kaggle/working"


def launch(gpu, cmd, log, **env):
    return subprocess.Popen(["bash", "-c", cmd], env={**base, 'CUDA_VISIBLE_DEVICES': gpu, **env},
                            stdout=open(f"{W}/{log}", "w"), stderr=subprocess.STDOUT)


def tail(log, n):
    return "".join(l for l in open(f"{W}/{log}") if " step " not in l)[-n:]


# 1) 2-epoch smokes, compile off on GPU0 and on on GPU1 in parallel (every code path incl. fold split, SWA, TTA, OOF); keep the faster
smoke = {c: launch(str(i), PY, f"log_smoke{c}.txt", COMPILE=c, FOLD='0', LIMIT='80', EPOCHS_MAX='2', TAG=f'_smoke{c}')
         for i, c in enumerate('01')}
speed = {}
for c, p in smoke.items():
    p.wait()
    m = re.search(r"([0-9.]+) s/step ->", open(f"{W}/log_smoke{c}.txt").read())
    speed[c] = float(m.group(1)) if p.returncode == 0 and m else float('inf')
COMPILE = min(speed, key=speed.get)
print('smoke s/step', speed, '-> COMPILE', COMPILE, flush=True)
for c in '01':
    print(f"===== smoke compile={c}")
    print(tail(f"log_smoke{c}.txt", 2500))

# 2) the four folds, two per GPU in sequence (';' so a failed fold does not stop the next)
runs = {g: launch(g, "; ".join(f"FOLD={f} TAG=_t{f} {PY}" for f in folds), f"log_gpu{g}.txt", COMPILE=COMPILE)
        for g, folds in (('0', '01'), ('1', '23'))}
codes = {g: p.wait() for g, p in runs.items()}
for g in runs:
    print(f"===== GPU{g}: exit {codes[g]}")
    print(tail(f"log_gpu{g}.txt", 6000))
for f in glob.glob(f"{W}/*smoke*") + glob.glob(f"{W}/m_t*_last.pt"):
    if not f.endswith(".txt"):
        os.remove(f)
print("oof files:", sorted(glob.glob(f"{W}/oof_t*.csv")))
# No assert: a failed fold must not mark the kernel as errored, or the other folds' outputs can't be attached.
