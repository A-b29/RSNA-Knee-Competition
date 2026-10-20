import os, glob, subprocess, sys
# Students on the denoised targets (labels v3 = report labels mixed with the 4-fold teacher's out-of-fold predictions), gold-58 held out.
# One process per GPU (DataParallel leaks on Kaggle). Each runs a 2-epoch smoke on 80 studies first, then the full run.
base = dict(os.environ, TARGETS='labels_v3_denoised.parquet', HOURS='2.6', EPOCHS_MAX='9', WORKERS='2', COMPILE='1')   # teacher smoke: 0.60 vs 0.74 s/step
MEMBERS = [('0', 'convnext_base.fb_in22k_ft_in1k_384', '1e-4', '0.3', '42', '_s3c'),
           ('1', 'convnext_small.fb_in22k_ft_in1k_384', '1e-4', '0.3', '11', '_s3d')]
PY = f"{sys.executable} -u /tmp/train_member.py"
runs = {}
for gpu, arch, lr, dp, seed, tag in MEMBERS:
    env = {**base, 'CUDA_VISIBLE_DEVICES': gpu, 'ARCH': arch, 'BB_LR': lr, 'DROP_PATH': dp, 'SEED': seed, 'TAG': tag}
    cmd = f"LIMIT=80 EPOCHS_MAX=2 TAG={tag}_smoke {PY} && {PY}"
    runs[tag] = subprocess.Popen(["bash", "-c", cmd], env=env, stdout=open(f"/kaggle/working/log{tag}.txt", "w"), stderr=subprocess.STDOUT)
codes = {tag: p.wait() for tag, p in runs.items()}
for tag in runs:
    print(f"===== {tag}: exit {codes[tag]}")
    print("".join(l for l in open(f"/kaggle/working/log{tag}.txt") if " step " not in l)[-5000:])
for f in set(glob.glob("/kaggle/working/m*_smoke*") + glob.glob("/kaggle/working/m*_last.pt")):   # set: m_X_smoke_last.pt matches both
    os.remove(f)
# No assert: a failed student must not mark the kernel as errored, or the other student's weights can't be attached.
