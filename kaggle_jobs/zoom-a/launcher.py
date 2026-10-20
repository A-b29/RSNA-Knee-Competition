import os, glob, subprocess, sys
# Close-up members: the same trainer, but every window is a fixed central ZOOM x 140 mm close-up (0.7 ~ 100 mm, R7's local-view size),
# for input diversity against the full-field Raptor models. Labels v2, so zoom is the only change vs the 0.936 members.
# One process per GPU (DataParallel leaks on Kaggle). Each runs a 2-epoch smoke on 80 studies first, then the full run.
base = dict(os.environ, ZOOM='0.7', HOURS='2.8', EPOCHS_MAX='12', WORKERS='2', COMPILE='1')
MEMBERS = [('0', '42', '_z7a'), ('1', '7', '_z7b')]
PY = f"{sys.executable} -u /tmp/train_member.py"
runs = {}
for gpu, seed, tag in MEMBERS:
    env = {**base, 'CUDA_VISIBLE_DEVICES': gpu, 'ARCH': 'convnext_small.fb_in22k_ft_in1k_384', 'BB_LR': '1e-4', 'DROP_PATH': '0.2', 'SEED': seed, 'TAG': tag}
    cmd = f"LIMIT=80 EPOCHS_MAX=2 TAG={tag}_smoke {PY} && {PY}"
    runs[tag] = subprocess.Popen(["bash", "-c", cmd], env=env, stdout=open(f"/kaggle/working/log{tag}.txt", "w"), stderr=subprocess.STDOUT)
codes = {tag: p.wait() for tag, p in runs.items()}
for tag in runs:
    print(f"===== {tag}: exit {codes[tag]}")
    print("".join(l for l in open(f"/kaggle/working/log{tag}.txt") if " step " not in l)[-5000:])
for f in set(glob.glob("/kaggle/working/m*_smoke*") + glob.glob("/kaggle/working/m*_last.pt")):   # set: m_X_smoke_last.pt matches both
    os.remove(f)
# No assert and nothing that can raise after training: an errored kernel can't be attached as a source.
