import os, subprocess, sys
# Pre-reset check of the zoom path: one 2-epoch smoke on 80 studies (keeps its SWA weights for an inference test). No assert.
env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', ARCH='convnext_small.fb_in22k_ft_in1k_384', ZOOM='0.7', LIMIT='80', EPOCHS_MAX='2',
           COMPILE='1', WORKERS='2', TAG='_zs')
rc = subprocess.run([sys.executable, '-u', '/tmp/train_member.py'], env=env).returncode
print('smoke exit', rc, os.listdir('/kaggle/working'))
