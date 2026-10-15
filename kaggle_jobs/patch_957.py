"""Variants of the top-957 notebook (R7 + Raptor 384/224): our ConvNeXt members after R7, and a fracture post-processing.

Members run as their own stage AFTER R7 (a patched copy of the Raptor stage script: our arms, 24 windows + mirror, a
deadline), so they never take R7's time (R7 is capped at 8.3 h from notebook start and decodes ~40 s/study on CPU).
Final cell: out = (1 - W_MEM) * base + W_MEM * mean member rank (base where no member scored a study), then
optionally Fracture <- rank(Fracture + ALPHA * Contusion).

usage: python -P patch_957.py OUT_DIR SLUG W_MEM ALPHA [--arm FILE=ARCH ...] [--kernel OWNER/SLUG ...]
"""
import argparse, json, re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("slug"); ap.add_argument("w_mem", type=float); ap.add_argument("alpha", type=float)
ap.add_argument("--arm", action="append", default=[]); ap.add_argument("--kernel", action="append", default=[])
a = ap.parse_args()
assert 0 <= a.w_mem <= 1 and a.alpha >= 0 and (a.w_mem == 0) == (not a.arm)
here = Path(__file__).parent
src = here / "top-957"
nb = json.loads((src / "rsna-knee-top-957.ipynb").read_text(encoding="utf-8"))
cells = nb["cells"]
assert "".join(cells[2]["source"]).startswith("%%writefile /tmp/raptor_run.py") and "Stage 3" in "".join(cells[8]["source"])


def sub(text, old, new):
    assert text.count(old) == 1, old[:70]
    return text.replace(old, new)


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


if a.arm:
    s = "".join(cells[2]["source"])
    s = sub(s, "%%writefile /tmp/raptor_run.py", "%%writefile /tmp/member_run.py")
    arms = "".join(f'    {{"file": "{f}", "arch": "{arch}", "res": 320, "w": 1.0}},\n' for f, arch in (x.split("=", 1) for x in a.arm))
    s, k = re.subn(r"ARMS = \[[^\n]*\n(?:    \{[^\n]*\n)+\]\n", "ARMS = [   # [members] our ConvNeXt members (trained on the Raptor-96 corpus)\n" + arms + "]\n", s, count=1)
    assert k == 1
    s = sub(s, "kw = dict(pretrained=pretrained, num_classes=0, in_chans=3, img_size=INPUT_RES)",
            "kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)\n"
            "    if not arch.startswith(\"convnext\"):   # [members] timm ConvNeXt takes no img_size\n"
            "        kw[\"img_size\"] = INPUT_RES")
    s = sub(s, "    assert observed_sha == '7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef', \"Bound checkpoint bytes differ\"\n", "")
    s = sub(s, 'assert int(ck.get("res", res_default)) == 384, "Expected original 384 checkpoint"',
            'assert int(ck.get("res", res_default)) == res_default, "checkpoint res differs from its arm"')
    s = sub(s, "K_EVAL = 94   # every window position of the 96-slice test volume (full coverage)",
            "K_EVAL = 24   # [members] 24 windows + mirror (the members' per-epoch gold eval; ~4x faster than 94)\n"
            "_DEADLINE = float(os.environ.get(\"MEMBER_DEADLINE\", \"inf\"))   # [members] studies after this stay unscored (0.5)")
    s = sub(s, "assert plain.shape==mirrored.shape==(94,3,320,320)", "assert plain.shape==mirrored.shape==(K_EVAL,3,320,320)")
    s = sub(s, "            sid, vol, mask = item; i = pos[sid]\n",
            "            sid, vol, mask = item; i = pos[sid]\n"
            "            if time.time() > _DEADLINE:   # [members] out of time: leave this study unscored\n"
            "                q.task_done(); continue\n")
    s = sub(s, '''    _raw = pd.DataFrame(arm_probs[0].astype(np.float32), columns=LAB)   # [blend] raw probabilities; a failed study stays 0.5
    _raw.insert(0, "StudyInstanceUID", test_ids)
    _raw.to_csv("/kaggle/working/raptor_raw.csv", index=False)
''', '''    for a_ in range(A):   # [members] raw probabilities per arm; an unscored study stays 0.5
        _raw = pd.DataFrame(arm_probs[a_].astype(np.float32), columns=LAB)
        _raw.insert(0, "StudyInstanceUID", test_ids)
        _raw.to_csv(f"/kaggle/working/member_raw_arm{a_}.csv", index=False)
''')
    s = sub(s, 'out = "/kaggle/working/raptor_sub.csv"', 'out = "/kaggle/working/member_sub.csv"')
    run = (f"# [members] Stage 2b: our {len(a.arm)} ConvNeXt member(s), AFTER R7 so they never take its time; capped so the notebook\n"
           "# ends well inside 9 h. Any failure here keeps the base submission (the final cell blends only what was scored).\n"
           "import os, sys, time, subprocess\n"
           "MEMBER_END = NB_T0 + 8.75 * 3600\n"
           "_tm = time.time()\n"
           "try:\n"
           "    subprocess.run([sys.executable, '-u', '/tmp/member_run.py'], env=dict(os.environ, MEMBER_DEADLINE=repr(MEMBER_END - 180)),\n"
           "                   timeout=max(60.0, MEMBER_END - time.time()))\n"
           "except Exception as e:\n"
           "    print('MEMBER_STAGE_FAILED', type(e).__name__, str(e)[:200], flush=True)\n"
           "MEMBER_SECONDS = time.time() - _tm\n"
           "print('MEMBER_STAGE_DONE seconds %.0f' % MEMBER_SECONDS, flush=True)\n")
    cells[8:8] = [code(s), code(run)]

fin = cells[-1]
s = "".join(fin["source"])
s = sub(s, "sub = pd.DataFrame(out.astype(np.float32), columns=labels)\n",
        f"""# [members] our ConvNeXt members (stage 2b) and the fracture post-processing, both fixed before the run
W_MEM = {a.w_mem}   # members' share of the final rank blend
N_MEM = {len(a.arm)}
FRAC_ALPHA = {a.alpha}   # Fracture <- rank(Fracture + FRAC_ALPHA * Contusion); gold-58: 0.868 -> 0.899 at 0.5 on our members
mem_scored = {{}}
if W_MEM > 0:
    mranks = []
    for a_ in range(N_MEM):
        f_ = WORK / f'member_raw_arm{{a_}}.csv'
        if not f_.is_file():
            continue
        m_ = pd.read_csv(f_, dtype={{'StudyInstanceUID': str}}).set_index('StudyInstanceUID').loc[test_ids, labels]
        ok_ = ~(m_.values == 0.5).all(axis=1)
        mem_scored[f_.name] = int(ok_.sum())
        mranks.append(np.stack([pct_on(m_[c].values, ok_) for c in labels], 1))
    if mranks:
        cnt = np.sum([~np.isnan(r) for r in mranks], 0)
        mr = np.nansum(mranks, 0) / np.maximum(cnt, 1)
        out = np.where(cnt > 0, (1.0 - W_MEM) * out + W_MEM * mr, out)
    print('MEMBERS', json.dumps(mem_scored), flush=True)
if FRAC_ALPHA > 0:
    jf, jc = labels.index('Fracture'), labels.index('Contusion')
    out[:, jf] = pct(out[:, jf] + FRAC_ALPHA * out[:, jc])
sub = pd.DataFrame(out.astype(np.float32), columns=labels)
""")
s = sub(s, "(WORK / 'blend_receipt.json').write_text(json.dumps(blend, indent=1))\n",
        "blend.update(w_mem=W_MEM, n_members=N_MEM, members_scored=mem_scored, frac_alpha=FRAC_ALPHA)   # [members]\n"
        "(WORK / 'blend_receipt.json').write_text(json.dumps(blend, indent=1))\n")
fin["source"] = s.splitlines(keepends=True)

for c in cells:
    if c["cell_type"] == "code":
        compile("".join(l for l in "".join(c["source"]).splitlines(keepends=True) if not l.startswith("%%")), "cell", "exec")
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
(out / f"{a.slug}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
meta = json.loads((src / "kernel-metadata.json").read_text(encoding="utf-8"))
meta.update(id=f"arnavkumarmd24/{a.slug}", title=a.slug, code_file=f"{a.slug}.ipynb", kernel_sources=a.kernel)
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("ok", out, a.slug, "w_mem", a.w_mem, "alpha", a.alpha, "arms", len(a.arm), "cells", len(cells))
