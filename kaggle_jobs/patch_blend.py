"""Add our ConvNeXt-small member (kernel rsna-knee-members-a) to the 0.956 top-blend notebook.

The member runs as a second arm of the Raptor stage (same decoded volumes, windows and mirror TTA) with w=0 there,
so raptor_sub.csv stays the public Raptor model alone; its raw probabilities go to raptor_raw_arm1.csv and the
final cell blends: Raptor family = (1 - W_MEMBER) * Raptor + W_MEMBER * member, then W_RAPTOR * family + rest * R7.

usage: python -P patch_blend.py SRC_DIR OUT_DIR SLUG W_RAPTOR W_MEMBER
"""
import json, shutil, sys
from pathlib import Path

src, out, slug, w_raptor, w_member = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
nb = json.loads((src / "rsna-knee-top-blend.ipynb").read_text(encoding="utf-8"))
cells = nb["cells"]


def sub(i, old, new):
    s = "".join(cells[i]["source"])
    assert s.count(old) == 1, (i, old[:60])
    cells[i]["source"] = s.replace(old, new).splitlines(keepends=True)


# Raptor stage: second arm, convnext builder, per-arm res check, member raw dump.
sub(3, '"res": 384, "w": 1.0},\n]',
    '"res": 384, "w": 1.0},\n'
    '    # our ConvNeXt-small (kernel rsna-knee-members-a, gold-58 held out: 0.9075); w=0 keeps raptor_sub.csv = the Raptor model,\n'
    '    # the final cell blends this arm from raptor_raw_arm1.csv\n'
    '    {"file": "m_cnxs_swa.pt", "arch": "convnext_small.fb_in22k_ft_in1k_384", "res": 320, "w": 0.0},\n]')
sub(3, 'kw = dict(pretrained=pretrained, num_classes=0, in_chans=3, img_size=INPUT_RES)',
    'kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)\n'
    '    if not arch.startswith("convnext"):   # timm ConvNeXt takes no img_size\n'
    '        kw["img_size"] = INPUT_RES')
sub(3, '''    assert observed_sha == '7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef', "Bound checkpoint bytes differ"''',
    '''    if os.path.basename(pt_path) == ARMS[0]["file"]:
        assert observed_sha == '7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef', "Bound checkpoint bytes differ"''')
sub(3, 'assert int(ck.get("res", res_default)) == 384, "Expected original 384 checkpoint"',
    'assert int(ck.get("res", res_default)) == res_default, "checkpoint res differs from its arm"')
sub(3, '    _raw.to_csv("/kaggle/working/raptor_raw.csv", index=False)\n',
    '    _raw.to_csv("/kaggle/working/raptor_raw.csv", index=False)\n'
    '    for a in range(1, A):   # our member arms: raw probabilities, blended by the final cell\n'
    '        _r = pd.DataFrame(arm_probs[a].astype(np.float32), columns=LAB)\n'
    '        _r.insert(0, "StudyInstanceUID", test_ids)\n'
    '        _r.to_csv(f"/kaggle/working/raptor_raw_arm{a}.csv", index=False)\n')

# Final cell: three-way blend.
sub(11, "W_RAPTOR, W_OURS = 0.5, 0.5\nW_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.5, 'PF OA': 0.5}",
    f"W_RAPTOR = {w_raptor}   # Raptor family (public Raptor + our ConvNeXt member); R7 gets the rest\n"
    "W_OURS = 1.0 - W_RAPTOR\n"
    f"W_MEMBER = {w_member}   # our ConvNeXt-small's share of the Raptor family\n"
    "W_RAPTOR_C = {}")
sub(11, "rsub = pd.read_csv(WORK / 'raptor_sub.csv', dtype={'StudyInstanceUID': str})\n",
    "rsub = pd.read_csv(WORK / 'raptor_sub.csv', dtype={'StudyInstanceUID': str})\n"
    "mem = pd.read_csv(WORK / 'raptor_raw_arm1.csv', dtype={'StudyInstanceUID': str})\n")
sub(11, "for df, nm in ((ours, 'ours'), (raw, 'raptor_raw'), (rsub, 'raptor_sub')):",
    "for df, nm in ((ours, 'ours'), (raw, 'raptor_raw'), (rsub, 'raptor_sub'), (mem, 'member_raw')):")
sub(11, "rsub = rsub.set_index('StudyInstanceUID').loc[test_ids, labels]\n",
    "rsub = rsub.set_index('StudyInstanceUID').loc[test_ids, labels]\n"
    "mem = mem.set_index('StudyInstanceUID').loc[test_ids, labels]\n")
sub(11, "rap_fail = (raw.values == 0.5).all(axis=1)             # Raptor leaves a failed study at 0.5 in every column\n",
    "rap_fail = (raw.values == 0.5).all(axis=1)             # Raptor leaves a failed study at 0.5 in every column\n"
    "mem_fail = rap_fail | (mem.values == 0.5).all(axis=1)   # a member failure falls back to the Raptor model alone\n")
sub(11, "    rr, ro = pct_on(raw[c].values, ~rap_fail), pct_on(ours[c].values, ~our_fail)   # [v3] each over its own non-failed rows\n",
    "    rr, ro = pct_on(raw[c].values, ~rap_fail), pct_on(ours[c].values, ~our_fail)   # [v3] each over its own non-failed rows\n"
    "    rr = np.where(mem_fail, rr, (1.0 - W_MEMBER) * rr + W_MEMBER * pct_on(mem[c].values, ~mem_fail))\n")
sub(11, "'raptor_failed': int(rap_fail.sum()),",
    "'raptor_failed': int(rap_fail.sum()), 'member': W_MEMBER, 'member_failed': int(mem_fail.sum()),")

out.mkdir(parents=True, exist_ok=True)
(out / "rsna-knee-top-blend.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
meta = json.loads((src / "kernel-metadata.json").read_text(encoding="utf-8"))
meta.pop("id_no", None)
meta["id"], meta["title"] = f"arnavkumarmd24/{slug}", slug
meta["kernel_sources"] = ["arnavkumarmd24/rsna-knee-members-a"]
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
for i in (3, 11):
    compile("".join(l for l in cells[i]["source"] if not l.startswith("%%")), f"cell{i}", "exec")
print("ok", out, slug, w_raptor, w_member)
