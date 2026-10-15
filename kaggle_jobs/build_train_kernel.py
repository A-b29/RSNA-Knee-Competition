"""Build a training kernel: cell 1 writes members-a/train_member.py to /tmp, cell 2 is the folder's launcher.py.

usage: python -P build_train_kernel.py DIR SLUG "markdown title + text" [extra dataset_sources...]
"""
import json, sys
from pathlib import Path

d, slug, md, extra = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4:]
here = Path(__file__).parent
src = (here / "members-a/train_member.py").read_text(encoding="utf-8")
launcher = (d / "launcher.py").read_text(encoding="utf-8")
compile(src, "train_member", "exec"); compile(launcher, "launcher", "exec")


def cell(kind, text):
    c = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


nb = {"cells": [cell("markdown", md), cell("code", "%%writefile /tmp/train_member.py\n" + src), cell("code", launcher)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
(d / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
meta = json.loads((here / "members-a/kernel-metadata.json").read_text(encoding="utf-8"))
meta.update(id=f"arnavkumarmd24/{slug}", title=slug, code_file=f"{slug}.ipynb")
meta["dataset_sources"] = meta["dataset_sources"] + [s for s in extra if s not in meta["dataset_sources"]]
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("ok", d / f"{slug}.ipynb")
