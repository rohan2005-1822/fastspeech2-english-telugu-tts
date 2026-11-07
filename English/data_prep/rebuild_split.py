"""Rebuild the English train/val split (documents the notebook step exactly).
Run from the English repo root after align_english.py and recover_english.py:
    python data_prep/rebuild_split.py
Steps: (1) merge recovered TextGrids into TextGrid/ without overwriting, (2) run preprocess.py to extract
mel/pitch/energy/duration, (3) rewrite train.txt/val.txt from the `set` column of english_split_meta.csv
(preprocess.py itself would put every clip in train.txt because val_size is 0).
"""
import glob, os, shutil, subprocess, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

pp = "preprocessed_data/English"
meta_csv = "data_prep/english_split_meta.csv"
m = pd.read_csv(meta_csv)
meta = {int(r.idx): (r.spk, r.set, float(r.dur)) for r in m.itertuples()}

# guard: clip id = 'en' + idx, and the speaker folder must agree with the metadata
for p in glob.glob(f"{pp}/TextGrid*/**/*.TextGrid", recursive=True):
    cid, folder = os.path.basename(p)[:-9], os.path.basename(os.path.dirname(p))
    assert int(cid[2:]) in meta and meta[int(cid[2:])][0] == folder, f"mapping problem: {cid}"

src_dir = Path(pp) / "TextGrid_recovered"
dst_dir = Path(pp) / "TextGrid"
if src_dir.exists():
    for src_file in src_dir.rglob("*"):
        if src_file.is_file():
            dst_file = dst_dir / src_file.relative_to(src_dir)
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            if not dst_file.exists():
                shutil.copy2(src_file, dst_file)
subprocess.run([sys.executable, "preprocess.py", "config/English/preprocess.yaml"], check=True)

lines = [l for f in ("train.txt", "val.txt") for l in open(f"{pp}/{f}", encoding="utf-8").read().splitlines() if l.strip()]
shutil.copy(f"{pp}/train.txt", f"{pp}/train_all_unsplit.txt")
tr = [l for l in lines if meta[int(l.split("|")[0][2:])][1] == "train"]
va = [l for l in lines if meta[int(l.split("|")[0][2:])][1] != "train"]
open(f"{pp}/train.txt", "w", encoding="utf-8").write("\n".join(tr) + "\n")
open(f"{pp}/val.txt", "w", encoding="utf-8").write("\n".join(va) + "\n")
print(f"train {len(tr)} clips | val {len(va)} clips")
