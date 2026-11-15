"""Stage 9: a small sample of the dataset with the same layout, zipped for sharing.

Contains two stacks of the same object seen by both missions (WISE 0855: a real
mover) with all their labels, one fixed injected realisation each, the PSFs, the
documentation, the tools and the QC figures. The loader works on the sample
exactly as on the full dataset.

Output: <project>/planetx-dataset-sample/ and planetx-dataset-sample.zip
"""
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config as C

STACKS = ["spherex:wise0855", "wise:wise0855"]
REALIZATIONS = ["spherex:wise0855:r0", "wise:wise0855:r0"]
NAME = "planetx-dataset-sample"


def main():
    out = C.ROOT / NAME
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    for f in ("README.md", "DATA_DICTIONARY.md", "CITATIONS.md", "splits.json", "excluded.json"):
        shutil.copy(C.OUT / f, out / f)
    shutil.copytree(C.OUT / "tools", out / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(C.OUT / "qc", out / "qc")
    shutil.copytree(C.OUT / "baseline", out / "baseline", ignore=shutil.ignore_patterns("*.parquet"))

    ix = pd.read_parquet(C.OUT / "index.parquet")
    keep = ix[ix.stack_id.isin(STACKS) & ((ix.sample_type != "injected") | ix.realization_id.isin(REALIZATIONS))]
    keep.to_parquet(out / "index.parquet", index=False)
    for rel in keep.path:
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(C.OUT / rel, out / rel)
    for sid in STACKS:
        mission, field = sid.split(":")
        m = out / mission
        (m / "psf").mkdir(exist_ok=True)
        shutil.copy(C.OUT / mission / "psf" / f"{field}.npz", m / "psf" / f"{field}.npz")
        meta = json.loads((C.OUT / mission / "stacks.json").read_text())
        (m / "stacks.json").write_text(json.dumps({sid: meta[sid]}, indent=1))
        for t in ("frames", "sources", "tracks"):
            pd.read_parquet(C.OUT / mission / f"{t}.parquet", filters=[("stack_id", "==", sid)]).to_parquet(m / f"{t}.parquet", index=False)
        for t in ("injected_objects", "injected_truth"):
            d = pd.read_parquet(C.OUT / mission / f"{t}.parquet", filters=[("stack_id", "==", sid)])
            d[d.realization_id.isin(REALIZATIONS)].to_parquet(m / f"{t}.parquet", index=False)

    full = pd.read_parquet(C.OUT / "index.parquet")
    n_full = full[full.sample_type == "real_stack"].groupby("mission").size().to_dict()
    (out / "SAMPLE.md").write_text(f"""# Sample of the Planet X / SPHEREx moving-object dataset

This is a small cut of the full dataset, in the same layout, so the code you write against it runs unchanged on the full one.

What is here: the field of the brown dwarf WISE 0855 as seen by SPHEREx (60 frames over a year, 6 detectors) and by WISE (84 exposures over ten years), with every label for those two stacks, one fixed synthetic-mover realisation for each, the per-frame PSFs, the documentation, the tools and the QC figures.

The full dataset has {n_full.get('spherex', 0)} SPHEREx stacks, {n_full.get('wise', 0)} WISE stacks, {n_full.get('comet', 0)} stacks of comet 3I/ATLAS and {n_full.get('neossat', 0)} NEOSSat stacks, and {int((full.sample_type == 'injected').sum())} synthetic realisations. The numbers in README.md describe the full dataset, not this sample.

## Try it

```python
import sys; sys.path.insert(0, ".")          # run from this folder
from tools.loader import Dataset
ds = Dataset(".")
s = ds.load("spherex:wise0855")               # QC-passed frames only
print(s["image"].shape)                       # (frames, 128, 128)
target = s["sources"][s["sources"].is_target] # the brown dwarf: class, motion_px, frames detected
track = s["tracks"][s["tracks"].src_id.isin(target.src_id)]   # its position and flux in every frame
inj = ds.load_injected("spherex:wise0855:r0") # same frames with 10 synthetic objects
print(inj["injected_objects"][["class", "snr_target", "motion_px"]])
aug = ds.inject("spherex:wise0855", seed=1)   # a new random set, for training
```

Needs numpy, pandas, pyarrow and scipy.

Read README.md first, then DATA_DICTIONARY.md for every column and unit. CITATIONS.md lists the data sources and their terms.
""", encoding="utf-8")

    zpath = C.ROOT / f"{NAME}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                z.write(p, Path(NAME) / p.relative_to(out))
    with zipfile.ZipFile(zpath) as z:
        assert z.testzip() is None
        n = len(z.namelist())
    print(f"{zpath.name}: {zpath.stat().st_size / 1e6:.1f} MB, {n} files")


if __name__ == "__main__":
    main()
