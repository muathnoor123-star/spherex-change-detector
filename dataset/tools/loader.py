"""Load stacks, labels and splits from this dataset. Needs numpy and pandas (scipy for injection).

    from tools.loader import Dataset
    ds = Dataset("path/to/dataset")
    for sid in ds.stacks(split="train", mission="spherex"):
        s = ds.load(sid)                    # QC-passed frames only (qc_only=True)
        s["image"].shape                    # [N, 128, 128] float32, background-subtracted
        s["tracks"]                         # per-frame positions + photometry of every known source
        s["sources"]                        # per-source labels (class, motion, detectability)

    # fixed benchmark realisation (for val/test)
    inj = ds.load_injected("spherex:wise0855:r0")
    # unlimited new realisations for training
    aug = ds.inject(sid, seed=123, n_objects=10)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, root):
        self.root = Path(root)
        self.splits = json.loads((self.root / "splits.json").read_text())
        self.index = pd.read_parquet(self.root / "index.parquet")
        self._tables = {}

    def _table(self, mission, name):
        key = (mission, name)
        if key not in self._tables:
            p = self.root / mission / f"{name}.parquet"
            self._tables[key] = pd.read_parquet(p) if p.exists() else pd.DataFrame()
        return self._tables[key]

    def stacks(self, split=None, mission=None, fold=None):
        ix = self.index[self.index.sample_type == "real_stack"]
        if mission:
            ix = ix[ix.mission == mission]
        if fold is not None:
            ix = ix[ix.fold == fold]
        if split:
            ix = ix[ix.fold.isin(self.splits["default_split"][split])]
        return ix.stack_id.tolist()

    def load(self, stack_id, qc_only=True):
        row = self.index[(self.index.stack_id == stack_id) & (self.index.sample_type == "real_stack")].iloc[0]
        mission = row.mission
        z = np.load(self.root / row.path)
        fr = self._table(mission, "frames")
        fr = fr[fr.stack_id == stack_id].sort_values("frame_index")
        keep = fr.qc_pass.to_numpy() if qc_only else np.ones(len(fr), bool)
        idx = np.where(keep)[0]
        out = {k: z[k][idx] for k in ("image", "sigma", "flags", "valid", "wave", "bandwidth")}
        out["valid"] = out["valid"].astype(bool)
        out["frames"] = fr.iloc[idx].reset_index(drop=True)
        tr = self._table(mission, "tracks")
        out["tracks"] = tr[(tr.stack_id == stack_id) & tr.frame_index.isin(fr.frame_index.iloc[idx])]
        src = self._table(mission, "sources")
        out["sources"] = src[src.stack_id == stack_id]
        psf_path = self.root / mission / "psf" / Path(row.path).name
        if psf_path.exists():
            out["psf"] = np.load(psf_path)["psf"][idx]
        out["grid"] = json.loads((self.root / mission / "stacks.json").read_text())[stack_id]["grid"]
        out["frame_index"] = fr.frame_index.to_numpy()[idx]
        return out

    def load_injected(self, realization_id):
        row = self.index[(self.index.sample_type == "injected") & (self.index.realization_id == realization_id)].iloc[0]
        base = self.load(row.stack_id, qc_only=True)
        z = np.load(self.root / row.path)
        assert np.array_equal(z["frame_index"], base["frame_index"]), "QC selection changed since the benchmark was built"
        base["image"], base["sigma"] = z["image"], z["sigma"]
        if "valid" in z.files:   # injected objects bright enough to saturate blank their own core
            base["valid"] = z["valid"].astype(bool)
        objs = self._table(row.mission, "injected_objects")
        truth = self._table(row.mission, "injected_truth")
        base["injected_objects"] = objs[objs.realization_id == realization_id]
        base["injected_truth"] = truth[truth.realization_id == realization_id]
        return base

    def inject(self, stack_id, seed, n_objects=10):
        """A new random realisation on top of the real stack (training augmentation)."""
        try:
            from .injection import inject
        except ImportError:
            from injection import inject
        s = self.load(stack_id)
        fr = s["frames"]
        g = s["grid"]
        frames = {"mjd": fr.mjd_mid.to_numpy(), "time_group": fr.time_group.to_numpy(),
                  "wave_um": fr.wave_center_um.to_numpy(), "obs_xyz": fr[["obs_x", "obs_y", "obs_z"]].to_numpy(),
                  "sky_noise": fr.aperture_sky_noise.to_numpy(), "shot_k": fr.shot_k.to_numpy(),
                  "sat_level": fr.sat_level.to_numpy() if "sat_level" in fr else None}
        grid = {"ra": g["crval"][0], "dec": g["crval"][1], "scale_arcsec": abs(g["cdelt_deg"][1]) * 3600, "npix": g["npix"]}
        img, sig, valid, objs, truth = inject(s, frames, s["psf"], grid, np.random.default_rng(seed), n_objects)
        s["image"], s["sigma"], s["valid"] = img, sig, valid
        s["injected_objects"] = pd.DataFrame(objs)
        t = pd.DataFrame(truth)
        t["frame_index"] = s["frame_index"][t.frame_index.to_numpy()]
        s["injected_truth"] = t
        return s
