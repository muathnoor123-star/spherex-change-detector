"""Stage 7: run the classical baseline on real stacks and the injection benchmark.

usage: python s07_baseline.py spherex|wise

Real stacks: recall of known solar-system objects and known movers, and the rate of
unexplained detections (an upper bound on false alarms: some may be real, unknown
transients, and they are kept as review candidates).
Injected stacks: completeness versus true S/N and object class, and false
detections per frame that are new relative to the uninjected stack.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "dataset" / "tools"))
import config as C
from baseline import detect

MATCH_PX = 2.0
SNR_BINS = [0, 3, 5, 10, 20, 50, 1e9]


def load_stack(mission, field, stack_id):
    out = C.OUT / mission
    z = np.load(out / "stacks" / f"{field}.npz")
    fr = pd.read_parquet(out / "frames.parquet", filters=[("stack_id", "==", stack_id)]).sort_values("frame_index")
    idx = np.where(fr.qc_pass.to_numpy())[0]
    band_col = "detector" if mission == "spherex" else "band"
    info = {"band": fr[band_col].to_numpy()[idx], "mjd": fr.mjd_mid.to_numpy()[idx], "tg": fr.time_group.to_numpy()[idx],
            "wave": fr.wave_center_um.to_numpy()[idx] if mission == "spherex" else None}
    return z, idx, info


def run_dets(image, sigma, valid, info):
    return pd.DataFrame(detect(image, sigma, valid, info["band"], info["mjd"], info["tg"], wave=info["wave"]))


def match(dets, pts_by_frame):
    """pts_by_frame: frame -> array [k,2]. Returns per-detection matched flag."""
    m = np.zeros(len(dets), bool)
    for f, g in dets.groupby("frame") if len(dets) else []:
        p = pts_by_frame.get(f)
        if p is None or len(p) == 0:
            continue
        d, _ = cKDTree(p).query(g[["x", "y"]].to_numpy())
        m[g.index] = d <= MATCH_PX
    return m


def real_task(args):
    mission, field, stack_id = args
    z, idx, info = load_stack(mission, field, stack_id)
    if len(idx) < 3:
        return stack_id, None, None
    dets = run_dets(z["image"][idx], z["sigma"][idx], z["valid"][idx], info)
    out = C.OUT / mission
    tr = pd.read_parquet(out / "tracks.parquet", filters=[("stack_id", "==", stack_id)])
    src = pd.read_parquet(out / "sources.parquet", filters=[("stack_id", "==", stack_id)])
    pos = {k: i for i, k in enumerate(idx)}
    tr = tr[tr.frame_index.isin(idx)].copy()
    tr["f"] = tr.frame_index.map(pos)
    # asteroids: blend-free difference S/N (raw S/N where no other epoch exists)
    snr_eff = np.where(tr.src_kind == "sso", np.where(np.isfinite(tr.snr_diff), tr.snr_diff, tr.snr), tr.snr)
    tr["vis"] = (np.nan_to_num(snr_eff) >= C.DETECTION_SNR) & (tr.aperture_valid_frac >= 0.95)
    off = np.hypot(tr.centroid_dx, tr.centroid_dy)
    clean = ~tr.blended.astype(bool) & ~tr.artifact_vetoes_mover & ~(off > 1.5)
    tr.loc[tr.src_kind == "sso", "vis"] &= clean[tr.src_kind == "sso"]
    sso_ids = set(src.loc[src.kind == "sso", "src_id"])
    mover_ids = set(src.loc[src.mover_detectable.fillna(False) & (src.kind != "sso"), "src_id"])
    known = {f: g[["x", "y"]].to_numpy() for f, g in tr[tr.vis | tr.saturated].groupby("f")}   # any saturation explains a peak
    res = {"stack_id": stack_id, "n_frames": len(idx), "n_detections": len(dets)}
    for name, ids in (("sso", sso_ids), ("mover", mover_ids)):
        t = tr[tr.src_id.isin(ids) & tr.vis]
        res[f"{name}_visible"] = len(t)
        if len(t) and len(dets):
            dd = {f: g[["x", "y"]].to_numpy() for f, g in dets.groupby("frame")}
            hit = [f in dd and cKDTree(dd[f]).query([x, y])[0] <= MATCH_PX for f, x, y in zip(t.f, t.x, t.y)]
            res[f"{name}_recovered"] = int(np.sum(hit))
        else:
            res[f"{name}_recovered"] = 0
    if len(dets):
        dets["explained"] = match(dets, known)
        res["unexplained"] = int((~dets.explained).sum())
        res["unexplained_coincident"] = int((~dets.explained & dets.coincident).sum())
        dets["stack_id"] = stack_id
        dets["frame_index"] = idx[dets.frame.to_numpy()]
    else:
        res["unexplained"] = res["unexplained_coincident"] = 0
    return stack_id, res, dets


def inj_task(args):
    mission, field, stack_id, r = args
    z, idx, info = load_stack(mission, field, stack_id)
    zi = np.load(C.OUT / mission / "injected" / f"{field}-r{r}.npz")
    valid = z["valid"][idx]
    base = run_dets(z["image"][idx], z["sigma"][idx], valid, info)
    dets = run_dets(zi["image"], zi["sigma"], zi["valid"] if "valid" in zi.files else valid, info)
    truth = pd.read_parquet(C.OUT / mission / "injected_truth.parquet",
                            filters=[("realization_id", "==", f"{stack_id}:r{r}")])
    objs = pd.read_parquet(C.OUT / mission / "injected_objects.parquet",
                           filters=[("realization_id", "==", f"{stack_id}:r{r}")])
    pos = {k: i for i, k in enumerate(idx)}
    truth["f"] = truth.frame_index.map(pos)
    # saturated injections are scored apart: their core is blanked, only the wings remain
    tv = truth[truth.present & truth.in_grid & (truth.masked_frac.fillna(1) < 0.5) & ~truth.saturated].copy()
    tv = tv.merge(objs[["obj_id", "class", "is_mover"]], on="obj_id")
    # what a change detector should flag: movers (>= 1 px), transients and variables.
    # Static sources and sub-pixel proper-motion stars are negatives.
    tv["should_change"] = tv["class"].isin(["transient", "variable"]) | tv.is_mover
    dd = {f: g[["x", "y"]].to_numpy() for f, g in dets.groupby("frame")} if len(dets) else {}
    tv["found"] = [f in dd and cKDTree(dd[f]).query([x, y])[0] <= MATCH_PX for f, x, y in zip(tv.f, tv.x, tv.y)]
    fp = 0
    if len(dets):
        inj_pts = {f: g[["x", "y"]].to_numpy() for f, g in truth[truth.present & truth.in_grid].groupby("f")}
        base_pts = {f: g[["x", "y"]].to_numpy() for f, g in base.groupby("frame")} if len(base) else {}
        new = ~match(dets, base_pts)
        # very bright or saturated injections leave wings and rings well beyond 2 px:
        # detections within 8 px of one belong to it and are not false alarms
        tb = truth[truth.present & truth.in_grid & ((truth.snr_true > 100) | truth.saturated)]
        near_bright = np.zeros(len(dets), bool)
        for f, g in dets.groupby("frame"):
            b = tb[tb.f == f]
            if len(b):
                d, _ = cKDTree(b[["x", "y"]].to_numpy()).query(g[["x", "y"]].to_numpy())
                near_bright[g.index] = d <= 8.0
        fp = int((new & ~match(dets, inj_pts) & ~near_bright).sum())
    tv["realization_id"] = f"{stack_id}:r{r}"
    return tv[["realization_id", "obj_id", "frame_index", "class", "is_mover", "should_change", "snr_true", "found"]], fp, len(idx)


def main(mission):
    out = C.OUT / mission
    meta = json.loads((out / "stacks.json").read_text())
    stacks = [(mission, v["field"], k) for k, v in meta.items()]
    objs = pd.read_parquet(out / "injected_objects.parquet")
    reals = sorted(objs.realization_id.unique())
    inj = [(mission, meta[r.rsplit(":", 1)[0]]["field"], r.rsplit(":", 1)[0], int(r.rsplit(":r", 1)[1])) for r in reals]
    with ProcessPoolExecutor(C.WORKERS) as ex:
        real = list(ex.map(real_task, stacks))
        injr = list(ex.map(inj_task, inj))
    bdir = C.OUT / "baseline"
    bdir.mkdir(parents=True, exist_ok=True)
    rs = pd.DataFrame([r for _, r, _ in real if r])
    cand = pd.concat([d[~d.explained] for _, _, d in real if d is not None and len(d)], ignore_index=True)
    rs.to_csv(bdir / f"{mission}-real-stacks.csv", index=False)
    cand.to_parquet(bdir / f"{mission}-unexplained-detections.parquet", index=False)
    tv = pd.concat([t for t, _, _ in injr], ignore_index=True)
    tv["snr_bin"] = pd.cut(tv.snr_true, SNR_BINS, right=False)
    pos = tv[tv.should_change]
    neg = tv[~tv.should_change]
    comp = pos.groupby("snr_bin", observed=True).found.agg(["mean", "size"]).rename(columns={"mean": "completeness", "size": "n"})
    comp.index = comp.index.astype(str)
    comp_cls = pos[pos.snr_true >= 10].groupby("class").found.agg(["mean", "size"]).rename(columns={"mean": "completeness_snr10", "size": "n"})
    static_fa = neg[neg.snr_true >= 10].groupby("class").found.agg(["mean", "size"]).rename(columns={"mean": "flagged_rate_snr10", "size": "n"})
    comp.to_csv(bdir / f"{mission}-completeness-by-snr.csv")
    comp_cls.to_csv(bdir / f"{mission}-completeness-by-class.csv")
    static_fa.to_csv(bdir / f"{mission}-static-flag-rate.csv")
    nfr = sum(n for _, _, n in injr)
    # held-out numbers: the default test fold only
    gf = {g: f for f, gs in C.FOLDS.items() for g in gs}
    test_stacks = {sid for sid, v in meta.items() if gf[v["group"]] in C.DEFAULT_SPLIT["test"]}
    pt = pos[pos.realization_id.str.rsplit(":", n=1).str[0].isin(test_stacks)]
    nt = neg[neg.realization_id.str.rsplit(":", n=1).str[0].isin(test_stacks)]
    ct = pt.groupby("snr_bin", observed=True).found.agg(["mean", "size"]).rename(columns={"mean": "completeness", "size": "n"})
    ct.index = ct.index.astype(str)
    cct = pt[pt.snr_true >= 10].groupby("class").found.agg(["mean", "size"]).rename(columns={"mean": "completeness_snr10", "size": "n"})
    sft = nt[nt.snr_true >= 10].groupby("class").found.agg(["mean", "size"]).rename(columns={"mean": "flagged_rate_snr10", "size": "n"})
    test_fp = [(f, n) for (m_, fld, sid, r), (_, f, n) in zip(inj, injr) if sid in test_stacks]
    rt = rs[rs.stack_id.isin(test_stacks)]
    test = {"stacks": sorted(test_stacks), "completeness_by_snr": ct.reset_index().to_dict("records"),
            "completeness_snr_ge_10_by_class": cct.reset_index().to_dict("records"),
            "static_sources_flagged_snr_ge_10": sft.reset_index().to_dict("records"),
            "new_false_detections_per_frame": float(sum(f for f, _ in test_fp) / max(sum(n for _, n in test_fp), 1)),
            "real_known_mover_visible": int(rt.mover_visible.sum()), "real_known_mover_recovered": int(rt.mover_recovered.sum()),
            "real_unexplained_per_frame": float(rt.unexplained.sum() / max(rt.n_frames.sum(), 1))}
    summary = {
        "mission": mission, "detector": "dataset/tools/baseline.py (median-reference difference imaging, 5 sigma)",
        "real": {"frames": int(rs.n_frames.sum()), "detections": int(rs.n_detections.sum()),
                 "known_sso_visible": int(rs.sso_visible.sum()), "known_sso_recovered": int(rs.sso_recovered.sum()),
                 "known_mover_visible": int(rs.mover_visible.sum()), "known_mover_recovered": int(rs.mover_recovered.sum()),
                 "unexplained_per_frame": float(rs.unexplained.sum() / max(rs.n_frames.sum(), 1)),
                 "unexplained_coincident_per_frame": float(rs.unexplained_coincident.sum() / max(rs.n_frames.sum(), 1))},
        "injected": {"realizations": len(inj), "object_frames_scored": int(len(tv)),
                     "new_false_detections_per_frame": float(sum(f for _, f, _ in injr) / max(nfr, 1)),
                     "completeness_by_snr": comp.reset_index().to_dict("records"),
                     "completeness_snr_ge_10_by_class": comp_cls.reset_index().to_dict("records"),
                     "static_sources_flagged_snr_ge_10": static_fa.reset_index().to_dict("records")},
        "test_fold": test,
    }
    (bdir / f"{mission}-summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main(sys.argv[1])
