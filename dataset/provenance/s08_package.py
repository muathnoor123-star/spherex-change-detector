"""Stage 8: index, splits, leakage checks, statistics, figures, exclusions and checksums."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from common import json_dump

MISSIONS = ["spherex", "wise", "comet", "neossat"]


def group_fold():
    gf = {}
    for f, groups in C.FOLDS.items():
        for g in groups:
            assert g not in gf, f"group {g} in two folds"
            gf[g] = f
    return gf


def split_of(fold):
    for name, folds in C.DEFAULT_SPLIT.items():
        if fold in folds:
            return name
    return None


def build_index(gf):
    rows = []
    for m in MISSIONS:
        p = C.OUT / m / "stacks.json"
        if not p.exists():
            continue
        meta = json.loads(p.read_text())
        fr = pd.read_parquet(C.OUT / m / "frames.parquet")
        src = pd.read_parquet(C.OUT / m / "sources.parquet")
        for sid, v in meta.items():
            g = v["group"]
            assert g in gf, f"group {g} of {sid} has no fold"
            f = fr[fr.stack_id == sid]
            s = src[src.stack_id == sid]
            base = {"mission": m, "stack_id": sid, "field": v["field"], "group": g, "fold": gf[g],
                    "split": split_of(gf[g]), "n_frames": len(f), "n_frames_qc": int(f.qc_pass.sum()),
                    "n_time_groups_qc": int(f.loc[f.qc_pass, "time_group"].nunique()),
                    "baseline_days": float(np.ptp(f.loc[f.qc_pass, "mjd_mid"])) if f.qc_pass.any() else 0.0,
                    "n_sources": len(s), "n_real_movers": int(s.mover_detectable.fillna(False).sum()),
                    "n_sso_visible": int(((s.kind == "sso") & (s.n_frames_detected > 0)).sum()),
                    "is_control": bool(v.get("is_control", False))}
            rows.append({**base, "sample_type": "real_stack", "realization_id": None,
                         "path": f"{m}/stacks/{v['field']}.npz"})
            cp = C.OUT / m / "stacks" / f"{v['field']}-coadd.npz"
            if cp.exists():
                rows.append({**base, "sample_type": "epoch_coadd", "realization_id": None,
                             "path": f"{m}/stacks/{v['field']}-coadd.npz"})
        op = C.OUT / m / "injected_objects.parquet"
        if op.exists():
            objs = pd.read_parquet(op)
            for rid, o in objs.groupby("realization_id"):
                sid = o.stack_id.iat[0]
                v = meta[sid]
                r = int(rid.rsplit(":r", 1)[1])
                rows.append({"mission": m, "stack_id": sid, "field": v["field"], "group": v["group"],
                             "fold": gf[v["group"]], "split": split_of(gf[v["group"]]), "sample_type": "injected",
                             "realization_id": rid, "path": f"{m}/injected/{v['field']}-r{r}.npz",
                             "n_injected": len(o), "n_injected_movers": int(o.is_mover.sum())})
    return pd.DataFrame(rows)


def leakage_checks(ix):
    """Different physical groups must not share sky or source files."""
    report = {"overlapping_footprints_different_groups": [], "near_pairs_within_1deg_different_groups": [],
              "shared_source_files_across_groups": []}
    cen = []
    for m in MISSIONS:
        p = C.OUT / m / "stacks.json"
        if p.exists():
            for sid, v in json.loads(p.read_text()).items():
                g = v["grid"]
                half_diag = g["npix"] * abs(g["cdelt_deg"][1]) / np.sqrt(2)
                cen.append((sid, v["group"], *g["crval"], half_diag))
    ra = np.radians([c[2] for c in cen])
    de = np.radians([c[3] for c in cen])
    rad = np.array([c[4] for c in cen])
    for i in range(len(cen)):
        cosd = np.sin(de[i]) * np.sin(de) + np.cos(de[i]) * np.cos(de) * np.cos(ra - ra[i])
        sep = np.degrees(np.arccos(np.clip(cosd, -1, 1)))
        for j in np.where(sep < 1.0)[0]:
            if j > i and cen[i][1] != cen[j][1]:
                rec = {"a": cen[i][0], "b": cen[j][0], "sep_deg": float(sep[j]),
                       "sum_half_diagonals_deg": float(rad[i] + rad[j])}
                key = "overlapping_footprints_different_groups" if sep[j] < rad[i] + rad[j] else "near_pairs_within_1deg_different_groups"
                report[key].append(rec)
    files = []
    for m in MISSIONS:
        p = C.OUT / m / "frames.parquet"
        if p.exists():
            f = pd.read_parquet(p)
            col = "sha256" if "sha256" in f else None
            if col:
                g = ix[ix.sample_type == "real_stack"].set_index("stack_id").group
                files.append(pd.DataFrame({"sha": f[col], "group": f.stack_id.map(g)}))
    if files:
        fa = pd.concat(files).dropna()
        multi = fa.groupby("sha").group.nunique()
        report["shared_source_files_across_groups"] = multi[multi > 1].index.tolist()
    return report


EXCLUDED = [
    ("wise/images/ (44 IRSA server cutouts of Epsilon Indi B)", "Same exposures as the full NEOWISE frames used here; the full frames carry the uncertainty and mask planes."),
    ("gaia/*-dr3-context.csv (10 bounded tables)", "Four were cut at 5,000 rows. Replaced by complete Gaia DR3 queries (no row limit) with DSC and variability classes."),
    ("gaia/extras/ (XP and RVS spectra, epoch photometry, AP tables)", "Not needed for motion labels; class probabilities come from the new Gaia query. Kept in the source folder for spectral work."),
    ("wise/unwise/ (unWISE neo7 time-domain catalogues)", "Detection catalogues, not images. Suitable for a separate tabular track-linking task; not used here."),
    ("spherex/spectral-context/ (12 DR1 HEALPix cubes)", "About 14 arcsec per pixel against 6.15 arcsec in the frames, and built from the same exposures, so not a valid independent subtraction template."),
    ("spherex/calibration/ (87 products)", "Already applied in the Level-2 frames."),
    ("spherex-context/ (SPLICES, spectral calibration tar, M101 cube)", "Selection-biased ice-candidate catalogue and laboratory calibration; no motion information."),
    ("neossat frames without a celestial WCS (214: all 192 of 1999 KW4, all 22 of the Tabby field)", "Cannot be placed on the sky without plate solving; listed in neossat/excluded-frames.csv."),
    ("asteroid positions from IMCCE SkyBoT (used in v1.0)", "Not a NASA or listed partner source and its reuse terms were not confirmed; removed in v1.1."),
    ("frames failing QC", "Kept in the stack arrays with qc_pass = False and the reason in qc_reasons; the loader drops them by default."),
]


def figures(ix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    qdir = C.OUT / "qc"
    qdir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8})

    # 1. every SPHEREx field: median of QC frames, target track, known asteroids
    meta = json.loads((C.OUT / "spherex" / "stacks.json").read_text())
    tr = pd.read_parquet(C.OUT / "spherex" / "tracks.parquet")
    src = pd.read_parquet(C.OUT / "spherex" / "sources.parquet")
    fr = pd.read_parquet(C.OUT / "spherex" / "frames.parquet")
    fig, axes = plt.subplots(3, 4, figsize=(11, 8.6))
    for ax, (sid, v) in zip(axes.ravel(), meta.items()):
        z = np.load(C.OUT / "spherex" / "stacks" / f"{v['field']}.npz")
        q = fr[fr.stack_id == sid].sort_values("frame_index").qc_pass.to_numpy()
        im = np.where(z["valid"][q].astype(bool), z["image"][q], np.nan)
        med = np.nanmedian(im, axis=0)
        lo, hi = np.nanpercentile(med, [5, 99.5])
        ax.imshow(med, origin="lower", cmap="gray_r", vmin=lo, vmax=hi)
        s = src[src.stack_id == sid]
        tid = s.loc[s.is_target.fillna(False), "src_id"]
        t = tr[(tr.stack_id == sid) & tr.src_id.isin(tid)]
        ax.plot(t.x, t.y, ".", color="#c0392b", ms=2)
        sso = tr[(tr.stack_id == sid) & (tr.src_kind == "sso") & tr.x.between(0, 127) & tr.y.between(0, 127)]
        ax.plot(sso.x, sso.y, "+", color="#1f77b4", ms=4, mew=0.7)
        ax.set_title(f"{v['field']}  ({int(q.sum())} QC frames, {s.kind.eq('sso').sum()} known SSOs)")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("The first 12 SPHEREx fields: median of QC-passed frames. Red: target position in each frame.", fontsize=9)
    fig.tight_layout()
    fig.savefig(qdir / "spherex-fields.png", dpi=110)
    plt.close(fig)

    # 2. Barnard's Star across ten years of WISE W2 epoch coadds
    zc = np.load(C.OUT / "wise" / "stacks" / "barnard-coadd.npz")
    keys = zc["key"]
    w2 = [i for i, k in enumerate(keys) if k.endswith("W2")]
    pick = [w2[0], w2[len(w2) // 3], w2[2 * len(w2) // 3], w2[-1]]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.1))
    for ax, i in zip(axes, pick):
        im = np.where(zc["valid"][i].astype(bool), zc["image"][i], np.nan)
        lo, hi = np.nanpercentile(im, [5, 99.5])
        ax.imshow(im, origin="lower", cmap="gray_r", vmin=lo, vmax=hi)
        ax.set_title(str(keys[i]).split(":")[0])
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("WISE W2 half-year coadds around Barnard's Star (2.75\"/px, north up): the saturated star moves ~38 px", fontsize=9)
    fig.tight_layout()
    fig.savefig(qdir / "wise-barnard-epochs.png", dpi=110)
    plt.close(fig)

    # 3. registration precision and QC outcome
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for m, c in (("spherex", "#2c7fb8"), ("wise", "#d95f0e"), ("neossat", "#31a354"), ("comet", "#756bb1")):
        p = C.OUT / m / "frames.parquet"
        if not p.exists():
            continue
        f = pd.read_parquet(p)
        e = f.reg_shift_err_px.dropna()
        axes[0].hist(np.clip(e, 0, 0.2), bins=40, histtype="step", color=c, label=f"{m} (n={len(e)})")
        r = f.loc[~f.qc_pass, "qc_reasons"].str.split(";").explode().value_counts()
        axes[1].barh([f"{m}: {k}" for k in r.index], r.values, color=c)
    axes[0].axvline(C.QC["max_registration_error_px"], color="k", lw=0.8, ls="--")
    axes[0].set_xlabel("uncertainty of applied registration shift (px)")
    axes[0].legend(frameon=False)
    axes[1].set_xlabel("frames rejected")
    fig.tight_layout()
    fig.savefig(qdir / "registration-and-qc.png", dpi=110)
    plt.close(fig)

    # 4. one injected frame with its truth
    rid = ix.loc[(ix.sample_type == "injected") & (ix.mission == "spherex"), "realization_id"].iloc[0]
    row = ix[ix.realization_id == rid].iloc[0]
    zi = np.load(C.OUT / row.path)
    zb = np.load(C.OUT / "spherex" / "stacks" / f"{row.field}.npz")
    truth = pd.read_parquet(C.OUT / "spherex" / "injected_truth.parquet", filters=[("realization_id", "==", rid)])
    objs = pd.read_parquet(C.OUT / "spherex" / "injected_objects.parquet", filters=[("realization_id", "==", rid)])
    k = 0
    fi = int(zi["frame_index"][k])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    a, b = zb["image"][fi], zi["image"][k]
    lo, hi = np.percentile(a, [5, 99.5])
    for ax, im, ttl in ((axes[0], a, "real frame"), (axes[1], b, "with injections"), (axes[2], b - a, "injected signal + photon noise")):
        ax.imshow(im, origin="lower", cmap="gray_r", vmin=lo if ttl != "injected signal + photon noise" else None,
                  vmax=hi if ttl != "injected signal + photon noise" else None)
        ax.set_title(ttl)
        ax.set_xticks([])
        ax.set_yticks([])
    t = truth[(truth.frame_index == fi) & truth.present & truth.in_grid & (truth.masked_frac < 0.5)].merge(
        objs[["obj_id", "class"]].rename(columns={"class": "cls"}), on="obj_id")
    for r in t.itertuples():
        axes[1].plot(r.x, r.y, "o", mfc="none", color="#c0392b", ms=9, mew=0.8)
        axes[1].annotate(f"{r.cls} S/N {r.snr_true:.0f}", (r.x, r.y), xytext=(4, 4),
                         textcoords="offset points", color="#c0392b", fontsize=6)
    fig.suptitle(f"{rid}, frame {fi}", fontsize=9)
    fig.tight_layout()
    fig.savefig(qdir / "injection-example.png", dpi=110)
    plt.close(fig)


def stats(ix):
    out = {"dataset_version": C.DATASET_VERSION, "missions": {}}
    for m in MISSIONS:
        p = C.OUT / m / "frames.parquet"
        if not p.exists():
            continue
        f = pd.read_parquet(p)
        s = pd.read_parquet(C.OUT / m / "sources.parquet")
        d = {"stacks": int((ix.mission.eq(m) & ix.sample_type.eq("real_stack")).sum()),
             "frames": len(f), "frames_qc_pass": int(f.qc_pass.sum()),
             "qc_rejections": f.loc[~f.qc_pass, "qc_reasons"].str.split(";").explode().value_counts().to_dict(),
             "registration_shift_err_px_median": float(f.reg_shift_err_px.median()),
             "registration_star_scatter_px_median": float(f.reg_rms.median()),
             "source_rows": len(s), "classes": s["class"].value_counts().to_dict(),
             "real_movers_detectable": int(s.mover_detectable.fillna(False).sum()),
             "sso_in_grid": int((s.kind == "sso").sum()),
             "sso_detected_snr5": int(((s.kind == "sso") & (s.n_frames_detected > 0)).sum())}
        if "psf_source" in f:
            d["psf_source"] = f.psf_source.value_counts().to_dict()
            d["psf_fwhm_px_median"] = float(f.psf_fwhm_px.median())
        op = C.OUT / m / "injected_objects.parquet"
        if op.exists():
            o = pd.read_parquet(op)
            d["injected_realizations"] = int(o.realization_id.nunique())
            d["injected_objects"] = o["class"].value_counts().to_dict()
        out["missions"][m] = d
    out["samples_by_split"] = ix.groupby(["split", "mission", "sample_type"]).size().reset_index(name="n").to_dict("records")
    return out


def main():
    gf = group_fold()
    ix = build_index(gf)
    ix.to_parquet(C.OUT / "index.parquet", index=False)
    splits = {"folds": {str(k): v for k, v in C.FOLDS.items()}, "default_split": C.DEFAULT_SPLIT,
              "group_to_fold": gf, "rule": "All products of one physical group (every mission, its offset control "
              "field, every derived and injected sample) share one fold. Use the folds for 5-fold cross-validation "
              "or the default split (test fold 0, validation fold 1)."}
    json_dump(splits, C.OUT / "splits.json")
    leak = leakage_checks(ix)
    json_dump(leak, C.OUT / "qc" / "leakage-checks.json")
    print("leakage:", {k: len(v) for k, v in leak.items()})
    json_dump([{"item": a, "reason": b} for a, b in EXCLUDED], C.OUT / "excluded.json")
    st = stats(ix)
    json_dump(st, C.OUT / "qc" / "summary.json")
    figures(ix)
    prov = C.OUT / "provenance"
    prov.mkdir(exist_ok=True)
    for p in Path(__file__).parent.glob("*.py"):
        shutil.copy(p, prov / p.name)
    lines = []
    for p in sorted(C.OUT.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.sha256":
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            lines.append(f"{h}  {p.relative_to(C.OUT).as_posix()}")
    (C.OUT / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    print(json.dumps(st, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
