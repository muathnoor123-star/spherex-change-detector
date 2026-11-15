"""Stage 4: aligned image stacks with per-frame quality control and per-source photometry.

usage: python s04_stacks.py spherex|wise

Outputs in dataset/<mission>/:
  stacks/<field>.npz     image, sigma, flags, valid, wave, bandwidth  [N, H, W]
  frames.parquet         one row per frame: time, detector/band, registration, QC verdict
  sources.parquet        one row per (stack, source): astrometry, class labels, motion, detectability
  tracks.parquet         one row per (frame, source): predicted position and forced photometry
  stacks.json            grid WCS and provenance per stack
WISE also gets stacks/<field>-coadd.npz with half-year epoch coadds.
"""
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from common import earth_bary_au, json_dump
from s03_ephemeris import SAME_TIME_DAYS, time_groups
from sources import build_field_sources


def build_sources(mission, fields, targets):
    out = {}
    d = C.CACHE / "sources"
    d.mkdir(parents=True, exist_ok=True)
    for key, f in fields.items():
        if f["mission"] != mission:
            continue
        slug = f["slug"]
        t = targets[mission][slug]
        ck = f["catalog_key"]
        gaia = pd.read_parquet(C.CACHE / "catalogs" / f"{ck}-gaia.parquet")
        aw = pd.read_parquet(C.CACHE / "catalogs" / f"{ck}-allwise.parquet")
        src = build_field_sources({**f, "key": key}, t, gaia, aw)
        p = d / f"{mission}-{slug}.parquet"
        src.to_parquet(p, index=False)
        out[slug] = p
        print(key, "sources", len(src), src["class"].value_counts().to_dict(), "target rows", int(src.is_target.sum()))
    return out


SSO_COLS = ["field", "src_id", "mjd_utc", "ra_obs", "dec_obs", "ra_rate_arcsec_h", "dec_rate_arcsec_h",
            "vmag", "type", "posunc_arcsec"]


def sso_table(mission):
    if not C.USE_SKYBOT:
        return pd.DataFrame({c: pd.Series(dtype="object" if c in ("field", "src_id", "type") else "float64") for c in SSO_COLS})
    sk = pd.read_parquet(C.CACHE / "ephem" / "skybot_observer.parquet")
    sk = sk[sk.query_key.str.startswith(mission + ":")].copy()
    sk["field"] = sk.query_key.str.split(":").str[1]
    num = sk.number.astype(str)
    sk["src_id"] = "sso:" + np.where(num.isin(["--", "", "nan", "None", "0"]), sk.name.astype(str), num)
    return sk


def frame_specs(mission, frames, fields, src_paths, sso):
    specs = []
    for slug, g in frames.groupby("field"):
        f = fields[f"{mission}:{slug}"]
        tmp = C.CACHE / "frames" / mission / slug
        tmp.mkdir(parents=True, exist_ok=True)
        s_f = sso[sso.field == slug]
        for i, r in enumerate(g.itertuples()):
            dt = np.abs(s_f.mjd_utc - r.mjd_mid)
            near = s_f[dt < SAME_TIME_DAYS].assign(_dt=dt[dt < SAME_TIME_DAYS])
            near = near.sort_values("_dt").drop_duplicates("src_id").drop(columns="_dt")
            spec = {"mission": mission, "stack_id": f"{mission}:{slug}", "frame_uid": f"{mission}:{slug}:{Path(r.path).name}",
                    "path": r.path, "mjd": float(r.mjd_mid), "grid_ra": f["ra"], "grid_dec": f["dec"],
                    "sources_path": str(src_paths[slug]), "sso": near if len(near) else None,
                    "out": str(tmp / f"{Path(r.path).stem}.npz")}
            if mission == "wise":
                spec.update(unc_path=r.unc_path, msk_path=r.msk_path, magzp=float(r.magzp), band=int(r.band))
            specs.append(spec)
    return specs


def qc_verdict(fr, mission):
    q = C.QC
    reasons = []
    if fr.get("read_error"):
        return False, "read_error"
    if fr["valid_frac"] < q["min_valid_fraction"]:
        reasons.append("low_coverage")
    # The target region must have been observed. Saturation there is not a reason to
    # drop the frame: a saturated target is plainly visible and is labelled as such.
    if fr["center_covered_frac"] < q["min_center_valid_fraction"]:
        reasons.append("target_region_not_covered")
    if fr["reg_status"] != "ok":
        reasons.append("registration_" + fr["reg_status"])
    elif (fr["reg_rms"] > q["max_registration_rms_px"]
          or 1.2533 * fr["reg_rms"] / np.sqrt(fr["reg_nref"]) > q["max_registration_error_px"]):
        reasons.append("registration_uncertain")
    if fr["noise_ratio"] > q["max_noise_ratio"]:
        reasons.append("noisy")
    return (len(reasons) == 0), ";".join(reasons)


def difference_photometry(tracks, fr, arrays, mission, src, min_gap_days=1.0):
    """For solar-system objects: flux at the predicted position minus the median flux
    at the same sky position in QC-passed frames of the same detector/band taken more
    than `min_gap_days` away, when the object was elsewhere. A static star under the
    predicted position cancels, so a blend is not mistaken for a detection."""
    tracks = tracks.assign(diff_flux=np.nan, diff_err=np.nan, snr_diff=np.nan, blended=False)
    sso = tracks.index[tracks.src_kind == "sso"]
    if len(sso) == 0:
        return tracks
    # blended: a catalogued source within 2.5 px whose median S/N over the whole stack
    # is >= 5 (its usual brightness; a passing object cannot inflate the median).
    # SPHEREx wavelength changes between frames, so such a star does not cancel exactly.
    from scipy.spatial import cKDTree
    # Stacks with fewer than three exposure times (single-epoch comet stacks) contain
    # the moving object in every frame, so there the catalogue brightness decides.
    allcat = tracks[tracks.src_kind == "catalog"]
    if fr.loc[fr.qc_pass, "time_group"].nunique() >= 3:
        med = allcat.groupby("src_id").snr.median()
        bright = set(med.index[med >= C.DETECTION_SNR])
    else:
        bright = set(src.loc[(src.gmag < 17) | (src.w1 < 14.5), "src_id"])
    cat = allcat[allcat.src_id.isin(bright)]
    trees = {f: cKDTree(g[["x", "y"]].to_numpy()) for f, g in cat.groupby("frame_index")}
    for k in sso:
        tr_ = trees.get(int(tracks.at[k, "frame_index"]))
        if tr_ is not None and tr_.query([tracks.at[k, "x"], tracks.at[k, "y"]])[0] <= 2.5:
            tracks.at[k, "blended"] = True
    from common import aperture_photometry
    rap, rin, rout = C.APERTURE["spherex" if mission == "comet" else mission]
    band_col = "detector" if mission in ("spherex", "comet") else "band"
    img = np.stack(arrays["image"])
    sig = np.stack(arrays["sigma"])
    val = np.stack(arrays["valid"]).astype(bool)
    bands = fr[band_col].to_numpy()
    mjd = fr.mjd_mid.to_numpy()
    q = fr.qc_pass.to_numpy()
    for k in sso:
        i = int(tracks.at[k, "frame_index"])
        x, y = float(tracks.at[k, "x"]), float(tracks.at[k, "y"])
        refs = np.where(q & (bands == bands[i]) & (np.abs(mjd - mjd[i]) > min_gap_days))[0]
        if len(refs) < 2:
            continue
        rf = []
        for j in refs:
            f, _, _, vf = aperture_photometry(img[j], sig[j] ** 2, val[j], np.array([x]), np.array([y]), rap, rin, rout)
            if np.isfinite(f[0]) and vf[0] >= 0.95:
                rf.append(f[0])
        if len(rf) < 2:
            continue
        ref = float(np.median(rf))
        e = float(tracks.at[k, "flux_err"])
        derr = e * np.sqrt(1 + 1.5708 / len(rf))
        tracks.at[k, "diff_flux"] = tracks.at[k, "flux"] - ref
        tracks.at[k, "diff_err"] = derr
        tracks.at[k, "snr_diff"] = (tracks.at[k, "flux"] - ref) / derr
    return tracks


def sed_residuals(tracks, fr, arrays, k=5, min_points=12):
    """SPHEREx measures every frame at a different wavelength, so a constant source
    still changes flux from frame to frame. For each catalogue source this adds the
    wavelength at its position (`wave_um`), a local spectrum estimate (`flux_sed`:
    the median flux of the k measurements nearest in log-wavelength that were taken
    at a different exposure time) and the deviation from it in units of the error
    plus 3% of the model (`sed_resid_snr`). A real brightness change stands out in
    `sed_resid_snr`; a spectral feature does not."""
    wave = np.stack(arrays["wave"])
    n, H, W = wave.shape
    xi = np.round(tracks.x.to_numpy()).astype(int)
    yi = np.round(tracks.y.to_numpy()).astype(int)
    fi = tracks.frame_index.to_numpy()
    inside = (xi >= 0) & (yi >= 0) & (xi < W) & (yi < H)
    wv = np.full(len(tracks), np.nan)
    wv[inside] = wave[fi[inside], yi[inside], xi[inside]]
    tracks["wave_um"] = wv.astype(np.float32)
    model = np.full(len(tracks), np.nan)
    tg = fr.set_index("frame_index").time_group.reindex(fi).to_numpy()
    qc = fr.set_index("frame_index").qc_pass.reindex(fi).to_numpy()
    flux = tracks.flux.to_numpy(float)
    good = qc & np.isfinite(flux) & np.isfinite(wv) & (tracks.aperture_valid_frac.to_numpy() >= 0.95) \
        & ~tracks.saturated.to_numpy() & (tracks.src_kind.to_numpy() == "catalog")
    order = np.argsort(tracks.src_id.to_numpy(), kind="stable")
    ids = tracks.src_id.to_numpy()[order]
    starts = np.r_[0, np.where(ids[1:] != ids[:-1])[0] + 1, len(ids)]
    for a, b in zip(starts[:-1], starts[1:]):
        rows = order[a:b]
        rows = rows[good[rows]]
        if len(rows) < min_points:
            continue
        lw = np.log(wv[rows])
        d = np.abs(lw[:, None] - lw[None, :])
        d[tg[rows][:, None] == tg[rows][None, :]] = np.inf   # leave out the same exposure time
        nn = np.argsort(d, axis=1)[:, :k]
        model[rows] = np.median(flux[rows][nn], axis=1)
    err = tracks.flux_err.to_numpy(float)
    tracks["flux_sed"] = model.astype(np.float32)
    tracks["sed_resid_snr"] = ((flux - model) / np.sqrt(err ** 2 + (0.03 * model) ** 2)).astype(np.float32)
    return tracks


BRIGHT_G, BRIGHT_W1 = 13.0, 9.5      # bright enough to saturate or to throw a halo
HALO_RADIUS_PX = 4.0


def flag_contamination(tracks, src):
    """Catalogue sources whose forced photometry is dominated by something else.

    contaminated: another catalogue source within 4 px is at least 5x brighter in
    that frame (or saturated). saturated_self: the saturation in the aperture is
    the source's own (it is the target or a bright catalogue star), not a
    neighbour's. Without these, faint sources in the glare of a bright star were
    measured at S/N > 30 and counted as detected."""
    from scipy.spatial import cKDTree
    bright_ids = set(src.loc[src.is_target.fillna(False) | (src.gmag < BRIGHT_G) | (src.w1 < BRIGHT_W1), "src_id"])
    tracks["saturated_self"] = tracks.saturated & (tracks.src_id.isin(bright_ids) | (tracks.src_kind == "sso"))
    cont = np.zeros(len(tracks), bool)
    pos = {k: i for i, k in enumerate(tracks.index)}
    cat = tracks[tracks.src_kind == "catalog"]
    for _, g in cat.groupby("frame_index"):
        xy = g[["x", "y"]].to_numpy()
        ok = np.isfinite(xy).all(1)
        if ok.sum() < 2:
            continue
        g = g[ok]
        xy = xy[ok]
        f = np.where(g.saturated_self, np.inf, np.nan_to_num(g.flux.to_numpy(), nan=0.0))
        strong = np.where((np.nan_to_num(g.snr.to_numpy()) >= C.DETECTION_SNR) | g.saturated_self.to_numpy())[0]
        if len(strong) == 0:
            continue
        tree = cKDTree(xy[strong])
        for i, nb in enumerate(tree.query_ball_point(xy, HALO_RADIUS_PX)):
            for j in nb:
                k = strong[j]
                if k != i and f[k] > 5 * max(f[i], 0):
                    cont[pos[g.index[i]]] = True
                    break
    tracks["contaminated"] = cont
    return tracks


def source_summary(tracks, frames, sources, stack_id, npix):
    """Per-source labels computed only from frames that passed QC. Solar-system objects
    use difference photometry, so a star underneath their track does not count."""
    good = frames.loc[frames.qc_pass, "frame_uid"]
    t = tracks[tracks.frame_uid.isin(good)].copy()
    t["in_grid"] = (t.x >= 0) & (t.y >= 0) & (t.x <= npix - 1) & (t.y <= npix - 1)
    is_sso = (t.src_kind == "sso").to_numpy()
    # without other-epoch frames to subtract (single-exposure comet stacks) use raw S/N
    snr_eff = np.where(is_sso, np.where(np.isfinite(t.snr_diff), t.snr_diff, t.snr), t.snr)
    t["detected"] = t.in_grid & (np.nan_to_num(snr_eff) >= C.DETECTION_SNR) & (t.aperture_valid_frac >= 0.95)
    # an asteroid detection must also be free of artefact flags and centre within
    # 1.5 px of the ephemeris position (where a centroid was measurable)
    off = np.hypot(t.centroid_dx, t.centroid_dy)
    clean = ~t.blended.astype(bool) & ~t.artifact_vetoes_mover & ~(off > 1.5)
    t.loc[is_sso, "detected"] &= clean[is_sso]
    # a catalogue detection must not be dominated by a brighter neighbour and, where a
    # centroid exists, must sit within 1.5 px of the prediction
    cat_clean = ~t.contaminated.astype(bool) & ~(off > 1.5) & ~(t.saturated & ~t.saturated_self)
    t.loc[~is_sso, "detected"] &= cat_clean[~is_sso]
    # a source saturated by its own light is plainly visible
    t["visible"] = t.detected | (t.in_grid & t.saturated_self & ~is_sso)
    g = t.groupby("src_id")
    s = pd.DataFrame({
        "n_frames_in_grid": g.in_grid.sum(), "n_frames_detected": g.detected.sum(),
        "n_frames_saturated": (t.in_grid & t.saturated_self).groupby(t.src_id).sum(),
        "n_frames_contaminated": (t.in_grid & t.contaminated).groupby(t.src_id).sum(),
        "n_frames_visible": g.visible.sum(),
        "median_snr": g.snr.median(), "max_snr": g.snr.max(),
        "max_snr_diff": g.snr_diff.max(),
        "x_min": g.x.min(), "x_max": g.x.max(), "y_min": g.y.min(), "y_max": g.y.max(),
        "frac_frames_artifact": g.artifact_in_aperture.mean(),
        # robust scatter of the wavelength-corrected residuals: about 1 for a constant source
        "sed_scatter": (g.sed_resid_snr.apply(lambda v: 1.4826 * np.nanmedian(np.abs(v - np.nanmedian(v))) if v.notna().sum() >= 8 else np.nan)
                        if "sed_resid_snr" in t else np.nan),
    })
    s["motion_px"] = np.hypot(s.x_max - s.x_min, s.y_max - s.y_min)
    s = s.reset_index()
    s["stack_id"] = stack_id
    out = sources.merge(s, on="src_id", how="right") if sources is not None else s
    out["is_mover"] = out.motion_px >= C.MOVER_MIN_PX
    out["mover_detectable"] = out.is_mover & (out.n_frames_visible >= 2)
    return out


def assemble(mission, slug, frames, rows, phots, field, src_path, sso):
    stack_id = f"{mission}:{slug}"
    npix = C.GRID[mission]["npix"]
    fr = frames.copy()
    fr["frame_uid"] = [f"{mission}:{slug}:{Path(p).name}" for p in fr.path]
    met = pd.DataFrame([r for r in rows if r["frame_uid"] in set(fr.frame_uid)])
    fr = fr.merge(met, on="frame_uid", how="left")
    band_col = "detector" if mission in ("spherex", "comet") else "band"
    fr["noise_ratio"] = fr.aperture_sky_noise / fr.groupby(band_col).aperture_sky_noise.transform("median")
    v = [qc_verdict(r, mission) for r in fr.to_dict("records")]
    fr["reg_shift_err_px"] = 1.2533 * fr.reg_rms / np.sqrt(fr.reg_nref.clip(lower=1))
    fr["qc_pass"] = [a for a, _ in v]
    fr["qc_reasons"] = [b for _, b in v]
    fr = fr.sort_values(["mjd_mid", band_col]).reset_index(drop=True)
    fr["time_group"] = time_groups(fr.mjd_mid.to_numpy())
    fr["stack_id"] = stack_id
    fr["frame_index"] = np.arange(len(fr))
    obs = earth_bary_au(fr.mjd_mid.to_numpy())   # barycentric Earth (AU), for parallax in injections
    fr["obs_x"], fr["obs_y"], fr["obs_z"] = obs[:, 0], obs[:, 1], obs[:, 2]

    arrays = {k: [] for k in ("image", "sigma", "flags", "valid", "wave", "bandwidth")}
    tmp = C.CACHE / "frames" / mission / slug
    for p in fr.path:
        z = np.load(tmp / f"{Path(p).stem}.npz")
        for k in arrays:
            arrays[k].append(z[k])
    out_dir = C.OUT / mission / "stacks"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{slug}.npz", **{k: np.stack(v) for k, v in arrays.items()},
                        frame_uid=fr.frame_uid.to_numpy().astype(str), mjd=fr.mjd_mid.to_numpy())

    tracks = pd.concat([p for p in phots if p is not None and len(p)], ignore_index=True)
    tracks = tracks[tracks.frame_uid.isin(set(fr.frame_uid))]
    tracks = tracks.merge(fr[["frame_uid", "frame_index"]], on="frame_uid")
    tracks.insert(0, "stack_id", stack_id)
    tracks = difference_photometry(tracks, fr, arrays, mission, pd.read_parquet(src_path))
    from stackcore import ARTIFACT
    veto = ARTIFACT[mission] & ~C.bitmask(C.SPHEREX_MOVER_TOLERATED_BITS if mission in ("spherex", "comet") else [])
    tracks["artifact_vetoes_mover"] = (tracks.aperture_flags.to_numpy() & veto) != 0
    tracks = flag_contamination(tracks, pd.read_parquet(src_path))
    if mission in ("spherex", "comet"):
        tracks = sed_residuals(tracks, fr, arrays)

    src = pd.read_parquet(src_path)
    s_cat = source_summary(tracks[tracks.src_kind == "catalog"], fr, src, stack_id, npix)
    sso_rows = sso[sso.src_id.isin(tracks.src_id)].sort_values("vmag").drop_duplicates("src_id")
    sso_src = pd.DataFrame({
        "src_id": sso_rows.src_id, "kind": "sso", "class": np.where(sso_rows.type.str.contains("Comet", case=False), "comet", "asteroid"),
        "subclass": sso_rows.type, "label_source": "imcce_skybot_ephemeris" if C.USE_SKYBOT else "jpl_horizons_ephemeris", "vmag": sso_rows.vmag,
        "posunc_arcsec": sso_rows.posunc_arcsec, "is_target": False, "group": field["group"], "field_key": stack_id,
    })
    s_sso = source_summary(tracks[tracks.src_kind == "sso"], fr, sso_src, stack_id, npix)
    sources = pd.concat([s_cat, s_sso], ignore_index=True)
    sources = sources[sources.n_frames_in_grid > 0].reset_index(drop=True)

    if mission == "wise":
        coadd(fr, arrays, out_dir / f"{slug}-coadd.npz")
    return fr, sources, tracks


def coadd(fr, arrays, path):
    """Inverse-variance mean per half-year bin and band over QC-passed frames.
    No outlier rejection: with two scans per bin it would delete real transients."""
    img, sig, val = np.stack(arrays["image"]), np.stack(arrays["sigma"]), np.stack(arrays["valid"]).astype(bool)
    keys, ims, sgs, vls, mj, nf = [], [], [], [], [], []
    for (b, band), g in fr[fr.qc_pass].groupby(["bin", "band"]):
        idx = g.index.to_numpy()
        w = np.where(val[idx], 1.0 / np.clip(sig[idx], 1e-12, None) ** 2, 0.0)
        ws = w.sum(0)
        ok = ws > 0
        ims.append(np.where(ok, (w * img[idx]).sum(0) / np.where(ok, ws, 1), 0).astype(np.float32))
        sgs.append(np.where(ok, 1 / np.sqrt(np.where(ok, ws, 1)), 0).astype(np.float32))
        vls.append(ok.astype(np.uint8))
        keys.append(f"{b}:W{band}")
        mj.append(float(g.mjd_mid.mean()))
        nf.append(len(idx))
    if keys:
        np.savez_compressed(path, image=np.stack(ims), sigma=np.stack(sgs), valid=np.stack(vls),
                            key=np.array(keys), mjd=np.array(mj), n_frames=np.array(nf))


def main(mission):
    fields = json.loads((C.CACHE / "fields.json").read_text())
    targets = json.loads((C.CACHE / "targets.json").read_text())
    frames = pd.read_parquet(C.CACHE / f"frames_{mission}.parquet")
    if mission == "wise":
        frames = frames[frames.complete]
    src_paths = build_sources(mission, fields, targets)
    sso = sso_table(mission)
    run(mission, frames, fields, src_paths, sso)


def run(mission, frames, fields, src_paths, sso):
    specs = frame_specs(mission, frames, fields, src_paths, sso)
    print("frames to process", len(specs), flush=True)
    with ProcessPoolExecutor(C.WORKERS) as ex:
        results = list(ex.map(__import__("stackcore").process_frame, specs, chunksize=2))
    rows = [r for r, _ in results]
    phots = [p for _, p in results]
    errs = [r for r in rows if r.get("read_error")]
    print("read errors", len(errs), errs[:3])

    all_fr, all_src, all_tr, meta = [], [], [], {}
    for slug, g in frames.groupby("field"):
        f = fields[f"{mission}:{slug}"]
        fr, src, tr = assemble(mission, slug, g, rows, phots, f, src_paths[slug], sso)
        all_fr.append(fr)
        all_src.append(src)
        all_tr.append(tr)
        gc = C.GRID[mission]
        meta[f"{mission}:{slug}"] = {"field": slug, "group": f["group"], "is_control": f["is_control"],
                                     "grid": {"ctype": ["RA---TAN", "DEC--TAN"], "crval": [f["ra"], f["dec"]],
                                              "crpix_1based": [(gc["npix"] + 1) / 2] * 2,
                                              "cdelt_deg": [-gc["scale_arcsec"] / 3600, gc["scale_arcsec"] / 3600],
                                              "npix": gc["npix"]},
                                     "n_frames": len(fr), "n_frames_qc_pass": int(fr.qc_pass.sum())}
        print(f"{mission}:{slug}", "frames", len(fr), "qc pass", int(fr.qc_pass.sum()),
              fr.loc[~fr.qc_pass, "qc_reasons"].value_counts().to_dict(),
              "reg rms med", round(float(fr.reg_rms.median()), 3), "movers", int(src.mover_detectable.sum()), flush=True)
    out = C.OUT / mission
    pd.concat(all_fr, ignore_index=True).to_parquet(out / "frames.parquet", index=False)
    pd.concat(all_src, ignore_index=True).to_parquet(out / "sources.parquet", index=False)
    pd.concat(all_tr, ignore_index=True).to_parquet(out / "tracks.parquet", index=False)
    json_dump(meta, out / "stacks.json")
    shutil.rmtree(C.CACHE / "frames" / mission, ignore_errors=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main(sys.argv[1])
