"""Stage 6: per-frame empirical PSFs, photon-noise slopes and a fixed injection benchmark.

usage: python s06_psf_inject.py spherex|wise

Outputs in dataset/<mission>/:
  psf/<field>.npz                psf [N,S,S] for every frame, n_stars, fwhm_px, fallback flag
  frames.parquet                 gains psf_fwhm_px, psf_n_stars, psf_fallback, shot_k
  injected/<field>-r<k>.npz      injected image and sigma for QC-passed frames
  injected_objects.parquet       one row per injected object (parameters, motion)
  injected_truth.parquet         one row per (object, frame): position, flux, S/N, masked fraction
"""
import json
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import shift as nd_shift
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "dataset" / "tools"))
import config as C
from injection import inject
from stackcore import SATURATED

STAMP = {"spherex": 15, "wise": 21}
REALIZATIONS = {"spherex": 6, "wise": 4}
N_OBJECTS = 10
MIN_PSF_STARS = 3


def star_stamps(img, pixel_noise, valid, stars, S, core_r=3.0):
    """Centred star stamps. Masked pixels in a stamp's wings are allowed (NaN); the
    5x5 core must be clean. Each stamp is normalised by its flux within core_r so
    that incomplete wings do not bias the scale."""
    h = S // 2
    pad = h + 2
    stamps = []
    H, W = img.shape
    yy, xx = np.mgrid[0:S, 0:S] - h
    core = np.hypot(xx, yy) <= core_r
    for x, y in stars:
        xc, yc = int(round(x)), int(round(y))
        if xc - pad < 0 or yc - pad < 0 or xc + pad >= W or yc + pad >= H:
            continue
        sub = img[yc - pad:yc + pad + 1, xc - pad:xc + pad + 1].astype(float)
        vm = valid[yc - pad:yc + pad + 1, xc - pad:xc + pad + 1].astype(float)
        if vm[pad - 2:pad + 3, pad - 2:pad + 3].min() < 1:
            continue
        # background plane fitted to the outer ring; stamps on a gradient (a galaxy, a
        # bright star's halo) that a plane cannot remove are rejected
        ry, rx = np.mgrid[0:2 * pad + 1, 0:2 * pad + 1] - pad
        ring = (np.hypot(rx, ry) >= pad - 2) & (vm > 0)
        if ring.sum() < 20:
            continue
        A = np.c_[np.ones(ring.sum()), rx[ring], ry[ring]]
        coef, *_ = np.linalg.lstsq(A, sub[ring], rcond=None)
        plane = coef[0] + coef[1] * rx + coef[2] * ry
        resid = sub[ring] - A @ coef
        peak = sub[pad, pad] - coef[0]
        scatter = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        # reject: structure in the surroundings beyond 2.5x the frame's measured sky
        # pixel noise (the resampled sigma map understates it), or a background
        # slope worth more than 5% of the peak across the stamp
        if peak <= 0 or scatter > 2.5 * pixel_noise or np.hypot(coef[1], coef[2]) * pad > 0.05 * peak:
            continue
        sub = np.where(vm > 0, sub - plane, 0.0)
        sh = nd_shift(sub, (yc - y, xc - x), order=3, mode="constant")[2:2 + S, 2:2 + S]
        msh = nd_shift(vm, (yc - y, xc - x), order=1, mode="constant")[2:2 + S, 2:2 + S]
        st = np.where(msh > 0.99, sh, np.nan)
        if np.isnan(st[core]).any():
            continue
        norm = st[core].sum()
        if norm <= 0:
            continue
        stamps.append(st / norm)
    return np.stack(stamps) if stamps else np.zeros((0, S, S))


def median_psf(stamps):
    p = np.nanmedian(stamps, axis=0)
    p = np.clip(np.nan_to_num(p), 0, None)
    return p / p.sum()


def fwhm(p, up=8):
    """Diameter of the circle with the same area as the region above half maximum
    (on an 8x upsampled PSF). Insensitive to faint wing pedestals."""
    from scipy.ndimage import zoom
    if p.sum() <= 0:
        return np.nan
    z = zoom(p, up, order=3)
    area = (z >= 0.5 * z.max()).sum() / up ** 2
    return 2 * np.sqrt(area / np.pi)


def shot_slope(img, sig, valid, pix_noise):
    v = valid.astype(bool)
    sky = v & (np.abs(img) < 2 * pix_noise)
    bright = v & (img > 30 * pix_noise)
    if sky.sum() < 100 or bright.sum() < 20:
        return 0.0
    s2 = np.median(sig[sky] ** 2)
    k = np.median((sig[bright] ** 2 - s2) / img[bright])
    return float(max(k, 0.0))


def collect_stamps(args):
    """Pass 1: star stamps for every frame of one stack, plus photon-noise slopes."""
    mission, field, stack_id = args
    out = C.OUT / mission
    z = np.load(out / "stacks" / f"{field}.npz")
    fr = pd.read_parquet(out / "frames.parquet", filters=[("stack_id", "==", stack_id)]).sort_values("frame_index")
    tr = pd.read_parquet(out / "tracks.parquet", filters=[("stack_id", "==", stack_id)])
    src = pd.read_parquet(out / "sources.parquet", filters=[("stack_id", "==", stack_id)])
    # PSF stars: confident Gaia stars only (no galaxy knots, quasars or unknown sources)
    good_ids = set(src.loc[(src["class"] == "star") & ((src.p_star > 0.95) | (src.parallax_snr > 5)) & ~src.is_target.fillna(False), "src_id"])
    S = STAMP[mission]
    stamps = []
    for i in range(len(fr)):
        t = tr[(tr.frame_index == i) & (tr.src_kind == "catalog")]
        if len(t) == 0:
            stamps.append(np.zeros((0, S, S)))
            continue
        tree = cKDTree(np.c_[t.x, t.y])
        snr = t.snr.to_numpy()
        cand = np.where(t.snr.between(20, 2000) & (t.aperture_valid_frac >= 0.9) & ~t.artifact_in_aperture
                        & ~t.saturated & t.centroid_dx.notna() & t.src_id.isin(good_ids))[0]
        stars = []
        for j in cand:
            # reject comparable neighbours only: the median over stars tolerates faint ones
            near = [k for k in tree.query_ball_point([t.x.iat[j], t.y.iat[j]], 3.0) if k != j]
            if any(snr[k] > 0.3 * snr[j] for k in near):
                continue
            stars.append((t.x.iat[j] + t.centroid_dx.iat[j], t.y.iat[j] + t.centroid_dy.iat[j]))
        stamps.append(star_stamps(z["image"][i], fr.pixel_noise.iat[i], z["valid"][i].astype(bool), stars, S))
    kk = np.array([shot_slope(z["image"][i], z["sigma"][i], z["valid"][i], fr.pixel_noise.iat[i]) for i in range(len(fr))])
    # brightest valid pixel of frames that contain saturated pixels: just below the saturation level
    satbits = SATURATED[mission]
    peak = np.full(len(fr), np.nan)
    for i in range(len(fr)):
        v = z["valid"][i].astype(bool)
        if ((z["flags"][i] & satbits) != 0).sum() >= 3 and v.any():
            peak[i] = float(z["image"][i][v].max())
    return stack_id, field, stamps, kk, peak


def inject_stack(args):
    """Pass 2: fixed benchmark realisations on QC-passed frames."""
    mission, field, stack_id = args
    out = C.OUT / mission
    z = np.load(out / "stacks" / f"{field}.npz")
    fr = pd.read_parquet(out / "frames.parquet", filters=[("stack_id", "==", stack_id)]).sort_values("frame_index")
    psf = np.load(out / "psf" / f"{field}.npz")["psf"]
    meta = json.loads((out / "stacks.json").read_text())[stack_id]
    q = fr.qc_pass.to_numpy()
    objs_all, truth_all = [], []
    if q.sum() < 2:
        return objs_all, truth_all
    idx = np.where(q)[0]
    frames = {"mjd": fr.mjd_mid.to_numpy()[idx], "time_group": fr.time_group.to_numpy()[idx],
              "wave_um": fr.wave_center_um.to_numpy()[idx], "obs_xyz": fr[["obs_x", "obs_y", "obs_z"]].to_numpy()[idx],
              "sky_noise": fr.aperture_sky_noise.to_numpy()[idx], "shot_k": fr.shot_k.to_numpy()[idx],
              "sat_level": fr.sat_level.to_numpy()[idx]}
    base = {"image": z["image"][idx], "sigma": z["sigma"][idx], "valid": z["valid"][idx]}
    g = meta["grid"]
    grid = {"ra": g["crval"][0], "dec": g["crval"][1], "scale_arcsec": abs(g["cdelt_deg"][1]) * 3600, "npix": g["npix"]}
    (out / "injected").mkdir(exist_ok=True)
    for r in range(REALIZATIONS[mission]):
        rid = f"{stack_id}:r{r}"
        seed = zlib.crc32(rid.encode())
        img, sig, valid, objs, truth = inject(base, frames, psf[idx], grid, np.random.default_rng(seed), N_OBJECTS,
                                              C.APERTURE[mission][0])
        np.savez_compressed(out / "injected" / f"{field}-r{r}.npz", image=img, sigma=sig,
                            valid=valid.astype(np.uint8), frame_index=idx)
        for o in objs:
            o.update(realization_id=rid, stack_id=stack_id, seed=seed)
        for t in truth:
            t.update(realization_id=rid, stack_id=stack_id, frame_index=int(idx[t["frame_index"]]))
        objs_all += objs
        truth_all += truth
    return objs_all, truth_all


def main(mission):
    out = C.OUT / mission
    meta = json.loads((out / "stacks.json").read_text())
    tasks = [(mission, v["field"], k) for k, v in meta.items()]
    with ProcessPoolExecutor(min(C.WORKERS, len(tasks))) as ex:
        res = list(ex.map(collect_stamps, tasks))
    fr = pd.read_parquet(out / "frames.parquet")
    band_col = "detector" if mission == "spherex" else "band"
    S = STAMP[mission]

    # PSF hierarchy: own frame (>= 3 stars) -> same detector/band in this stack (>= 5)
    # -> same detector/band across the whole mission
    mission_pool = {}
    for sid, field, stamps, _, _ in res:
        bands = fr[fr.stack_id == sid].sort_values("frame_index")[band_col].to_numpy()
        for b, st in zip(bands, stamps):
            if len(st):
                mission_pool.setdefault(b, []).append(st)
    mission_psf = {b: median_psf(np.concatenate(v)) for b, v in mission_pool.items()}
    stats = []
    pos_k = {}
    peaks = {}
    for sid, field, stamps, kk, peak in res:
        f = fr[fr.stack_id == sid].sort_values("frame_index")
        for b, pk in zip(f[band_col], peak):
            if np.isfinite(pk):
                peaks.setdefault(b, []).append(pk)
        for b, k in zip(f[band_col], kk):
            if k > 0:
                pos_k.setdefault(b, []).append(k)
    med_k = {b: float(np.median(v)) for b, v in pos_k.items()}
    sat_level = {b: float(np.median(v)) for b, v in peaks.items() if len(v) >= 5}
    for sid, field, stamps, kk, _ in res:
        f = fr[fr.stack_id == sid].sort_values("frame_index")
        bands = f[band_col].to_numpy()
        n = len(f)
        psf = np.zeros((n, S, S), np.float32)
        src = np.empty(n, dtype=object)
        nst = np.array([len(s) for s in stamps])
        stack_psf = {}
        for b in np.unique(bands):
            pool = [s for s, bb in zip(stamps, bands) if bb == b and len(s)]
            if pool and sum(len(p) for p in pool) >= 5:
                stack_psf[b] = median_psf(np.concatenate(pool))
        for i in range(n):
            if nst[i] >= MIN_PSF_STARS:
                psf[i], src[i] = median_psf(stamps[i]), "frame"
            elif bands[i] in stack_psf:
                psf[i], src[i] = stack_psf[bands[i]], "stack_band"
            else:
                psf[i], src[i] = mission_psf[bands[i]], "mission_band"
        kfix = np.array([k if k > 0 else med_k.get(b, 0.0) for k, b in zip(kk, bands)])
        fw = np.array([fwhm(p) for p in psf])
        (out / "psf").mkdir(exist_ok=True)
        np.savez_compressed(out / "psf" / f"{field}.npz", psf=psf, n_stars=nst, fwhm_px=fw, source=src.astype(str))
        stats.append(pd.DataFrame({"stack_id": sid, "frame_index": f.frame_index.to_numpy(), "psf_fwhm_px": fw,
                                   "psf_n_stars": nst, "psf_source": src, "shot_k": kfix,
                                   "sat_level": [sat_level.get(b, np.nan) for b in bands],
                                   "shot_k_measured": kk > 0}))
    stats = pd.concat(stats, ignore_index=True)
    fr = fr.drop(columns=[c for c in stats.columns if c not in ("stack_id", "frame_index") and c in fr.columns]
                 + [c for c in ("psf_fallback",) if c in fr.columns])
    fr = fr.merge(stats, on=["stack_id", "frame_index"], how="left")
    fr.to_parquet(out / "frames.parquet", index=False)

    with ProcessPoolExecutor(min(C.WORKERS, len(tasks))) as ex:
        res2 = list(ex.map(inject_stack, tasks))
    objs = pd.DataFrame([o for r in res2 for o in r[0]])
    truth = pd.DataFrame([t for r in res2 for t in r[1]])
    objs.to_parquet(out / "injected_objects.parquet", index=False)
    truth.to_parquet(out / "injected_truth.parquet", index=False)
    print(mission, "PSF source", stats.psf_source.value_counts().to_dict(),
          "median FWHM px", round(float(np.nanmedian(stats.psf_fwhm_px)), 2))
    print("injected objects", len(objs), objs["class"].value_counts().to_dict() if len(objs) else {})


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main(sys.argv[1])
