"""Stage 3b: proper motion and parallax of WISE 0855 and Luhman 16 from the WISE frames.

Neither has a Gaia DR3 proper motion. Instead of taking catalogue values from
SIMBAD, the motion is measured here: the AllWISE catalogue (NASA) gives a start
position and a rough motion, the object is then centroided in every W2 half-year
coadd (2014-2024), and position, proper motion and parallax are fitted by least
squares. SIMBAD's values are printed only as a cross-check.

Needs dataset/wise (stage 4 run once). Output: cache/target_astrometry.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from common import centroid, earth_bary_au, grid_wcs, json_dump, mjd_to_jyear
from sources import ALLWISE_JYEAR

REF_JYEAR = 2016.0
# WISE 0855: the position printed in the challenge statement. Luhman 16: brightest
# fast-moving AllWISE source near the field centre.
SEEDS = {"wise0855": {"ra": 133.7951292, "dec": -7.2451472, "radius_arcsec": 6.0, "shift_and_stack": True},
         "luhman16": {"radius_arcsec": 120.0, "w2_max": 9.0}}


def parallax_factors(ra, dec, mjd):
    o = earth_bary_au(mjd)
    a, d = np.radians(ra), np.radians(dec)
    pra = o[:, 0] * np.sin(a) - o[:, 1] * np.cos(a)
    pdec = o[:, 0] * np.sin(d) * np.cos(a) + o[:, 1] * np.sin(d) * np.sin(a) - o[:, 2] * np.cos(d)
    return pra, pdec


def allwise_seed(slug, field):
    aw = pd.read_parquet(C.CACHE / "catalogs" / f"{field['catalog_key']}-allwise.parquet")
    for c in ("pmra", "pmdec", "sigpmra", "w2mpro"):
        aw[c] = pd.to_numeric(aw[c], errors="coerce")
    s = SEEDS[slug]
    ra0, dec0 = s.get("ra", field["ra"]), s.get("dec", field["dec"])
    sep = np.hypot((aw.ra - ra0) * np.cos(np.radians(dec0)), aw.dec - dec0) * 3600
    c = aw[sep < s["radius_arcsec"]].copy()
    if "w2_max" in s:
        c = c[c.w2mpro < s["w2_max"]]
        c = c.assign(pm=np.hypot(c.pmra, c.pmdec)).sort_values("pm", ascending=False)
    else:
        c = c.assign(sep=sep[c.index]).sort_values("sep")
    r = c.iloc[0]
    return {"designation": str(r.designation), "ra": float(r.ra), "dec": float(r.dec),
            "pmra": float(r.pmra), "pmdec": float(r.pmdec), "w2": float(r.w2mpro)}


def fit(slug):
    fields = json.loads((C.CACHE / "fields.json").read_text())
    field = fields[f"wise:{slug}"]
    seed = allwise_seed(slug, field)
    z = np.load(C.OUT / "wise" / "stacks" / f"{slug}-coadd.npz")
    keep = np.array([str(k).endswith("W2") for k in z["key"]])
    img, val, mjd = z["image"][keep], z["valid"][keep].astype(bool), z["mjd"][keep]
    g = C.GRID["wise"]
    w = grid_wcs(field["ra"], field["dec"], g["npix"], g["scale_arcsec"])
    cosd = np.cos(np.radians(seed["dec"]))
    pra, pdec = parallax_factors(seed["ra"], seed["dec"], mjd)
    dt = mjd_to_jyear(mjd) - REF_JYEAR
    # model parameters: offsets (mas) from the seed position at REF_JYEAR, pm (mas/yr), parallax (mas)
    d0 = REF_JYEAR - ALLWISE_JYEAR
    p = np.array([seed["pmra"] * d0, seed["pmdec"] * d0, seed["pmra"], seed["pmdec"], 0.0])
    if SEEDS[slug].get("shift_and_stack"):
        # Faint target: find the motion that maximises the summed signal along the
        # track from the AllWISE position (epoch 2010.56), on a 0.1 arcsec/yr grid.
        sm = np.stack([gaussian_filter(np.where(val[i], img[i], 0.0), 1.0) for i in range(len(dt))])
        sm = sm / np.array([1.4826 * np.median(np.abs(s_[v_] - np.median(s_[v_]))) for s_, v_ in zip(sm, val)])[:, None, None]
        tt = mjd_to_jyear(mjd) - ALLWISE_JYEAR
        best = (-np.inf, 0.0, 0.0)
        for pmr in np.arange(-10000, 10001, 100.0):
            for pmd in np.arange(-10000, 10001, 100.0):
                if np.hypot(pmr, pmd) < 1000:
                    continue
                x, y = w.wcs_world2pix(seed["ra"] + pmr * tt / 3.6e6 / cosd, seed["dec"] + pmd * tt / 3.6e6, 0)
                xi_, yi_ = np.round(x).astype(int), np.round(y).astype(int)
                inb = (xi_ >= 0) & (yi_ >= 0) & (xi_ < g["npix"]) & (yi_ < g["npix"])
                if inb.sum() < 0.7 * len(tt):
                    continue
                v = sm[np.where(inb)[0], yi_[inb], xi_[inb]]
                score = np.sum(np.clip(v, -2, 8)) / np.sqrt(inb.sum())
                if score > best[0]:
                    best = (score, pmr, pmd)
        print(f"   shift-and-stack: best motion ({best[1]:.0f}, {best[2]:.0f}) mas/yr, score {best[0]:.1f}")
        p = np.array([best[1] * d0, best[2] * d0, best[1], best[2], 0.0])
    A_ra = np.c_[np.ones_like(dt), np.zeros_like(dt), dt, np.zeros_like(dt), pra]
    A_de = np.c_[np.zeros_like(dt), np.ones_like(dt), np.zeros_like(dt), dt, pdec]
    search = 4.0 if not SEEDS[slug].get("shift_and_stack") else 2.5
    for it in range(5):
        xi = (A_ra @ p) / 3.6e6 / cosd + seed["ra"]
        et = (A_de @ p) / 3.6e6 + seed["dec"]
        px, py = w.wcs_world2pix(xi, et, 0)
        mx, my = np.full(len(dt), np.nan), np.full(len(dt), np.nan)
        for i in range(len(dt)):
            x, y = px[i], py[i]
            if it == 0:   # the catalogue motion is rough: start from the brightest smoothed peak nearby
                sm = gaussian_filter(np.where(val[i], img[i], 0.0), 1.0)
                yy, xx = np.mgrid[0:sm.shape[0], 0:sm.shape[1]]
                m = np.hypot(xx - x, yy - y) <= search
                if not m.any():
                    continue
                k = np.argmax(np.where(m, sm, -np.inf))
                y, x = np.unravel_index(k, sm.shape)
            cx, cy = centroid(img[i], val[i], float(x), float(y), r=2.5)
            if np.isfinite(cx) and np.hypot(cx - px[i], cy - py[i]) <= (search if it == 0 else 2.0):
                mx[i], my[i] = cx, cy
        ok = np.isfinite(mx)
        mra, mde = w.wcs_pix2world(mx[ok], my[ok], 0)
        y_ra = (mra - seed["ra"]) * cosd * 3.6e6
        y_de = (mde - seed["dec"]) * 3.6e6
        A = np.r_[A_ra[ok], A_de[ok]]
        yv = np.r_[y_ra, y_de]
        keep_rows = np.ones(len(yv), bool)
        for _ in range(3):
            p, *_ = np.linalg.lstsq(A[keep_rows], yv[keep_rows], rcond=None)
            res = yv - A @ p
            s = 1.4826 * np.median(np.abs(res[keep_rows]))
            keep_rows = np.abs(res) < 3.5 * max(s, 50.0)
    n = int(keep_rows.sum())
    cov = np.linalg.inv(A[keep_rows].T @ A[keep_rows]) * (np.sum(res[keep_rows] ** 2) / max(n - 5, 1))
    err = np.sqrt(np.diag(cov))
    out = {"slug": slug, "source": "least-squares fit to W2 half-year coadd centroids, 2014-2024; AllWISE seed " + seed["designation"],
           "ra": seed["ra"] + p[0] / 3.6e6 / cosd, "dec": seed["dec"] + p[1] / 3.6e6, "ref_jyear": REF_JYEAR,
           "pmra": p[2], "pmdec": p[3], "plx": p[4], "pmra_err": err[2], "pmdec_err": err[3], "plx_err": err[4],
           "pos_err_mas": float(np.hypot(err[0], err[1])), "residual_rms_mas": float(np.sqrt(np.mean(res[keep_rows] ** 2))),
           "n_epochs_used": int(ok.sum()), "n_measurements": n, "allwise_seed": seed}
    return out


def main():
    targets = json.loads((C.CACHE / "targets.json").read_text())["wise"]
    out = {}
    for slug in SEEDS:
        r = fit(slug)
        out[slug] = r
        t = targets[slug]
        print(f"{slug}: pm = ({r['pmra']:.0f} +- {r['pmra_err']:.0f}, {r['pmdec']:.0f} +- {r['pmdec_err']:.0f}) mas/yr, "
              f"plx = {r['plx']:.0f} +- {r['plx_err']:.0f} mas, rms {r['residual_rms_mas']:.0f} mas, {r['n_epochs_used']} epochs")
        print(f"   cross-check, SIMBAD: pm = ({t['pmra']:.0f}, {t['pmdec']:.0f}), plx = {t['plx']}")
    json_dump(out, C.CACHE / "target_astrometry.json")


if __name__ == "__main__":
    main()
