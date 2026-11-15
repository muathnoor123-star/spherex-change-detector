"""Frame processing shared by SPHEREx and WISE: read, register, resample, measure."""
import warnings
import zlib

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

import config as C
from common import (aperture_photometry, centroid, earth_bary_au, grid_wcs, resample, robust_median_std,
                    sky_aperture_noise, strip_sip)
from sources import reference_star_mask, source_positions

warnings.simplefilter("ignore")

HARD = {"spherex": C.bitmask(C.SPHEREX_HARD_BITS), "wise": C.bitmask(C.WISE_HARD_BITS)}
ARTIFACT = {"spherex": C.bitmask(C.SPHEREX_ARTIFACT_BITS), "wise": C.bitmask(C.WISE_ARTIFACT_BITS)}
HARD["comet"], ARTIFACT["comet"] = HARD["spherex"], ARTIFACT["spherex"]
HARD["neossat"], ARTIFACT["neossat"] = 3, 0     # bit 0 = saturated, bit 1 = outside the data section
# bits meaning "too bright to measure": the source is there, its core value is lost
SATURATED = {"spherex": C.bitmask([1, 2]), "comet": C.bitmask([1, 2]), "wise": C.bitmask(range(10, 20)), "neossat": 1}
SOURCE_BIT = 1 << 21
REF_G = {"spherex": (10.0, 17.0), "comet": (10.0, 17.0), "wise": (9.0, 16.5), "neossat": (9.0, 15.5)}
CENTROID_R = {"spherex": 2.0, "comet": 2.0, "wise": 2.5, "neossat": 2.5}
NEOSSAT_SATURATION_ADU = 60000.0

_SRC_CACHE = {}


def load_sources(path):
    if path not in _SRC_CACHE:
        s = pd.read_parquet(path)
        s["refstar"] = False
        for m in REF_G:
            s[f"ref_{m}"] = reference_star_mask(s, *REF_G[m])
        _SRC_CACHE[path] = s
    return _SRC_CACHE[path]


# --------------------------------------------------------------------------- readers
def read_spherex(path):
    with fits.open(C.SRC / path, memmap=False) as h:
        hd = h["IMAGE"].header
        img = h["IMAGE"].data.astype(np.float64) - h["ZODI"].data.astype(np.float64)
        var = h["VARIANCE"].data.astype(np.float64)
        flags = h["FLAGS"].data.astype(np.uint32)
        zodi = h["ZODI"].data.astype(np.float64)
        wcs = WCS(hd)
        wave_wcs = WCS(strip_sip(hd), fobj=h, key="W")
    hard = ((flags & HARD["spherex"]) != 0) | ~np.isfinite(img) | ~np.isfinite(var) | (var <= 0)
    return {"img": img, "var": var, "flags": flags, "hard": hard, "wcs": wcs, "wave_wcs": wave_wcs,
            "zodi_median": float(np.nanmedian(zodi)), "native_bkg": 0.0}


def mesh_background(img, valid, box=64):
    ny, nx = img.shape
    gy, gx = int(np.ceil(ny / box)), int(np.ceil(nx / box))
    mesh = np.full((gy, gx), np.nan)
    for i in range(gy):
        for j in range(gx):
            sl = (slice(i * box, (i + 1) * box), slice(j * box, (j + 1) * box))
            v = img[sl][valid[sl]]
            if v.size > box * box * 0.3:
                mesh[i, j] = robust_median_std(v)[0]
    if np.isnan(mesh).all():
        return np.zeros_like(img)
    fill = np.nanmedian(mesh)
    mesh = np.where(np.isfinite(mesh), mesh, fill)
    from scipy.ndimage import median_filter, zoom
    mesh = median_filter(mesh, size=3, mode="nearest")
    bk = zoom(mesh, (ny / gy, nx / gx), order=1, mode="nearest")
    return bk[:ny, :nx]


def read_wise(path, unc_path, msk_path, magzp):
    with fits.open(C.SRC / path, memmap=False) as h:
        hd = h[0].header
        img = h[0].data.astype(np.float64)
        wcs = WCS(hd)
    with fits.open(C.SRC / unc_path, memmap=False) as h:
        unc = h[0].data.astype(np.float64)
    with fits.open(C.SRC / msk_path, memmap=False) as h:
        msk = h[0].data.astype(np.int64).astype(np.uint32)
    scale = 10 ** (-0.4 * (magzp - 22.5))   # DN -> flux with mag = 22.5 - 2.5 log10(flux), Vega
    img *= scale
    var = (unc * scale) ** 2
    hard = ((msk & HARD["wise"]) != 0) | ~np.isfinite(img) | ~np.isfinite(var) | (var <= 0)
    bk = mesh_background(img, ~hard)
    img -= bk
    return {"img": img, "var": var, "flags": msk, "hard": hard, "wcs": wcs, "wave_wcs": None,
            "zodi_median": np.nan, "native_bkg": float(np.median(bk))}


def _section(s):
    """FITS section '[x0:x1,y0:y1]' (1-based, inclusive) -> numpy slices."""
    x, y = s.strip("[]").split(",")
    x0, x1 = (int(v) for v in x.split(":"))
    y0, y1 = (int(v) for v in y.split(":"))
    return slice(y0 - 1, y1), slice(x0 - 1, x1)


def read_neossat(path):
    """NEOSSat corrected frame. No variance or mask is supplied, so both are built
    here: CCD noise from the header gain and read noise, saturation from the ADU
    level, and the overscan strip (outside DATASEC) is masked."""
    with fits.open(C.SRC / path, memmap=False) as h:
        hd = h[0].header
        raw = h[0].data.astype(np.float64)
        wcs = WCS(hd)
    gain = float(hd.get("GAIN", 1.1))
    rdn = float(hd.get("RDNOISE", 8.0))
    flags = np.zeros(raw.shape, np.uint32)
    datasec = hd.get("DATASEC", "")
    if datasec:
        inside = np.zeros(raw.shape, bool)
        inside[_section(datasec)] = True
        flags[~inside] |= 2
    biassec = hd.get("BIASSEC", "")
    bias = float(np.median(raw[_section(biassec)])) if biassec else 0.0
    flags[raw >= NEOSSAT_SATURATION_ADU] |= 1
    img = raw - bias
    var = np.clip(img, 0, None) / gain + (rdn / gain) ** 2
    hard = (flags != 0) | ~np.isfinite(img)
    bk = mesh_background(img, ~hard)
    img -= bk
    return {"img": img, "var": var, "flags": flags, "hard": hard, "wcs": wcs, "wave_wcs": None,
            "zodi_median": np.nan, "native_bkg": float(np.median(bk))}


# --------------------------------------------------------------------------- registration
def refine_registration(mission, d, src, mjd, obs):
    ra, dec = source_positions(src, mjd, obs)
    ref = src[f"ref_{mission}"].to_numpy()
    px, py = d["wcs"].all_world2pix(ra[ref], dec[ref], 0, quiet=True)
    ny, nx = d["img"].shape
    inside = (px > 4) & (py > 4) & (px < nx - 5) & (py < ny - 5)
    px, py = px[inside], py[inside]
    valid = ~d["hard"]
    dx, dy = [], []
    sig = np.sqrt(np.where(valid, d["var"], np.inf))
    for x, y in zip(px, py):
        cx, cy = centroid(d["img"], valid, x, y, r=CENTROID_R[mission])
        if not np.isfinite(cx):
            continue
        xi, yi = int(round(cx)), int(round(cy))
        if d["img"][yi, xi] < 10 * sig[yi, xi]:   # peak S/N >= 10
            continue
        dx.append(cx - x)
        dy.append(cy - y)
    dx, dy = np.asarray(dx), np.asarray(dy)
    out = {"reg_nref": int(len(dx)), "reg_dx": 0.0, "reg_dy": 0.0, "reg_rms": np.nan, "reg_status": "insufficient_stars"}
    if len(dx) < C.QC["min_reference_stars"]:
        return out
    keep = np.ones(len(dx), bool)
    for _ in range(3):
        mx, my = np.median(dx[keep]), np.median(dy[keep])
        r = np.hypot(dx - mx, dy - my)
        s = 1.4826 * np.median(r[keep]) + 1e-3
        keep = r < max(3 * s, 0.3)
    mx, my = float(np.median(dx[keep])), float(np.median(dy[keep]))
    rx, ry = dx[keep] - mx, dy[keep] - my
    rms = float(np.sqrt(0.5 * ((1.4826 * np.median(np.abs(rx))) ** 2 + (1.4826 * np.median(np.abs(ry))) ** 2)))
    out.update(reg_nref=int(keep.sum()), reg_dx=mx, reg_dy=my, reg_rms=rms, reg_status="ok")
    if np.hypot(mx, my) > C.QC["max_registration_shift_px"]:
        out["reg_status"] = "shift_too_large"
    return out


# --------------------------------------------------------------------------- one frame
def process_frame(spec):
    """spec: dict with mission, stack_id, frame_uid, paths, mjd, grid centre, sources path,
    sso rows (DataFrame or None), out path. Writes arrays to an .npz, returns
    (frame row, photometry DataFrame)."""
    m = spec["mission"]
    try:
        if m in ("spherex", "comet"):
            d = read_spherex(spec["path"])
        elif m == "wise":
            d = read_wise(spec["path"], spec["unc_path"], spec["msk_path"], spec["magzp"])
        else:
            d = read_neossat(spec["path"])
    except Exception as e:  # tellscan: ignore C05 -- recorded as read_error and rejected by QC
        return {"frame_uid": spec["frame_uid"], "read_error": repr(e)[:300]}, None
    src = load_sources(spec["sources_path"])
    mjd = spec["mjd"]
    obs = earth_bary_au([mjd])[0]
    reg = refine_registration(m, d, src, mjd, obs)
    use_shift = reg["reg_status"] == "ok"

    gcfg = C.GRID[m]
    npx = gcfg["npix"]
    gw = grid_wcs(spec["grid_ra"], spec["grid_dec"], npx, gcfg["scale_arcsec"])
    ref_m = "spherex" if m == "comet" else m
    yy, xx = np.mgrid[0:npx, 0:npx].astype(float)
    gra, gdec = gw.wcs_pix2world(xx, yy, 0)
    xin, yin = d["wcs"].all_world2pix(gra.ravel(), gdec.ravel(), 0, quiet=True)
    if use_shift:
        xin = xin + reg["reg_dx"]
        yin = yin + reg["reg_dy"]
    xin = xin.reshape(npx, npx)
    yin = yin.reshape(npx, npx)
    res, valid, flags, covered = resample(xin, yin, d["img"].shape,
                                 {"img": (d["img"], False), "var": (d["var"], True)}, d["hard"], d["flags"])
    img, var = res["img"], res["var"]

    if d["wave_wcs"] is not None:
        wv, bw = d["wave_wcs"].pixel_to_world_values(xin, yin)
        wave = np.where(valid, wv, np.nan).astype(np.float32)
        band = np.where(valid, bw, np.nan).astype(np.float32)
    else:
        lam = {1: 3.4, 2: 4.6}.get(spec.get("band"), 0.6)   # WISE W1/W2; NEOSSat CLEAR ~0.6 um
        wave = np.where(valid, lam, np.nan).astype(np.float32)
        band = np.full_like(wave, np.nan)

    # residual background on the grid (pixels not flagged as known sources)
    sky = valid & ((flags & SOURCE_BIT) == 0) if ref_m == "spherex" else valid
    bkg, pix_std = robust_median_std(img[sky])
    if np.isfinite(bkg):
        img = np.where(valid, img - bkg, 0.0).astype(np.float32)

    # photometry of every catalogue source and every known solar-system object
    ra, dec = source_positions(src, mjd, obs)
    sx, sy = gw.wcs_world2pix(ra, dec, 0)
    margin = 3
    inn = (sx > -margin) & (sy > -margin) & (sx < npx - 1 + margin) & (sy < npx - 1 + margin)
    ids = src.src_id.to_numpy()[inn]
    sx, sy = sx[inn], sy[inn]
    kinds = np.full(len(ids), "catalog", dtype=object)
    sso = spec.get("sso")
    if sso is not None and len(sso):
        dt_h = (mjd - sso.mjd_utc.to_numpy()) * 24
        cosd = np.cos(np.radians(sso.dec_obs.to_numpy()))
        sra = sso.ra_obs.to_numpy() + sso.ra_rate_arcsec_h.to_numpy() * dt_h / 3600 / cosd
        sdec = sso.dec_obs.to_numpy() + sso.dec_rate_arcsec_h.to_numpy() * dt_h / 3600
        ox, oy = gw.wcs_world2pix(sra, sdec, 0)
        ids = np.r_[ids, sso.src_id.to_numpy()]
        sx, sy = np.r_[sx, ox], np.r_[sy, oy]
        kinds = np.r_[kinds, np.full(len(ox), "sso", dtype=object)]
    rap, rin, rout = C.APERTURE[ref_m]
    flux, ferr_var, lbkg, vfrac = aperture_photometry(img, var, valid, sx, sy, rap, rin, rout)
    # Empty-sky aperture noise. Only bright sources are avoided; the robust (MAD)
    # scatter tolerates faint sources falling in some apertures, and in crowded
    # fields that confusion is part of the real noise.
    bright = ((src.gmag.to_numpy() < 16) | (src.w1.to_numpy() < 13))[inn]
    avoid = list(zip(sx[: len(bright)][bright], sy[: len(bright)][bright]))
    seed = zlib.crc32(spec["frame_uid"].encode())
    sky_noise, n_sky = sky_aperture_noise(img, valid, avoid, rap, seed=seed, avoid_r=5.0)
    if n_sky < 30:
        sky_noise, n_sky = sky_aperture_noise(img, valid, [], rap, seed=seed)
    sky_part = np.nanmedian(ferr_var) if np.isfinite(ferr_var).any() else 0.0
    ferr = np.sqrt(np.nan_to_num(sky_noise) ** 2 + np.clip(ferr_var ** 2 - sky_part ** 2, 0, None))
    snr = flux / ferr
    # artefact and hard-mask bits inside each aperture
    ap_flags = np.zeros(len(sx), np.uint32)
    for k in range(len(sx)):
        if not (np.isfinite(sx[k]) and np.isfinite(sy[k])):
            continue
        xc, yc = int(round(sx[k])), int(round(sy[k]))
        x0, x1, y0, y1 = max(xc - 2, 0), min(xc + 3, npx), max(yc - 2, 0), min(yc + 3, npx)
        if x1 > x0 and y1 > y0:
            ap_flags[k] = np.bitwise_or.reduce(flags[y0:y1, x0:x1].ravel())
    cx = np.full(len(sx), np.nan)
    cy = np.full(len(sx), np.nan)
    for k in np.where(snr > 5)[0]:
        cx[k], cy[k] = centroid(img, valid, sx[k], sy[k], r=CENTROID_R[m])
    phot = pd.DataFrame({
        "frame_uid": spec["frame_uid"], "src_id": ids, "src_kind": kinds, "x": sx.astype(np.float32),
        "y": sy.astype(np.float32), "flux": flux.astype(np.float32), "flux_err": ferr.astype(np.float32),
        "snr": snr.astype(np.float32), "aperture_valid_frac": vfrac.astype(np.float32),
        "local_bkg": lbkg.astype(np.float32), "aperture_flags": ap_flags,
        "artifact_in_aperture": (ap_flags & ARTIFACT[m]) != 0,
        "saturated": (ap_flags & SATURATED[m]) != 0,
        "centroid_dx": (cx - sx).astype(np.float32), "centroid_dy": (cy - sy).astype(np.float32),
    })

    c0, c1 = npx // 2 - 16, npx // 2 + 16
    row = {
        "frame_uid": spec["frame_uid"], **reg,
        "valid_frac": float(valid.mean()), "center_valid_frac": float(valid[c0:c1, c0:c1].mean()),
        "covered_frac": float(covered.mean()), "center_covered_frac": float(covered[c0:c1, c0:c1].mean()),
        "grid_bkg_removed": float(bkg) if np.isfinite(bkg) else np.nan, "pixel_noise": pix_std,
        "aperture_sky_noise": sky_noise, "n_sky_apertures": n_sky,
        "native_bkg": d["native_bkg"], "zodi_median": d["zodi_median"],
        "wave_center_um": float(np.nanmedian(wave[c0:c1, c0:c1])) if np.isfinite(wave[c0:c1, c0:c1]).any() else np.nan,
        "wave_min_um": float(np.nanmin(wave)) if np.isfinite(wave).any() else np.nan,
        "wave_max_um": float(np.nanmax(wave)) if np.isfinite(wave).any() else np.nan,
        "artifact_frac": float(((flags & ARTIFACT[m]) != 0)[valid].mean()) if valid.any() else np.nan,
    }
    sig = np.where(valid, np.sqrt(np.clip(var, 0, None)), 0.0).astype(np.float32)
    np.savez_compressed(spec["out"], image=np.where(valid, img, 0.0).astype(np.float32), sigma=sig,
                        flags=flags, valid=valid.astype(np.uint8), wave=wave, bandwidth=band)
    return row, phot
