"""Stage 1: parse target records and index every image file that the dataset uses.

Outputs (cache/): targets.json, frames_spherex.parquet, frames_comet.parquet,
frames_wise.parquet, frames_neossat.parquet
"""
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from common import json_dump, parse_sesame


def rel(path):
    """Path relative to the raw download; files fetched later (cache/raw_extra) keep an absolute path."""
    p = Path(path)
    try:
        return str(p.relative_to(C.SRC)).replace("\\", "/")
    except ValueError:
        return p.as_posix()


def sha256_file(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spherex_header(path):
    with fits.open(path, memmap=False) as h:
        hd = h["IMAGE"].header
        ny, nx = h["IMAGE"].shape
        return {
            "path": rel(path),
            "obsid": hd.get("OBSID"), "detector": int(hd["DETECTOR"]),
            "mjd_mid": float(hd["MJD-AVG"]), "mjd_beg": float(hd["MJD-BEG"]),
            "exptime_s": float(hd.get("XPOSURE", np.nan)), "psf_fwhm_arcsec": float(hd.get("PSF_FWHM", np.nan)),
            "nx": nx, "ny": ny, "time_scale": hd.get("TIMESYS", "UTC"),
        }


def wise_header(path):
    with fits.open(path, memmap=False) as h:
        hd = h[0].header
        p = Path(path)
        return {
            "path": str(p.relative_to(C.SRC)).replace("\\", "/"),
            "frame_key": p.name.split("-")[0], "band": int(hd["BAND"]),
            "bin": p.parent.name, "mjd_mid": float(hd["MJD_OBS"]), "exptime_s": float(hd["EXPTIME"]),
            "magzp": float(hd["MAGZP"]), "moonsep_deg": float(hd.get("MOONSEP", np.nan)),
            "scan": hd.get("SCAN"), "nx": hd["NAXIS1"], "ny": hd["NAXIS2"], "time_scale": "UTC",
        }


def neossat_header(path):
    with fits.open(path, memmap=False) as h:
        hd = h[0].header
        try:
            w = WCS(hd)
            has_wcs = bool(w.has_celestial) and "RA" in str(hd.get("CTYPE1", ""))
        except Exception:
            has_wcs = False
        exp = float(hd.get("EXPOSURE", hd.get("AEXPTIME", np.nan)))
        t0 = Time(hd["DATE-OBS"], scale="utc")
        p = Path(path)
        return {
            "path": str(p.relative_to(C.SRC)).replace("\\", "/"), "group_dir": p.parent.name,
            "object": hd.get("OBJECT"), "mjd_mid": float(t0.mjd + exp / 2 / 86400.0), "exptime_s": exp,
            "mode": hd.get("MODE"), "has_wcs": has_wcs, "nx": hd["NAXIS1"], "ny": hd["NAXIS2"],
            "ra_vel": hd.get("RA_VEL"), "dec_vel": hd.get("DEC_VEL"), "time_scale": "UTC",
        }


def run(fn, paths):
    with ProcessPoolExecutor(C.WORKERS) as ex:
        return list(ex.map(fn, paths, chunksize=max(1, len(paths) // (C.WORKERS * 8))))


def main():
    C.CACHE.mkdir(parents=True, exist_ok=True)
    sha = {}
    with open(C.SRC / "file-index.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sha[row["relative_path"].replace("\\", "/")] = row["sha256"]

    targets = {"spherex": {}, "wise": {}}
    fitted = {}
    if (C.CACHE / "target_astrometry.json").exists():   # motion measured from the WISE frames (stage 3b)
        import json
        fitted = json.loads((C.CACHE / "target_astrometry.json").read_text())

    def with_fit(t, group):
        f = fitted.get(group)
        if f:
            t = {**t, **{k: f[k] for k in ("ra", "dec", "pmra", "pmdec", "plx", "ref_jyear")},
                 "astrometry_source": "fit_to_wise_frames"}
        return t

    for slug, (group, xml) in C.SPHEREX_FIELDS.items():
        if xml is None:   # selected from Gaia DR3 by proper motion, no name resolver involved
            e = C.EXTRA_FIELDS[slug]
            t = {"name": e["name"], "otype": None, "sptype": None, "rv": None,
                 "identity_source": "gaia_dr3_proper_motion_selection",
                 **{k: e[k] for k in ("ra", "dec", "pmra", "pmdec", "plx", "ref_jyear")}}
        else:
            t = with_fit(parse_sesame(C.SRC / xml), group)
        targets["spherex"][slug] = {**t, "group": group, "field_ra": t["ra"], "field_dec": t["dec"]}
    for slug, (group, xml, dra) in C.WISE_FIELDS.items():
        t = with_fit(parse_sesame(C.SRC / xml), group)
        targets["wise"][slug] = {**t, "group": group, "is_control": dra != 0.0,
                                 "field_ra": t["ra"] + dra, "field_dec": t["dec"]}
    json_dump(targets, C.CACHE / "targets.json")

    sx = []
    for slug, d in C.SPHEREX_FRAME_DIRS.items():
        sx += [(slug, str(p)) for p in sorted((C.SRC / d).glob("*.fits"))]   # an absolute d replaces C.SRC
    rows = run(spherex_header, [p for _, p in sx])
    df = pd.DataFrame(rows)
    df.insert(0, "field", [s for s, _ in sx])
    df["sha256"] = df["path"].map(sha)
    missing = df.sha256.isna()
    if missing.any():
        with ProcessPoolExecutor(C.WORKERS) as ex:
            df.loc[missing, "sha256"] = list(ex.map(sha256_file, df.loc[missing, "path"], chunksize=8))
    df.to_parquet(C.CACHE / "frames_spherex.parquet", index=False)
    print("spherex frames", len(df), df.groupby("field").size().to_dict())

    cp = sorted((C.SRC / C.COMET_3I["folder"]).glob("*.fits"))
    dc = pd.DataFrame(run(spherex_header, [str(p) for p in cp]))
    dc["sha256"] = dc["path"].map(sha)
    dc.to_parquet(C.CACHE / "frames_comet.parquet", index=False)
    print("comet frames", len(dc))

    wp = []
    for slug in C.WISE_FIELDS:
        wp += [(slug, str(p)) for p in sorted((C.SRC / "wise/fullframes" / slug).glob("*/*-int-1b.fits"))]
    dw = pd.DataFrame(run(wise_header, [p for _, p in wp]))
    dw.insert(0, "field", [s for s, _ in wp])
    base = dw["path"].str.replace("-int-1b.fits", "", regex=False)
    dw["unc_path"] = base + "-unc-1b.fits.gz"
    dw["msk_path"] = base + "-msk-1b.fits.gz"
    for c in ("path", "unc_path", "msk_path"):
        dw[c.replace("path", "sha256")] = dw[c].map(sha)
    missing = [p for c in ("unc_path", "msk_path") for p in dw[c] if not (C.SRC / p).exists()]
    dw["complete"] = [(C.SRC / u).exists() and (C.SRC / m).exists() for u, m in zip(dw.unc_path, dw.msk_path)]
    dw.to_parquet(C.CACHE / "frames_wise.parquet", index=False)
    print("wise int frames", len(dw), "missing companions", len(missing))

    npth = sorted((C.SRC / "neossat/fits").glob("*/*.fits"))
    dn = pd.DataFrame(run(neossat_header, [str(p) for p in npth]))
    dn["sha256"] = dn["path"].map(sha)
    dn.to_parquet(C.CACHE / "frames_neossat.parquet", index=False)
    print("neossat frames", len(dn), "with WCS", int(dn.has_wcs.sum()))
    print(dn.groupby(["group_dir", "has_wcs"]).size())


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
