"""Stage 5: real solar-system movers.

comet   - SPHEREx frames of interstellar comet 3I/ATLAS. One stack per exposure time
          (the two detectors that observed simultaneously), centred on the comet
          position seen from SPHEREx.
neossat - NEOSSat sequences of comets/asteroids and two stellar control fields.
          Frames without a celestial WCS are excluded. A stack is one target-night,
          split when the object travels more than 7 arcmin or the sequence pauses
          for more than an hour. Targets come from Horizons as seen from NEOSSat,
          other known objects from SkyBoT moved to the NEOSSat position.
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from common import json_dump, topocentric_shift
from s02_catalogs import query_allwise, query_gaia
from s03_ephemeris import sc_lookup, skybot_task, time_groups
from s04_stacks import run
from sources import build_field_sources

EPH = C.CACHE / "ephem"
CAT = C.CACHE / "catalogs"
MAX_PATH_DEG = 7.0 / 60
MAX_GAP_DAYS = 1.0 / 24


def interp_eph(eph, mjd):
    m = eph.mjd_utc.to_numpy()
    o = np.argsort(m)
    out = {}
    for c in ("RA", "DEC", "RA_rate", "DEC_rate", "delta"):
        v = eph[c].to_numpy(float)[o]
        if c == "RA":
            v = np.degrees(np.unwrap(np.radians(v)))
        out[c] = np.interp(mjd, m[o], v)
    out["RA"] = out["RA"] % 360
    return out


def mag_col(eph):
    for c in ("Tmag", "V", "Nmag"):
        if c in eph.columns and pd.to_numeric(eph[c], errors="coerce").notna().any():
            return pd.to_numeric(eph[c], errors="coerce").to_numpy()
    return np.full(len(eph), np.nan)


def fetch_catalogs(fields):
    with ThreadPoolExecutor(4) as ex:
        futs = [ex.submit(query_gaia, f["catalog_key"], f["ra"], f["dec"], CAT) for f in fields.values()]
        futs += [ex.submit(query_allwise, f["catalog_key"], f["ra"], f["dec"], CAT) for f in fields.values()]
        log = [fu.result() for fu in futs]
    bad = [r for r in log if not str(r[1]).startswith(("ok", "cached"))]
    print("catalog queries", len(log), "failed", bad)


def make_sources(mission, fields):
    d = C.CACHE / "sources"
    d.mkdir(parents=True, exist_ok=True)
    paths = {}
    dummy = {"ra": 0.0, "dec": 0.0, "pmra": 0.0, "pmdec": 0.0, "plx": 0.0, "ref_jyear": 2000.0, "otype": None}
    for key, f in fields.items():
        gaia = pd.read_parquet(CAT / f"{f['catalog_key']}-gaia.parquet")
        aw = pd.read_parquet(CAT / f"{f['catalog_key']}-allwise.parquet")
        src = build_field_sources({**f, "key": key, "is_control": True}, dummy, gaia, aw)
        p = d / f"{mission}-{f['slug']}.parquet"
        src.to_parquet(p, index=False)
        paths[f["slug"]] = p
    return paths


# --------------------------------------------------------------------------- 3I/ATLAS
def comet():
    fc = pd.read_parquet(C.CACHE / "frames_comet.parquet")
    eph = pd.read_parquet(EPH / "eph_3iatlas.parquet")
    fc["tg"] = time_groups(fc.mjd_mid.to_numpy())
    fields, rows = {}, []
    for k, g in fc.groupby("tg"):
        slug = f"3iatlas-t{k:02d}"
        p = interp_eph(eph, np.array([g.mjd_mid.mean()]))
        fields[f"comet:{slug}"] = {"mission": "comet", "slug": slug, "group": C.COMET_3I["group"],
                                   "ra": float(p["RA"][0]), "dec": float(p["DEC"][0]), "is_control": False,
                                   "catalog_key": f"comet_{slug}"}
        e = interp_eph(eph, g.mjd_mid.to_numpy())
        for i, r in enumerate(g.itertuples()):
            rows.append({"field": slug, "src_id": "sso:3I/ATLAS", "mjd_utc": r.mjd_mid, "ra_obs": e["RA"][i],
                         "dec_obs": e["DEC"][i], "ra_rate_arcsec_h": e["RA_rate"][i],
                         "dec_rate_arcsec_h": e["DEC_rate"][i], "vmag": np.nanmedian(mag_col(eph)),
                         "type": "Comet (interstellar)", "posunc_arcsec": np.nan})
    fc["field"] = fc.tg.map(lambda k: f"3iatlas-t{k:02d}")
    fetch_catalogs(fields)
    src_paths = make_sources("comet", fields)
    run("comet", fc, fields, src_paths, pd.DataFrame(rows))
    json_dump(fields, C.OUT / "comet" / "fields.json")


# --------------------------------------------------------------------------- NEOSSat
def wcs_centre(path):
    with fits.open(C.SRC / path, memmap=False) as h:
        w = WCS(h[0].header)
        ny, nx = h[0].data.shape
    ra, dec = w.wcs_pix2world([[nx / 2, ny / 2]], 0)[0]
    return float(ra), float(dec)


def neossat():
    fn = pd.read_parquet(C.CACHE / "frames_neossat.parquet")
    excluded = fn[~fn.has_wcs][["path", "group_dir"]].assign(reason="no celestial WCS in header")
    fn = fn[fn.has_wcs].copy()
    fn["proc"] = fn.path.str.extract(r"_(clean|cor|cord)\.fits$")[0]
    cen = [wcs_centre(p) for p in fn.path]
    fn["ptg_ra"] = [c[0] for c in cen]
    fn["ptg_dec"] = [c[1] for c in cen]
    vec = pd.read_parquet(EPH / "sc_vectors_neossat.parquet")

    fields, sso_rows, chunk_of = {}, [], {}
    for gdir, g in fn.groupby("group_dir"):
        group, hid, _, cls = C.NEOSSAT_GROUPS[gdir]
        eph = pd.read_parquet(EPH / f"eph_{group}.parquet") if hid else None
        g = g.sort_values("mjd_mid")
        if eph is not None:
            e = interp_eph(eph, g.mjd_mid.to_numpy())
            ra, dec = e["RA"], e["DEC"]
        else:
            ra, dec = g.ptg_ra.to_numpy(), g.ptg_dec.to_numpy()
        night = np.floor(g.mjd_mid.to_numpy()).astype(int)
        chunk, start, c = [], 0, 0
        for i in range(len(g)):
            if i > 0:
                sep = np.hypot((ra[i] - ra[start]) * np.cos(np.radians(dec[i])), dec[i] - dec[start])
                gap = g.mjd_mid.iat[i] - g.mjd_mid.iat[i - 1]
                if night[i] != night[i - 1] or gap > MAX_GAP_DAYS or sep > MAX_PATH_DEG:
                    c += 1
                    start = i
            chunk.append(c)
        g = g.assign(chunk=chunk)
        for k, gg in g.groupby("chunk"):
            idx = g.index.get_indexer(gg.index)
            slug = f"{gdir}-c{k:02d}"
            for fi in gg.index:
                chunk_of[fi] = slug
            mid = len(idx) // 2
            fields[f"neossat:{slug}"] = {"mission": "neossat", "slug": slug, "group": group,
                                         "ra": float(np.median(ra[idx])) if eph is None else float(ra[idx][mid]),
                                         "dec": float(np.median(dec[idx])) if eph is None else float(dec[idx][mid]),
                                         "is_control": eph is None, "catalog_key": f"neossat_{slug}",
                                         "target_class": cls, "mjd_start": float(gg.mjd_mid.min()),
                                         "mjd_end": float(gg.mjd_mid.max())}
            if eph is not None:
                for j, r in enumerate(gg.itertuples()):
                    sso_rows.append({"field": slug, "src_id": f"sso:{hid}", "mjd_utc": r.mjd_mid,
                                     "ra_obs": ra[idx][j], "dec_obs": dec[idx][j],
                                     "ra_rate_arcsec_h": e["RA_rate"][idx][j], "dec_rate_arcsec_h": e["DEC_rate"][idx][j],
                                     "vmag": float(np.nanmedian(mag_col(eph))), "type": cls.capitalize() + " (NEOSSat target)",
                                     "posunc_arcsec": np.nan})
    fn["field"] = fn.index.map(chunk_of)
    fn["band"] = 0
    print("neossat stacks", len(fields), "frames", len(fn))

    # other known objects in each stack: SkyBoT at the stack mid time, then moved to NEOSSat per frame
    tasks = []
    for key, f in fields.items():
        rad = C.GRID["neossat"]["npix"] * C.GRID["neossat"]["scale_arcsec"] / 3600 / np.sqrt(2) + 0.03
        tasks.append((key, f["ra"], f["dec"], rad, 0.5 * (f["mjd_start"] + f["mjd_end"])))
    sk_path = EPH / "skybot_neossat.parquet"
    if not sk_path.exists() and C.USE_SKYBOT:
        with ThreadPoolExecutor(6) as ex:
            res = list(ex.map(skybot_task, tasks))
        status = {k: s for k, s, _ in res}
        json_dump(status, EPH / "skybot_neossat-status.json")
        parts = [d for _, _, d in res if d is not None]
        (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()).to_parquet(sk_path, index=False)
    sk = pd.read_parquet(sk_path) if (C.USE_SKYBOT and sk_path.exists()) else pd.DataFrame()
    if len(sk):
        for key, s in sk.groupby("query_key"):
            slug = key.split(":", 1)[1]
            frames = fn[fn.field == slug]
            for r in frames.itertuples():
                dt_h = (r.mjd_mid - s.mjd_utc.to_numpy()) * 24
                cosd = np.cos(np.radians(s.dec_geo.to_numpy()))
                gra = s.ra_geo.to_numpy() + s.ra_rate_arcsec_h.to_numpy() * dt_h / 3600 / cosd
                gdec = s.dec_geo.to_numpy() + s.dec_rate_arcsec_h.to_numpy() * dt_h / 3600
                sc = sc_lookup(vec, np.full(len(s), r.mjd_mid))
                ora, odec, _ = topocentric_shift(gra, gdec, s.geodist_au.to_numpy(), sc)
                num = s.number.astype(str)
                ids = "sso:" + np.where(num.isin(["--", "", "nan", "None", "0"]), s.name.astype(str), num)
                for j in range(len(s)):
                    sso_rows.append({"field": slug, "src_id": ids[j], "mjd_utc": r.mjd_mid, "ra_obs": ora[j],
                                     "dec_obs": odec[j], "ra_rate_arcsec_h": s.ra_rate_arcsec_h.iat[j],
                                     "dec_rate_arcsec_h": s.dec_rate_arcsec_h.iat[j], "vmag": s.vmag.iat[j],
                                     "type": s.type.iat[j], "posunc_arcsec": s.posunc_arcsec.iat[j]})
    sso = pd.DataFrame(sso_rows)
    # A NEOSSat target also returned by SkyBoT (possibly under another name) would be
    # listed twice: drop SkyBoT rows within 20" of the Horizons target in the same frame.
    sso = sso.drop_duplicates(["field", "src_id", "mjd_utc"], keep="first")
    tgt_ids = {f"sso:{v[1]}" for v in C.NEOSSAT_GROUPS.values() if v[1]}
    tg = sso[sso.src_id.isin(tgt_ids)][["field", "mjd_utc", "ra_obs", "dec_obs"]]
    m = sso.merge(tg, on=["field", "mjd_utc"], how="left", suffixes=("", "_t"))
    sep = np.hypot((m.ra_obs - m.ra_obs_t) * np.cos(np.radians(m.dec_obs)), m.dec_obs - m.dec_obs_t) * 3600
    dup = (~m.src_id.isin(tgt_ids)) & (sep < 20)
    print("SkyBoT rows duplicating a NEOSSat target", int(dup.sum()), m.loc[dup, "src_id"].unique()[:5])
    sso = sso[~dup.to_numpy()].reset_index(drop=True)
    fetch_catalogs(fields)
    src_paths = make_sources("neossat", fields)
    run("neossat", fn, fields, src_paths, sso)
    out = C.OUT / "neossat"
    json_dump(fields, out / "fields.json")
    excluded.to_csv(out / "excluded-frames.csv", index=False)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    which = sys.argv[1]
    {"comet": comet, "neossat": neossat}[which]()
