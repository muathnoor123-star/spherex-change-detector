"""Stage 3: solar-system ground truth.

1. Geocentric spacecraft vectors (JPL Horizons) at every frame time, so that
   ephemeris positions can be moved from the geocentre to the actual observer.
2. IMCCE SkyBoT cone searches: every known asteroid/comet inside each SPHEREx and
   WISE grid at each exposure time.
3. Horizons ephemerides seen from the spacecraft for the named targets
   (3I/ATLAS for SPHEREx, the NEOSSat comets and asteroids).
Outputs in cache/ephem/.
"""
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.time import Time

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from common import json_dump, topocentric_shift

OUT = C.CACHE / "ephem"
SAME_TIME_DAYS = 0.002   # frames closer than ~3 min share one exposure time


def time_groups(mjd):
    """Integer group id per frame; frames within SAME_TIME_DAYS share a group."""
    order = np.argsort(mjd)
    gid = np.empty(len(mjd), int)
    g = -1
    last = -1e9
    for i in order:
        if mjd[i] - last > SAME_TIME_DAYS:
            g += 1
        gid[i] = g
        last = mjd[i]
    return gid


def horizons_retry(fn, tries=4):
    err = None
    for k in range(tries):
        try:
            return fn()
        except Exception as e:  # network or server-side errors are retried, then reported
            err = e
            time.sleep(5 * (k + 1))
    raise err


def spacecraft_vectors(mission, mjds):
    """Geocentric ICRF position of the spacecraft (AU) at UTC MJDs."""
    from astroquery.jplhorizons import Horizons
    mjds = np.unique(np.round(np.asarray(mjds), 6))
    jd_tdb = Time(mjds, format="mjd", scale="utc").tdb.jd
    rows = []
    for i in range(0, len(jd_tdb), 40):
        chunk = list(jd_tdb[i:i + 40])
        v = horizons_retry(lambda: Horizons(id=C.HORIZONS_SPACECRAFT_ID[mission], location="500@399",
                                            epochs=chunk, id_type=None).vectors(refplane="earth"))
        for jd, x, y, z in zip(v["datetime_jd"], v["x"], v["y"], v["z"]):
            rows.append((float(jd), float(x), float(y), float(z)))
    df = pd.DataFrame(rows, columns=["jd_tdb", "x_au", "y_au", "z_au"]).sort_values("jd_tdb")
    # map back to the requested UTC times
    df["mjd_utc"] = np.interp(df["jd_tdb"], np.sort(jd_tdb), np.sort(mjds))
    return df


def sc_lookup(vec, mjd):
    m = vec["mjd_utc"].to_numpy()
    return np.stack([np.interp(mjd, m, vec[c].to_numpy()) for c in ("x_au", "y_au", "z_au")], -1)


def skybot(ra, dec, radius_deg, mjd):
    from astroquery.imcce import Skybot
    try:
        t = Skybot.cone_search(SkyCoord(ra * u.deg, dec * u.deg), radius_deg * u.deg,
                               Time(mjd, format="mjd", scale="utc"), location="500", cache=False)
    except RuntimeError as e:
        if "No solar system object" in str(e):
            return "empty", None
        raise
    if t is None or len(t) == 0:   # astroquery >= 0.4.11 returns an empty table plus a warning
        return "empty", None
    df = pd.DataFrame({
        "number": [str(x) for x in t["Number"]], "name": [str(x) for x in t["Name"]],
        "ra_geo": t["RA"].to_value(u.deg), "dec_geo": t["DEC"].to_value(u.deg),
        "type": [str(x) for x in t["Type"]], "vmag": np.asarray(t["V"], float),
        "posunc_arcsec": t["posunc"].to_value(u.arcsec),
        "ra_rate_arcsec_h": t["RA_rate"].to_value(u.arcsec / u.h),
        "dec_rate_arcsec_h": t["DEC_rate"].to_value(u.arcsec / u.h),
        "geodist_au": t["geodist"].to_value(u.au), "heliodist_au": t["heliodist"].to_value(u.au),
    })
    return "ok", df


SKYBOT_CACHE = C.CACHE / "ephem" / "skybot_queries"


def skybot_task(args):
    """One cone search, cached on disk so an interrupted run resumes where it stopped."""
    import hashlib
    from astroquery.imcce import conf
    conf.timeout = 60
    key, ra, dec, rad, mjd = args
    SKYBOT_CACHE.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1(f"{key}|{ra:.6f}|{dec:.6f}|{rad:.4f}|{mjd:.6f}".encode()).hexdigest()[:16]
    pq, empty = SKYBOT_CACHE / f"{h}.parquet", SKYBOT_CACHE / f"{h}.empty"
    if pq.exists():
        return key, "ok", pd.read_parquet(pq)
    if empty.exists():
        return key, "empty", None
    for k in range(3):
        try:
            status, df = skybot(ra, dec, rad, mjd)
            if df is not None:
                df.insert(0, "query_key", key)
                df.insert(1, "mjd_utc", mjd)
                df.to_parquet(pq, index=False)
            else:
                empty.touch()
            return key, status, df
        except Exception as e:
            err = repr(e)[:300]
            time.sleep(5 * (k + 1))
    return key, "error: " + err, None


def target_ephemeris(target_id, id_type, location, mjds, comet):
    from astroquery.jplhorizons import Horizons
    jd = list(Time(np.asarray(mjds), format="mjd", scale="utc").jd)
    rows = []
    for i in range(0, len(jd), 40):
        chunk = jd[i:i + 40]
        kw = {"closest_apparition": True, "no_fragments": True} if comet else {}
        e = horizons_retry(lambda: Horizons(id=target_id, id_type=id_type, location=location,
                                            epochs=chunk).ephemerides(extra_precision=True, **kw))
        cols = [c for c in ("datetime_jd", "RA", "DEC", "delta", "RA_rate", "DEC_rate", "V", "Tmag", "Nmag", "RA_3sigma", "DEC_3sigma") if c in e.colnames]
        rows.append(e[cols].to_pandas())
    df = pd.concat(rows, ignore_index=True)
    df["mjd_utc"] = df["datetime_jd"] - 2400000.5
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fields = json.loads((C.CACHE / "fields.json").read_text())
    fs = pd.read_parquet(C.CACHE / "frames_spherex.parquet")
    fc = pd.read_parquet(C.CACHE / "frames_comet.parquet")
    fw = pd.read_parquet(C.CACHE / "frames_wise.parquet")
    fn = pd.read_parquet(C.CACHE / "frames_neossat.parquet")

    # 1. spacecraft vectors
    vec = {}
    for mission, mjds in (("spherex", np.r_[fs.mjd_mid, fc.mjd_mid]), ("wise", fw.mjd_mid.to_numpy()),
                          ("neossat", fn.mjd_mid.to_numpy())):
        p = OUT / f"sc_vectors_{mission}.parquet"
        if not p.exists():
            spacecraft_vectors(mission, mjds).to_parquet(p, index=False)
        vec[mission] = pd.read_parquet(p)
        r_km = np.linalg.norm(vec[mission][["x_au", "y_au", "z_au"]].to_numpy(), axis=1) * 149597870.7
        print(mission, "vectors", len(vec[mission]), "geocentric distance km min/max", r_km.min().round(), r_km.max().round())

    # 2. SkyBoT searches, one per field and exposure time
    tasks = []
    half_diag = {m: C.GRID[m]["npix"] * C.GRID[m]["scale_arcsec"] / 3600 / np.sqrt(2) for m in ("spherex", "wise")}
    for mission, frames in (("spherex", fs), ("wise", fw)):
        for slug, g in frames.groupby("field"):
            f = fields[f"{mission}:{slug}"]
            gid = time_groups(g.mjd_mid.to_numpy())
            for k in np.unique(gid):
                mjd = float(g.mjd_mid.to_numpy()[gid == k].mean())
                tasks.append((f"{mission}:{slug}:{mjd:.6f}", f["ra"], f["dec"], half_diag[mission] + 0.03, mjd))
    done = OUT / "skybot.parquet"
    status_path = OUT / "skybot-status.json"
    if not done.exists():
        print("skybot queries", len(tasks), flush=True)
        results, status = [], {}
        from concurrent.futures import as_completed
        with ThreadPoolExecutor(8) as ex:
            futs = [ex.submit(skybot_task, t) for t in tasks]
            for i, fu in enumerate(as_completed(futs), 1):
                key, st, df = fu.result()
                status[key] = st
                if df is not None:
                    results.append(df)
                if i % 25 == 0 or i == len(tasks):
                    n_err = sum(v.startswith("error") for v in status.values())
                    print(f"skybot {i}/{len(tasks)} done, {n_err} errors", flush=True)
        sk = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
        sk.to_parquet(done, index=False)
        json_dump(status, status_path)
    sk = pd.read_parquet(done)
    status = json.loads(status_path.read_text())
    errors = {k: v for k, v in status.items() if v.startswith("error")}
    print("skybot rows", len(sk), "queries", len(status), "errors", len(errors))

    # move each SkyBoT position from the geocentre to the spacecraft
    mission = sk["query_key"].str.split(":").str[0]
    out = []
    for m in ("spherex", "wise"):
        s = sk[mission == m].copy()
        if s.empty:
            continue
        sc = sc_lookup(vec[m], s["mjd_utc"].to_numpy())
        ra, dec, dist = topocentric_shift(s.ra_geo.to_numpy(), s.dec_geo.to_numpy(), s.geodist_au.to_numpy(), sc)
        s["ra_obs"], s["dec_obs"], s["dist_obs_au"] = ra, dec, dist
        s["parallax_shift_arcsec"] = SkyCoord(ra * u.deg, dec * u.deg).separation(
            SkyCoord(s.ra_geo.to_numpy() * u.deg, s.dec_geo.to_numpy() * u.deg)).arcsec
        out.append(s)
    sk = pd.concat(out, ignore_index=True)
    sk.to_parquet(OUT / "skybot_observer.parquet", index=False)
    print("observer-frame shift arcsec median/max", sk.parallax_shift_arcsec.median().round(2), sk.parallax_shift_arcsec.max().round(2))

    # 3. named targets from the spacecraft
    p = OUT / "eph_3iatlas.parquet"
    if not p.exists():
        target_ephemeris(C.COMET_3I["horizons_id"], None, C.HORIZONS_OBSERVER["spherex"],
                         fc.mjd_mid.to_numpy(), comet=True).to_parquet(p, index=False)
    for folder, (group, hid, idtype, cls) in C.NEOSSAT_GROUPS.items():
        if hid is None:
            continue
        p = OUT / f"eph_{group}.parquet"
        if p.exists():
            continue
        mj = fn.loc[fn.group_dir == folder, "mjd_mid"].to_numpy()
        try:
            target_ephemeris(hid, idtype, C.HORIZONS_OBSERVER["neossat"], mj, comet=(cls == "comet")).to_parquet(p, index=False)
            print("ephemeris", group, len(mj))
        except Exception as e:  # tellscan: ignore C05 -- printed; the missing file stops stage 5 loudly
            print("ephemeris FAILED", group, repr(e)[:400])


if __name__ == "__main__":
    main()
