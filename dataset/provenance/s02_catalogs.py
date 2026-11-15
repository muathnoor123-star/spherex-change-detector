"""Stage 2: complete reference catalogues for every sky field.

Gaia DR3 (no row limit) with DSC class probabilities and variability classes, plus
the AllWISE catalogue so that infrared sources without a Gaia counterpart are also
known. (Server-side joins to the 2MASS/AllWISE neighbour tables timed out with HTTP
408, so the AllWISE cross-match is done locally in stage 4.)
Outputs: cache/catalogs/<field_key>-gaia.parquet, <field_key>-allwise.parquet, fields.json
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from common import earth_bary_au, json_dump, mjd_to_jyear, propagate

GAIA_ADQL = """
SELECT g.source_id, g.ra, g.dec, g.ref_epoch, g.ra_error, g.dec_error,
       g.pmra, g.pmdec, g.pmra_error, g.pmdec_error, g.parallax, g.parallax_error,
       g.astrometric_params_solved, g.ruwe, g.duplicated_source, g.non_single_star,
       g.phot_g_mean_mag, g.phot_bp_mean_mag, g.phot_rp_mean_mag, g.bp_rp, g.radial_velocity,
       g.phot_variable_flag, g.classprob_dsc_combmod_star, g.classprob_dsc_combmod_galaxy,
       g.classprob_dsc_combmod_quasar, g.in_qso_candidates, g.in_galaxy_candidates,
       v.best_class_name AS vari_class, v.best_class_score AS vari_class_score
FROM gaiadr3.gaia_source AS g
LEFT OUTER JOIN gaiadr3.vari_classifier_result AS v ON v.source_id = g.source_id
WHERE 1 = CONTAINS(POINT('ICRS', g.ra, g.dec), CIRCLE('ICRS', {ra:.7f}, {dec:.7f}, {rad}))
"""

ALLWISE_COLS = ("designation,ra,dec,sigra,sigdec,w1mpro,w1sigmpro,w1snr,w2mpro,w2sigmpro,w2snr,"
                "w3mpro,w4mpro,cc_flags,ext_flg,ph_qual,nb,na,pmra,sigpmra,pmdec,sigpmdec")


def field_centres():
    """Grid centre = the target's apparent position at the mean epoch of that
    field's frames, so the target stays near the middle of every stack."""
    import json
    targets = json.loads((C.CACHE / "targets.json").read_text())
    fs = pd.read_parquet(C.CACHE / "frames_spherex.parquet")
    fw = pd.read_parquet(C.CACHE / "frames_wise.parquet")
    out = {}
    for mission, frames in (("spherex", fs), ("wise", fw)):
        for slug, t in targets[mission].items():
            mjd = float(frames.loc[frames.field == slug, "mjd_mid"].mean())
            obs = earth_bary_au([mjd])[0]
            ra, dec = propagate(t["ra"], t["dec"], t["pmra"], t["pmdec"], t["plx"] or 0.0,
                                t["ref_jyear"], mjd_to_jyear(mjd), obs)
            ra = float(ra) + (t["field_ra"] - t["ra"])
            out[f"{mission}:{slug}"] = {"mission": mission, "slug": slug, "group": t["group"],
                                         "ra": ra, "dec": float(dec), "mean_mjd": mjd,
                                         "is_control": bool(t.get("is_control", False))}
    return out


def catalog_key(f):
    return f"{f['group']}{'_control' if f['is_control'] else ''}"


ESA_SYNC = "https://gea.esac.esa.int/tap-server/tap/sync"
IRSA_STATE = {"down": False}   # set on the first IRSA failure so later fields do not wait for the timeout


def esa_sync(adql, timeout=300):
    """ESA Gaia archive, synchronous TAP. Returns a DataFrame with lower-case columns."""
    import io
    import requests
    r = requests.post(ESA_SYNC, data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": adql}, timeout=timeout)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.lower() for c in df.columns]
    return df


def query_gaia(key, ra, dec, outdir):
    path = outdir / f"{key}-gaia.parquet"
    if path.exists():
        return key, "cached", len(pd.read_parquet(path))
    err = ""
    for attempt in range(4):
        try:
            df = esa_sync(GAIA_ADQL.format(ra=ra, dec=dec, rad=C.CATALOG_RADIUS_DEG))
            df["source_id"] = df["source_id"].astype("int64")
            df["vari_class"] = df["vari_class"].astype(object).where(df["vari_class"].notna(), "").astype(str)
            for c in ("duplicated_source", "in_qso_candidates", "in_galaxy_candidates"):
                if df[c].dtype == object:
                    df[c] = df[c].astype(str).str.lower().isin(["true", "1", "t"])
            df = df.drop_duplicates("source_id").reset_index(drop=True)
            df.to_parquet(path, index=False)
            return key, "ok", len(df)
        except Exception as e:  # tellscan: ignore C05 -- retried, then returned as FAILED and printed
            err = repr(e)
            time.sleep(15 * (attempt + 1))
    return key, "FAILED " + err[:200], 0


def query_allwise(key, ra, dec, outdir):
    """AllWISE sources in the field: IRSA first, then ESA's copy of the same catalogue
    (gaiadr1.allwise_original_valid) when IRSA is unavailable. The ESA copy has no
    S/N or motion columns; S/N is derived from the magnitude error."""
    from astroquery.ipac.irsa import Irsa
    path = outdir / f"{key}-allwise.parquet"
    if path.exists():
        return key, "cached", len(pd.read_parquet(path))
    err = "IRSA skipped after an earlier failure in this run"
    try:
        if IRSA_STATE["down"]:
            raise ConnectionError(err)
        Irsa.TIMEOUT = 90
        tab = Irsa.query_region(SkyCoord(ra * u.deg, dec * u.deg), catalog="allwise_p3as_psd",
                                spatial="Cone", radius=C.CATALOG_RADIUS_DEG * u.deg, columns=ALLWISE_COLS)
        df = tab.to_pandas()
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype(str)
        df.to_parquet(path, index=False)
        return key, "ok", len(df)
    except Exception as e:  # tellscan: ignore C05 -- falls through to the ESA copy below
        err = repr(e)
        IRSA_STATE["down"] = True
    for attempt in range(4):
        try:
            d = esa_sync("SELECT designation, ra, dec, w1mpro, w1mpro_error, w2mpro, w2mpro_error, w3mpro, w4mpro, "
                         "cc_flags, ext_flag, ph_qual FROM gaiadr1.allwise_original_valid WHERE 1=CONTAINS(POINT('ICRS',ra,dec),"
                         f"CIRCLE('ICRS',{ra:.7f},{dec:.7f},{C.CATALOG_RADIUS_DEG}))")
            df = pd.DataFrame({
                "designation": d.designation.astype(str), "ra": d.ra, "dec": d.dec,
                "w1mpro": d.w1mpro, "w1sigmpro": d.w1mpro_error, "w1snr": 1.0857 / d.w1mpro_error,
                "w2mpro": d.w2mpro, "w2sigmpro": d.w2mpro_error, "w2snr": 1.0857 / d.w2mpro_error,
                "w3mpro": d.w3mpro, "w4mpro": d.w4mpro, "cc_flags": d.cc_flags.astype(str),
                "ext_flg": d.ext_flag, "ph_qual": d.ph_qual.astype(str),
                "pmra": np.nan, "sigpmra": np.nan, "pmdec": np.nan, "sigpmdec": np.nan,
            })
            df.to_parquet(path, index=False)
            return key, "ok (ESA copy of AllWISE)", len(df)
        except Exception as e:  # tellscan: ignore C05 -- retried, then returned as FAILED and printed
            err = repr(e)
            time.sleep(15 * (attempt + 1))
    return key, "FAILED " + err[:200], 0


def main():
    outdir = C.CACHE / "catalogs"
    outdir.mkdir(parents=True, exist_ok=True)
    fields = field_centres()
    queries = {}
    for k, f in fields.items():
        ck = catalog_key(f)
        f["catalog_key"] = ck
        queries.setdefault(ck, (f["ra"], f["dec"]))
    # a field seen by both missions shares one query; check the two centres agree
    for k, f in fields.items():
        qra, qdec = queries[f["catalog_key"]]
        sep = SkyCoord(qra * u.deg, qdec * u.deg).separation(SkyCoord(f["ra"] * u.deg, f["dec"] * u.deg)).arcmin
        f["catalog_centre_offset_arcmin"] = float(sep)
        assert sep < 1.5, (k, sep)
    json_dump(fields, C.CACHE / "fields.json")

    log = []
    with ThreadPoolExecutor(4) as ex:
        futs = [ex.submit(query_gaia, k, ra, dec, outdir) for k, (ra, dec) in queries.items()]
        futs += [ex.submit(query_allwise, k, ra, dec, outdir) for k, (ra, dec) in queries.items()]
        for fu in futs:
            r = fu.result()
            log.append(r)
            print(r, flush=True)
    json_dump(log, C.CACHE / "catalogs" / "query-log.json")

    # sky separation between distinct physical groups: overlapping fields would leak across folds
    keys = list(queries)
    cs = SkyCoord([queries[k][0] for k in keys] * u.deg, [queries[k][1] for k in keys] * u.deg)
    for i in range(len(keys)):
        sep = cs[i].separation(cs).deg
        for j in np.where((sep < 2 * C.CATALOG_RADIUS_DEG + 0.05) & (np.arange(len(keys)) != i))[0]:
            gi, gj = keys[i].replace("_control", ""), keys[j].replace("_control", "")
            if gi != gj:
                print("WARNING overlapping fields from different groups:", keys[i], keys[j], sep[j])


if __name__ == "__main__":
    main()
