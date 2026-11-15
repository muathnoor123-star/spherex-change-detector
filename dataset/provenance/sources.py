"""Per-field source lists: Gaia DR3 + AllWISE-only infrared sources + the named target.

Every source gets an astrometric model (position, reference epoch, proper motion,
parallax) so it can be placed in any frame, plus catalogue-derived class labels.
"""
import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import config as C
from common import mjd_to_jyear, propagate

ALLWISE_JYEAR = mjd_to_jyear(55400.0)   # AllWISE mean epoch, 2010.56
MATCH_ARCSEC = 3.0

TARGET_CLASS = {
    "barnard": "star", "proxima": "star", "ross128": "star", "teegarden": "star", "wolf359": "star",
    "lalande21185": "star", "61cyg_a": "star", "wise0855": "brown_dwarf", "epsilon_indi_b": "brown_dwarf",
    "luhman16": "brown_dwarf", "r_draconis": "variable_star", "rw_aurigae": "variable_star",
    "m31": "galaxy", "m51": "galaxy", "m101": "galaxy",
}


def gaia_class(df):
    cls = np.full(len(df), "unknown_point", dtype=object)
    sub = np.full(len(df), "", dtype=object)
    pstar = df.classprob_dsc_combmod_star.to_numpy(float)
    pgal = df.classprob_dsc_combmod_galaxy.to_numpy(float)
    pqso = df.classprob_dsc_combmod_quasar.to_numpy(float)
    plx_snr = (df.parallax / df.parallax_error).to_numpy(float)
    star = (pstar > 0.9) | (plx_snr > 5)
    cls[star] = "star"
    cls[(pgal > 0.9) & ~star] = "galaxy"
    cls[(pqso > 0.9) & ~star] = "quasar"
    # Gaia returns an empty string, not null, when a source has no variability class
    vc = df.vari_class.astype(str).str.strip()
    var = ~vc.isin(["", "None", "nan", "--", "<NA>"]).to_numpy()
    cls[var] = "variable_star"
    sub[var] = df.vari_class[var].astype(str)
    src = np.full(len(df), "gaia_dsc_or_parallax", dtype=object)
    src[var] = "gaia_vari_classifier"
    src[cls == "unknown_point"] = "none"
    return cls, sub, src


def build_field_sources(field, target, gaia, allwise):
    """Return the source table for one sky field."""
    g = gaia.copy()
    cls, sub, src = gaia_class(g)
    t = pd.DataFrame({
        "src_id": "gaia:" + g.source_id.astype(str), "kind": "gaia", "ra": g.ra, "dec": g.dec,
        "ref_jyear": g.ref_epoch.astype(float), "pmra": g.pmra, "pmdec": g.pmdec, "plx": g.parallax,
        "gmag": g.phot_g_mean_mag, "bp_rp": g.bp_rp, "ruwe": g.ruwe,
        "parallax_snr": g.parallax / g.parallax_error,
        "astrometric_params": g.astrometric_params_solved,
        # true/false from ESA, 0/1 from the IRSA copy: store one boolean type
        "duplicated_source": g.duplicated_source.map(lambda v: str(v).strip().lower() in ("true", "1", "1.0", "t")),
        "p_star": g.classprob_dsc_combmod_star, "p_galaxy": g.classprob_dsc_combmod_galaxy,
        "p_quasar": g.classprob_dsc_combmod_quasar, "phot_variable_flag": g.phot_variable_flag.astype(str),
        "vari_class_score": g.vari_class_score, "class": cls, "subclass": sub, "label_source": src,
        "is_target": False, "w1": np.nan, "w2": np.nan, "w1_snr": np.nan, "w2_snr": np.nan, "ext_flg": np.nan,
    })

    # target: match the SIMBAD astrometry to Gaia at J2016 (barycentric, no parallax)
    tra, tdec = propagate(target["ra"], target["dec"], target["pmra"], target["pmdec"], 0.0,
                          target["ref_jyear"], 2016.0, (0.0, 0.0, 0.0))
    d = np.hypot((t.ra - tra) * np.cos(np.radians(tdec)), t.dec - tdec) * 3600
    near = (d < MATCH_ARCSEC).to_numpy().copy()
    # A fast-moving target must also agree in proper motion: this rejects background
    # stars that happen to lie near its 2016 position and Gaia entries without a
    # proper-motion solution (which would freeze the target's track). Binary
    # components share the system motion and still match.
    tpm = np.hypot(target["pmra"], target["pmdec"])
    if tpm > 100:
        dpm = np.hypot(t.pmra - target["pmra"], t.pmdec - target["pmdec"]).to_numpy()
        near &= np.isfinite(dpm) & (dpm < 0.2 * tpm)
    tclass = TARGET_CLASS.get(field["group"], "star")
    target_rows = None
    if not field.get("is_control", False):
        if near.any():
            t.loc[near, "is_target"] = True
            t.loc[near, "class"] = tclass
            t.loc[near, "subclass"] = target["otype"] or ""
            t.loc[near, "label_source"] = target.get("identity_source", "simbad_target_identity")
        else:
            target_rows = pd.DataFrame([{
                "src_id": f"target:{field['group']}", "kind": "target", "ra": target["ra"], "dec": target["dec"],
                "ref_jyear": target["ref_jyear"], "pmra": target["pmra"], "pmdec": target["pmdec"],
                "plx": target["plx"] or 0.0, "class": tclass, "subclass": target["otype"] or "",
                "label_source": ("simbad_identity_wise_fit_astrometry" if target.get("astrometry_source")
                                 else "simbad_target_identity"), "is_target": True,
            }])

    # AllWISE: attach W1/W2 to Gaia sources, keep unmatched infrared sources
    aw = allwise.copy()
    cosd = np.cos(np.radians(field["dec"]))

    def xy(ra, dec):
        return np.c_[(np.asarray(ra) - field["ra"]) * cosd * 3600, (np.asarray(dec) - field["dec"]) * 3600]

    gra, gdec = propagate(t.ra, t.dec, t.pmra, t.pmdec, 0.0, t.ref_jyear, ALLWISE_JYEAR, (0.0, 0.0, 0.0))
    ref_xy = [xy(gra, gdec)]
    ref_rows = [np.arange(len(t))]
    if target_rows is not None:
        r = target_rows.iloc[0]
        a, b = propagate(r.ra, r.dec, r.pmra, r.pmdec, 0.0, r.ref_jyear, ALLWISE_JYEAR, (0.0, 0.0, 0.0))
        ref_xy.append(xy([a], [b]))
        ref_rows.append(np.array([-1]))
    tree = cKDTree(np.vstack(ref_xy))
    rows = np.concatenate(ref_rows)
    dist, idx = tree.query(xy(aw.ra, aw.dec), distance_upper_bound=MATCH_ARCSEC)
    matched = np.isfinite(dist)
    gi = rows[np.where(matched, idx, 0)]
    best = {}
    for k in np.where(matched)[0]:
        j = gi[k]
        if j >= 0 and (j not in best or dist[k] < dist[best[j]]):
            best[j] = k
    for j, k in best.items():
        for col, src_col in (("w1", "w1mpro"), ("w2", "w2mpro"), ("w1_snr", "w1snr"), ("w2_snr", "w2snr"), ("ext_flg", "ext_flg")):
            t.iat[j, t.columns.get_loc(col)] = pd.to_numeric(aw[src_col].iloc[k], errors="coerce")
    if target_rows is not None:
        hit = np.where(matched & (gi == -1))[0]
        if len(hit):
            k = hit[np.argmin(dist[hit])]
            target_rows["w1"] = pd.to_numeric(aw.w1mpro.iloc[k], errors="coerce")
            target_rows["w2"] = pd.to_numeric(aw.w2mpro.iloc[k], errors="coerce")
    un = aw[~matched]
    ext = pd.to_numeric(un.ext_flg, errors="coerce").fillna(0).to_numpy()
    awt = pd.DataFrame({
        "src_id": "allwise:" + un.designation.astype(str), "kind": "allwise_only", "ra": un.ra, "dec": un.dec,
        "ref_jyear": ALLWISE_JYEAR, "pmra": 0.0, "pmdec": 0.0, "plx": 0.0,
        "w1": pd.to_numeric(un.w1mpro, errors="coerce"), "w2": pd.to_numeric(un.w2mpro, errors="coerce"),
        "w1_snr": pd.to_numeric(un.w1snr, errors="coerce"), "w2_snr": pd.to_numeric(un.w2snr, errors="coerce"),
        "ext_flg": ext, "class": np.where(ext > 0, "extended_ir", "ir_only_point"), "subclass": "",
        "label_source": "allwise_no_gaia_counterpart", "is_target": False,
    })
    parts = [t, awt] + ([target_rows] if target_rows is not None else [])
    out = pd.concat(parts, ignore_index=True)
    out["pm_total_masyr"] = np.hypot(out.pmra.fillna(0), out.pmdec.fillna(0))
    out["field_key"] = field["key"]
    out["group"] = field["group"]
    return out


def source_positions(src, mjd, obs_xyz):
    return propagate(src.ra.to_numpy(), src.dec.to_numpy(), src.pmra.to_numpy(), src.pmdec.to_numpy(),
                     src.plx.to_numpy(), src.ref_jyear.to_numpy(), mjd_to_jyear(mjd), obs_xyz)


def reference_star_mask(src, gmin, gmax):
    """Isolated, well-measured Gaia stars suitable for registration."""
    g = (src.kind == "gaia") & src.gmag.between(gmin, gmax) & (src.astrometric_params >= 31)
    g &= (src.ruwe.fillna(9) < 1.4) & ~src.duplicated_source.fillna(True).astype(bool) & ~src.is_target
    # Must be a star, not a cluster or knot inside a nearby galaxy: confident Gaia
    # star probability, or a significant parallax (foreground Milky Way star).
    plx_snr = (src.parallax_snr if "parallax_snr" in src else pd.Series(np.nan, index=src.index)).fillna(0)
    g &= (src.p_star.fillna(0) > 0.95) | (plx_snr > 5)
    cosd = np.cos(np.radians(src.dec.mean()))
    pts = np.c_[src.ra * cosd * 3600, src.dec * 3600]
    tree = cKDTree(pts)
    mags = src.gmag.fillna(15).to_numpy()  # IR-only sources count as moderately bright neighbours
    mags = np.where(src.kind == "allwise_only", np.nan_to_num(src.w1.to_numpy(), nan=15) + 2.0, mags)
    ok = g.to_numpy().copy()
    for i in np.where(ok)[0]:
        for j in tree.query_ball_point(pts[i], 15.0):
            if j != i and mags[j] < mags[i] + 3.0:
                ok[i] = False
                break
    return ok
