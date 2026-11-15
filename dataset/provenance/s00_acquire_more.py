"""Stage 0 (optional): add SPHEREx fields around the fastest-moving stars in Gaia DR3.

The original download has three real movers in SPHEREx. This adds every Gaia DR3
source with total proper motion above 3 arcsec/yr that is not already covered,
selected from Gaia alone (no name resolver). Frames are found with IRSA's TAP
service and downloaded as 15 arcmin cutouts centred on the star's position at
the SPHEREx epoch, ten per detector spread over the available dates: the same
recipe the original download used.

Outputs: cache/raw_extra/spherex/<slug>/*.fits and cache/raw_extra/fields.json
"""
import csv
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from common import earth_bary_au, mjd_to_jyear, propagate

OUT = C.CACHE / "raw_extra"
MIN_PM_MASYR = 3000.0
SPHEREX_MID_MJD = 61050.0
TAP = "https://irsa.ipac.caltech.edu/TAP/sync"


def gaia_fast_movers():
    """Gaia DR3 sources with total proper motion above the limit, from IRSA's copy
    of the catalogue (ESA's archive was answering HTTP 429 when this was built)."""
    import pandas as pd
    cache = OUT / "gaia_fast_movers.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    q = ("SELECT source_id, ra, dec, ref_epoch, pmra, pmdec, parallax, phot_g_mean_mag FROM gaia_dr3_source "
         f"WHERE pm > {MIN_PM_MASYR} ORDER BY pm DESC")
    txt = get(TAP, {"QUERY": q, "LANG": "ADQL", "RESPONSEFORMAT": "CSV"}).text
    t = pd.read_csv(io.StringIO(txt))
    t.columns = [c.lower() for c in t.columns]
    t.to_parquet(cache, index=False)
    return t


def get(url, params=None, tries=4):
    err = None
    for k in range(tries):
        try:
            r = requests.get(url, params=params, timeout=120)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            err = e
            time.sleep(5 * (k + 1))
    raise err


def plan_field(ra, dec):
    q = ("SELECT a.uri,p.time_bounds_lower,p.time_bounds_upper,p.energy_bandpassname FROM spherex.artifact a "
         f"JOIN spherex.plane p ON a.planeid=p.planeid WHERE 1=CONTAINS(POINT('ICRS',{ra},{dec}),p.poly) "
         "ORDER BY p.time_bounds_lower")
    txt = get(TAP, {"QUERY": q, "LANG": "ADQL", "RESPONSEFORMAT": "CSV"}).text
    rows = [r for r in csv.DictReader(io.StringIO(txt)) if ("/spherex/dr1/" in r.get("uri", "") or "/spherex/qr3/" in r.get("uri", "")) and r["uri"].endswith(".fits")]
    chosen = []
    for det in range(1, 7):
        g = sorted({r["uri"]: r for r in rows if r["energy_bandpassname"] == f"SPHEREx-D{det}"}.values(),
                   key=lambda r: float(r["time_bounds_lower"]))
        if g:
            for i in np.unique(np.round(np.linspace(0, len(g) - 1, min(10, len(g)))).astype(int)):
                chosen.append(g[i])
    return chosen, len(rows)


def download(job):
    path, url = job
    if path.exists() and path.stat().st_size > 100000:
        return "cached"
    try:
        r = get(url, tries=6)
    except requests.RequestException:
        return "failed"   # counted and reported; the field is kept if enough frames arrive
    if not r.content.startswith(b"SIMPLE"):
        return "not_fits"
    path.write_bytes(r.content)
    return "ok"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    g = gaia_fast_movers()
    existing = json.loads((C.CACHE / "fields.json").read_text())
    # compare against the original fields only; fields this script added earlier are named gaia-<id>
    ex = np.array([[f["ra"], f["dec"]] for f in existing.values()
                   if f["mission"] == "spherex" and not f["slug"].startswith("gaia-")])
    fields, jobs = {}, []
    for r in g.itertuples():
        obs = earth_bary_au([SPHEREX_MID_MJD])[0]
        ra, dec = propagate(r.ra, r.dec, r.pmra, r.pmdec, r.parallax, r.ref_epoch, mjd_to_jyear(SPHEREX_MID_MJD), obs)
        ra, dec = float(ra), float(dec)
        sep = np.hypot((ex[:, 0] - ra) * np.cos(np.radians(dec)), ex[:, 1] - dec)
        if sep.min() < 0.3 or any(np.hypot((f["ra"] - ra) * np.cos(np.radians(dec)), f["dec"] - dec) < 0.3 for f in fields.values()):
            continue   # already covered (an existing field, or the companion of a star just added)
        slug = f"gaia-{int(r.source_id)}"
        plan_file = OUT / "plans" / f"{slug}.json"
        plan_file.parent.mkdir(parents=True, exist_ok=True)
        if plan_file.exists():
            chosen, n_all = json.loads(plan_file.read_text())
        else:
            chosen, n_all = plan_field(ra, dec)
            plan_file.write_text(json.dumps([chosen, n_all]))
        if len(chosen) < 12:
            print(slug, "skipped: only", len(chosen), "frames", flush=True)
            continue
        d = OUT / "spherex" / slug
        d.mkdir(parents=True, exist_ok=True)
        for row in chosen:
            parent = "https://irsa.ipac.caltech.edu/" + row["uri"].lstrip("/")
            url = parent + "?" + urlencode({"center": f"{ra},{dec}deg", "size": "15arcmin"})
            jobs.append((d / Path(row["uri"]).name, url))
        fields[slug] = {"slug": slug, "group": slug.replace("-", "_"), "source_id": str(int(r.source_id)),
                        "name": f"Gaia DR3 {int(r.source_id)}", "ra": float(r.ra), "dec": float(r.dec),
                        "ref_jyear": float(r.ref_epoch), "pmra": float(r.pmra), "pmdec": float(r.pmdec),
                        "plx": float(r.parallax), "gmag": float(r.phot_g_mean_mag), "cutout_ra": ra, "cutout_dec": dec,
                        "n_frames_available": n_all, "n_frames_planned": len(chosen)}
        print(slug, "G", round(r.phot_g_mean_mag, 1), "pm", round(float(np.hypot(r.pmra, r.pmdec))), "frames", len(chosen), "of", n_all, flush=True)
    with ThreadPoolExecutor(6) as ex_:
        res = list(ex_.map(download, jobs))
    print("downloads", {k: res.count(k) for k in set(res)})
    for slug, f in fields.items():
        f["n_frames_downloaded"] = len(list((OUT / "spherex" / slug).glob("*.fits")))
    (OUT / "fields.json").write_text(json.dumps(fields, indent=1))


if __name__ == "__main__":
    main()
