"""Shared numerical helpers: grids, resampling, astrometry, photometry."""
import json
import re
import warnings
import xml.etree.ElementTree as ET

import numpy as np
from astropy.coordinates import get_body_barycentric
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS, FITSFixedWarning

warnings.simplefilter("ignore", FITSFixedWarning)

ARCSEC = np.pi / (180 * 3600)
AU_KM = 149597870.7


# --------------------------------------------------------------------------- targets
def parse_sesame(path):
    """Read a CDS Sesame XML response: position (J2000, epoch J2000), motion, type."""
    root = ET.parse(path).getroot()
    res = root.find(".//Resolver")

    def f(tag):
        el = res.find(tag)
        return float(el.text) if el is not None and el.text else None

    pm = res.find("pm")
    out = {
        "name": (res.findtext("oname") or root.findtext(".//name") or "").replace("NAME ", ""),
        "otype": res.findtext("otype"),
        "sptype": res.findtext("spType"),
        "ra": f("jradeg"), "dec": f("jdedeg"),
        "pmra": float(pm.findtext("pmRA")) if pm is not None and pm.findtext("pmRA") else 0.0,
        "pmdec": float(pm.findtext("pmDE")) if pm is not None and pm.findtext("pmDE") else 0.0,
        "plx": None, "rv": None, "ref_jyear": 2000.0,
    }
    plx = res.find("plx")
    if plx is not None and plx.findtext("v"):
        out["plx"] = float(plx.findtext("v"))
    vel = res.find("Vel")
    if vel is not None and vel.findtext("v"):
        out["rv"] = float(vel.findtext("v"))
    return out


# --------------------------------------------------------------------------- grids
def grid_wcs(ra, dec, npix, scale_arcsec):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [(npix + 1) / 2.0, (npix + 1) / 2.0]
    w.wcs.cdelt = [-scale_arcsec / 3600.0, scale_arcsec / 3600.0]
    w.wcs.radesys = "ICRS"
    return w


def grid_world(w, npix):
    yy, xx = np.mgrid[0:npix, 0:npix].astype(float)
    ra, dec = w.wcs_pix2world(xx, yy, 0)
    return ra, dec


def strip_sip(header):
    h = header.copy()
    for k in list(h.keys()):
        if re.match(r"^(A|B|AP|BP)_(\d+_\d+|ORDER|DMAX)$", k):
            del h[k]
    for k in ("CTYPE1", "CTYPE2"):
        if k in h:
            h[k] = h[k].replace("-SIP", "")
    return h


def resample(xin, yin, src_shape, fields, invalid, flags=None):
    """Bilinear resampling at native pixel coordinates (xin, yin).

    fields: dict name -> 2-D array to interpolate (image-like)
    var_fields: names in `fields` that are variances (weights are squared)
    invalid: 2-D bool, native pixels that must not contribute.
    An output pixel is valid only when all four contributing native pixels are
    inside the frame and valid. Flags are OR-combined over the contributors.
    """
    ny, nx = src_shape
    x0 = np.floor(xin).astype(int)
    y0 = np.floor(yin).astype(int)
    fx = xin - x0
    fy = yin - y0
    inside = (x0 >= 0) & (y0 >= 0) & (x0 + 1 < nx) & (y0 + 1 < ny)
    xa = np.clip(x0, 0, nx - 2)
    ya = np.clip(y0, 0, ny - 2)
    corners = [(ya, xa, (1 - fx) * (1 - fy)), (ya, xa + 1, fx * (1 - fy)),
               (ya + 1, xa, (1 - fx) * fy), (ya + 1, xa + 1, fx * fy)]
    valid = inside.copy()
    for yy, xx, wgt in corners:
        valid &= ~(invalid[yy, xx] & (wgt > 1e-6))
    out = {}
    for name, (arr, is_var) in fields.items():
        acc = np.zeros(xin.shape, dtype=np.float64)
        for yy, xx, wgt in corners:
            v = arr[yy, xx]
            v = np.where(np.isfinite(v), v, 0.0)
            acc += (wgt ** 2 if is_var else wgt) * v
        out[name] = np.where(valid, acc, 0.0).astype(np.float32)
    fl = None
    if flags is not None:
        fl = np.zeros(xin.shape, dtype=np.uint32)
        for yy, xx, wgt in corners:
            fl |= np.where(wgt > 1e-6, flags[yy, xx], 0).astype(np.uint32)
        fl = np.where(inside, fl, 0).astype(np.uint32)
    return out, valid, fl, inside


# --------------------------------------------------------------------------- astrometry
def earth_bary_au(mjd_utc):
    t = Time(np.atleast_1d(mjd_utc), format="mjd", scale="utc")
    p = get_body_barycentric("earth", t)
    return np.vstack([p.x.to_value("AU"), p.y.to_value("AU"), p.z.to_value("AU")]).T


def propagate(ra, dec, pmra, pmdec, plx, ref_jyear, jyear, obs_xyz):
    """Apparent ICRS position at `jyear` for an observer at barycentric `obs_xyz` (AU).

    Linear proper motion in the tangent plane plus annual parallax. ra/dec in deg,
    pm in mas/yr (pmra includes cos dec), plx in mas. Missing pm/plx count as 0.
    Aberration is not applied because the image WCS solutions are tied to
    astrometric catalogue positions.
    """
    ra = np.asarray(ra, float)
    dec = np.asarray(dec, float)
    pmra = np.nan_to_num(np.asarray(pmra, float))
    pmdec = np.nan_to_num(np.asarray(pmdec, float))
    plx = np.nan_to_num(np.asarray(plx, float))
    plx = np.where(plx > 0, plx, 0.0)
    dt = jyear - np.asarray(ref_jyear, float)
    a = np.radians(ra)
    d = np.radians(dec)
    X, Y, Z = obs_xyz
    dra_star = pmra * dt + plx * (X * np.sin(a) - Y * np.cos(a))                     # mas
    ddec = pmdec * dt + plx * (X * np.sin(d) * np.cos(a) + Y * np.sin(d) * np.sin(a) - Z * np.cos(d))
    new_dec = dec + ddec / 3.6e6
    new_ra = ra + dra_star / 3.6e6 / np.cos(np.radians(0.5 * (dec + new_dec)))
    return new_ra, new_dec


def topocentric_shift(ra, dec, delta_au, sc_geo_xyz_au):
    """Move a geocentric astrometric position to a spacecraft observer."""
    a = np.radians(ra)
    d = np.radians(dec)
    vec = np.stack([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)], -1) * np.asarray(delta_au)[..., None]
    v = vec - np.asarray(sc_geo_xyz_au)
    r = np.linalg.norm(v, axis=-1)
    return (np.degrees(np.arctan2(v[..., 1], v[..., 0])) % 360.0,
            np.degrees(np.arcsin(v[..., 2] / r)), r)


def mjd_to_jyear(mjd):
    return 2000.0 + (np.asarray(mjd, float) - 51544.5) / 365.25


# --------------------------------------------------------------------------- statistics
def robust_median_std(x):
    x = x[np.isfinite(x)]
    if x.size < 5:
        return np.nan, np.nan
    for _ in range(3):
        med = np.median(x)
        mad = 1.4826 * np.median(np.abs(x - med))
        if mad <= 0:
            break
        keep = np.abs(x - med) < 3 * mad
        if keep.sum() < 5:
            break
        x = x[keep]
    med = np.median(x)
    return float(med), float(1.4826 * np.median(np.abs(x - med)))


# --------------------------------------------------------------------------- photometry
_AP_CACHE = {}


def _aperture_kernel(r, sub=5):
    """Fractional pixel weights of a circular aperture for 25 sub-pixel phases."""
    key = (r, sub)
    if key in _AP_CACHE:
        return _AP_CACHE[key]
    half = int(np.ceil(r)) + 1
    size = 2 * half + 1
    phases = np.linspace(-0.4, 0.4, 5)
    kern = np.zeros((5, 5, size, size))
    s = (np.arange(sub) + 0.5) / sub - 0.5
    for i, py in enumerate(phases):
        for j, px in enumerate(phases):
            yy, xx = np.mgrid[0:size, 0:size] - half
            yy = yy[..., None, None] + s[:, None]
            xx = xx[..., None, None] + s[None, :]
            inside = ((xx - px) ** 2 + (yy - py) ** 2) <= r * r
            kern[i, j] = inside.mean(axis=(2, 3))
    _AP_CACHE[key] = (kern, half, phases)
    return _AP_CACHE[key]


def aperture_photometry(img, var, valid, xs, ys, r, rin, rout):
    """Aperture sums at grid positions (0-based pixel centres).

    Returns flux, error from the variance map, local background, fraction of the
    aperture that is valid, and the number of valid annulus pixels.
    """
    kern, half, phases = _aperture_kernel(r)
    ny, nx = img.shape
    ro = int(np.ceil(rout))
    n = len(xs)
    flux = np.full(n, np.nan)
    ferr = np.full(n, np.nan)
    bkg = np.full(n, np.nan)
    vfrac = np.zeros(n)
    yy_a, xx_a = np.mgrid[-ro:ro + 1, -ro:ro + 1]
    for k in range(n):
        x, y = xs[k], ys[k]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        xc, yc = int(np.round(x)), int(np.round(y))
        if xc - half < 0 or yc - half < 0 or xc + half >= nx or yc + half >= ny:
            continue
        j = int(np.clip(np.round((x - xc) / 0.2) + 2, 0, 4))
        i = int(np.clip(np.round((y - yc) / 0.2) + 2, 0, 4))
        w = kern[i, j]
        sl = (slice(yc - half, yc + half + 1), slice(xc - half, xc + half + 1))
        vm = valid[sl]
        vfrac[k] = (w * vm).sum() / w.sum()
        y0, y1 = max(yc - ro, 0), min(yc + ro + 1, ny)
        x0, x1 = max(xc - ro, 0), min(xc + ro + 1, nx)
        dy = yy_a[y0 - yc + ro:y1 - yc + ro, x0 - xc + ro:x1 - xc + ro] + (yc - y)
        dx = xx_a[y0 - yc + ro:y1 - yc + ro, x0 - xc + ro:x1 - xc + ro] + (xc - x)
        rr = np.hypot(dx, dy)
        ann = (rr >= rin) & (rr <= rout) & valid[y0:y1, x0:x1]
        if ann.sum() < 8:
            continue
        b, _ = robust_median_std(img[y0:y1, x0:x1][ann])
        bkg[k] = b
        flux[k] = (w * (img[sl] - b) * vm).sum()
        ferr[k] = np.sqrt((w ** 2 * var[sl] * vm).sum())
    return flux, ferr, bkg, vfrac


def centroid(img, valid, x, y, r=2.0, iters=3):
    """Iterative background-subtracted first-moment centroid."""
    ny, nx = img.shape
    for _ in range(iters):
        if not (np.isfinite(x) and np.isfinite(y)):
            return np.nan, np.nan
        xc, yc = int(np.round(x)), int(np.round(y))
        h = int(np.ceil(r)) + 3
        if xc - h < 0 or yc - h < 0 or xc + h >= nx or yc + h >= ny:
            return np.nan, np.nan
        sub = img[yc - h:yc + h + 1, xc - h:xc + h + 1]
        vm = valid[yc - h:yc + h + 1, xc - h:xc + h + 1] & np.isfinite(sub)
        sub = np.where(vm, sub, 0.0)
        yy, xx = np.mgrid[yc - h:yc + h + 1, xc - h:xc + h + 1]
        rr = np.hypot(xx - x, yy - y)
        ring = (rr > r + 1) & vm
        if ring.sum() < 6:
            return np.nan, np.nan
        b = np.median(sub[ring])
        core = (rr <= r) & vm
        if core.sum() < max(4, 0.7 * np.pi * r * r):
            return np.nan, np.nan
        wgt = np.clip(sub - b, 0, None) * core
        s = wgt.sum()
        if s <= 0:
            return np.nan, np.nan
        x, y = (wgt * xx).sum() / s, (wgt * yy).sum() / s
    return x, y


def sky_aperture_noise(img, valid, avoid_xy, r, n_trials=300, seed=0, avoid_r=6.0):
    """Empirical noise of aperture sums placed on empty sky (includes the pixel
    correlations introduced by resampling and faint-source confusion)."""
    rng = np.random.default_rng(seed)
    kern, half, _ = _aperture_kernel(r)
    w = kern[2, 2]
    ny, nx = img.shape
    sums = []
    ax = np.asarray([p[0] for p in avoid_xy]) if len(avoid_xy) else np.zeros(0)
    ay = np.asarray([p[1] for p in avoid_xy]) if len(avoid_xy) else np.zeros(0)
    for _ in range(n_trials * 4):
        if len(sums) >= n_trials:
            break
        xc = rng.integers(half + 8, nx - half - 8)
        yc = rng.integers(half + 8, ny - half - 8)
        if ax.size and np.min((ax - xc) ** 2 + (ay - yc) ** 2) < avoid_r ** 2:
            continue
        sl = (slice(yc - half, yc + half + 1), slice(xc - half, xc + half + 1))
        if not valid[sl][w > 0].all():
            continue
        ann = img[max(yc - 10, 0):yc + 11, max(xc - 10, 0):xc + 11]
        vann = valid[max(yc - 10, 0):yc + 11, max(xc - 10, 0):xc + 11]
        b = np.median(ann[vann])
        sums.append((w * (img[sl] - b)).sum())
    if len(sums) < 30:
        return np.nan, len(sums)
    return robust_median_std(np.asarray(sums))[1], len(sums)


def json_dump(obj, path):
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        return str(o)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=default))
