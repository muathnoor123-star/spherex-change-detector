"""Synthetic source injection into real image stacks, with exact per-frame truth.

Motion model (all classes): offset(t) = pm * (t - t_ref) + parallax * (P(t) - P(t_ref)),
where P(t) is the annual parallax factor from the observer's barycentric position.
This one model covers nearby stars and brown dwarfs (proper motion plus parallax)
and distant solar-system bodies, whose apparent motion is dominated by the
observer's reflex (parallax = 206265 / distance_au arcsec).

Each frame's PSF is the empirical PSF measured from real stars in that frame
(psf/<stack>.npz). Injected flux follows a power-law spectrum across wavelength,
and photon noise is added with the frame's measured variance-versus-signal slope.

Needs numpy and scipy only.
"""
import numpy as np
from scipy.ndimage import shift as nd_shift

CLASSES = {
    # name: (weight, description)
    "pm_star": (2.0, "proper motion 0.05-10 arcsec/yr, parallax 0-0.5 arcsec"),
    "fast_mover": (1.0, "nearby brown dwarf-like: proper motion 10-60 arcsec/yr, parallax 0.1-0.8 arcsec"),
    "distant_planet": (1.5, "solar-system body at 100-2000 AU: reflex parallax plus slow orbital motion"),
    "transient": (1.5, "present in one exposure time only (asteroid-like or flare)"),
    "variable": (1.0, "static, brightness varies 0.3-1.5 mag"),
    "static": (1.0, "static, constant brightness (negative control)"),
}


def parallax_factors(ra_deg, dec_deg, obs_xyz):
    a, d = np.radians(ra_deg), np.radians(dec_deg)
    X, Y, Z = obs_xyz[:, 0], obs_xyz[:, 1], obs_xyz[:, 2]
    p_ra = X * np.sin(a) - Y * np.cos(a)
    p_dec = X * np.sin(d) * np.cos(a) + Y * np.sin(d) * np.sin(a) - Z * np.cos(d)
    return p_ra, p_dec


def render(psf, x, y, shape):
    """PSF centred at (x, y) on an image of `shape`; returns (stamp, y0, x0) clipped to the image."""
    s = psf.shape[0]
    h = s // 2
    xc, yc = int(np.round(x)), int(np.round(y))
    st = nd_shift(psf, (y - yc, x - xc), order=3, mode="constant", cval=0.0)
    st = np.clip(st, 0, None)
    tot = st.sum()
    if tot > 0:
        st = st * (psf.sum() / tot)
    y0, x0 = yc - h, xc - h
    ys, xs = max(0, -y0), max(0, -x0)
    ye, xe = min(s, shape[0] - y0), min(s, shape[1] - x0)
    if ye <= ys or xe <= xs:
        return None, 0, 0
    return st[ys:ye, xs:xe], y0 + ys, x0 + xs


def aperture_fraction(psf, r):
    s = psf.shape[0]
    yy, xx = np.mgrid[0:s, 0:s] - s // 2
    return float(psf[np.hypot(xx, yy) <= r].sum() / max(psf.sum(), 1e-12))


def draw_objects(rng, n, frames, npix, scale_arcsec, lam_ref, snr_max=3000.0):
    """Random object parameters. Positions are defined at a reference frame chosen among the stack."""
    names = list(CLASSES)
    w = np.array([CLASSES[k][0] for k in names])
    groups = np.unique(frames["time_group"])
    objs = []
    for i in range(n):
        cls = names[rng.choice(len(names), p=w / w.sum())]
        o = {"obj_id": i, "class": cls, "pm_ra": 0.0, "pm_dec": 0.0, "plx": 0.0, "dist_au": np.nan,
             "amp_mag": 0.0, "period_d": np.nan, "phase": 0.0, "transient_group": -1,
             "sed_alpha": rng.uniform(-2, 3), "snr_target": float(np.exp(rng.uniform(np.log(3), np.log(snr_max)))),
             "lam_ref_um": lam_ref}
        ang = rng.uniform(0, 2 * np.pi)
        if cls == "pm_star":
            pm = np.exp(rng.uniform(np.log(0.05), np.log(10)))
            o.update(pm_ra=pm * np.cos(ang), pm_dec=pm * np.sin(ang), plx=rng.uniform(0, 0.5))
        elif cls == "fast_mover":
            pm = rng.uniform(10, 60)
            o.update(pm_ra=pm * np.cos(ang), pm_dec=pm * np.sin(ang), plx=rng.uniform(0.1, 0.8))
        elif cls == "distant_planet":
            d = np.exp(rng.uniform(np.log(100), np.log(2000)))
            orbital = 2 * np.pi * d ** -1.5 * 206265.0   # arcsec/yr for a circular orbit seen face-on
            o.update(dist_au=d, plx=206265.0 / d, pm_ra=orbital * np.cos(ang), pm_dec=orbital * np.sin(ang))
        elif cls == "transient":
            o["transient_group"] = int(rng.choice(groups))
        elif cls == "variable":
            o.update(amp_mag=rng.uniform(0.3, 1.5), period_d=float(np.exp(rng.uniform(0, np.log(400)))),
                     phase=rng.uniform(0, 2 * np.pi))
        ref = int(rng.integers(len(frames["mjd"])))
        if cls == "transient":
            ref = int(rng.choice(np.where(frames["time_group"] == o["transient_group"])[0]))
        o.update(ref_frame=ref, x_ref=rng.uniform(0.1, 0.9) * (npix - 1), y_ref=rng.uniform(0.1, 0.9) * (npix - 1))
        objs.append(o)
    return objs


def trajectory(o, frames, grid_ra, grid_dec, scale_arcsec):
    mjd = frames["mjd"]
    p_ra, p_dec = parallax_factors(grid_ra, grid_dec, frames["obs_xyz"])
    r = o["ref_frame"]
    dt = (mjd - mjd[r]) / 365.25
    dra = o["pm_ra"] * dt + o["plx"] * (p_ra - p_ra[r])        # arcsec, east positive
    ddec = o["pm_dec"] * dt + o["plx"] * (p_dec - p_dec[r])
    x = o["x_ref"] - dra / scale_arcsec                          # east is to the left
    y = o["y_ref"] + ddec / scale_arcsec
    return x, y


def flux_curve(o, frames):
    lam = frames["wave_um"]
    f = (lam / o["lam_ref_um"]) ** o["sed_alpha"]
    if o["class"] == "variable":
        f = f * 10 ** (-0.4 * o["amp_mag"] * np.sin(2 * np.pi * frames["mjd"] / o["period_d"] + o["phase"]))
    present = np.ones(len(lam), bool)
    if o["class"] == "transient":
        present = frames["time_group"] == o["transient_group"]
    return f, present


def inject(base, frames, psf, grid, rng, n_objects=10, aperture_r=2.0, snr_max=3000.0):
    """Inject `n_objects` sources into a stack.

    Target S/N is log-uniform from 3 to `snr_max`. Where the real image plus the
    injected light exceeds the frame's saturation level (frames["sat_level"], if
    given), the pixel is blanked and marked invalid, as the detector would do.

    base:   dict with image, sigma, valid arrays [N, H, W] (only QC-passed frames)
    frames: dict of 1-D arrays: mjd, time_group, wave_um, obs_xyz [N,3], sky_noise, shot_k
    psf:    [N, S, S] per-frame normalised PSF
    grid:   dict ra, dec, scale_arcsec, npix
    Returns new image, new sigma, new valid mask, object table (list of dicts), truth rows (list of dicts).
    """
    img = base["image"].copy()
    sig = base["sigma"].copy()
    valid = base["valid"].astype(bool).copy()
    sat = frames.get("sat_level")
    n, H, W = img.shape
    lam_ref = float(np.nanmedian(frames["wave_um"]))
    objs = draw_objects(rng, n_objects, frames, grid["npix"], grid["scale_arcsec"], lam_ref, snr_max)
    apf = np.array([aperture_fraction(psf[i], aperture_r) for i in range(n)])
    ref_noise = float(np.nanmedian(frames["sky_noise"] / np.clip(apf, 1e-3, None)))
    truth = []
    for o in objs:
        x, y = trajectory(o, frames, grid["ra"], grid["dec"], grid["scale_arcsec"])
        fshape, present = flux_curve(o, frames)
        total = o["snr_target"] * ref_noise
        o["flux_ref"] = total
        for i in range(n):
            ftot = total * fshape[i] if present[i] else 0.0
            inside = (-2 <= x[i] <= W + 1) and (-2 <= y[i] <= H + 1)
            row = {"obj_id": o["obj_id"], "frame_index": i, "x": float(x[i]), "y": float(y[i]),
                   "present": bool(present[i]), "in_grid": bool(0 <= x[i] <= W - 1 and 0 <= y[i] <= H - 1),
                   "flux_total": float(ftot), "ap_flux_true": float(ftot * apf[i]),
                   "snr_true": float(ftot * apf[i] / frames["sky_noise"][i]) if frames["sky_noise"][i] > 0 else np.nan,
                   "masked_frac": np.nan, "saturated": False}
            if present[i] and inside and ftot > 0:
                st, y0, x0 = render(psf[i], x[i], y[i], (H, W))
                if st is not None:
                    sl = (i, slice(y0, y0 + st.shape[0]), slice(x0, x0 + st.shape[1]))
                    model = st * ftot
                    vm = valid[sl]
                    row["masked_frac"] = float(1 - (model * vm).sum() / max(model.sum(), 1e-30))
                    k = max(frames["shot_k"][i], 0.0)
                    noise = rng.normal(0, 1, model.shape) * np.sqrt(k * model)
                    img[sl] = np.where(vm, img[sl] + model + noise, img[sl])
                    sig[sl] = np.where(vm, np.sqrt(sig[sl] ** 2 + k * model), sig[sl])
                    if sat is not None and np.isfinite(sat[i]):
                        over = vm & (img[sl] > sat[i])
                        if over.any():
                            row["saturated"] = True
                            img[sl] = np.where(over, 0.0, img[sl])
                            sig[sl] = np.where(over, 0.0, sig[sl])
                            valid[sl] = vm & ~over
            truth.append(row)
        xs, ys = x[present], y[present]
        o["motion_px"] = float(np.hypot(np.ptp(xs), np.ptp(ys))) if present.sum() > 1 else 0.0
        o["is_mover"] = bool(o["motion_px"] >= 1.0)
    return img.astype(np.float32), sig.astype(np.float32), valid, objs, truth
