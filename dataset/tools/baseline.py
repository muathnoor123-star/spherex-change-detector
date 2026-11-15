"""Classical difference-imaging detector: the reference score a learned model should beat.

For each frame, the reference is the per-pixel median of frames from the same
detector or band taken more than `min_gap_days` earlier or later. When per-frame
wavelengths are given (SPHEREx), only the four such frames closest in wavelength
are used, because a star's brightness changes with wavelength. The difference
is divided by the combined noise, rescaled by its robust scatter (resampling makes
the stored sigma too optimistic), smoothed with a small Gaussian matched filter
and searched for local maxima. A detection in a frame that has a simultaneous
partner (the other SPHEREx detector of the pair, or the other WISE band) is marked
`coincident` when the partner frame also shows a >= 3 sigma peak within 2 pixels.

Needs numpy and scipy only.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter


def robust_std(x):
    x = x[np.isfinite(x)]
    if x.size < 10:
        return np.nan
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))


def score_maps(image, sigma, valid, band, mjd, min_gap_days=1.0, min_refs=3, smooth_px=1.0, wave=None, n_wave_refs=4):
    n = image.shape[0]
    val = valid.astype(bool)
    scores = np.zeros_like(image, dtype=np.float32)
    ok = np.zeros(image.shape, bool)
    for i in range(n):
        same = (band == band[i]) & (np.arange(n) != i)
        refs = np.where(same & (np.abs(mjd - mjd[i]) > min_gap_days))[0]
        if len(refs) < min_refs:
            refs = np.where(same & (np.abs(mjd - mjd[i]) > 0.01))[0]
        if len(refs) < 2:
            continue
        if wave is not None and len(refs) > n_wave_refs:
            # SPHEREx changes wavelength from frame to frame: compare with the frames closest in wavelength
            refs = refs[np.argsort(np.abs(wave[refs] - wave[i]))[:n_wave_refs]]
        stack = np.where(val[refs], image[refs], np.nan)
        ref = np.nanmedian(stack, axis=0)
        nref = np.sum(val[refs], axis=0)
        rsig = 1.2533 * np.nanmedian(np.where(val[refs], sigma[refs], np.nan), axis=0) / np.sqrt(np.clip(nref, 1, None))
        good = val[i] & (nref >= 2) & np.isfinite(ref)
        d = np.where(good, (image[i] - ref) / np.sqrt(sigma[i] ** 2 + rsig ** 2 + 1e-30), 0.0)
        s = robust_std(d[good])
        if not np.isfinite(s) or s == 0:
            continue
        d = d / s
        if smooth_px > 0:
            w = gaussian_filter(good.astype(float), smooth_px)
            d = np.where(good, gaussian_filter(d, smooth_px) / np.clip(w, 1e-3, None), 0.0)
            s2 = robust_std(d[good])
            d = d / s2 if s2 and np.isfinite(s2) else d
        scores[i] = d
        ok[i] = good
    return scores, ok


def peaks(score, ok, threshold, sep=3):
    mx = maximum_filter(np.where(ok, score, -np.inf), size=2 * sep + 1)
    y, x = np.where(ok & (score >= threshold) & (score == mx))
    return x.astype(float), y.astype(float), score[y, x]


def detect(image, sigma, valid, band, mjd, time_group, threshold=5.0, partner_threshold=3.0, match_px=2.0, wave=None):
    """Returns a list of detections: dict(frame, x, y, score, coincident, has_partner)."""
    scores, ok = score_maps(image, sigma, valid, band, mjd, wave=wave)
    dets = []
    for i in range(image.shape[0]):
        if not ok[i].any():
            continue
        x, y, s = peaks(scores[i], ok[i], threshold)
        partners = [j for j in np.where(time_group == time_group[i])[0] if j != i and ok[j].any()]
        for xx, yy, ss in zip(x, y, s):
            coin = False
            for j in partners:
                y0, y1 = int(max(yy - match_px, 0)), int(min(yy + match_px + 1, image.shape[1]))
                x0, x1 = int(max(xx - match_px, 0)), int(min(xx + match_px + 1, image.shape[2]))
                win = np.where(ok[j][y0:y1, x0:x1], scores[j][y0:y1, x0:x1], -np.inf)
                if win.size and win.max() >= partner_threshold:
                    coin = True
            dets.append({"frame": i, "x": float(xx), "y": float(yy), "score": float(ss),
                         "coincident": coin, "has_partner": bool(partners)})
    return dets
