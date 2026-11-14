from pathlib import Path
import argparse
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, maximum_filter
from detector import proposals, score


def diagnose(root, model_path):
    folder = root / "spherex"
    sid = "spherex:wise0855"
    frames = pd.read_parquet(folder / "frames.parquet").query("stack_id == @sid and qc_pass")
    frames = frames.set_index("frame_index").sort_index()
    target = pd.read_parquet(folder / "sources.parquet").query("stack_id == @sid and is_target").iloc[0]
    track = pd.read_parquet(folder / "tracks.parquet", filters=[("stack_id", "=", sid), ("src_id", "=", target.src_id)])
    track = track.set_index("frame_index").loc[frames.index]
    path = folder / "stacks/wise0855.npz"
    model, threshold = joblib.load(model_path)
    blind = score(model, proposals(path, frames, False))
    with np.load(path) as stack:
        image = np.where(stack["valid"], stack["image"], np.nan)
    gy, gx = np.indices(image.shape[1:])
    rows = []
    for fid, t in track.iterrows():
        if t.saturated or t.contaminated or not (t.aperture_valid_frac >= .95 and t.snr >= 5):
            continue
        f = frames.loc[fid]
        refs = frames[(frames.detector == f.detector) & (abs(frames.mjd_mid - f.mjd_mid) > 1)]
        gap = abs(np.log(refs.wave_center_um / f.wave_center_um))
        ids = refs.index[np.argsort(gap.to_numpy())[:3]]
        row = dict(frame_index=fid, detector=int(f.detector), mjd=f.mjd_mid, x=t.x, y=t.y,
                   aperture_snr=t.snr, reference_count=len(ids))
        if len(ids) < 2:
            rows.append(dict(row, outcome="insufficient_references"))
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ref = np.nanmedian(image[ids], axis=0)
        good = np.isfinite(image[fid]) & np.isfinite(ref)
        diff = gaussian_filter(np.where(good, image[fid]-ref, 0), .85)
        noise = max(1e-8, 1.4826*np.median(abs(diff[good]-np.median(diff[good]))))
        strength = abs(diff)/noise
        region = good & (np.hypot(gx-t.x, gy-t.y) <= 2)
        row["max_abs_diff_z"] = strength[region].max() if region.any() else np.nan
        row["median_reference_offset_px"] = np.median(np.hypot(track.loc[ids].x-t.x, track.loc[ids].y-t.y))
        peaks = blind[(blind.frame_index == fid) & (np.hypot(blind.x-t.x, blind.y-t.y) <= 2)]
        extrema = region & (strength >= 3.5) & (strength == maximum_filter(strength,5))
        assert extrema.sum() == len(peaks)
        row["proposal_count"] = len(peaks)
        if peaks.empty:
            reason = "no_valid_pixels" if not region.any() else "residual_below_threshold" if row["max_abs_diff_z"] < 3.5 else "no_local_extremum"
        else:
            best = peaks.loc[peaks.priority.idxmax()]
            row.update(priority=best.priority, change_type=best.change_type, motion_score=best.score_motion)
            accepted = peaks[peaks.priority >= threshold]
            reason = "below_review_threshold" if accepted.empty else "motion_alert" if accepted.change_type.eq("motion").any() else "other_class_alert"
        rows.append(dict(row,outcome=reason))
    return pd.DataFrame(rows), threshold, target.motion_px


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model", type=Path, default=Path("results/detector.joblib"))
    args = parser.parse_args()
    data, threshold, motion = diagnose(args.dataset, args.model)
    out = Path("diagnostics")
    out.mkdir(exist_ok=True)
    data.to_csv(out/"wise0855_frames.csv",index=False)
    print(data.outcome.value_counts(), threshold, motion)
    print(data.groupby(["detector","outcome"]).size())
    print(data.describe().to_string())
