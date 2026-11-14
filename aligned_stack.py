"""Compare signed residual stacks on the catalogue track and at its fixed mean position."""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, map_coordinates


def residual(image, frames, fid):
    frame = frames.loc[fid]
    refs = frames[(frames.detector == frame.detector) & ((frames.mjd_mid - frame.mjd_mid).abs() > 1)]
    gap = np.abs(np.log(refs.wave_center_um / frame.wave_center_um))
    ids = refs.index[np.argsort(gap.to_numpy())[:3]]
    if len(ids) < 2:
        raise ValueError(f"Insufficient reference frames for {fid}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref = np.nanmedian(image[ids], axis=0)
    good = np.isfinite(image[fid]) & np.isfinite(ref)
    diff = gaussian_filter(np.where(good, image[fid] - ref, 0), 0.85)
    values = diff[good]
    noise = max(1e-8, 1.4826 * np.median(abs(values - np.median(values))))
    return np.where(good, diff / noise, np.nan)


def run(root, out):
    folder = root / "spherex"
    sid = "spherex:wise0855"
    frames = pd.read_parquet(folder / "frames.parquet").query("stack_id == @sid and qc_pass").set_index("frame_index").sort_index()
    target, = pd.read_parquet(folder / "sources.parquet").query("stack_id == @sid and is_target").src_id
    tracks = pd.read_parquet(folder / "tracks.parquet", filters=[("stack_id", "=", sid), ("src_id", "=", target)])
    track = tracks.set_index("frame_index").loc[frames.index].query(
        "not saturated and not contaminated and aperture_valid_frac >= 0.95 and snr >= 5")
    track = track.join(frames[["mjd_mid"]]).sort_values("mjd_mid")
    with np.load(folder / "stacks/wise0855.npz") as stack:
        image = np.where(stack["valid"], stack["image"], np.nan)
    # The first offset is the target. Surrounding grid points translate the entire track.
    offsets = [(0, 0)] + [(dx, dy) for dy in range(-48, 49, 6) for dx in range(-48, 49, 6)
                          if 8 <= np.hypot(dx, dy) <= 48]
    dx, dy = np.array(offsets).T
    anchor = track[["x", "y"]].mean()
    moving, fixed = [], []
    for fid, row in track.iterrows():
        z = residual(image, frames, fid)
        def sample(x, y):
            return map_coordinates(z, [y + dy, x + dx], order=1, mode="constant", cval=np.nan)
        moving.append(sample(row.x, row.y))
        fixed.append(sample(anchor.x, anchor.y))
    moving, fixed = np.array(moving), np.array(fixed)
    complete = np.isfinite(moving).all(axis=0) & np.isfinite(fixed).all(axis=0)
    if not complete[0] or complete.sum() < 2:
        raise ValueError("Incomplete coverage for the target or comparison grid")
    scores = pd.DataFrame({"dx":dx[complete], "dy":dy[complete],
                           "aligned":moving[:,complete].mean(axis=0), "fixed":fixed[:,complete].mean(axis=0)})
    scores["gain"] = scores.aligned - scores.fixed
    track = track[["mjd_mid","x","y"]].assign(aligned=moving[:,0], fixed=fixed[:,0])
    track["gain"] = track.aligned - track.fixed
    controls = scores.iloc[1:]
    result = dict(frames=len(track), proposed_controls=len(offsets)-1, complete_controls=len(controls),
                  target=scores.iloc[0].to_dict(),
                  controls_at_least_target={k:int((controls[k]>=scores[k].iloc[0]).sum()) for k in ["aligned","gain"]},
                  control_95th_percentile=controls[["aligned","gain"]].quantile(.95).to_dict(),
                  chronological_half_gains=[float(track.gain.iloc[i].mean()) for i in np.array_split(np.arange(len(track)),2)])
    out.mkdir(exist_ok=True, parents=True)
    scores.to_csv(out/"positions.csv",index=False)
    track.to_csv(out/"frames.csv")
    (out/"summary.json").write_text(json.dumps(result,indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", type=Path, default=Path("experiments/aligned_stack"))
    args = parser.parse_args()
    print(json.dumps(run(args.dataset,args.out),indent=2))
