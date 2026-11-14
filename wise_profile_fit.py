"""Compare fixed and catalogue-centred Gaussian profiles on original WISE 0855 exposures."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def run(root, out):
    data = root/"spherex"
    sid = "spherex:wise0855"
    frames = pd.read_parquet(data/"frames.parquet").query("stack_id == @sid and qc_pass").set_index("frame_index")
    target, = pd.read_parquet(data/"sources.parquet").query("stack_id == @sid and is_target").src_id
    track = pd.read_parquet(data/"tracks.parquet",filters=[("stack_id","=",sid),("src_id","=",target)])
    track = track.set_index("frame_index").join(frames[["psf_fwhm_px","pixel_noise"]],how="inner")
    track = track.query("not saturated and not contaminated and aperture_valid_frac >= 0.95 and snr >= 5")
    anchor = track[["x","y"]].mean()
    with np.load(data/"stacks/wise0855.npz") as stack:
        image = np.where(stack["valid"],stack["image"],np.nan)
    scores = []
    for dx,dy in [(0,0)]+[(x,y) for x in range(-48,49,6) for y in range(-48,49,6) if 8<=np.hypot(x,y)<=48]:
        x0,y0 = anchor.x+dx,anchor.y+dy
        yy,xx = np.mgrid[-4:5,-4:5]
        xx,yy = xx+round(x0),yy+round(y0)
        gains = []
        for fid,row in track.iterrows():
            patch = image[fid,yy,xx]
            valid = np.isfinite(patch)
            if valid.sum()<60:
                break
            x,y,flux = xx[valid],yy[valid],patch[valid].astype(float)
            errors = []
            for cx,cy in [(x0,y0),(row.x+dx,row.y+dy)]:
                profile = np.exp(-0.5*((x-cx)**2+(y-cy)**2)/(row.psf_fwhm_px/2.355)**2)
                basis = np.column_stack([np.ones(len(x)),x-x0,y-y0])
                # Remove the common planar background before fitting nonnegative flux.
                signal = flux-basis@np.linalg.lstsq(basis,flux,rcond=None)[0]
                shape = profile-basis@np.linalg.lstsq(basis,profile,rcond=None)[0]
                amplitude = max(0,signal@shape/(shape@shape))
                errors.append(np.mean(((signal-amplitude*shape)/row.pixel_noise)**2))
            gains.append(errors[0]-errors[1])
        if len(gains)==len(track) and np.isfinite(gains).all():
            scores.append(dict(dx=dx,dy=dy,gain=np.mean(gains),preferred_frames=int((np.array(gains)>0).sum())))
    positions=pd.DataFrame(scores)
    assert len(positions)>1 and positions.iloc[0].dx==positions.iloc[0].dy==0
    gain=float(positions.iloc[0].gain)
    report=dict(frames=len(track),target_gain=gain,preferred_frames=int(positions.iloc[0].preferred_frames),
                controls=len(positions)-1,controls_at_least_target=int((positions.gain.iloc[1:]>=gain).sum()))
    out.mkdir(parents=True,exist_ok=True)
    positions.to_csv(out/"psf_positions.csv",index=False)
    (out/"psf_summary.json").write_text(json.dumps(report,indent=2))
    print(report)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("dataset",type=Path)
    parser.add_argument("--out",type=Path,default=Path("experiments/review_priority"))
    args=parser.parse_args()
    run(args.dataset,args.out)
