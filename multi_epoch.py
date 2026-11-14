"""Targeted multi-epoch motion experiment on aligned SPHEREx images."""
from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from detector import prepare, proposals, score


def motion_fit(image, valid, frames, x, y):
    """Fit one velocity to image centroids, allowing a separate origin per detector."""
    cx, cy = round(x), round(y)
    yy, xx = np.mgrid[cy-4:cy+5, cx-4:cx+5]
    empty = dict(score=np.nan, dx_px=np.nan, dy_px=np.nan, dates=0, frames=0)
    if xx.min()<0 or yy.min()<0 or xx.max()>=image.shape[2] or yy.max()>=image.shape[1]:
        return empty
    patch = image[:,yy,xx]
    mask = valid[:,yy,xx] & np.isfinite(patch)
    radius = np.hypot(xx-x, yy-y)
    core = radius <= 3
    ring = radius >= 3.6
    keep = mask[:,core].all(axis=1) & (mask[:,ring].sum(axis=1)>=12)
    if not keep.any():
        return empty
    patch, mask = patch[keep], mask[keep]
    sky = np.nanmedian(np.where(mask[:,ring],patch[:,ring],np.nan),axis=1)
    weights = np.maximum(patch[:,core]-sky[:,None],0)
    flux = weights.sum(axis=1)
    meta = frames.iloc[np.flatnonzero(keep)].copy()
    meta["significance"] = flux / meta.aperture_sky_noise.clip(lower=1e-10)
    for axis, grid in (("x",xx),("y",yy)):
        meta[axis] = (weights*grid[core]).sum(axis=1)/np.maximum(flux,1e-10)
    meta = meta[meta.significance>=5].copy()
    meta["date"] = np.floor(meta.mjd_mid)
    obs = meta.groupby(["detector","date"])[["mjd_mid","x","y"]].median().reset_index()
    by = obs.groupby("detector")
    obs = obs[by.date.transform("count")>=3].copy()
    result = dict(empty,dates=obs.date.nunique(),frames=len(meta))
    if result["dates"]<4 or obs.mjd_mid.max()-obs.mjd_mid.min()<60:
        return result
    values = obs[["mjd_mid","x","y"]]
    centered = values-values.groupby(obs.detector).transform("mean")
    t = centered.mjd_mid.to_numpy()
    points = centered[["x","y"]].to_numpy()
    slope = (t @ points)/(t @ t)
    residual = points - t[:,None]*slope
    dof = len(t)-obs.detector.nunique()-1
    scatter = max(.07,np.sqrt(np.square(residual).sum()/(2*dof)))
    shift = slope*(obs.mjd_mid.max()-obs.mjd_mid.min())
    result.update(score=np.linalg.norm(slope)*np.sqrt(t@t)/scatter, dx_px=shift[0], dy_px=shift[1])
    return result


def run(root, out, model_path):
    out.mkdir(exist_ok=True,parents=True)
    ix, frames, eligible = prepare(root)
    objects = pd.read_parquet(root/"spherex/injected_objects.parquet")
    model, cut = joblib.load(model_path)
    rows = []
    for row in ix.query("sample_type == 'injected' and fold in [0,1]").itertuples():
        meta = frames[frames.stack_id==row.stack_id].set_index("frame_index").sort_index()
        with np.load(root/row.path) as z:
            assert np.array_equal(z["frame_index"],meta.index)
            image, valid = z["image"], z["valid"].astype(bool)
        labels = eligible[eligible.realization_id==row.realization_id]
        old = proposals(root/row.path,meta,True)
        old = score(model,old).query("priority >= @cut and change_type == 'motion'")
        for obj in objects[objects.realization_id==row.realization_id].to_dict("records"):
            if not (obj["class"]=="static" or obj["is_mover"]):
                continue
            track = labels[labels.obj_id==obj["obj_id"]]
            if track.empty:
                continue
            pairs = track.merge(old[["frame_index","x","y"]],on="frame_index",suffixes=("_truth","_peak"))
            hit = np.any(np.hypot(pairs.x_truth-pairs.x_peak,pairs.y_truth-pairs.y_peak)<=2)
            fit = motion_fit(image,valid,meta,obj["x_ref"],obj["y_ref"])
            rows.append(dict(field=row.field,fold=row.fold,realization=row.realization_id,obj_id=obj["obj_id"],
                             mover=bool(obj["is_mover"]),motion_px=obj["motion_px"],previous=bool(hit),**fit))
    data = pd.DataFrame(rows)
    controls = data.query("fold == 1 and not mover").score.dropna()
    threshold = float(controls.quantile(.95,interpolation="higher"))
    data["alert"] = data.score>threshold
    data.to_csv(out/"objects.csv",index=False)
    sid = "spherex:wise0855"
    meta = frames[frames.stack_id==sid].set_index("frame_index").sort_index()
    track = pd.read_parquet(root/"spherex/tracks.parquet",filters=[("stack_id","=",sid),("src_id","=","target:wise0855")])
    seed = track.set_index("frame_index").loc[meta.index[0]]
    with np.load(root/"spherex/stacks/wise0855.npz") as z:
        fit = motion_fit(z["image"][meta.index],z["valid"][meta.index].astype(bool),meta,seed.x,seed.y)
    info=dict(threshold=threshold,validation_static_scored=len(controls),wise0855=fit,wise_alert=bool(fit["score"]>threshold))
    (out/"summary.json").write_text(json.dumps(info,indent=2,default=float))
    print(info)
    print(data.query("fold==0").groupby("mover")[["previous","alert","score"]].agg(["sum","count"]))
    print(pd.crosstab([data[data.fold==0].mover,data[data.fold==0].previous],data[data.fold==0].alert))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("dataset",type=Path)
    parser.add_argument("--out",type=Path,default=Path("experiments/multi_epoch"))
    parser.add_argument("--model",type=Path,default=Path("results/detector.joblib"))
    args=parser.parse_args()
    run(args.dataset,args.out,args.model)
