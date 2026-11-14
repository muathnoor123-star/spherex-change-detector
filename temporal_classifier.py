"""Compare temporal photometry, transient oversampling, and hard-negative mining."""
import argparse
import json
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import convolve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from detector import FEATURES, KEY, collect, prepare, proposals

HISTORY = ["prior_flux", "prior_max", "repeat_fraction", "brightness_excursion",
           "history_scatter", "history_count", "partner_signal"]


def photometry(path, meta, peaks, injected):
    with np.load(path) as stack:
        image, valid = stack["image"], stack["valid"].astype(bool)
        if injected:
            assert np.array_equal(stack["frame_index"], meta.index)
        else:
            image, valid = image[meta.index], valid[meta.index]
    aperture = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.float32)[None]
    valid &= np.isfinite(image)
    flux = convolve(np.where(valid,image,0), aperture, mode="constant")
    flux[convolve(valid.astype("f4"),aperture,mode="constant") < 5] = np.nan
    t = meta.mjd_mid.to_numpy()
    wave = meta.wave_center_um.to_numpy()
    noise = np.maximum(meta.aperture_sky_noise.to_numpy(), 1e-8)
    peaks = peaks.copy()
    for fid, group in peaks.groupby("frame_index"):
        i = meta.index.get_loc(fid)
        refs = np.flatnonzero((meta.detector.to_numpy()==meta.detector.iloc[i]) & (abs(t-t[i])>1))
        refs = refs[np.argsort(abs(np.log(wave[refs]/wave[i])))[:5]]
        x,y = group.x.to_numpy(),group.y.to_numpy()
        hist = flux[refs[:,None],y,x]
        snr = hist/noise[refs,None]
        now = flux[i,y,x]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",RuntimeWarning)
            median = np.nanmedian(hist,axis=0)
            scatter = 1.4826*np.nanmedian(abs(hist-median),axis=0)
            partners = np.flatnonzero((meta.time_group.to_numpy()==meta.time_group.iloc[i]) &
                                      (meta.detector.to_numpy()!=meta.detector.iloc[i]))
            partner = np.nanmax(flux[partners[:,None],y,x]/noise[partners,None],axis=0) if len(partners) else np.full(len(x),np.nan)
            count = np.isfinite(hist).sum(axis=0)
            values = [median/noise[i],np.nanmax(snr,axis=0),(snr>=3).sum(axis=0)/np.maximum(count,1),
                      (now-median)/np.hypot(scatter,noise[i]), scatter/noise[i],count,partner]
        peaks.loc[group.index,HISTORY] = np.nan_to_num(np.column_stack(values),nan=-999.)
    return peaks


def probabilities(model, table):
    return pd.DataFrame(model.predict_proba(table[list(model.feature_names_in_)]),
                        columns=model.classes_,index=table.index)


def scan(root, sid, bundle):
    model, threshold = joblib.load(bundle)
    index = pd.read_parquet(root/"index.parquet")
    row = index.query("stack_id == @sid and sample_type == 'real_stack'").iloc[0]
    frames = pd.read_parquet(root/"spherex/frames.parquet")
    meta = frames.query("stack_id == @sid and qc_pass").set_index("frame_index").sort_index()
    path = root/row.path
    peaks = proposals(path,meta,False)
    if peaks.empty:
        return peaks.assign(priority=pd.Series(dtype=float),change_type=pd.Series(dtype=str))
    prob = probabilities(model, photometry(path,meta,peaks,False))
    return peaks.assign(priority=1-prob.background,change_type=prob.drop(columns="background").idxmax(axis=1)).query("priority >= @threshold").sort_values("priority",ascending=False)


def measure(table, prob, flag, truth):
    actual = table.label.ne("background")
    kind = prob.drop(columns="background").idxmax(axis=1)
    hits = table[flag & kind.eq(table.label)][KEY].drop_duplicates()
    objects = hits[["realization_id","obj_id"]].drop_duplicates().merge(
        truth[["realization_id","obj_id","label"]].drop_duplicates(),how="inner")
    tp = int((actual & flag).sum())
    return dict(alerts=int(flag.sum()),precision=tp/max(1,int(flag.sum())),recall=tp/int(actual.sum()),
                f1=f1_score(actual,flag),average_precision=average_precision_score(actual,1-prob.background),
                object_hits=objects.label.value_counts().to_dict(),
                transient_alerts=int((flag & kind.eq("transient")).sum()),
                transient_correct=int((flag & kind.eq("transient") & table.label.eq("transient")).sum()))


def run(root, out):
    out.mkdir(parents=True,exist_ok=True)
    ix, frames, truth = prepare(root)
    data = collect(root,ix,frames,truth,enrich=photometry)
    train = data[data.fold>=2]
    pos = train[train.label!="background"]
    neg = train[train.label=="background"]
    sampled = neg.sample(min(len(neg),8*len(pos)),random_state=42)
    base, base_cut = joblib.load("results/detector.joblib")
    models = {"baseline":base}
    def fit(extra):
        rows = pd.concat([pos,sampled,*extra])
        return RandomForestClassifier(n_estimators=100,min_samples_leaf=6,max_depth=14,n_jobs=2,random_state=42).fit(rows[FEATURES+HISTORY],rows.label)
    models["history"] = fit([])
    # Mine only training candidates absent from the initial classifier fit.
    pool = neg.drop(sampled.index)
    hard_scores = 1-probabilities(models["history"],pool).background
    hard = pool.loc[hard_scores.nlargest(len(pos)).index]
    repeats = [pos[pos.label=="transient"]]*7
    models["history_transients"] = fit(repeats)
    models["history_hard_negatives"] = fit([hard])
    models["history_both"] = fit([*repeats,hard])
    cuts = {"baseline":base_cut}
    report = {"training_transient_candidates":len(pos[pos.label=="transient"]),"added_hard_negatives":len(hard)}
    for name,fold in [("validation",1),("test",0)]:
        sample = data[data.fold==fold]
        labels = truth[truth.realization_id.isin(sample.realization_id.unique()) & truth.label.ne("background")]
        budget = int(((1-probabilities(base,sample).background)>=base_cut).sum())
        results = {}
        for method,model in models.items():
            prob = probabilities(model,sample)
            rank = 1-prob.background
            if fold==1 and method!="baseline":
                p,r,th = precision_recall_curve(sample.label!="background",rank)
                cuts[method]=float(th[np.argmax((2*p*r/np.maximum(p+r,1e-12))[:-1])])
            top = rank.rank(method="first",ascending=False)<=budget
            results[method] = dict(equal_budget=measure(sample,prob,top,labels),
                                   operating_point=measure(sample,prob,rank>=cuts[method],labels))
        report[name]=results
        if fold==1:
            winner = max(results,key=lambda k: results[k]["equal_budget"]["f1"])
            report["selected_on_validation"]=winner
    report["thresholds"]=cuts
    (out/"metrics.json").write_text(json.dumps(report,indent=2))
    joblib.dump((models[winner],cuts[winner]),out/"selected.joblib",compress=3)
    print(winner, {k: v["equal_budget"] for k,v in report["test"].items()},flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("dataset",type=Path)
    parser.add_argument("--out",type=Path,default=Path("experiments/temporal_classifier"))
    args=parser.parse_args()
    run(args.dataset,args.out)
