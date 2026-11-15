"""Extract blind features and compare classifiers on the Planet X v1.3 SPHEREx split."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, precision_recall_curve)
from threadpoolctl import threadpool_limits

from detector import CLASSES, FEATURES, KEY, collect, prepare, proposals
from evaluate_targets import audit_targets
from morphology_experiment import MORPH
from temporal_classifier import HISTORY, photometry

OUT = Path("experiments/v13")
CONTEXT = ["other_time_near", "other_time_same_sign", "other_time_distance",
           "other_time_strength", "frame_density", "nearest_peak"]


def context(path, meta, peaks, injected):
    peaks = photometry(path, meta, peaks, injected)
    xy = peaks[["x", "y"]].to_numpy()
    signed = peaks.diff_z.to_numpy()
    times = meta.loc[peaks.frame_index, "mjd_mid"].to_numpy()
    for fid, group in peaks.groupby("frame_index"):
        idx = group.index.to_numpy()
        pts = xy[idx]
        frame_tree = cKDTree(pts)
        density = frame_tree.query_ball_point(pts, 12, return_length=True) - 1
        nearest = frame_tree.query(pts, k=2)[0][:, 1].clip(max=64)
        other = np.flatnonzero(np.abs(times-times[idx[0]]) > 1)
        if len(other):
            tree = cKDTree(xy[other])
            distances, neighbors = tree.query(pts, k=list(range(1, 9)))
            strengths = signed[other[neighbors.clip(max=len(other)-1)]]
            near = distances <= 2
            columns = [near.sum(axis=1), (near & (strengths*signed[idx,None]>0)).sum(axis=1),
                       distances[:,0].clip(max=64), np.where(near, abs(strengths), 0).max(axis=1)]
        else:
            columns = [np.zeros(len(idx)) for _ in range(4)]
            columns[2][:] = 64
        peaks.loc[idx, CONTEXT] = np.column_stack([*columns, density, nearest])
    return peaks


def extract(root, cache):
    ix, frames, truth = prepare(root)
    assert ix.groupby("group").fold.nunique().max() == 1
    blocks = []
    for row in ix.query("sample_type == 'injected'").itertuples():
        block = collect(root, ix.loc[[row.Index]], frames, truth, enrich=context, morphology=True)
        blocks.append(block.assign(stack_id=row.stack_id))
        if len(blocks) % 20 == 0:
            print("Extracted", len(blocks), flush=True)
    pd.concat(blocks, ignore_index=True).to_parquet(cache, index=False)


def prob(model, sample):
    return pd.DataFrame(model.predict_proba(sample[list(model.feature_names_in_)]),
                        columns=model.classes_, index=sample.index)[CLASSES].to_numpy()


def classify(p, cut):
    return np.where(1-p[:,0] >= cut, np.array(CLASSES)[1+p[:,1:].argmax(1)], "background")


def assess(sample, p, cut):
    actual = sample.label.ne("background")
    pred = classify(p, cut)
    report = classification_report(actual, pred != "background", output_dict=True)
    return dict(accuracy=accuracy_score(actual, pred != "background"),
                average_precision=average_precision_score(actual, 1-p[:,0]),
                precision=report["True"]["precision"], recall=report["True"]["recall"],
                f1=report["True"]["f1-score"], alerts=int((pred != "background").sum()))


def run(root, cache):
    if not cache.exists():
        extract(root, cache)
    data = pd.read_parquet(cache)
    train = data[data.fold>=2]
    val, test = (data[data.fold==i] for i in (1,0))
    results, best = {}, -1
    for name, cols in [("spatial_temporal", FEATURES+HISTORY+MORPH),
                       ("with_neighbors", FEATURES+HISTORY+MORPH+CONTEXT)]:
        model = HistGradientBoostingClassifier(max_iter=180, max_leaf_nodes=31,
                    learning_rate=.12, l2_regularization=2, early_stopping=False, random_state=42)
        model.fit(train[cols], train.label)
        p = prob(model, val)
        pr, re, th = precision_recall_curve(val.label!="background", 1-p[:,0])
        cut = float(th[np.argmax((2*pr*re/np.maximum(pr+re,1e-15))[:-1])])
        results[name] = assess(val, p, cut)
        if results[name]["f1"] > best:
            best = results[name]["f1"]
            selected, winner, threshold = name, model, cut
        print(name, results[name], flush=True)
    outputs(root, test, winner, selected, threshold, results)


def scan(root, row, frames, model, cut):
    meta = frames[frames.stack_id==row.stack_id].set_index("frame_index").sort_index()
    path = root/row.path
    peaks = proposals(path, meta, False, morphology=True)
    if len(peaks):
        p = prob(model, context(path, meta, peaks, False))
        peaks = peaks.assign(priority=1-p[:,0], change_type=classify(p, cut))
        return peaks[peaks.change_type!="background"].assign(stack_id=row.stack_id)
    return pd.DataFrame()


def outputs(root, test, model, name, cut, results):
    ix, frames, truth = prepare(root)
    p = prob(model, test)
    pred = classify(p, cut)
    actual = test.label
    report = dict(validation=results, selected=name, threshold=cut, test=assess(test,p,cut),
                  candidates=len(test), fields=test.stack_id.nunique(), positives=int((actual!="background").sum()),
                  multiclass=classification_report(actual,pred,output_dict=True,zero_division=0))
    pd.DataFrame(confusion_matrix(actual,pred,labels=CLASSES),index=CLASSES,columns=CLASSES).to_csv(OUT/"confusion.csv")
    eligible = truth[truth.realization_id.isin(ix[ix.fold==0].realization_id) & truth.label.ne("background")]
    hits = test.loc[(actual==pred) & (actual!="background"),KEY].drop_duplicates().assign(hit=True)
    recovered = eligible.merge(hits,on=KEY,how="left")
    recovered["hit"] = recovered.hit.eq(True)
    recovered.groupby(["label","realization_id","obj_id"]).hit.any().groupby("label").agg(["sum","count"]).to_csv(OUT/"objects.csv")
    (OUT/"metrics.json").write_text(json.dumps(report,indent=2))
    joblib.dump((model,cut),OUT/"model.joblib",compress=3)
    alerts = pd.concat([scan(root,row,frames,model,cut) for row in ix.query("sample_type=='real_stack'").itertuples()])
    alerts[["stack_id","frame_index","x","y","priority","change_type"]].to_csv(OUT/"real_alerts.csv.gz",index=False)
    audit_targets(root,alerts).to_csv(OUT/"named_targets.csv",index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root",type=Path)
    parser.add_argument("cache",type=Path,help="Parquet cache path reserved for this v1.3 extraction")
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        run(args.root,args.cache)
