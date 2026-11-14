"""Matched-budget ablation of signed residual morphology on unchanged peak proposals."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from detector import FEATURES, prepare, collect, proposals
from temporal_classifier import HISTORY, photometry, probabilities, measure
from evaluate_targets import TARGETS, audit_targets

MORPH = ["opposite_fraction", "opposite_peak_ratio", "core_fraction", "lobe_offset", "spatial_extent"]


def run(root, out):
    out.mkdir(exist_ok=True, parents=True)
    ix, frames, truth = prepare(root)
    # Process one field at a time, retaining candidate tables only.
    chunks = []
    for number, (_, group) in enumerate(ix.query("sample_type == 'injected'").groupby("stack_id", sort=False)):
        chunk = collect(root, group, frames, truth, enrich=photometry, morphology=True)
        chunks.append(chunk)
        if number % 5 == 0:
            print("Extracted fields:", number + 1, flush=True)
    data = pd.concat(chunks, ignore_index=True)
    # Restore the original index order for reproducible negative sampling.
    order = dict(zip(ix.realization_id, range(len(ix))))
    data = data.sort_values("realization_id", key=lambda s:s.map(order), kind="stable").reset_index(drop=True)
    train = data[data.fold >= 2]
    pos = train[train.label != "background"]
    neg = train[train.label == "background"]
    sampled = neg.sample(min(len(neg),8*len(pos)), random_state=42)
    history = FEATURES + HISTORY
    def fit(rows, cols):
        return RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_leaf=6,
                                      random_state=42, n_jobs=2).fit(rows[cols],rows.label)
    print("Mining training negatives", flush=True)
    miner = fit(pd.concat([pos,sampled]),history)
    pool = neg.drop(sampled.index)
    hard = pool.loc[(1-probabilities(miner,pool).background).nlargest(len(pos)).index]
    print("Fitting spatial model", flush=True)
    model = fit(pd.concat([pos,sampled,hard]),history+MORPH)
    previous, prior_cut = joblib.load("experiments/temporal_classifier/selected.joblib")
    base, base_cut = joblib.load("results/detector.joblib")
    report = {}
    for fold, name in [(1,"validation"),(0,"test")]:
        s = data[data.fold==fold]
        labels = truth[truth.realization_id.isin(s.realization_id.unique()) & truth.label.ne("background")]
        budget = ((1-probabilities(base,s).background)>=base_cut).sum()
        report[name] = {}
        for label, clf in [("previous",previous),("morphology",model)]:
            prob = probabilities(clf,s)
            score = 1-prob.background
            if fold==1 and label=="morphology":
                p,r,t = precision_recall_curve(s.label!="background",score)
                cut = float(t[np.argmax((2*p*r/np.maximum(p+r,1e-12))[:-1])])
            threshold = prior_cut if label=="previous" else cut
            report[name][label] = dict(equal_budget=measure(s,prob,score.rank(method="first",ascending=False)<=budget,labels),
                                       operating_point=measure(s,prob,score>=threshold,labels))
    report["morphology_threshold"] = cut
    (out/"metrics.json").write_text(json.dumps(report,indent=2))
    joblib.dump((model,cut),out/"model.joblib",compress=3)
    alerts = []
    for row in ix.query("sample_type == 'real_stack'").itertuples():
        if row.field not in TARGETS:
            continue
        meta = frames[frames.stack_id==row.stack_id].set_index("frame_index").sort_index()
        path = root/row.path
        peaks = proposals(path,meta,False,morphology=True)
        if peaks.empty:
            continue
        prob = probabilities(model,photometry(path,meta,peaks,False))
        alerts.append(peaks.assign(stack_id=row.stack_id,change_type=prob.drop(columns="background").idxmax(axis=1))[1-prob.background>=cut])
    audit_targets(root,pd.concat(alerts)).to_csv(out/"named_targets.csv",index=False)
    print(json.dumps(report["test"]),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",type=Path)
    parser.add_argument("--out",type=Path,default=Path("experiments/morphology"))
    args = parser.parse_args()
    run(args.dataset,args.out)
