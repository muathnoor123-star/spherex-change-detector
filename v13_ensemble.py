"""Blend the previous frozen cutout CNN with the v1.3 validation-selected tree model."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from threadpoolctl import threadpool_limits
from detector import CLASSES, KEY, prepare, proposals
from evaluate_targets import audit_targets
from experiments.kaggle_cnn.training import patches
from v13_experiment import OUT, assess, classify, context, prob



def network():
    saved = torch.load("experiments/kaggle_cnn/model.pt",map_location="cpu",weights_only=True)
    assert saved["classes"] == CLASSES
    model = nn.Sequential(nn.Conv2d(4,24,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
                          nn.Conv2d(24,48,3,padding=1), nn.ReLU(), nn.Flatten(),
                          nn.Linear(48*5*5,64), nn.ReLU(), nn.Linear(64,len(CLASSES)))
    model.load_state_dict(saved["state_dict"])
    return model.eval()


def cnn_prob(cnn, path, meta, peaks, injected):
    images = torch.from_numpy(patches(path,meta,peaks.reset_index(drop=True),injected))
    with torch.inference_mode():
        return torch.cat([cnn(x.float()).softmax(1) for x in images.split(256)]).numpy()


def blend(tree, cnn, weight):
    return (1-weight)*tree + weight*cnn


def scan_field(root, row, frames, model, cnn, weight, threshold):
    meta = frames[frames.stack_id==row.stack_id].set_index("frame_index").sort_index()
    path = root/row.path
    peaks = proposals(path,meta,False,morphology=True)
    if peaks.empty:
        return pd.DataFrame(columns=["stack_id","frame_index","x","y","priority","change_type"])
    tree = prob(model,context(path,meta,peaks,False))
    image = cnn_prob(cnn,path,meta,peaks,False)
    p = blend(tree,image,weight)
    ranked = peaks.assign(stack_id=row.stack_id,priority=1-p[:,0],change_type=classify(p,threshold))
    return ranked[ranked.change_type!="background"].sort_values("priority",ascending=False)


def run(root, cache):
    cnn = network()
    model, _ = joblib.load(OUT/"model.joblib")
    data = pd.read_parquet(cache)
    ix, frames, truth = prepare(root)
    settings = {}
    for fold in [1,0]:
        blocks = []
        for row in ix.query("sample_type=='injected' and fold==@fold").itertuples():
            meta = frames[frames.stack_id==row.stack_id].set_index("frame_index").sort_index()
            peaks = data[data.realization_id.eq(row.realization_id)]
            if len(peaks):
                scores = cnn_prob(cnn,root/row.path,meta,peaks,True)
                blocks.append(peaks.assign(**dict(zip(CLASSES,scores.T))))
        sample = pd.concat(blocks)
        tree, image = prob(model,sample), sample[CLASSES].to_numpy()
        if fold == 1:
            for weight in [0,.25,.5,.75,1]:
                p = blend(tree,image,weight)
                pr,re,t = precision_recall_curve(sample.label!="background",1-p[:,0])
                cut = float(t[np.argmax((2*pr*re/np.maximum(pr+re,1e-12))[:-1])])
                settings[weight] = dict(threshold=cut, **assess(sample,p,cut))
            chosen = max(settings,key=lambda w:settings[w]["f1"])
            cut = settings[chosen]["threshold"]
            print("Selected CNN weight", chosen, flush=True)
    p = blend(tree,image,chosen)
    pred = classify(p,cut)
    field_rows = []
    for field, indices in sample.groupby("stack_id").indices.items():
        group = sample.iloc[indices]
        field_rows.append(dict(field=field, candidates=len(group),
                               positives=int(group.label.ne("background").sum()),
                               **assess(group,p[indices],cut)))
    pd.DataFrame(field_rows).to_csv(OUT/"ensemble_fields.csv",index=False)
    binary = assess(sample,p,cut)
    eligible = truth[truth.realization_id.isin(ix[ix.fold==0].realization_id) & truth.label.ne("background")]
    hits = sample.loc[(sample.label==pred)&(pred!="background"),KEY].drop_duplicates().assign(hit=True)
    outcome = eligible.merge(hits,on=KEY,how="left")
    outcome["hit"] = outcome.hit.eq(True)
    outcome.groupby(["label","realization_id","obj_id"]).hit.any().groupby("label").agg(["sum","count"]).to_csv(OUT/"ensemble_objects.csv")
    outcome.groupby("label").hit.agg(["sum","count"]).to_csv(OUT/"ensemble_frames.csv")
    report = dict(validation=settings,cnn_weight=chosen,threshold=cut,test=binary,
                  multiclass=classification_report(sample.label,pred,output_dict=True,zero_division=0))
    (OUT/"ensemble.json").write_text(json.dumps(report,indent=2))
    pd.DataFrame(confusion_matrix(sample.label,pred,labels=CLASSES),index=CLASSES,columns=CLASSES).to_csv(OUT/"ensemble_confusion.csv")
    print(binary,flush=True)
    alerts = [scan_field(root, row, frames, model, cnn, chosen, cut)
              for row in ix.query("sample_type=='real_stack'").itertuples()]
    alerts = pd.concat(alerts)
    alerts[["stack_id","frame_index","x","y","priority","change_type"]].to_csv(OUT/"ensemble_alerts.csv.gz",index=False)
    audit_targets(root,alerts).to_csv(OUT/"ensemble_targets.csv",index=False)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("cache",type=Path)
    args = parser.parse_args()
    torch.set_num_threads(3)
    with threadpool_limits(limits=3):
        run(args.root,args.cache)
