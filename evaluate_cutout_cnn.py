"""Evaluate the frozen Kaggle checkpoint on catalogue targets and held-out transients."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from detector import CLASSES, collect, prepare, proposals
from evaluate_targets import TARGETS, audit_targets
from experiments.kaggle_cnn.training import patches


def run(root, out):
    torch.set_num_threads(2)
    checkpoint = torch.load(out/"model.pt", map_location="cpu", weights_only=True)
    reference = json.loads((out/"results.json").read_text())
    threshold = checkpoint["threshold"]
    assert checkpoint["classes"] == CLASSES and threshold == reference["threshold"]
    model = nn.Sequential(nn.Conv2d(4,24,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                          nn.Conv2d(24,48,3,padding=1),nn.ReLU(),nn.Flatten(),
                          nn.Linear(48*5*5,64),nn.ReLU(),nn.Linear(64,len(CLASSES)))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    index, frames, truth = prepare(root)
    selected = index.query("sample_type == 'injected' and fold == 0")
    real = index[index.sample_type.eq("real_stack") & index.field.isin(TARGETS)]
    originals, injections = [], []
    for row in pd.concat([real,selected]).itertuples():
        meta = frames[frames.stack_id==row.stack_id].set_index("frame_index").sort_index()
        injected = row.sample_type == "injected"
        path = root/row.path
        if injected:
            peaks = collect(root,index.loc[[row.Index]],frames,truth)
        else:
            peaks = proposals(path,meta,False)
        if peaks.empty:
            continue
        cuts = torch.from_numpy(patches(path,meta,peaks,injected))
        with torch.inference_mode():
            prob = torch.cat([model(batch.float()).softmax(1) for batch in cuts.split(256)]).numpy()
        assert np.isfinite(prob).all()
        peaks = peaks.assign(stack_id=row.stack_id,priority=1-prob[:,0],
                             change_type=np.array(CLASSES)[1+prob[:,1:].argmax(axis=1)])
        (injections if injected else originals).append(peaks)
    original = pd.concat(originals)
    named = audit_targets(root,original[original.priority>=threshold])
    named.to_csv(out/"named_targets.csv",index=False)
    sample = pd.concat(injections,ignore_index=True)
    assert len(sample)==reference["test_candidates"]
    flag = sample.priority.rank(method="first",ascending=False)<=reference["equal_budget"]["alerts"]
    keys = ["realization_id","obj_id"]
    objects = truth[truth.realization_id.isin(selected.realization_id) & truth.label.eq("transient")][keys].drop_duplicates()
    for name, mask in [("proposed",sample.label.eq("transient")),("alerted",sample.label.eq("transient") & flag),
                       ("correct",sample.label.eq("transient") & flag & sample.change_type.eq("transient"))]:
        objects[name] = pd.MultiIndex.from_frame(objects[keys]).isin(pd.MultiIndex.from_frame(sample.loc[mask,keys]))
    assert len(objects)==reference["eligible_objects"]["transient"]
    assert int(objects.correct.sum())==reference["equal_budget"]["object_hits"]["transient"]
    objects.to_csv(out/"transient_objects.csv",index=False)
    print(named.to_string(index=False), flush=True)
    print(objects[["proposed","alerted","correct"]].sum())
    print(sample[flag & sample.change_type.eq("transient")].label.value_counts())


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",type=Path)
    parser.add_argument("--out",type=Path,default=Path("experiments/kaggle_cnn"))
    args = parser.parse_args()
    run(args.dataset,args.out)

