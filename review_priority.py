"""Validate a transient review quota under a fixed total alert budget."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from detector import CLASSES, collect, prepare
from experiments.kaggle_cnn.training import patches


def evaluate(table, budget, fraction):
    # Reserve a few slots for the highest transient scores, then fill by change score.
    reserved = table.nlargest(round(budget * fraction), "transient_score")
    general = table.drop(reserved.index).nlargest(budget - len(reserved), "priority")
    queue = pd.concat([reserved, general])
    actual = queue[queue.label != "background"]
    objects = actual.drop_duplicates(["realization_id", "obj_id"])
    return dict(quota=fraction, alerts=len(queue), true_alerts=len(actual),
                precision=len(actual)/len(queue), recovered=objects.label.value_counts().to_dict(),
                reserved_slots=len(reserved), reserved_true_transient=int(reserved.label.eq("transient").sum()))


def run(root, out):
    torch.set_num_threads(2)
    folder = Path("experiments/kaggle_cnn")
    checkpoint = torch.load(folder/"model.pt",map_location="cpu",weights_only=True)
    assert checkpoint["classes"] == CLASSES
    model = nn.Sequential(nn.Conv2d(4,24,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                          nn.Conv2d(24,48,3,padding=1),nn.ReLU(),nn.Flatten(),
                          nn.Linear(48*5*5,64),nn.ReLU(),nn.Linear(64,len(CLASSES)))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    ix, frames, truth = prepare(root)
    results = {}
    for fold, name, budget in [(1,"validation",6064),(0,"test",4453)]:
        chunks = []
        selection = ix.query("sample_type == 'injected' and fold == @fold")
        for j,row in enumerate(selection.itertuples()):
            meta = frames[frames.stack_id==row.stack_id].set_index("frame_index").sort_index()
            peaks = collect(root,ix.loc[[row.Index]],frames,truth)
            if peaks.empty:
                continue
            cuts = torch.from_numpy(patches(root/row.path,meta,peaks,True))
            with torch.inference_mode():
                prob = torch.cat([model(batch.float()).softmax(1) for batch in cuts.split(256)]).numpy()
            assert np.isfinite(prob).all()
            chunks.append(peaks[["realization_id","obj_id","label"]].assign(priority=1-prob[:,0],transient_score=prob[:,CLASSES.index("transient")]))
            if j % 12 == 0:
                print(name, j+1, "/",len(selection),flush=True)
        data = pd.concat(chunks, ignore_index=True)
        baseline = evaluate(data,budget,0)
        if name=="validation":
            trials = [evaluate(data,budget,q) for q in [0,.01,.025,.05,.1]]
            feasible = [v for v in trials if v["precision"] >= baseline["precision"]-.02]
            selected = max(feasible,key=lambda v:(v["recovered"].get("transient",0),v["precision"],-v["quota"]))["quota"]
            results["validation_trials"] = trials
        results[name] = dict(baseline=baseline,selected=evaluate(data,budget,selected))
        print(results[name],flush=True)
    out.mkdir(parents=True,exist_ok=True)
    (out/"metrics.json").write_text(json.dumps(results,indent=2))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",type=Path)
    parser.add_argument("--out",type=Path,default=Path("experiments/review_priority"))
    args = parser.parse_args()
    run(args.dataset,args.out)
