"""Train a cutout classifier and export the run for review."""
import copy
import json
import warnings
from detector import CLASSES, prepare, collect
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_recall_fscore_support


def patches(path, meta, peaks, injected):
    with np.load(path) as archive:
        image = np.where(archive["valid"], archive["image"], np.nan)
        if injected:
            assert np.array_equal(archive["frame_index"], meta.index)
        else:
            image = image[meta.index.to_numpy()]
    cuts = np.empty((len(peaks), 4, 11, 11), dtype=np.float16)
    dy, dx = np.mgrid[-5:6, -5:6]
    for fid, group in peaks.groupby("frame_index"):
        i = meta.index.get_loc(fid)
        row = meta.iloc[i]
        refs = np.flatnonzero((meta.detector == row.detector) & (abs(meta.mjd_mid-row.mjd_mid)>1))
        refs = refs[np.argsort(abs(np.log(meta.wave_center_um.to_numpy()[refs]/row.wave_center_um)))[:3]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ref = np.nanmedian(image[refs], axis=0)
        y = group.y.to_numpy()[:,None,None] + dy
        x = group.x.to_numpy()[:,None,None] + dx
        inside = (y>=0)&(x>=0)&(y<ref.shape[0])&(x<ref.shape[1])
        region = y.clip(0,ref.shape[0]-1), x.clip(0,ref.shape[1]-1)
        a, b = image[i][region], ref[region]
        valid = inside & np.isfinite(a) & np.isfinite(b)
        noise = float(row.pixel_noise)
        if not np.isfinite(noise) or noise <= 0:
            pixels = image[i][np.isfinite(image[i])].astype(np.float64)
            noise = 1.4826 * np.median(np.abs(pixels - np.median(pixels)))
            if noise <= 0:
                noise = pixels.std()
        if not np.isfinite(noise):
            raise ValueError(f"Cannot estimate pixel noise in {path}, frame {fid}")
        noise = max(noise, 1e-8)
        channels = [np.arcsinh(np.where(valid,v,0)/noise).clip(-16,16) for v in (a,b,a-b)]
        cuts[group.index] = np.stack([*channels,valid],axis=1)
    if not np.isfinite(cuts).all():
        raise ValueError(f"Non-finite image cutouts in {path}")
    return cuts


def train_cutouts(root, out, reference, smoke=False):
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out.mkdir(parents=True,exist_ok=True)
    ix, frames, truth = prepare(root)
    ix = ix.query("sample_type == 'injected'").copy()
    assert ix.groupby("group").fold.nunique().max()==1
    if smoke:
        ix = ix.groupby("fold",sort=False).head(1)
    cut_blocks, labels = [], []
    for j, row in enumerate(ix.itertuples()):
        meta = frames[frames.stack_id==row.stack_id].set_index("frame_index").sort_index()
        block = collect(root, ix.iloc[j:j+1], frames, truth)
        if row.fold >= 2:
            bg = block.index[block.label=="background"]
            take = rng.choice(bg,min(len(bg),max(128,8*(len(block)-len(bg)))),replace=False)
            block = block.loc[np.sort(np.r_[block.index[block.label!="background"],take])]
        block = block.reset_index(drop=True)
        cut_blocks.append(patches(root/row.path,meta,block,True))
        labels.append(block[["fold","label","realization_id","obj_id"]])
        if j % 20 == 0:
            print("Prepared stacks", j+1, "/",len(ix),flush=True)
    table = pd.concat(labels,ignore_index=True)
    X = torch.from_numpy(np.concatenate(cut_blocks))
    del cut_blocks
    y = torch.tensor(pd.Categorical(table.label,categories=CLASSES).codes.astype("int64"))
    def loader(mask,shuffle=False):
        idx = torch.from_numpy(np.flatnonzero(mask))
        return DataLoader(idx,batch_size=256,shuffle=shuffle)
    model = nn.Sequential(nn.Conv2d(4,24,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                          nn.Conv2d(24,48,3,padding=1),nn.ReLU(),nn.Flatten(),
                          nn.Linear(48*5*5,64),nn.ReLU(),nn.Linear(64,len(CLASSES))).to(device)
    train = table.fold>=2
    weights = torch.bincount(y[train.to_numpy()],minlength=4).clamp(min=1).float().rsqrt()
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    opt = torch.optim.Adam(model.parameters(),lr=.001)
    def predict(mask):
        model.eval()
        with torch.no_grad():
            prob = np.concatenate([model(X[idx].to(device,dtype=torch.float32)).softmax(1).cpu().numpy() for idx in loader(mask)])
            if not np.isfinite(prob).all():
                raise FloatingPointError("Non-finite model predictions before evaluation")
            return prob
    validation = table.fold==1
    best = -1.
    history = []
    for epoch in range(1, (1 if smoke else 8)+1):
        model.train()
        for idx in loader(train,True):
            opt.zero_grad()
            loss = loss_fn(model(X[idx].to(device,dtype=torch.float32)),y[idx].to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss in epoch {epoch}")
            loss.backward()
            opt.step()
        scores = 1-predict(validation)[:,0]
        actual = table.loc[validation,"label"]!="background"
        ap = float(average_precision_score(actual,scores))
        history.append(ap)
        if ap>best:
            best, state, selected = ap, copy.deepcopy(model.state_dict()), epoch
            p,r,t = precision_recall_curve(actual,scores)
            cut = float(t[np.argmax((2*p*r/np.maximum(p+r,1e-10))[:-1])])
        print("Epoch",epoch,"validation average precision",round(ap,4),flush=True)
    model.load_state_dict(state)
    test = table.fold==0
    data = table[test].reset_index(drop=True)
    prob = predict(test.to_numpy())
    score = 1-prob[:,0]
    kinds = np.array(CLASSES)[1+prob[:,1:].argmax(axis=1)]
    actual = data.label.ne("background")
    eligible = truth[truth.realization_id.isin(ix.loc[ix.fold==0,"realization_id"])]
    totals = eligible.drop_duplicates(["realization_id","obj_id"]).label.value_counts()
    def metrics(flag):
        p,r,f,_ = precision_recall_fscore_support(actual,flag,average="binary",zero_division=0)
        recovered = data[flag & data.label.eq(kinds)].drop_duplicates(["realization_id","obj_id"]).label.value_counts()
        return dict(alerts=int(flag.sum()),precision=float(p),recall=float(r),f1=float(f),
                    object_hits={c:int(recovered.get(c,0)) for c in CLASSES[1:]},
                    transient_alerts=int((flag & (kinds=="transient")).sum()),
                    transient_correct=int((flag & (kinds=="transient") & data.label.eq("transient")).sum()))
    budget = reference["alerts"]
    if smoke:
        budget = max(1, round(len(data)*4453/117825))
    else:
        assert len(data)==117825 and len(ix)==198, "Dataset does not match the published benchmark"
    top = np.zeros(len(data),dtype=bool)
    top[np.argsort(-score,kind="stable")[:budget]] = True
    matched = metrics(top)
    result = dict(status="complete",smoke=smoke,device=str(device),torch_version=str(torch.__version__),
                  seed=42,selected_epoch=selected,validation_ap=history,threshold=cut,
                  test_candidates=len(data),test_average_precision=float(average_precision_score(actual,score)),
                  eligible_objects={c:int(totals.get(c,0)) for c in CLASSES[1:]},
                  equal_budget=matched,operating_point=metrics(score>=cut))
    if not smoke:
        result.update(reference_equal_budget=reference,f1_change_points=100*(matched["f1"]-reference["f1"]))
    torch.save(dict(state_dict={k:v.cpu() for k,v in state.items()},classes=CLASSES,threshold=cut),out/"model.pt")
    (out/"results.json").write_text(json.dumps(result,indent=2))
    with zipfile.ZipFile(out/"run.zip","w",zipfile.ZIP_DEFLATED) as archive:
        for name in ["model.pt","results.json"]:
            archive.write(out/name,name)
    return result
