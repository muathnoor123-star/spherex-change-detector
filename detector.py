"""Blind peak proposals and supervised SPHEREx change classification."""
from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve, average_precision_score

FEATURES = ["peak", "flux_z", "reference_z", "diff_z", "dip", "rise",
            "centroid_shift", "flux_contrast", "valid_fraction", "wave_gap"]
CLASSES = ["background", "motion", "variable", "transient"]
KEY = ["realization_id", "frame_index", "obj_id"]


def proposals(path, frames, injected):
    """Return peaks in calibrated, aligned images without access to object labels."""
    with np.load(path) as stack:
        ids = frames.index.to_numpy()
        if injected and not np.array_equal(ids, stack["frame_index"]):
            raise ValueError("Frame metadata does not match image order")
        image, valid = stack["image"], stack["valid"].astype(bool)
        if not injected:
            image, valid = image[ids], valid[ids]
    image = np.where(valid, image, np.nan)
    times = frames.mjd_mid.to_numpy()
    wave = frames.wave_center_um.to_numpy()
    detectors = frames.detector.to_numpy()
    yy, xx = np.mgrid[-2:3, -2:3]
    result = []
    for i, frame_id in enumerate(ids):
        refs = np.flatnonzero((detectors == detectors[i]) & (abs(times - times[i]) > 1))
        if len(refs) < 2:
            continue
        refs = refs[np.argsort(abs(np.log(wave[refs] / wave[i])))[:3]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ref = np.nanmedian(image[refs], axis=0)
        good = np.isfinite(image[i]) & np.isfinite(ref)
        current = np.where(good, image[i], 0)
        reference = np.where(good, ref, 0)
        diff = gaussian_filter(current - reference, 0.85)
        noise = max(1e-8, 1.4826 * np.median(abs(diff[good] - np.median(diff[good])))) if good.any() else 1
        z = diff / noise
        strength = abs(z)
        peaks = good & (strength >= 3.5) & (strength == maximum_filter(strength, 5))
        peaks[:2] = peaks[-2:] = False
        peaks[:, :2] = peaks[:, -2:] = False
        y, x = np.where(peaks)
        if not len(x):
            continue
        region = (y[:, None, None] + yy, x[:, None, None] + xx)
        a, b, d = current[region], reference[region], z[region]
        def centroid(patch):
            weights = np.maximum(patch - patch.min((1, 2), keepdims=True), 0)
            total = np.maximum(weights.sum((1, 2)), 1e-8)
            return np.array([(weights * axis).sum((1, 2)) / total for axis in (xx, yy)])
        signal = max(float(frames.pixel_noise.iloc[i]), 1e-8)
        flux_a, flux_b = a.sum((1, 2)), b.sum((1, 2))
        features = [strength[y, x], flux_a / signal, flux_b / signal, z[y, x],
                    d.min((1, 2)), d.max((1, 2)), np.linalg.norm(centroid(a)-centroid(b), axis=0),
                    (flux_a-flux_b)/(abs(flux_a)+abs(flux_b)+signal), good[region].mean((1, 2)),
                    np.full(len(x), np.max(abs(np.log(wave[refs]/wave[i]))))]
        table = pd.DataFrame(np.column_stack(features), columns=FEATURES)
        table["frame_index"], table["x"], table["y"] = frame_id, x, y
        result.append(table)
    return pd.concat(result, ignore_index=True) if result else pd.DataFrame(columns=FEATURES+["frame_index","x","y"])


def prepare(root):
    ix = pd.read_parquet(root/"index.parquet").query("mission == 'spherex'")
    frames = pd.read_parquet(root/"spherex/frames.parquet").query("qc_pass")
    objects = pd.read_parquet(root/"spherex/injected_objects.parquet")
    truth = pd.read_parquet(root/"spherex/injected_truth.parquet").merge(
        objects[["realization_id","obj_id","class","is_mover"]], on=["realization_id","obj_id"], validate="many_to_one")
    truth["label"] = np.select([truth["class"].eq("variable"), truth["class"].eq("transient"), truth.is_mover],
                               ["variable","transient","motion"], default="background")
    eligible = truth.query("present and in_grid and masked_frac < 0.5 and not saturated").copy()
    return ix, frames, eligible


def collect(root, ix, frames, eligible, enrich=None):
    samples = []
    for row in ix.query("sample_type == 'injected'").itertuples():
        meta = frames[frames.stack_id == row.stack_id].set_index("frame_index").sort_index()
        candidates = proposals(root/row.path, meta, True)
        if enrich is not None:
            candidates = enrich(root/row.path, meta, candidates, injected=True)
        candidates["obj_id"], candidates["label"] = -1, "background"
        labels = eligible[eligible.realization_id == row.realization_id]
        for fid, group in candidates.groupby("frame_index"):
            stars = labels[labels.frame_index == fid]
            if stars.empty:
                continue
            dist, idx = cKDTree(stars[["x","y"]]).query(group[["x","y"]])
            near = dist <= 2
            candidates.loc[group.index[near], ["obj_id","label"]] = stars.iloc[idx[near]][["obj_id","label"]].to_numpy()
        candidates["realization_id"], candidates["fold"] = row.realization_id, row.fold
        samples.append(candidates)
    return pd.concat(samples, ignore_index=True)


def score(model, table):
    table = table.copy()
    probabilities = model.predict_proba(table[FEATURES])
    for label, values in zip(model.classes_, probabilities.T):
        table["score_"+label] = values
    table["priority"] = 1-table.score_background
    table["change_type"] = table[["score_"+k for k in CLASSES[1:]]].idxmax(axis=1).str[6:]
    return table


def train(root, out):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ix, frames, eligible = prepare(root)
    assert ix.groupby("group").fold.nunique().max() == 1
    data = collect(root, ix, frames, eligible)
    train_rows = data[data.fold.isin([2,3,4])]
    positives = train_rows.query("label != 'background'")
    negatives = train_rows.query("label == 'background'")
    train_rows = pd.concat([positives, negatives.sample(min(len(negatives),8*len(positives)),random_state=42)])
    model = RandomForestClassifier(n_estimators=100, min_samples_leaf=6, max_depth=14, n_jobs=2, random_state=42)
    model.fit(train_rows[FEATURES], train_rows.label)
    val = score(model, data[data.fold == 1])
    precision, recall, thresholds = precision_recall_curve(val.label != "background", val.priority)
    f1 = 2*precision*recall / np.maximum(precision+recall, 1e-12)
    threshold = float(thresholds[np.argmax(f1[:-1])])
    joblib.dump((model, threshold),out/"detector.joblib",compress=3)
    test = score(model, data[data.fold == 0])
    flagged = test[test.priority >= threshold]
    actual = test.label != "background"
    metrics = {"training_candidates":len(train_rows), "training_counts":train_rows.label.value_counts().to_dict(),
               "validation_candidates":len(val),"test_candidates":len(test),"test_positive_candidates":int(actual.sum()),
               "test_fields":int(ix.query("fold == 0").stack_id.nunique()), "threshold":threshold,
               "test_average_precision":average_precision_score(actual,test.priority),
               "test_candidate_report":classification_report(test.label,np.where(test.priority>=threshold,test.change_type,"background"),
                    labels=CLASSES,output_dict=True,zero_division=0)}
    eligible_test = eligible[eligible.realization_id.isin(ix[ix.fold==0].realization_id)].query("label != 'background'")
    correct = flagged[flagged.label == flagged.change_type][KEY].drop_duplicates().assign(recovered=True)
    recovery = eligible_test.merge(correct,on=KEY,how="left")
    recovery["recovered"] = recovery.recovered.eq(True)
    recovery.groupby("label").recovered.agg(["sum","count","mean"]).to_csv(out/"frame_recovery.csv")
    recovery.groupby(["label","realization_id","obj_id"]).recovered.any().groupby("label").agg(["sum","count","mean"]).to_csv(out/"object_recovery.csv")
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2))
    return model, threshold, metrics


def scan(root, stack_id, model, threshold):
    root = Path(root)
    ix = pd.read_parquet(root/"index.parquet")
    row = ix[(ix.stack_id==stack_id)&(ix.sample_type=="real_stack")].iloc[0]
    frames = pd.read_parquet(root/"spherex/frames.parquet")
    meta = frames[(frames.stack_id==stack_id)&frames.qc_pass].set_index("frame_index").sort_index()
    peaks = proposals(root/row.path,meta,False)
    if peaks.empty:
        return pd.DataFrame(columns=["frame_index","x","y","priority","change_type","mjd_mid","detector"])
    ranked = score(model,peaks).join(meta[["mjd_mid","detector"]],on="frame_index")
    return ranked[ranked.priority>=threshold].sort_values("priority",ascending=False)
