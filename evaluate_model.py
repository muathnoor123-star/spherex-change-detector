"""Evaluate the saved detector at its frozen operating point on fold zero."""
from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix
from detector import CLASSES, KEY, prepare, collect, score


def evaluate(root, model_path, out):
    index, frames, truth = prepare(root)
    test = index.query("fold == 0 and sample_type == 'injected'")
    model, threshold = joblib.load(model_path)
    peaks = score(model, collect(root, test, frames, truth))
    flagged = peaks.priority >= threshold
    actual = peaks.label != "background"
    prediction = np.where(flagged, peaks.change_type, "background")
    matrix = confusion_matrix(peaks.label, prediction, labels=CLASSES)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).rename_axis("actual").to_csv(out/"confusion.csv")
    eligible = truth[truth.realization_id.isin(test.realization_id) & truth.label.ne("background")]
    hits = peaks[flagged & peaks.label.eq(peaks.change_type)][KEY].drop_duplicates().assign(hit=True)
    matched = eligible.merge(hits,on=KEY,how="left",validate="one_to_one")
    matched["hit"] = matched.hit.eq(True)
    per_object = matched.groupby(["label","realization_id","obj_id"]).hit.any()
    report = {"fields":test.stack_id.nunique(),"realizations":len(test),"candidates":len(peaks),
              "threshold":threshold,"alerts":int(flagged.sum()),"matched_change_alerts":int((actual & flagged).sum()),
              "binary":classification_report(actual,flagged,output_dict=True,zero_division=0)["True"],
              "average_precision":average_precision_score(actual,peaks.priority),
              "class_metrics":classification_report(peaks.label,prediction,labels=CLASSES,output_dict=True,zero_division=0),
              "frame_recovery":matched.groupby("label").hit.agg(["sum","count"]).to_dict("index"),
              "object_recovery":per_object.groupby(level="label").agg(["sum","count"]).to_dict("index")}
    (out/"metrics.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model", type=Path, default=Path("results/detector.joblib"))
    parser.add_argument("--out", type=Path, default=Path("evaluation"))
    args = parser.parse_args()
    evaluate(args.dataset, args.model, args.out)
