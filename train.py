from pathlib import Path
import pandas as pd
from detector import train, scan

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    model, threshold, metrics = train(args.dataset, args.out)
    print(metrics, flush=True)
    index = pd.read_parquet(args.dataset/"index.parquet")
    targets = index.query("mission == 'spherex' and sample_type == 'real_stack'")
    print(targets.field.tolist(),flush=True)
    candidates = []
    for row in targets.itertuples():
        table = scan(args.dataset, row.stack_id, model, threshold)
        table["stack_id"] = row.stack_id
        candidates.append(table)
    pd.concat(candidates,ignore_index=True).to_csv(args.out/"real_candidates.csv.gz",index=False)
    from evaluate_targets import audit_targets
    audit_targets(args.dataset,pd.concat(candidates,ignore_index=True)).to_csv(args.out/"named_targets.csv",index=False)
