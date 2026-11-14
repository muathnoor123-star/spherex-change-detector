"""Compare blind real-image detections with the five named catalogue targets."""
from pathlib import Path
import numpy as np
import pandas as pd

TARGETS = {"barnards-star":"motion","epsilon-indi-b":"motion","wise0855":"motion",
           "rw-aurigae":"variable","r-draconis":"variable"}


def audit_targets(root, candidates):
    root = Path(root)
    index = pd.read_parquet(root/"index.parquet").query("mission == 'spherex' and sample_type == 'real_stack'")
    source = pd.read_parquet(root/"spherex/sources.parquet").query("is_target")
    frames = pd.read_parquet(root/"spherex/frames.parquet").query("qc_pass")
    rows = []
    for field, expected in TARGETS.items():
        row = index[index.field == field].iloc[0]
        sid = row.stack_id
        targets = source[source.stack_id == sid]
        columns = ["stack_id","src_id","frame_index","x","y","saturated","contaminated","aperture_valid_frac","snr"]
        tracks = pd.read_parquet(root/"spherex/tracks.parquet",columns=columns,filters=[("stack_id","=",sid)])
        tracks = tracks[tracks.src_id.isin(targets.src_id) & tracks.frame_index.isin(frames[frames.stack_id==sid].frame_index)]
        usable = tracks[(~tracks.saturated)&(~tracks.contaminated)&(tracks.aperture_valid_frac>=0.95)&(tracks.snr>=5)]
        hits = set()
        nearest = candidates[(candidates.stack_id==sid)&(candidates.change_type==expected)]
        for t in usable.itertuples():
            same = nearest[nearest.frame_index==t.frame_index]
            if np.any(np.hypot(same.x-t.x,same.y-t.y)<=2):
                hits.add(t.frame_index)
        rows.append(dict(target=field,split=row.split,expected=expected,
                         targets=len(targets),qc_frames=frames[frames.stack_id==sid].frame_index.nunique(),
                         saturated_frames=tracks.loc[tracks.saturated,"frame_index"].nunique(),
                         usable_frames=usable.frame_index.nunique(),matched_frames=len(hits)))
    return pd.DataFrame(rows)
