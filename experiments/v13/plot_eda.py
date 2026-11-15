"""Plot the released SPHEREx coverage and eligible synthetic labels."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

folder = Path(__file__).resolve().parent
eda = folder / "dataset_overlay" / "eda"
splits = ["train", "val", "test"]
frames = pd.read_csv(eda/"split_summary.csv").query("mission=='spherex'").set_index("split").loc[splits]
objects = pd.read_csv(eda/"synthetic_distribution.csv").query("mission=='spherex'")
table = objects.pivot(index="object_class",columns="split",values="eligible_rows")[splits]

fig, axes = plt.subplots(1,2,figsize=(10,4.2),layout="constrained")
axes[0].bar(splits,frames.frames,color="#d5dde4",label="Failed QC")
axes[0].bar(splits,frames.passing,color="#176b79",label="Passed QC")
for i,row in enumerate(frames.itertuples()):
    axes[0].text(i,row.frames+18,f"{row.passing}/{row.frames}",ha="center")
axes[0].set(ylim=(0,1150),ylabel="Original exposures",title="SPHEREx frame coverage")
axes[0].legend(frameon=False)
table.plot.barh(ax=axes[1],color=["#176b79","#f2a541","#655992"],width=.76,logx=True)
axes[1].set(xlabel="Eligible injected object-frames (log scale)",ylabel="",title="Supervision by class and split")
axes[1].legend(title="",frameon=False)
for ax in axes:
    ax.spines[["top","right"]].set_visible(False)
fig.savefig(folder/"eda.png",dpi=160)
