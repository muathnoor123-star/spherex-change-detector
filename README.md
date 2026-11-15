# SPHEREx change detector

Planet X v1.3.0 is available with the complete dataset, EDA and trained models. The validation-selected CNN and boosted-tree blend achieved 77.22% test F1 and 98.17% binary candidate accuracy across eight previously evaluated fields. [Experiments and reproduction](experiments/v13/README.md) · [Dataset download](https://github.com/muathnoor123-star/spherex-change-detector/releases/tag/dataset-v1.3.0).

for a real time demo you may use this link

https://github.com/muathnoor123-star/spherex-change-detector

## Original v1.2 detector

Train a CPU model to locate candidate changes in aligned SPHEREx images and rank them for review. The input is Planet X v1.2.0. See the [evaluation report](REPORT.md) for test results and limitations. DATA_CITATIONS.md reproduces the dataset credits.

## Reproduce the run

1. Use the original dataset ZIP, or download all twenty data-*.zip assets from the dataset-v1.2.0 GitHub Release and run `python restore_dataset.py path/to/downloads`. It reconstructs the archive and checks SHA-256.
2. For Jupyter or Kaggle, open **SPHEREx_AI.ipynb**, attach the ZIP or extracted directory and run all cells. Optionally set DATASET_PATH in the first cell. Kaggle execution has not been verified. Locally, install the pinned requirements in a Python 3.11 Jupyter kernel.
3. For a terminal run, extract the archive, install dependencies with `python -m pip install -r requirements.txt`, then run `python train.py path/to/planetx-dataset-v1.2.0`. The default output directory is results.
4. To use the saved model, load the trusted project artifact and call the same scanner used for evaluation.

```python
from pathlib import Path
import joblib
from detector import scan

model, threshold = joblib.load("results/detector.joblib")
ranked = scan(Path("data/planetx-dataset-v1.2.0"), "spherex:wise0855", model, threshold)
print(ranked[["frame_index", "mjd_mid", "x", "y", "change_type", "priority"]].head())
```

The notebook embeds the code and can run independently. The trained outputs are also distributed as results.zip; extract it into the project directory to restore results/. real_candidates.csv.gz contains all above-threshold candidates in all 33 unmodified fields. Positions are zero-based pixels on the aligned grid. MJD is in days. Each row is a detection in one frame. Multiple rows can refer to the same physical source.

## Training design

Each usable exposure is compared with the median of up to three other quality-passed exposures from the same detector, at least one day away and nearest in log wavelength. At least two references are required. Local extrema in the absolute smoothed difference above 3.5 robust noise units become proposals. The 100-tree random forest uses the signed residual, positive and negative structure, two patch centroids, local flux contrast, mask coverage and wavelength gap. The comparison approximates spectral matching. A sharp spectrum or masking near a bright star can still produce a residual.

The fixed injections on real images provide labels. Positive peaks lie within two pixels of an eligible injected mover (motion at least one pixel), variable or transient. Static and subpixel injections, plus unmatched peaks, supply background examples. Unlabelled real changes can enter that background. The training set samples eight backgrounds per positive at most. Folds 2–4 train the forest, fold 1 selects the binary change threshold by maximizing candidate F1, and fold 0 is evaluated afterwards. Class predictions are mutually exclusive.

Priority is one minus the forest background score. It ranks evidence for a reviewable change. It is uncalibrated and does not quantify rarity, scientific value or the probability of a discovery. Candidate classification metrics are conditional on a peak being proposed. The separate recovery tables include missed proposals in their denominators. Object recovery requires at least one correctly classified eligible frame. The published dataset baseline uses a different matching metric, so the report makes no superiority claim.

OpenAI Codex implemented and trained this detector, conducted the evaluations, and prepared the notebook and report under the project owner's direction.
