# Temporal photometry and training examples

The experiment compares the saved baseline with four classifiers on identical image peaks. Each temporal classifier measures flux in a five-pixel aperture at the candidate position, even at epochs without a detection. It uses up to five exposures from the same detector more than one day away, choosing the closest available wavelengths. References can occur before or after the target exposure. A simultaneous measurement from another detector supplies an additional feature when available. This is an offline analysis. The method approximates wavelength matching and does not fit a spectrum.

1. History alone adds flux, repeatability, excursion, scatter, coverage and simultaneous signal features.
2. History plus transient oversampling presents each labelled training transient eight times in total. These are repetitions of existing examples.
3. History plus hard negatives adds previously unsampled training background peaks with the highest change scores from the history-only forest.
4. The combined variant uses both changes to the training sample.

Training uses folds 2–4. The original saved detector sets the number of available review slots on each split. Each method fills exactly that number with its highest-ranked peaks, breaking score ties by input order. Selection uses binary F1 at this fixed budget on validation fold 1. Each variant also has a deployable score threshold chosen by validation F1. The known baseline retains its original threshold. Fold 0 is the project's previously examined test set. It supports a paired comparison on the existing benchmark. Independent confirmation requires new sky fields.

Run from the repository root with the original model extracted:

```sh
python temporal_classifier.py data/planetx-dataset-v1.2.0
```

For offline inference on an original field, use the experimental scanner:

```python
from pathlib import Path
from temporal_classifier import scan

alerts = scan(Path("data/planetx-dataset-v1.2.0"), "spherex:wise0855",
              Path("experiments/temporal_classifier/selected.joblib"))
```

At 4,453 test alerts, binary F1 rose from 51.26 to 57.58 points with history and to 60.02 with history plus hard negatives. The validation-selected model recovered 116 of 127 movers and 44 of 60 variable sources at that budget. It correctly classified zero of 76 transient objects. Adding oversampling to history and hard negatives recovered six transient objects, with seven correct transient alerts out of 61, and binary F1 of 59.68.

At its frozen validation threshold the selected model issued 4,101 test alerts, with 66.50% precision, 55.09% recall and binary F1 of 60.26 points. These are candidate-level injection benchmark results. On real observations Epsilon Indi B matches fell from 13 to 8 of 42 eligible frames. WISE 0855 remained at zero of 29 and RW Aurigae at zero of two. The other two named targets had no eligible unsaturated frames. This is an experimental checkpoint.

[All comparisons](metrics.json) · [Real targets](named_targets.csv) · [Notebook](experiment.ipynb)

To load the trained checkpoint from GitHub, run `python experiments/temporal_classifier/restore_model.py` once. It joins the 19 binary parts and verifies their SHA-256 before writing selected.joblib.
