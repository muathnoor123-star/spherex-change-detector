# Signed residual morphology

The added spatial measurements improved test candidate F1 from 60.02 to 64.66 points at 4,453 alerts. Both classifiers received identical blind peak proposals and used the same temporal measurements, training fields, negative sampling procedure and forest settings. Five new features describe the signed residual in a 9 by 9 pixel window: opposite-sign flux fraction and peak ratio, central flux concentration, offset of the opposite-sign centroid, and spatial extent. These features do not require target coordinates.

| Fixed test budget | Previous | With morphology |
| --- | ---: | ---: |
| Precision | 63.37% | 68.27% |
| Recall among proposed positives | 57.01% | 61.41% |
| Binary F1 | 60.02% | 64.66% |
| Average precision across thresholds | 62.26% | 69.97% |
| Movers recovered, correct class | 116/127 | 117/127 |
| Variables recovered, correct class | 44/60 | 46/60 |
| Transients recovered, correct class | 0/76 | 0/76 |

At the validation budget of 6,064 alerts, F1 rose from 66.10 to 72.59 points. At the new validation-selected threshold of 0.412442, the test set produced 4,282 alerts, 70.08% precision, 60.63% recall and 65.01% F1. Score thresholds and training examples were not chosen on the test fold. The test fields have been evaluated in earlier experiments; this is a comparison on a recurring benchmark, not independent confirmation on new fields.

On the five original target fields, expected-class matches within two pixels at the frozen threshold rose from 8 to 10 of 42 eligible Epsilon Indi B frames. WISE 0855 stayed at 0/29 and RW Aurigae at 0/2. Barnard's Star and R Draconis had no eligible frames. Frame matches do not establish an orbit, discovery or catalogue completeness. Per-object recovery requires at least one correctly classified frame. The binary candidate metrics combine motion, variability and transients.

## Reproduce

From the repository root, restore the dataset and extract results.zip as described in the main README. Run the following with the project dependencies installed:

~~~bash
python experiments/temporal_classifier/restore_model.py
python morphology_experiment.py data/planetx-dataset-v1.2.0
~~~

[Training notebook](experiment.ipynb) | [Metrics](metrics.json) | [Target audit](named_targets.csv)

To load the already trained forest, run the script below from the repository root to assemble and verify the checkpoint. The model expects the five morphology features as well as the temporal and original features; use the feature extraction and scoring in morphology_experiment.py.

~~~bash
python experiments/morphology/restore_model.py
~~~

The new model is useful for prioritizing candidates. Its ability to detect transients or recover WISE 0855 has not improved. A useful next experiment is to measure whether labelled transients and predicted WISE positions generate a candidate at all before changing the classifier.
