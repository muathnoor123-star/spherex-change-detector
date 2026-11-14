# Where WISE 0855 drops out

We scored the real WISE 0855 stack using the frozen morphology checkpoint. Catalogue positions were used for evaluation only. In each of the 29 eligible frames we searched within two pixels of the target and independently checked the proposal count against the extrema in the residual image. Eligibility and matching follow the five-target audit.

| Stage | Frames remaining |
| --- | ---: |
| Eligible target position | 29 |
| Blind peak proposal within two pixels | 8 |
| Change priority at least 0.412442 | 0 |
| Above-threshold motion alert | 0 |

All 29 frames had three reference exposures. In 21 frames even the strongest valid residual within the matching radius was below the proposal threshold (3.5 noise units). In the other eight there was exactly one nearby proposal per frame. Their change priorities ranged from 0.0064 to 0.0493. Among the non-background classes, seven favoured variability and one favoured motion. Spatial proximity alone does not prove these peaks belong to the target.

The largest loss occurs before classification (21/29); the classifier also rejects every surviving proposal. Lowering only the review cutoff cannot recover the 21 missing proposals. The median across frames of the target's median reference offset was 0.77 pixels. Partial cancellation during image subtraction is plausible but not established. This is a diagnostic of one known target, not a new model or an updated benchmark.

A next controlled test could compare motion-aligned residual stacks with stationary stacks at matched control positions, choosing settings on separate validation fields and tracking false alerts. This tests subthreshold residuals directly and complements the previous centroid-fit experiment.

From the repository root with the dataset and morphology checkpoint restored:

~~~bash
python diagnose_wise0855.py data/planetx-dataset-v1.2.0 --morphology --model experiments/morphology/model.joblib --out experiments/morphology/wise0855
~~~

[Frame-by-frame measurements](wise0855_frames.csv)
