# Combining motion evidence across dates

WISE 0855 scored **0.157** against a validation threshold of **2.609**. The fit used measurements spanning 16 dates, with 42 frames passing the initial pixel and signal cuts. The fitted displacement was 0.030 pixels in x and −0.021 pixels in y. This weak fitted trend does not measure the object's true displacement reliably.

Starting from one reference position, we estimated the centre of light inside a fixed three-pixel radius on each exposure. We combined measurements from the same detector and day, then fitted a common linear velocity with a separate position offset for each detector. The score compares the fitted slope with centroid scatter. We fixed the threshold at the upper 95th percentile of 63 scored static validation injections. Its numerical value is an empirical decision threshold, not a calibrated significance.

| Object-level outcome in eight test fields | Existing blind detector | Targeted multi-date fit | Either method |
|---|---:|---:|---:|
| Eligible injected movers flagged (127 total) | 115 | 68 | 117 |
| Eligible injected static sources flagged (70 total) | 1 | 5 | 6 |

The added fit recovered two of the twelve previously missed movers and flagged five additional static controls. It returned scores for 99 of the 127 movers and 69 of the 70 controls. Objects without usable fits count as missed or unflagged. The new fit starts at the known injection reference position or the first catalogue position for WISE 0855. It does not demonstrate blind discovery. The false alerts here are counted only at the tested static source positions.

A fixed aperture can track the combined light of neighbouring sources and lose a fast mover that leaves the aperture. A constant velocity also cannot capture annual parallax. This experiment does not establish which effect suppressed the WISE 0855 fit. The test fields have been used in earlier project evaluations. An independent assessment would require new fields. We kept this method experimental.

Run `python multi_epoch.py path/to/planetx-dataset-v1.2.0` from the repository root after extracting the saved model. See [object-level comparisons](objects.csv) and [the target fit](summary.json).
