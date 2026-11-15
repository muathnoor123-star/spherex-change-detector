# PlanetX dataset v1.3.0 - EDA and training preparation

The task is unchanged: find objects that move or change across sky images taken at different times. This release preserves every original image array, catalogue label, fixed injection and sky-group split from v1.2.0. It adds reproducible EDA tables and an optional framework-neutral training adapter. No model was trained and model performance is not guaranteed.

Start with DATA_GUIDE.pdf. The original 17-page guide is retained, with small version/navigation updates and one new EDA and training-preparation section on page 18. For detailed counts use eda/summary.json and eda/*.csv.

## Training

Install requirements-training.txt. Add this package directory to Python's module path, then:

```python
from tools.training import TrainingDataset
ds = TrainingDataset('path/to/planetx-dataset-v1.3.0')
rows = ds.sequences(split='train', mission='wise')
s = ds.load_sequence(rows.iloc[0].stack_id)
# s['image']: [N,H,W], original physical values, grayscale
# s['time_days']: elapsed days, ordered with images
# s['frames']: original times, bands, quality, plus sequence_index
# s['observations']: x/y points, classes, visibility and uncertainty
# s['tracks'], s['sources']: original labels and measurements
# s['valid'], s['sigma']: pixel usability and uncertainty
aug = ds.inject_training(rows.iloc[0].stack_id, seed=1300)
fixed = ds.load_fixed('wise:wise0855:r0')  # test sample; do not train on it
```

Do not feed catalogue class, track position, mover flags, measured label photometry, or expected source identity as model inputs. Those are supervision/evaluation data. Model inputs are the images, times, masks, noise and detector/wavelength metadata. The targets are point positions and temporal changes; no boxes or segmentation masks have been invented. Keep all original classes, small motions and uncertain labels available. A missing catalogue label does not establish empty background.

Use sequence_index to index returned images. frame_index is the original array index, which can differ after QC and chronological ordering. Zero-based x is column and y is row. Elapsed times use days and are float64. gap_from_previous_frame_days can be zero for simultaneous observations. time_group identifies simultaneous detectors. Data may have variable-length sequences; batching/padding and model architecture remain the training team's choices. No global normalization is imposed because missions have different units. Fit any learned normalization on training data only and preserve masks.

load_sequence excludes failed frames, matching the original loader. sequences excludes empty stacks, but original files and index rows remain. require_multiple_times=True is optional; single-time comet data are retained and should not silently be dropped from other tasks. load_fixed uses the existing benchmark and its updated injected valid mask. In injected samples, real-source quality flags are PRE-injection measurements: injected truth is exact, but real-source photometry/visibility must not be interpreted as remeasured after overlap with a synthetic object. Fresh injection is restricted to WISE/SPHEREx training groups through inject_training. The upstream loader remains unchanged for compatibility.

## Added files

- training/sequences.parquet: usable sequence inventory, sky split, time coverage and gaps.
- training/<mission>_frames.parquet: frame metadata plus elapsed time and gaps, including original failed rows.
- training/<mission>_observations.parquet: every original track row with source class, timing and explicit quality status. Unmatched source labels occur only outside usable image domains and remain null.
- training/<mission>_sources.parquet: original source rows plus visible_distinct_times and motion_observable_two_times. Original is_mover and mover_detectable are unchanged.
- training/<mission>_injected_truth.parquet: original synthetic truth joined to object class/split, with unchanged benchmark eligibility and change-target rules.
- eda/: exact table counts, class ratios, numerical distributions, missing values, image statistics, time coverage, quality reasons, split summaries, synthetic coverage, visual examples and validation records.

## Important limitations

Labels are catalogue-based and not exhaustive human annotations. Source counts count objects per stack and can repeat the same object. Track rows are object-frame observations, not independent images. Asteroids passing through WISE/SPHEREx fields are not exhaustively labelled. NEOSSat has zero QC-passed training frames under the preserved split. Its asteroid coverage is limited to Phaethon and is insufficient for a standalone asteroid benchmark. Five added SPHEREx fields lack variability classifications. Some new fields cover only about 200 days and show small movement. Keep these limitations visible rather than inventing targets or silently changing splits.

WISE 0855 is a hard test case, not corrupted data: the supplied reference detector recovered 0/32 visible SPHEREx object-frames and 8/9 WISE object-frames. The SPHEREx motion is about 1.35 pixels over a year. These are reference-detector outcomes, not limits on a trained model. It remains held out.

No new synthetic population, image resizing, balancing, imputation, label remapping, object deletion, box generation or split reassignment was performed. Synthetic generation already existed and was tested, including same-seed repeatability. The original files were hash-verified before preparation and unchanged scientific payload files were reverified after it.

## EDA method

Every original Parquet table was inspected for row counts, column types, missing values and numerical distributions. Real frames were checked for key uniqueness and track joins. All NPZ arrays were opened, inventoried and checked for shape; usable image pixels and their uncertainties were validated. Classes and quality were summarized separately by mission and split. Time span and observation gaps were measured per sequence. Frame visibility uses the original s04_stacks.py criteria, including source saturation and contamination. Distinct-time visibility is an extra diagnostic, not a changed label. Source-file hash sharing and source-ID overlap across splits were checked; upstream footprint checks remain in qc/. Fixed injection order, object joins and benchmark eligibility were validated. Every usable real sequence and all fixed injections were loaded through the new adapter. A seeded fresh injection was tested for each supported mission.

The audit covers this local v1.2.0 package. Original external raw downloads are not included and cannot be reconstructed from EDA. File paths in provenance may refer to the builder's computer. They are not needed to train on this processed package.
