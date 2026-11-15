# Planet X / SPHEREx moving-object dataset, v1.2.0

Aligned image stacks from SPHEREx, NEOWISE and NEOSSat, with object-level labels for every catalogued star, galaxy, infrared source and named target in every frame, plus a fixed benchmark of synthetic movers injected into the real images. It is built for training and testing models that find things that move or change between exposures: high proper-motion stars, brown dwarfs, distant solar-system bodies, asteroids and variables.

Built on 13 November 2026 from the raw download (now at `Codex/2026-11-13/see/outputs/excluded-not-for-submission/original-delivery/training-data`) by the scripts in `../pipeline/` (copied to `provenance/`). Nothing in the raw download was modified.

[DATA_GUIDE.pdf](DATA_GUIDE.pdf) explains each data source in plain words, one chapter each, for readers without a technical background. Every file, array, column, unit and flag bit is described in [DATA_DICTIONARY.md](DATA_DICTIONARY.md). Every data source, its license and the AI-use statement the hackathon requires are in [CITATIONS.md](CITATIONS.md).

Version 1.2 changes: 21 more SPHEREx fields around the fastest-moving stars in Gaia; the motion of WISE 0855 and Luhman 16 is now measured from the WISE frames instead of taken from SIMBAD; synthetic objects reach S/N 3000 and saturate like real stars; SPHEREx photometry carries a wavelength-corrected residual; the baseline compares frames at similar wavelengths. Version 1.1 removed the SkyBoT asteroid tracks, stopped counting faint sources in the glare of bright stars as detected, and moved baseline scores to the test fold.

## For the training team

- **Inputs per sample:** a stack of aligned frames (`image`, `sigma`, `valid`, `flags`, `wave`) plus per-frame metadata (time, detector or band, `time_group`, noise, PSF). Frames in the same `time_group` were taken at the same moment, so a real object appears in all of them and a detector artifact usually in one.
- **Labels:**
  - per-frame positions and photometry of every known source: `tracks.parquet`
  - per-source class and motion: `sources.parquet`
  - exact truth for synthetic movers: `injected_truth.parquet`
- **Positives:** real data has few true movers (3 SPHEREx, 9 WISE, 7 NEOSSat stack entries, plus the 3I/ATLAS comet). Use `Dataset.inject()` for as many synthetic movers as training needs, and keep the fixed `injected/` realisations for validation and test so scores stay comparable.
- **Splits:** use `splits.json` (folds by sky group). Never split by frame or row.
- **Floor to beat:** the baseline numbers below, computed on the same benchmark.
- **Read before trusting a label:** the "Known limits" section.

| | SPHEREx | NEOWISE | 3I/ATLAS (SPHEREx) | NEOSSat |
|---|---:|---:|---:|---:|
| Stacks | 33 | 10 | 14 | 69 |
| Frames / passing QC | 1,980 / 1,856 | 802 / 733 | 24 / 23 | 614 / 228 |
| Registration: median shift uncertainty | 0.012 px | 0.012 px | 0.008 px | 0.105 px |
| Registration: median star scatter after shift | 0.048 px (0.30") | 0.061 px (0.17") | 0.043 px | 0.32 px |
| Median PSF FWHM | 1.82 px | 2.80 px | | |
| Labelled sources (stack rows) | 69,412 | 4,734 | 19,716 | 19,352 |
| Real movers visible in 2+ frames (1 px or more) | 4 | 9 | comet in 9 stacks | 7 |
| Injected realisations / objects | 198 / 1,980 | 40 / 400 | | |

About 2.4 GB in total. Every file is listed with its SHA-256 in `MANIFEST.sha256`.

The real movers in SPHEREx are Barnard's Star, Epsilon Indi B, WISE 0855 and Gaia DR3 4034171629042489088 (1.1-1.8 px). Twelve more targets move 0.4-0.9 px: sub-pixel motion that a model can still measure on a bright star (`motion_px` gives the value for every source). In WISE, nine stars and brown dwarfs move 10-38 px over ten years, including 61 Cyg B. In NEOSSat the movers are comet C/2020 F3 (NEOWISE) and asteroid Phaethon. Where 3I/ATLAS is bright and unblended, its measured centre is within 0.06 px of the Horizons position computed for SPHEREx in 11 of 14 measurements; the other three are faint or blended and sit 0.25, 0.33 and 1.04 px away. Comet NEOWISE lies within 0.29 px (median) of the position computed for NEOSSat.

The 21 added fields are every Gaia DR3 source with proper motion above 3 arcsec/yr that was not already covered, chosen from Gaia alone. They were downloaded during an IRSA outage that blocked the later QR3 frames, so they span about 200 days, against 370-530 days for the original 12 fields. That is why most of them stay below one pixel. `pipeline/s00_acquire_more.py` now asks for both releases; rerunning it when IRSA is back roughly doubles their motion.

Known asteroids crossing the SPHEREx and WISE fields are not labelled. Version 1.0 took their positions from IMCCE SkyBoT; that analysis found none of them (V 20.5-24) cleanly detectable in a single frame. An unlabelled moving point in these stacks can therefore be a known faint asteroid.

## What is in a stack

A stack is every usable exposure of one sky field, resampled onto one north-up tangent-plane grid centred on the target's position at the mean epoch.

| Mission | Grid | Frames per stack | Wavelength |
|---|---|---|---|
| SPHEREx | 128 x 128 at 6.15"/px (13.1') | 60 (6 detectors x 10 visits) | 0.75-5.0 um, changes from frame to frame |
| NEOWISE | 128 x 128 at 2.75"/px (5.9') | 48-84 single exposures, 2014-2024 | W1 3.4 um, W2 4.6 um |
| NEOSSat | 256 x 256 at 3.0"/px (12.8') | one target-night | optical, about 0.6 um |
| 3I/ATLAS | 128 x 128 at 6.15"/px | the 2 simultaneous detectors of one exposure | SPHEREx |

`<mission>/stacks/<field>.npz` holds arrays of shape `[N, H, W]`:

- `image`: surface brightness with the zodiacal model (SPHEREx) or a 64-pixel mesh background (WISE, NEOSSat) removed, then a residual sky median removed on the grid. SPHEREx is in MJy/sr; WISE is in flux units where Vega mag = 22.5 - 2.5 log10(flux), from each frame's MAGZP; NEOSSat is in bias-subtracted ADU.
- `sigma`: 1-sigma per pixel, propagated through the resampling. Resampling correlates neighbouring pixels, so this understates aperture noise; use `aperture_sky_noise` in `frames.parquet` for detection thresholds.
- `valid`: 1 where the value is usable. A grid pixel is valid only if all four native pixels feeding it are valid.
- `flags`: the native bitmask, OR-combined over the four contributing pixels. Bits that remove a pixel are listed under Quality control; artifact bits (ghosts, crosstalk, persistence, cosmic-ray repairs and the like) are kept as information so a model can learn to discount them rather than never seeing them.
- `wave`, `bandwidth`: per-pixel central wavelength and bandwidth in um (SPHEREx, from the WAVE-TAB lookup).

WISE stacks also have `<field>-coadd.npz`: inverse-variance coadds per half-year and band. They use no outlier rejection, because with two scans per half-year rejection would also delete real transients.

`<mission>/psf/<field>.npz` gives an empirical PSF for every frame, normalised to unit sum, with `source` telling where it came from: the frame's own stars (3 or more isolated, unsaturated Gaia stars on a flat background), else all frames of that detector in the field, else that detector across the mission. A single 7.7 s WISE exposure rarely has three such stars, so most WISE PSFs are pooled per field and band.

## Tables

- `frames.parquet`: one row per frame. Time (`mjd_mid`, UTC), detector or band, `time_group` (exposures taken at the same moment, such as SPHEREx detector pairs D1/D4, D2/D5, D3/D6 or WISE W1/W2), registration result, coverage, noise, PSF width, photon-noise slope `shot_k`, observer position for parallax, source file and its SHA-256, and the QC verdict with reasons.
- `sources.parquet`: one row per source per stack. Astrometry, magnitudes, Gaia class probabilities, the label (`class`, `subclass`, `label_source`), and measured outcomes: frames inside the grid, frames detected at S/N >= 5, frames saturated, total motion in pixels, `is_mover` (motion >= 1 px) and `mover_detectable` (mover visible in at least two frames).
- `tracks.parquet`: one row per source per frame. Predicted position on the grid and forced aperture photometry there: flux, error, S/N, valid fraction of the aperture, flag bits inside it, `saturated`, and the centroid offset where S/N > 10.
- `injected_objects.parquet` and `injected_truth.parquet`: the synthetic benchmark (below).

## How the labels were made

Positions are predicted for each frame, then measured. Nothing is labelled from a file name.

- **Stars and galaxies**: Gaia DR3, queried for every field with no row limit (the original tables stopped at 5,000 rows; the Proxima field alone has 38,735 sources). Positions are propagated to each frame time with proper motion and annual parallax. Class comes from Gaia's DSC probabilities (> 0.9) or a parallax above 5 sigma; `variable_star` comes from Gaia's variability classifier. Five of the added fields were fetched from IRSA's copy of Gaia DR3 while ESA's archive was refusing queries; that copy has no variability classes, so those fields have no `variable_star` labels.
- **Infrared-only sources**: AllWISE sources with no Gaia counterpart within 3" (matched at the AllWISE epoch), `ir_only_point` or `extended_ir`. Cold brown dwarfs and red galaxies live here. For the added fields AllWISE came from ESA's copy of the catalogue (IRSA was down); it has no S/N columns, so S/N there is 1.0857 divided by the magnitude error.
- **Named targets**: for the original fields, SIMBAD says which object the field is about; a Gaia source becomes the target only if it lies within 3" and, for a fast target, its proper motion agrees within 20%. The astrometry then comes from Gaia. WISE 0855 and Luhman 16 have no Gaia proper motion, so theirs is measured here: starting from the AllWISE position, the object is centroided in every W2 half-year coadd from 2014 to 2024 and position, proper motion and parallax are fitted (`pipeline/s03b_target_astrometry.py`). The fit gives (-8120 +- 35, 687 +- 35) mas/yr and 396 +- 114 mas for WISE 0855 and (-2751 +- 18, 357 +- 18) mas/yr and 414 +- 98 mas for Luhman 16; SIMBAD lists (-8124, 673), 439 and (-2768, 357), 502. For the added fields the target is the Gaia source the field was selected for, with no name resolver involved (`label_source = gaia_dr3_proper_motion_selection`).
- **Wavelength-corrected photometry (SPHEREx)**: every frame is taken at a different wavelength, so a constant source still changes flux between frames. `tracks.parquet` gives the wavelength at each source (`wave_um`), a local spectrum estimate from the measurements nearest in wavelength taken at other times (`flux_sed`) and the deviation from it (`sed_resid_snr`). Its robust scatter per source, `sed_scatter`, has a median of 1.2 for ordinary stars, so real variability shows up as values well above that.
- **Comets and the NEOSSat targets**: JPL Horizons ephemerides computed for the spacecraft itself (SPHEREx -163182, NEOSSat -139089), for 3I/ATLAS, C/2020 F3 and Phaethon. A solar-system object counts as detected in a frame only if its S/N is at least 5 (difference S/N where other-day frames exist), no bright catalogued source lies within 2.5 px, no artifact flag other than OUTLIER is in the aperture, and its centroid is within 1.5 px of the ephemeris position.
- **Glare from bright stars**: a catalogue source counts as detected only if, where a centroid could be measured (S/N > 5), it lies within 1.5 px of the prediction, and no source at least 5 times brighter sits within 4 px (`contaminated`). Before this rule, hundreds of faint sources near Epsilon Indi A, Wolf 359 and Lalande 21185 were measured at S/N above 30 from the bright star's light alone.
- **OUTLIER flag**: the SPHEREx pipeline sets OUTLIER on the 3I/ATLAS comet in every frame where it is bright. A moving object can therefore carry this flag. It is kept as information, never used to mask pixels, and does not veto a mover.
- **Saturation**: a source whose aperture contains saturation bits is labelled `saturated` in that frame. It counts as visible when the saturation is its own (`saturated_self`: the target, or a star with G < 13 or W1 < 9.5), not a neighbour's. Barnard's Star, Proxima, R Dra, RW Aur and Ross 128 saturate in most SPHEREx frames, so their photometry is meaningless there while their position is not.

The labels are complete only for what the catalogues contain: roughly G < 21 for Gaia and the AllWISE depth for infrared sources. An unlabelled source in an image is not proof of a new object, and the baseline's unexplained detections are review candidates, not discoveries.

## Quality control

Frames are never deleted from the stacks; failures carry `qc_pass = False` and a reason. The loader skips them by default.

- Registration: the WCS is corrected by the median offset of isolated, confident Gaia stars measured in the native frame. A frame fails with fewer than 5 such stars, a shift above 3 px, or a shift uncertainty (1.2533 x scatter / sqrt(n)) above 0.1 px.
- Coverage: at least 50% of the grid valid, and at least 80% of the central 32 x 32 pixels inside the exposure footprint. Saturation in the centre does not fail a frame; a saturated target is plainly visible and labelled.
- Noise: aperture sky noise more than 3x the median of the same detector or band in that field.
- Pixels removed: SPHEREx OVERFLOW, SUR_ERROR, NONFUNC, MISSING_DATA, HOT, COLD, NONLINEAR; WISE bits 1, 3, 4, 6, 9-19 and 30 (the set the WISE pipeline already blanks, Explanatory Supplement Table 7); NEOSSat saturation (>= 60,000 ADU) and the overscan strip.

## Synthetic benchmark

Real data contain only a handful of verified movers, so each real stack also comes with fixed injected realisations (6 per SPHEREx field, 4 per WISE field, 10 objects each), stored in `<mission>/injected/`. The injection code is `tools/injection.py`; `Dataset.inject()` makes unlimited new realisations for training.

Every object follows offset(t) = proper motion x (t - t_ref) + parallax x (P(t) - P(t_ref)), with P(t) the annual parallax factor from the observer's real position at each frame time. Classes:

| Class | Parameters |
|---|---|
| pm_star | 0.05-10 "/yr, parallax 0-0.5" |
| fast_mover | 10-60 "/yr, parallax 0.1-0.8" (nearby brown dwarf) |
| distant_planet | 100-2000 AU: parallax 206265/d arcsec plus orbital motion. Most of its apparent motion is the observer's reflex, as it would be for a real Planet X |
| transient | present in one exposure time only |
| variable | static, 0.3-1.5 mag sinusoid |
| static | constant (negative control) |

Flux follows a power law in wavelength (index -2 to 3), the target S/N is drawn log-uniformly from 3 to 3000, each frame's own empirical PSF is used, and photon noise is added from that frame's measured variance-versus-signal slope. Where the real image plus the injected light exceeds the detector's saturation level (`sat_level` in `frames.parquet`, estimated per detector from the brightest unsaturated pixels next to real saturated stars), the pixel is blanked and marked invalid, as it would be for a real bright star; 39% of SPHEREx injected objects saturate in at least one frame. The saturation level is an estimate, good to a factor of about two. Injections are not placed on masked pixels. `injected_truth` records, per object and frame, the position, whether it is present and inside the grid, the true aperture flux and S/N, whether it saturated, and the fraction of its light that fell on masked pixels. Each injected `.npz` carries its own `valid` mask.

## Splits

`splits.json` assigns every physical group to one of five folds. A group is one sky region and target: all missions, its offset control field, and every derived and injected sample stay together, so no target or overlapping exposure appears on both sides of a split. The default split uses fold 0 for test, fold 1 for validation and folds 2-4 for training; with only 16 sky groups plus the NEOSSat objects, five-fold cross-validation gives steadier numbers. `qc/leakage-checks.json` confirms that no two groups share a source file or lie within 1 degree of each other.

| Fold | Groups |
|---|---|
| 0 (test) | WISE 0855 (+ offset control), R Dra, M31, Wolf 359 (WISE and SPHEREx), NEOSSat comet NEOWISE, 4 added Gaia fields |
| 1 (validation) | Epsilon Indi B, RW Aur, M101, Lalande 21185 (WISE and SPHEREx), NEOSSat Phaethon, 4 added Gaia fields |
| 2 | Luhman 16, M51, 61 Cyg (WISE and SPHEREx), NEOSSat C/2021 S3, NEOSSat Tabby field, 4 added Gaia fields |
| 3 | Barnard's Star (+ offset control), Proxima, NEOSSat T CrB, NEOSSat 1999 KW4, 3 added Gaia fields |
| 4 | Teegarden's Star, Ross 128, 3I/ATLAS, NEOSSat 12P, 3 added Gaia fields |

| Default split | SPHEREx stacks (QC frames) | WISE stacks (QC frames) | NEOSSat stacks (QC frames) | 3I/ATLAS stacks | Injected realisations |
|---|---:|---:|---:|---:|---:|
| train | 17 (973) | 5 (345) | 39 (0) | 14 | 122 |
| validation | 8 (454) | 2 (152) | 10 (136) | | 56 |
| test | 8 (429) | 3 (236) | 20 (92) | | 60 |

NEOSSat contributes no QC-passed training frames: all 12P and C/2021 S3 sequences fail registration because their 60-272 s exposures show smeared stars (star positions scatter 1.3-3 px and an affine fit does not reduce it), and the T CrB control has too few stars in its 5 s exposures. Treat NEOSSat as a test of transfer to optical data. Two near pairs cross folds without overlapping footprints: one Phaethon stack (NEOSSat, fold 1) lies 0.32-0.33 degrees from the Wolf 359 fields (WISE and SPHEREx, fold 0). The exact group-to-fold map, including the added fields by Gaia source id, is in `splits.json`.

## Baseline

`tools/baseline.py` is a plain difference-imaging detector: subtract the median of the same detector's frames taken more than a day away, normalise by the robust scatter, smooth with a 1-pixel Gaussian, keep 5-sigma peaks, and mark a peak as coincident if the simultaneous partner frame shows 3 sigma within 2 px. A model trained on this dataset should beat these numbers.

Held-out numbers: the test fold only (8 SPHEREx stacks, 3 WISE stacks). Completeness counts injected objects that should be flagged (movers of 1 px or more, transients and variables) and are found within 2 px in a frame where they are present, inside the grid, less than half masked and not saturated:

| True S/N | 3-5 | 5-10 | 10-20 | 20-50 | 50+ |
|---|---:|---:|---:|---:|---:|
| SPHEREx | 0.05 | 0.26 | 0.41 | 0.51 | 0.71 |
| WISE | 0.11 | 0.41 | 0.67 | 0.57 | 0.86 |

| At S/N >= 10 | distant_planet | fast_mover | transient | moving pm_star | variable | truly static flagged (false alarm) |
|---|---:|---:|---:|---:|---:|---:|
| SPHEREx | 0.95 | 0.78 | 0.90 | 0.58 | 0.32 | 0.09 |
| WISE | 0.98 | 0.97 | 1.00 | 0.88 | 0.35 | 0.00 |

The S/N 50+ column now reaches S/N 3000. Bright stars that move less than a pixel are flagged in 23% (SPHEREx) and 59% (WISE) of frames; that is a real signal, since a small shift of a very bright star is measurable, so it is reported apart from the false-alarm column. New false detections caused by injection, not counting anything within 8 px of an injected object brighter than S/N 100: 0.31 per SPHEREx frame and 0.64 per WISE frame. Numbers over all folds are in `baseline/<mission>-summary.json`.

Wavelength matching, tested on the original 12 SPHEREx fields: the baseline's detections drop from 15.0 to 12.1 per frame while real-mover recovery stays the same (30 against 31 of 72 unsaturated frames).

Real movers in the test fold: the baseline finds WISE 0855 in 0 of its 32 visible SPHEREx frames (it moves 1.35 px in a year and is faint) and real movers in 8 of 9 visible WISE frames. Over all folds it recovers real movers in 33 of 78 SPHEREx frames and 50 of 51 WISE frames. It also returns about 4.0 unexplained detections per SPHEREx test frame and 4.5 per WISE test frame. Most are residuals of bright, red or saturated stars, and some may be the unlabelled faint asteroids; they are listed in `baseline/<mission>-unexplained-detections.parquet` for review, and none should be called a discovery without follow-up.

## Using it

```python
import sys; sys.path.insert(0, "dataset")
from tools.loader import Dataset
ds = Dataset("dataset")
for sid in ds.stacks(split="train", mission="spherex"):
    s = ds.load(sid)                        # QC-passed frames only
    x = s["image"], s["sigma"], s["valid"], s["wave"]
    y = s["tracks"], s["sources"]
    aug = ds.inject(sid, seed=7)            # fresh synthetic movers on top
test = ds.load_injected("spherex:wise0855:r0")
```

Needs numpy, pandas and pyarrow; scipy for injection and the baseline.

## Known limits

- SPHEREx measures each frame at a different wavelength across its linear variable filter. Brightness differences between frames of one detector can be spectral rather than temporal; compare frames at similar `wave`, or let the model use it.
- Over one year of SPHEREx only four targets move a full pixel, and the 21 added fields cover about 200 days each (see above). WISE's ten years show all its targets moving 10-38 px.
- Thirty-three SPHEREx fields, ten WISE fields and seven NEOSSat groups are enough to develop and compare methods, not to claim performance on the whole sky.
- NEOSSat fluxes are not calibrated, and 214 NEOSSat frames without sky coordinates are excluded (`neossat/excluded-frames.csv`).
- Catalogue labels inherit catalogue errors: Gaia class probabilities are model outputs.
- The NEOSSat movers in the default split are all in the test fold (7); validation has none.
- What was left out of the raw download, and why, is in `excluded.json`.

## Rebuilding

From the project folder, with the packages in `../pipeline/` installed (numpy, scipy, astropy, pandas, pyarrow, astroquery, matplotlib; Python 3.14 was used):

```
python pipeline/s00_acquire_more.py       # optional: SPHEREx fields for Gaia's fastest stars (network, ~5 GB)
python pipeline/s01_index.py              # index raw frames, parse targets
python pipeline/s02_catalogs.py           # Gaia DR3 + AllWISE per field (network)
python pipeline/s03_ephemeris.py          # Horizons spacecraft vectors and target ephemerides (network; its SkyBoT step is unused while USE_SKYBOT is False)
python pipeline/s04_stacks.py spherex
python pipeline/s04_stacks.py wise
python pipeline/s03b_target_astrometry.py # fit WISE 0855 / Luhman 16 motion from the WISE stacks, then rerun s01 and s04
python pipeline/s05_comet_neossat.py comet
python pipeline/s05_comet_neossat.py neossat   # network for per-stack catalogues
python pipeline/s06_psf_inject.py spherex
python pipeline/s06_psf_inject.py wise
python pipeline/s07_baseline.py spherex
python pipeline/s07_baseline.py wise
python pipeline/s08_package.py            # index, splits, checks, figures, manifest
```

All thresholds live in `pipeline/config.py`. Network results are cached in `../cache/`, so reruns are offline after the first pass. Injection seeds come from the realisation id, so the benchmark is reproducible. Archive contents can change; the cached query results are what this version used.

## Credit

See [CITATIONS.md](CITATIONS.md) for every source, its terms, the required acknowledgement texts and the AI-use statement. Keep that file with any derived dataset or model.
