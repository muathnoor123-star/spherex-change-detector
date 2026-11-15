# v1.3.0 additions

See README.md for the training/ and eda/ schemas. Original scientific field meanings below are unchanged. Historical narrative counts may be stale; eda/table_inventory.csv is authoritative for actual rows.

# Data dictionary

Every file, array and column in this dataset, with units and the conventions that matter when you train on it. Start with [README.md](README.md) for what the dataset is and how it was made.

## Conventions used everywhere

- **Pixel coordinates.** `x` is the column, `y` the row, both 0-based, with (0, 0) the centre of the first pixel. Arrays are indexed `image[frame, y, x]`. North is +y, east is -x (RA increases to the left).
- **Grid WCS.** Each stack's grid is in `<mission>/stacks.json` under `grid`: TAN projection, `crval` = centre (deg), `crpix_1based` = FITS reference pixel (1-based, so 0-based centre is `crpix - 1`), `cdelt_deg` = pixel size with a negative x step. With astropy: `w = WCS(naxis=2); w.wcs.ctype = g["ctype"]; w.wcs.crval = g["crval"]; w.wcs.crpix = g["crpix_1based"]; w.wcs.cdelt = g["cdelt_deg"]`, then `w.wcs_world2pix(ra, dec, 0)`.
- **Time.** `mjd_mid` is the exposure midpoint, MJD in UTC, from the FITS header. Never a file or release date.
- **Identifiers.**
  - `stack_id` = `<mission>:<field>`, for example `spherex:wise0855`.
  - `frame_uid` = `<stack_id>:<source FITS file name>`.
  - `frame_index` = position of the frame inside its stack arrays and in `frames.parquet` (sorted by time, then detector or band). In every table, `frame_index` refers to the full stack, including frames that failed QC.
  - `src_id` = `gaia:<source_id>`, `allwise:<designation>`, `target:<group>` (a named target taken from SIMBAD because no Gaia entry matched) or `sso:<number or designation>`.
- **Groups and folds.** `group` is the physical sky region or target. Everything sharing a group shares a fold (`splits.json`). Split by group, never by frame or by row.
- **QC.** `qc_pass` is per frame. The loader returns only QC-passed frames unless asked otherwise; raw arrays still contain failed frames.

## Folder layout

```
DATA_GUIDE.pdf             plain-language guide, one chapter per data source
README.md                  what it is, how it was built, limits
CITATIONS.md               sources, licenses, acknowledgements, AI-use statement
DATA_DICTIONARY.md         this file
index.parquet              one row per sample (start here in code)
splits.json                folds, default split, group -> fold
excluded.json              raw inputs not used, with reasons
MANIFEST.sha256            SHA-256 of every file
<mission>/                 spherex | wise | comet | neossat
    stacks/<field>.npz       aligned image stack
    stacks/<field>-coadd.npz WISE only: half-year coadds
    psf/<field>.npz          SPHEREx and WISE only: per-frame PSF
    injected/<field>-r<k>.npz SPHEREx and WISE only: fixed synthetic benchmark
    stacks.json              grid WCS and counts per stack
    frames.parquet           one row per frame
    sources.parquet          one row per (stack, source)
    tracks.parquet           one row per (frame, source)
    injected_objects.parquet SPHEREx and WISE only
    injected_truth.parquet   SPHEREx and WISE only
    fields.json              comet and NEOSSat: stack centres and time spans
    excluded-frames.csv      NEOSSat: frames dropped for lacking sky coordinates
baseline/                  reference detector results
qc/                        figures, summary.json, leakage-checks.json
tools/                     loader.py, injection.py, baseline.py
provenance/                copy of the pipeline that built this
```

## `index.parquet`

One row per sample. 227 rows.

| Column | Meaning |
|---|---|
| `sample_type` | `real_stack`, `epoch_coadd` (WISE half-year coadds) or `injected` (fixed synthetic realisation) |
| `mission` | `spherex`, `wise`, `comet` (SPHEREx frames of 3I/ATLAS) or `neossat` |
| `stack_id`, `field`, `group` | see Conventions |
| `fold`, `split` | fold 0-4 and the default split (`train`, `val`, `test`) |
| `path` | file relative to the dataset root |
| `realization_id` | `<stack_id>:r<k>` for injected rows, else null |
| `n_frames`, `n_frames_qc` | frames in the stack, and those passing QC |
| `n_time_groups_qc` | distinct exposure times among QC frames |
| `baseline_days` | span of QC frame times |
| `n_sources` | rows in `sources.parquet` for this stack |
| `n_real_movers` | sources with `mover_detectable` |
| `n_sso_visible` | solar-system objects (3I/ATLAS, NEOSSat targets) detected in at least one frame |
| `is_control` | WISE offset background field (0.4 deg east of its target) |
| `n_injected`, `n_injected_movers` | injected rows only: objects, and how many move >= 1 px |

## Stack arrays: `<mission>/stacks/<field>.npz`

Shape `[N, H, W]`. N is every frame of the stack in `frame_index` order. H = W = 128 (256 for NEOSSat).

| Array | dtype | Meaning |
|---|---|---|
| `image` | float32 | Background-subtracted surface brightness. 0 where `valid` is 0. Units: SPHEREx and comet MJy/sr; WISE flux scaled so Vega mag = 22.5 - 2.5 log10(sum); NEOSSat bias-subtracted ADU. |
| `sigma` | float32 | 1-sigma per pixel after resampling. Neighbouring pixels are correlated, so per-pixel sigma understates noise in sums; use `frames.aperture_sky_noise` for thresholds. |
| `valid` | uint8 | 1 = usable value. A grid pixel is valid only if all four native pixels feeding it are valid and inside the exposure. |
| `flags` | uint32 | Native bitmask OR-combined over the four contributing native pixels (bit tables below). |
| `wave` | float32 | Central wavelength per pixel, um. SPHEREx from the WAVE-TAB lookup; WISE 3.4 (W1) or 4.6 (W2); NEOSSat 0.6. NaN where invalid. |
| `bandwidth` | float32 | SPHEREx bandwidth per pixel, um; NaN elsewhere. |
| `frame_uid` | str | per frame, matches `frames.parquet` |
| `mjd` | float64 | per frame, UTC |

Converting SPHEREx brightness to flux: one grid pixel is (6.15")^2 = 8.890e-10 sr, so `image` x 889.0 gives uJy per pixel.

### WISE coadds: `wise/stacks/<field>-coadd.npz`

`image`, `sigma`, `valid` with shape `[K, 128, 128]`, plus `key` (`"2014-H1:W2"`: half-year and band), `mjd` (mean time) and `n_frames` (frames averaged, usually 2). These are inverse-variance means of QC-passed frames with no outlier rejection, so a transient seen in one scan survives at reduced amplitude.

### Flag bits

SPHEREx (from the FITS headers):

| Bits | Names | Treatment |
|---|---|---|
| 1, 2, 6, 9, 10, 11, 15 | OVERFLOW, SUR_ERROR, NONFUNC, MISSING_DATA, HOT, COLD, NONLINEAR | pixel invalid |
| 0, 7, 14, 17, 20, 22-28 | TRANSIENT, DICHROIC, PHANMISS, PERSIST, CROSSTALK, GHOST, GHOST_FPA, GHOST_EXT, STREAK, BLOOM, SNOWBALL, HALO | kept; counted as artifacts |
| 19 | OUTLIER | kept; also set on real moving objects (the 3I/ATLAS comet), so it is not a reason to reject a mover |
| 12, 21 | FULLSAMPLE, SOURCE | information only |

Saturation in tracks means bits 1 or 2.

WISE (Explanatory Supplement, Table 7):

| Bits | Treatment |
|---|---|
| 1, 3, 4, 6, 9-19, 30 | pixel invalid (the set the WISE pipeline itself blanks) |
| 21, 26, 28 | kept; artifacts (transient bad pixel, unreliable nonlinearity, replaced spike) |

Saturation in tracks means bits 10-19.

NEOSSat (built here, since the files have no mask): bit 0 = at or above 60,000 ADU (saturated), bit 1 = outside DATASEC (overscan). Both are invalid.

## `frames.parquet`

One row per frame, with the same columns in every mission except where noted.

| Column | Meaning |
|---|---|
| `stack_id`, `frame_index`, `frame_uid`, `field` | identifiers |
| `path`, `sha256` | source FITS file relative to the raw download, and its hash. WISE adds `unc_path`, `msk_path`, `unc_sha256`, `msk_sha256`. |
| `mjd_mid`, `time_scale` | exposure midpoint and its scale (always UTC) |
| `exptime_s` | exposure time, s |
| `time_group` | frames sharing one exposure time: SPHEREx detector pairs D1/D4, D2/D5 and D3/D6, or WISE W1/W2. A real source appears in every frame of its group. |
| `detector` | SPHEREx/comet: 1-6 (1-3 short-wave, 4-6 long-wave) |
| `band` | WISE: 1 or 2; NEOSSat: 0 |
| `bin`, `frame_key`, `scan`, `magzp`, `moonsep_deg` | WISE: half-year bin, frame id, scan id, photometric zero point, Moon separation (deg) |
| `obsid`, `mjd_beg`, `psf_fwhm_arcsec` | SPHEREx: observation id, start time, header PSF FWHM |
| `group_dir`, `object`, `proc`, `mode`, `ra_vel`, `dec_vel`, `ptg_ra`, `ptg_dec` | NEOSSat: target folder, header OBJECT, processing family (`clean`, `cor`, `cord`), pointing mode, slew rates, frame centre |
| `wave_center_um`, `wave_min_um`, `wave_max_um` | wavelength at the grid centre (central 32 x 32 median) and its range over the grid |
| `reg_status`, `reg_nref` | registration result and number of reference stars used |
| `reg_dx`, `reg_dy` | shift applied to the header WCS, native pixels |
| `reg_rms` | robust scatter of reference-star offsets after the shift, native px |
| `reg_shift_err_px` | uncertainty of the shift, 1.2533 x `reg_rms` / sqrt(`reg_nref`) |
| `valid_frac`, `center_valid_frac` | valid fraction of the grid and of the central 32 x 32 |
| `covered_frac`, `center_covered_frac` | the same for "inside the exposure footprint", counting saturated pixels as covered |
| `grid_bkg_removed` | residual sky median subtracted on the grid |
| `native_bkg` | median of the mesh background removed before resampling (WISE, NEOSSat) |
| `zodi_median` | median of the zodiacal model removed (SPHEREx) |
| `pixel_noise` | robust scatter of sky pixels on the grid |
| `aperture_sky_noise`, `n_sky_apertures` | robust scatter of aperture sums on empty sky, and how many apertures measured it. This is the right noise for S/N. |
| `noise_ratio` | `aperture_sky_noise` / median of the same detector or band in the stack |
| `artifact_frac` | fraction of valid pixels carrying artifact bits |
| `obs_x`, `obs_y`, `obs_z` | Earth's barycentric position, AU, ICRS; used for parallax |
| `psf_fwhm_px`, `psf_n_stars`, `psf_source` | SPHEREx/WISE: PSF width (half-maximum area), stars used, and PSF origin (`frame`, `stack_band` or `mission_band`) |
| `sat_level` | SPHEREx/WISE: estimated saturation level of that detector or band, in image units; used to saturate bright injected objects |
| `shot_k`, `shot_k_measured` | SPHEREx/WISE: photon-noise slope (variance added per unit signal), and whether it was measured in this frame or borrowed from the band median |
| `qc_pass`, `qc_reasons` | verdict, and `;`-separated reasons when it fails (below) |

QC reasons: `low_coverage` (valid fraction < 0.5); `target_region_not_covered` (central footprint < 0.8); `registration_insufficient_stars` (< 5); `registration_shift_too_large` (> 3 px); `registration_uncertain` (shift error > 0.1 px or star scatter > 1 px); `noisy` (noise ratio > 3); `read_error`.

## `sources.parquet`

One row per source per stack. Only sources that fall inside the grid in at least one QC frame are kept.

Identity and astrometry:

| Column | Meaning |
|---|---|
| `stack_id`, `src_id`, `group`, `field_key` | identifiers |
| `kind` | `gaia`, `allwise_only`, `target` (SIMBAD) or `sso` |
| `ra`, `dec`, `ref_jyear` | position (deg, ICRS) at reference epoch: 2016.0 for Gaia, 2000.0 for SIMBAD, 2010.56 for AllWISE |
| `pmra`, `pmdec`, `pm_total_masyr` | proper motion, mas/yr (`pmra` includes cos dec) |
| `plx`, `parallax_snr` | parallax (mas) and its S/N |
| `ruwe`, `astrometric_params`, `duplicated_source` | Gaia astrometric quality |
| `vmag`, `posunc_arcsec` | SSO only: predicted V magnitude and ephemeris uncertainty |

Photometry and catalogue classes:

| Column | Meaning |
|---|---|
| `gmag`, `bp_rp` | Gaia G and BP-RP |
| `w1`, `w2`, `w1_snr`, `w2_snr`, `ext_flg` | AllWISE Vega magnitudes, their S/N and extended-source flag (matched within 3" for Gaia sources) |
| `p_star`, `p_galaxy`, `p_quasar` | Gaia DSC class probabilities |
| `phot_variable_flag`, `vari_class_score` | Gaia variability flag and classifier score |

Labels:

| Column | Meaning |
|---|---|
| `class` | `star`, `galaxy`, `quasar`, `variable_star`, `brown_dwarf`, `unknown_point` (Gaia, no confident class), `ir_only_point`, `extended_ir`, `asteroid`, `comet` |
| `subclass` | Gaia variability class, SIMBAD object type for targets, object type for solar-system targets |
| `label_source` | `gaia_dsc_or_parallax`, `gaia_vari_classifier`, `simbad_target_identity` (SIMBAD named the target; astrometry from Gaia), `simbad_identity_wise_fit_astrometry` (WISE 0855, Luhman 16: motion fitted from the WISE frames), `gaia_dr3_proper_motion_selection` (added fields), `allwise_no_gaia_counterpart`, `jpl_horizons_ephemeris` (comet and NEOSSat targets), `none` |
| `is_target` | the named target of the field (binary components are both targets) |

Measured in QC frames only:

| Column | Meaning |
|---|---|
| `n_frames_in_grid` | frames where the predicted position is inside the grid |
| `n_frames_detected` | frames with S/N >= 5 and aperture >= 95% valid, not `contaminated`, and with a centroid (where measured) within 1.5 px of the prediction. Solar-system objects use difference S/N where available and must also be unblended and free of artifact flags other than OUTLIER. |
| `n_frames_saturated` | frames where the source saturates by its own light (`saturated_self`) |
| `n_frames_contaminated` | frames where a much brighter neighbour dominates the aperture |
| `n_frames_visible` | detected or saturated; for SSOs, detected only |
| `median_snr`, `max_snr` | aperture S/N over frames in the grid |
| `max_snr_diff` | SSOs: best difference S/N |
| `x_min`, `x_max`, `y_min`, `y_max` | extent of the predicted track |
| `motion_px` | diagonal of that extent: total apparent motion within the stack |
| `frac_frames_artifact` | fraction of frames with artifact bits in the aperture |
| `sed_scatter` | SPHEREx: robust scatter of `sed_resid_snr` over the source's frames (8 or more needed). About 1.2 for a constant star; larger means real variability |
| `is_mover` | `motion_px` >= 1 |
| `mover_detectable` | `is_mover` and visible in at least 2 frames |

## `tracks.parquet`

One row per (frame, source): every catalogued source, and in the comet and NEOSSat stacks the solar-system target, predicted inside the grid or within 3 px of its edge. 3.0 M rows for SPHEREx. This is the per-frame label table.

| Column | Meaning |
|---|---|
| `stack_id`, `frame_uid`, `frame_index`, `src_id` | identifiers |
| `src_kind` | `catalog` or `sso` |
| `x`, `y` | predicted grid position, proper motion and parallax applied for that frame's time (SSOs: from the ephemeris, seen from the spacecraft) |
| `flux`, `flux_err`, `snr` | forced aperture photometry at (x, y). Radius 2.0 px for SPHEREx and comet, 2.5 px for WISE, 3.0 px for NEOSSat; local background from an annulus. Not aperture-corrected. `flux_err` combines empty-sky aperture noise with the source's own photon noise. |
| `aperture_valid_frac` | valid fraction of the aperture |
| `local_bkg` | annulus median |
| `aperture_flags` | OR of flag bits within 2 px |
| `artifact_in_aperture`, `saturated` | decoded from `aperture_flags` |
| `artifact_vetoes_mover` | artifact bits other than SPHEREx OUTLIER |
| `centroid_dx`, `centroid_dy` | measured centroid minus prediction, px, where S/N > 5 |
| `saturated_self` | the saturation in the aperture belongs to this source: it is the target or a bright star (G < 13 or W1 < 9.5) |
| `contaminated` | catalogue sources: another source within 4 px is at least 5 times brighter in this frame, or saturated |
| `diff_flux`, `diff_err`, `snr_diff` | SSOs only: flux minus the median at the same position in the same detector's frames from more than a day away. NaN when fewer than 2 such frames exist (comet stacks). |
| `wave_um` | SPHEREx: wavelength at the source position in this frame |
| `flux_sed` | SPHEREx: local spectrum estimate, the median flux of the 5 measurements nearest in log-wavelength taken at a different exposure time (needs 12 usable measurements) |
| `sed_resid_snr` | SPHEREx: (`flux` - `flux_sed`) / sqrt(`flux_err`^2 + (0.03 `flux_sed`)^2) |
| `blended` | SSOs only: a catalogued source of median S/N >= 5 lies within 2.5 px. In stacks with fewer than 3 exposure times: Gaia G < 17 or W1 < 14.5. |

## PSFs: `<mission>/psf/<field>.npz`

| Array | Meaning |
|---|---|
| `psf` [N, S, S] | unit-sum PSF per frame on the grid (S = 15 SPHEREx, 21 WISE), centred on the middle pixel |
| `n_stars` | stars measured in that frame |
| `fwhm_px` | width from the half-maximum area |
| `source` | `frame` (>= 3 own stars), `stack_band` (same detector or band in this stack), `mission_band` (same detector or band in the whole mission) |

The comet and NEOSSat folders have no PSFs and no injections.

## Synthetic benchmark

`<mission>/injected/<field>-r<k>.npz` holds `image`, `sigma` and `valid` of shape `[Q, H, W]` for the Q QC-passed frames only, plus `frame_index` [Q] giving each row's index in the full stack. `valid` differs from the real stack where an injected object saturated. Use `flags` and `wave` from the real stack at those indices. `Dataset.load_injected()` does this for you.

`injected_objects.parquet`, one row per object:

| Column | Meaning |
|---|---|
| `realization_id`, `stack_id`, `obj_id`, `seed` | identifiers; `seed` regenerates the realisation |
| `class` | `pm_star`, `fast_mover`, `distant_planet`, `transient`, `variable`, `static` |
| `pm_ra`, `pm_dec` | arcsec/yr, east and north |
| `plx` | parallax, arcsec |
| `dist_au` | distant_planet only |
| `amp_mag`, `period_d`, `phase` | variable only |
| `transient_group` | transient only: the `time_group` in which it exists |
| `sed_alpha`, `lam_ref_um` | flux proportional to (lambda / lam_ref)^sed_alpha |
| `snr_target`, `flux_ref` | intended aperture S/N (3 to 3000) and the total flux it implies, in image units x pixels |
| `ref_frame`, `x_ref`, `y_ref` | reference frame (index into the QC-passed frame list of that realisation) and the position there |
| `motion_px`, `is_mover` | apparent motion across the frames where it is present, and whether that is >= 1 px |

`injected_truth.parquet`, one row per object per QC frame:

| Column | Meaning |
|---|---|
| `realization_id`, `stack_id`, `obj_id` | identifiers |
| `frame_index` | index in the full stack |
| `x`, `y` | true position, which may lie outside the grid |
| `present` | false for a transient outside its exposure time |
| `in_grid` | position within the grid |
| `flux_total`, `ap_flux_true` | injected total flux and the part inside the photometry aperture |
| `snr_true` | `ap_flux_true` / `aperture_sky_noise` |
| `saturated` | the object pushed at least one pixel above `sat_level` in this frame; those pixels are blanked and invalid |
| `masked_frac` | fraction of the object's light on invalid pixels (that light was not added). NaN when not rendered. |

Score only rows with `present & in_grid & masked_frac < 0.5 & ~saturated`, and report saturated objects separately. That is the rule the baseline uses.

## `splits.json`

`folds` maps fold to groups, `group_to_fold` the reverse, and `default_split` gives `{"test": [0], "val": [1], "train": [2, 3, 4]}`.

## `baseline/`

Results of `tools/baseline.py` (median-reference difference imaging, 5 sigma):

| File | Contents |
|---|---|
| `<mission>-summary.json` | headline numbers |
| `-completeness-by-snr.csv`, `-completeness-by-class.csv` | completeness on injected objects that should be flagged |
| `-static-flag-rate.csv` | how often static or sub-pixel objects get flagged |
| `-real-stacks.csv` | per real stack: detections, known movers recovered, unexplained detections |
| `-unexplained-detections.parquet` | detections matching no known source: review candidates, not labels |

## `qc/`

| File | Contents |
|---|---|
| `summary.json` | counts per mission |
| `leakage-checks.json` | footprint overlap and shared source files across groups (none) |
| `spherex-fields.png` | every SPHEREx field with the target position in each frame |
| `wise-barnard-epochs.png` | Barnard's Star moving across WISE coadds |
| `registration-and-qc.png` | registration precision and rejection reasons |
| `injection-example.png` | one injected frame with its truth |
