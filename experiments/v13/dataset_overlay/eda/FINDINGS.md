# EDA findings and decisions

## Ready to use

- 3,420 original real frames, with 2,840 passing the original quality rules.
- 126 original real stacks. 84 contain usable frames; 58 have at least two distinct usable observation times.
- 113,214 catalogue source rows and 5,574,833 object-frame track rows. These are not counts of unique physical objects or images.
- 238 fixed synthetic realizations, containing 2,380 simulated objects. Preserve the supplied validation and test realizations.
- All original manifest-listed files were verified before preparation. Original scientific payload files and splits are retained byte-for-byte.

## Preparation added

The training adapter sorts frames by time and aligns every returned target with sequence_index. It supplies elapsed days, original time groups, detector/band information and masks. It excludes failed frames and empty sequences from the normal training view without deleting their data. It keeps the original classes and temporal targets.

Visibility uses the original quality rules, not a new threshold chosen from the test data. Original mover_detectable can count simultaneous detector frames. The added motion_observable_two_times field distinguishes actual visibility at two different observation times, without rewriting mover_detectable. Single-time comet stacks remain accessible.

## Issues retained and documented

| Finding | What it means | Decision |
|---|---|---|
| Some tracks lack a source-table row | All such rows are outside the usable image domain | Preserve rows and null source labels; do not invent a class |
| Real labels are catalogue-limited | Unlabelled sources can exist | Do not interpret all unlabelled pixels as verified background |
| Many sources are too faint or contaminated | A predicted location is not a confirmed visible training example | Add explicit visibility and uncertainty status |
| A few real movers dominate the motion examples | Large track counts do not imply diverse motion training | Keep simulations available and report real and synthetic results separately |
| No passing NEOSSat training frames | Optical training coverage is missing in the existing split | Preserve splits; document this as a transfer/evaluation limitation |
| Limited asteroid coverage | This is not a representative standalone asteroid training set | Do not relabel brown-dwarf simulations as realistic nearby asteroids |
| Five added SPHEREx fields lack variability classifications | Missing variability labels do not prove constant brightness | Preserve missingness |
| WISE 0855 is hard in SPHEREx | Small movement and faintness defeated the supplied reference detector | Keep as held-out test data; no image or target manipulation |
| Real-source measurements predate synthetic injection | An injection may overlap a real source | Explicitly mark real-source quality as pre-injection in synthetic samples |
| Some frames have saturated pixels | Bright source positions may still be informative | Preserve saturation information and masks; report separately |

## Handoff boundaries

No model training, model-specific normalization, rebalancing, resizing, invented bounding boxes, synthetic population changes or split changes were performed. The training team chooses architecture, batching and losses using the task contract. Any additional astrophysical labels or new raw observations require their own review. EDA establishes consistency and coverage, not perfect ground truth or guaranteed model accuracy.

The inspection covers the local processed package. External original raw downloads are not included in this release.
