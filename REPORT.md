# Training and evaluating the SPHEREx change detector

We trained a random forest on a local CPU to locate changes in aligned SPHEREx images, classify them and rank candidates for review. Planet X v1.2.0 supplied 198 image sequences with synthetic objects injected into observations of 33 sky fields. We used 17 fields for training, eight for validation and eight for testing. We then scanned the original observations from all 33 fields.

## Training and review priority

For each exposure, the detector selects reference images of the same field taken with the same detector more than a day apart, choosing the closest available wavelengths. It searches the difference image for positive and negative peaks. Only after locating the peaks does it use the injection labels to train the classifier. The training sample contained 122,283 candidates, including 8,741 labelled as motion, 4,620 as brightness variation and 226 as transients.

The review score measures how strongly the classifier favors a change. The dataset contains no expert ratings of scientific importance. Estimating the probability of a discovery would require additional evidence.

## Held-out test results

An injected object was eligible if it appeared in at least one frame that met the coverage and saturation criteria. We counted an object as recovered when the detector identified the correct change type within two pixels in at least one eligible frame. Frame recovery counts every eligible opportunity, including frames where the detector proposed no peak. Alert precision is the fraction of alerts for a class that matched that class.

| Change type | Objects recovered at least once | Eligible frames with a correct detection | Alert precision |
|---|---:|---:|---:|
| Motion of at least one pixel | 115 of 127 (90.6%) | 1,668 of 3,830 (43.6%) | 46.9% |
| Brightness variation | 43 of 60 (71.7%) | 631 of 2,625 (24.0%) | 71.2% |
| Transient | 0 of 76 | 0 of 108 | No correct alerts |

Average precision for ranking change candidates was 0.501. The test set contained 117,825 proposed peaks, of which 4,950 (4.2%) matched injected changes. We selected a review threshold of 0.592 on the validation fields. The 90.6% motion recovery rate counts each injected object once across all its eligible frames.

## Five astronomical targets

We searched for an alert of the expected type within two pixels of the target's catalogue track. Eligible frames passed quality control, had at least 95% valid aperture coverage and a signal-to-noise ratio of at least five, and showed neither saturation nor contamination at the target. For RW Aurigae, we examined both components and counted each frame once.

| Target | Field split | Eligible frames | Frames with a matching alert |
|---|---|---:|---|
| Barnard's Star | Training | 0 | All 58 frames were saturated |
| Epsilon Indi B | Validation | 42 | 13 |
| WISE 0855 | Test | 29 | 0 |
| RW Aurigae | Validation | 2 | 0 |
| R Draconis | Test | 0 | All 56 frames were saturated |

## Limitations

- The scan produced 226 alerts in the original images. Each alert marks one position in one frame, so a source can appear several times. These alerts still need astronomical review.
- The classifier failed to identify any held-out transient correctly. Distinguishing transients from motion and brightness variation requires more evidence across exposures and a fresh independent test.
- Wavelength differences and residual light from bright stars can mimic brightness changes. The detector analyzes local image patches and does not link detections into orbits or long tracks. Training treats unmatched peaks as background, which can include real, unlabelled changes.
- Training and evaluation covered SPHEREx. Transfer to WISE and NEOSSat remains untested. The supplied baseline uses a different recovery criterion, so these results do not establish a performance advantage.
- The implementation covers the model and notebook. We ran training locally. Kaggle execution remains unverified. OpenAI Codex wrote the code, ran training and prepared this report under the project owner's direction.

## Outputs

The [results archive](results.zip) contains the fitted model and threshold, test metrics, object and frame recovery tables, the five-target evaluation and the alert list. Each alert includes its position, exposure time, detector and class scores. The [README](README.md) explains how to run the notebook, load the saved model and reconstruct the original dataset ZIP from the twenty release files.
