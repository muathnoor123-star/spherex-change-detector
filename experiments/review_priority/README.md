# Review priority and the WISE 0855 profile fit

The frozen CNN scores each candidate with the trained Kaggle weights. To test a concrete definition of review value, this experiment rewards putting more distinct injected transient objects in the queue. It counts objects regardless of the predicted class. This models a reviewer who can inspect the original images and correct a class assignment. Expert ratings of scientific importance are unavailable.

The reference queue ranks all candidates by one minus their background score. The alternative reserves 0, 1, 2.5, 5 or 10 percent of the same slots for the highest transient scores and fills the rest using the reference ranking. Each candidate appears at most once. On validation, we maximize the number of transient objects represented subject to a maximum two percentage point drop in overall precision. Ties favour higher precision and then smaller quotas. We apply the selected quota once to the recurring test fold. The available review budgets are 6,064 validation candidates and 4,453 test candidates. The quota changes queue membership. Class predictions come from the same frozen model.

The previous [checkpoint audit](../kaggle_cnn/RUN_REPORT.md) evaluated all five named targets and traced the 76 eligible injected test transients through the detector. It found 71 transients with proposals, 59 represented in the queue and six correctly classified. Among 82 transient alerts, 58 matched movers, 18 matched background and six matched transients.

## Motion in the original WISE 0855 exposures

For each of the 29 eligible frames we fitted a Gaussian with the frame's measured or supplied PSF width, nonnegative brightness and a local planar background. We compared the catalogue track with its fixed mean position on identical valid pixels in a 9 by 9 window. At least 60 valid pixels were required per exposure. The score is the average reduction in squared noise-normalized residual per pixel, with equal weight per frame.

The moving profile reduced this score by 0.255 on average and fitted better in 13 of 29 frames. Of 146 nearby translated control tracks meeting identical coverage requirements, 10 tied or exceeded the target's gain. This is a targeted comparison with a simplified source profile in one field. Control ranks describe this sample and do not establish an independent detection. The blind CNN benchmark previously matched zero of 29 eligible WISE frames.

Run from the repository root with the restored dataset and checkpoint.

~~~bash
python review_priority.py data/planetx-dataset-v1.2.0
python wise_profile_fit.py data/planetx-dataset-v1.2.0
~~~

## Review queue result

Validation selected the 1% quota by the precision tie-break (4,793 versus 4,792 correct alerts and 68 transient objects in both queues). On test, both queues contained 3,317 true alerts out of 4,453, or 74.49% precision. Each represented 59 transient, 119 moving and 51 variable objects regardless of predicted class. Four of the 45 reserved alerts matched transients. This policy did not improve the measured review outcome. We retain the original ranking.

[Review metrics](metrics.json) | [Target fit](psf_summary.json) | [Comparison positions](psf_positions.csv)
