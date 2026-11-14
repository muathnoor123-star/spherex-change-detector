# WISE 0855 motion-aligned residual stack

Following the catalogue track yielded a mean signed residual of 0.2423 at the target. Sampling the same 29 exposures at the mean catalogue position yielded 0.2960. Alignment reduced the score by 0.0537. The first 15 exposures in time order had a mean gain of -0.1211 and the remaining 14 had a mean gain of +0.0185.

Each exposure uses the existing detector's signed, smoothed difference against the median of up to three reference frames. Each reference comes from the same detector more than a day away and is selected by wavelength proximity. The score averages the residual divided by its frame's robust noise scale. Bilinear sampling at the predicted subpixel position gives the central value of the aligned stack. Frames receive equal weight, including those that did not generate blind proposals. Fixed-position sampling uses the same interpolation. The score is in per-exposure noise units. Shared references and repeated visits make its sampling uncertainty different from that of independent exposures.

We repeated both measurements on a six-pixel grid of translated tracks between eight and 48 pixels from the target. Of 192 control positions, 37 had valid interpolation support in both stacks for every exposure. Seven of those 37 equalled or exceeded the target's aligned score, and 30 equalled or exceeded its alignment gain. The control comparison describes this field and its coverage. The controls were selected by geometry and valid coverage and can include real sources.

This targeted diagnostic found no enhancement from motion alignment on these images. It uses a supplied trajectory and measures a signed mean, which can lose signal if residual signs vary between exposures. It measures neither blind discovery performance nor a new classifier F1.

Run from the repository root with the restored dataset and project dependencies installed.

~~~bash
python aligned_stack.py data/planetx-dataset-v1.2.0
~~~

[Summary](summary.json) | [Target by frame](frames.csv) | [Target and controls](positions.csv)
