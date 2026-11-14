# Kaggle GPU run

The cutout CNN improved binary candidate F1 from 64.66 to 70.55 points at 4,453 alerts on the existing injection benchmark. It produced 3,317 true change alerts versus 3,040 for the morphology random forest. Precision rose from 68.27% to 74.49% and recall from 61.41% to 67.01%.

| Correctly classified objects | Random forest | CNN |
| --- | ---: | ---: |
| Moving | 117/127 | 118/127 |
| Variable | 46/60 | 48/60 |
| Transient | 0/76 | 6/76 |

The CNN issued 82 transient alerts at this budget. Six matched a labelled transient, for 7.32% transient precision. Object counts credit each recovered source once. Alert counts refer to individual candidates.

The full run used CUDA and eight epochs. Epoch six had the highest validation average precision. At the validation-selected threshold, the CNN issued 3,927 test alerts with 79.86% precision, 63.35% recall and 70.65 points of F1. The evaluation includes 117,825 candidates across the eight recurring test fields. Results describe injections on real images. The uploaded metrics contain no target-specific evaluation on the original observations.

[Original Kaggle results](results.json)
