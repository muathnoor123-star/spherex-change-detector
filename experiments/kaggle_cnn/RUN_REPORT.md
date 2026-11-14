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

## Frozen checkpoint audit

We loaded the supplied Kaggle weights on CPU. The checkpoint class order and threshold matched the archived run. At the frozen cutoff, the CNN matched the expected class within two pixels in 2/42 eligible Epsilon Indi B frames (previous forest 10/42), 0/29 WISE 0855 frames and 0/2 RW Aurigae frames. Barnard's Star and R Draconis had no eligible unsaturated frames. Each classifier used its own validation-selected cutoff. The comparison measures performance at those operating points.

Among 76 eligible injected transient objects, 71 generated a matched candidate, 59 had a candidate among the 4,453 reviewed alerts, and six received a correct transient alert. Thus five lacked a proposal, 12 had proposals below the review budget and 53 reached review only under another class. Of the 82 alerts classified as transient, 58 matched injected motion, 18 were labelled background and six matched transients. These counts distinguish unique injected objects from individual alerts.

Run from the repository root with PyTorch and the project dependencies installed:

~~~bash
python evaluate_cutout_cnn.py data/planetx-dataset-v1.2.0
~~~

[Checkpoint](model.pt) | [Target measurements](named_targets.csv) | [Transient object outcomes](transient_objects.csv)
