# Train the SPHEREx image classifier on Kaggle

Import [SPHEREx_Cutout_CNN.ipynb](SPHEREx_Cutout_CNN.ipynb) into a notebook and attach the extracted PlanetX v1.2.0 dataset or the original planetx-dataset-v1.2.0.zip as a private input. Select a GPU accelerator, run all cells and save the executed version with its outputs. The notebook embeds the candidate detector and previous benchmark. A repository clone and GitHub credentials are unnecessary.

Find these files under the spherex-cnn output directory:

- **results.json** contains the device, selected epoch, validation history, frozen cutoff, test precision, recall, F1, average precision, transient alert precision counts, eligible objects and recovery by class. Send this file for review.
- **run.zip** contains results.json and model.pt for anyone who needs the trained weights.
- **model.pt** contains the selected state dictionary, class order and threshold. The training cell defines the architecture.

The CNN learns from current, reference and residual image patches plus a validity mask. Performance is evaluated on every candidate from the eight existing test fields. The comparison with the published random forest uses exactly 4,453 test alerts. The operating-point metrics also show what happens when applying the cutoff selected on validation. Test fields have been used in previous project evaluations. Precision and recall are in the range 0 to 1. The F1 change is reported in percentage points.

A local smoke run uses one realization per fold and one epoch. Its metrics check the pipeline and must not be used for model selection. The Kaggle notebook runs the full experiment by default.
