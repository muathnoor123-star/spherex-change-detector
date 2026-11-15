# Reproduce the data audit

Use Python with numpy, pandas, pyarrow and Pillow installed. Keep an intact v1.2.0 source package. Run the following from the v1.3.0 package directory, with paths adjusted to your machine:

```powershell
python provenance/eda_v13_build.py --source "../planetx-dataset-v1.2.0/planetx-dataset-v1.2.0" --output "../planetx-eda-reproduction"
```

Use a separate output directory. This reconstructs the EDA tables, training metadata and class examples from the upstream scientific payload. It does not regenerate the edited guide or publish a release archive. The original source is read-only during this operation. Numeric distributions use all finite values; missingness is reported separately and never filled. Image display examples use percentile contrast for viewing only, and the original arrays remain unchanged.

To check the integrity of this prepared release after copying or downloading it:

```powershell
python tools/verify_release.py
```

The delivered CSV check records describe the full build-time validation, including all usable sequences, fixed injected samples, seeded augmentation and preservation of upstream payload files. File hashes verify exact bytes, not scientific truth.
