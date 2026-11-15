# Sources, licenses and AI use

The NASA Space Apps Participant Terms require every source to be cited, and the license of any non-NASA material to be named. This file lists everything that went into the dataset. Copy it into the project submission.

## Data in the dataset

| Source | Provider | What it supplies here | Listed on the challenge page | Terms |
|---|---|---|---|---|
| SPHEREx Level-2 spectral images, DR1 (doi:10.26131/IRSA678) | NASA / JPL-Caltech, via IRSA | SPHEREx frames of 33 fields and of comet 3I/ATLAS | Yes (IRSA SPHEREx page) | IRSA data-use terms: public data, acknowledgement requested |
| NEOWISE-R single-exposure images | NASA / JPL-Caltech, via IRSA | WISE W1/W2 frames, uncertainty and mask images | WISE is listed through WiseView; the listed IRSA page links the IRSA cloud bucket that holds these files | IRSA data-use terms |
| AllWISE Source Catalog | NASA / IPAC, via IRSA | infrared sources with no Gaia counterpart; W1/W2 magnitudes; start position for the motion fit of WISE 0855 and Luhman 16 | Not listed by name; NASA data. For the 21 added fields it was read from ESA's copy (`gaiadr1.allwise_original_valid`) during an IRSA outage | IRSA data-use terms |
| Gaia DR3 | ESA / Gaia / DPAC | star positions, proper motions, parallaxes, class probabilities, variability classes | Yes (Gaia Mission Data). Queried from ESA's Gaia archive, and for the selection of the added fields and five of their star lists from IRSA's copy (`gaia_dr3_source`), rather than the listed STScI page; same catalogue | Open, free use with credit to ESA/Gaia/DPAC |
| NEOSSat astronomy data | Canadian Space Agency, via CADC | NEOSSat frames | Yes (CADC and the NEOSSat dataset record) | Open Government Licence - Canada |
| JPL Horizons | NASA / JPL | spacecraft positions (SPHEREx, WISE, NEOSSat); ephemerides of 3I/ATLAS, C/2020 F3 and Phaethon as seen from the spacecraft | Not listed; NASA service | Public NASA service |
| SIMBAD (Sesame name resolver) | CDS, Strasbourg | the identity and catalogue position of each named target in the original 12 SPHEREx and 10 WISE fields, used to choose field centres when the frames were downloaded and to pick the target among the Gaia sources. No astrometry in the labels comes from SIMBAD any more: WISE 0855 and Luhman 16 are fitted from the WISE frames, and SIMBAD's values serve only as a printed cross-check. The 21 added fields do not use SIMBAD at all | Not listed; not NASA or a partner | Free public service. Its exact reuse licence was not confirmed: check CDS's terms before submitting |

Required acknowledgements:

- SPHEREx: "This publication makes use of data products from the Spectro-Photometer for the History of the Universe, Epoch of Reionization and Ices Explorer (SPHEREx), which is a joint project of the Jet Propulsion Laboratory and the California Institute of Technology, and is funded by the National Aeronautics and Space Administration."
- WISE/NEOWISE: credit NASA, JPL-Caltech, UCLA and IRSA (see https://irsa.ipac.caltech.edu/ack.html).
- Gaia: "This work has made use of data from the European Space Agency (ESA) mission Gaia (https://www.cosmos.esa.int/gaia), processed by the Gaia Data Processing and Analysis Consortium (DPAC)."
- NEOSSat: "Contains information licensed under the Open Government Licence - Canada." Credit the Canadian Space Agency, DRDC and CADC.
- SIMBAD: "This research has made use of the SIMBAD database, operated at CDS, Strasbourg, France."

## Tried and not used

- **IMCCE SkyBoT** (asteroid positions). Version 1.0 of this dataset carried asteroid tracks from SkyBoT. They were removed in 1.1 because SkyBoT is neither NASA nor a listed partner source and its reuse terms were not confirmed. No SkyBoT-derived row remains. The earlier analysis did show that none of those asteroids (V 20.5-24) was cleanly detectable in a single SPHEREx or WISE frame, which is why losing them costs little.
- **JPL Small-Body Identification API** (NASA), tested as the replacement: about 65 s per query and no distances, so the 880 searches needed were not practical.
- **CatWISE2020** (NASA/IRSA), tested for the motion of WISE 0855 and Luhman 16: it splits these fast movers into several entries. Their motion is measured from the WISE frames instead.
- **NASA/IPAC NED**, tested for target positions: it does not resolve the nearby-star names.

Not carried over from the raw download: Gaia XP/RVS spectra and epoch photometry, unWISE time-domain catalogues, SPHEREx spectral cubes, calibration files, the SPLICES catalogue and the 44 IRSA WISE cutouts (reasons in `excluded.json`).

## Software

All open source under OSI-approved licenses: Python (PSF-2.0), NumPy, SciPy, pandas, Astropy and astroquery (BSD-3-Clause), pyarrow (Apache-2.0), Matplotlib (Matplotlib license, BSD-compatible).

The pipeline code in `provenance/` and the tools in `tools/` were written for this project. Under the Participant Terms, original content is open source; if no other OSI license is named, NASA distributes it under Apache-2.0.

## AI use

The Participant Terms require a description of how AI tools were used. For this dataset:

- The raw files were selected and downloaded by an AI coding assistant (OpenAI Codex), which also wrote a source-provenance audit of that download.
- The processing pipeline, the labels, the synthetic-object injection, the baseline detector, the checks and this documentation were written and run by an AI coding assistant (Anthropic Claude, in Claude Code), directed by the team.
- No AI model generated or altered image pixels. Synthetic objects are added by the documented injection code, and every one is recorded in `injected_truth.parquet`.

## Dates

The raw download and versions 1.0 and 1.1 of this dataset were produced on 13 November 2026, the day challenge resources were published and the day before the hackathon. The 21 added fields were downloaded overnight on 13-14 November and version 1.2 was built on 14 November. File dates were left as they are. The FAQ says teams may not begin working on the challenges before the hackathon; whether this preparation is allowed is for the team to settle with the organizers.
