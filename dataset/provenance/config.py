"""Paths, field definitions, quality rules and split folds for the baseline dataset.

Every threshold used by the pipeline lives here so that the dataset card can quote
it and a rerun with different rules only needs edits in one place.
"""
from pathlib import Path

SRC = Path(r"C:\Users\fares\Documents\Codex\2026-11-13\see\outputs\excluded-not-for-submission\original-delivery\training-data")
# Asteroid positions from IMCCE SkyBoT are not used: it is neither NASA nor a listed partner
# source. (JPL sb_ident was tested as a replacement: ~65 s per query, no distances.)
USE_SKYBOT = False
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dataset"
CACHE = ROOT / "cache"

WORKERS = 24
DATASET_VERSION = "1.2.0"

# ---------------------------------------------------------------------------
# Fields. A "physical group" is one sky region / physical target. Every product
# of a group (all missions, its offset control field, every derived sample) goes
# to the same fold, so no target can appear on both sides of a split.
# ---------------------------------------------------------------------------
# slug -> (physical group, resolver xml relative to SRC)
SPHEREX_FIELDS = {
    "barnards-star": ("barnard", "spherex/metadata/barnards-star-resolver.xml"),
    "epsilon-indi-b": ("epsilon_indi_b", "spherex/metadata/epsilon-indi-b-resolver.xml"),
    "luhman16": ("luhman16", "spherex/metadata/luhman16-resolver.xml"),
    "m101": ("m101", "spherex/metadata/m101-resolver.xml"),
    "m31": ("m31", "spherex/metadata/m31-resolver.xml"),
    "m51": ("m51", "spherex/metadata/m51-resolver.xml"),
    "proxima": ("proxima", "spherex/metadata/proxima-resolver.xml"),
    "r-draconis": ("r_draconis", "spherex/metadata/r-draconis-resolver.xml"),
    "ross128": ("ross128", "spherex/metadata/ross128-resolver.xml"),
    "rw-aurigae": ("rw_aurigae", "spherex/metadata/rw-aurigae-resolver.xml"),
    "wise0855": ("wise0855", "spherex/metadata/wise0855-resolver.xml"),
    # the SPHEREx copy of this resolver response is empty; the WISE copy is the same SIMBAD record
    "teegarden": ("teegarden", "wise/metadata/teegarden-sesame.xml"),
}
SPHEREX_FRAME_DIRS = {s: ("spherex/extra-target-frames/" if s == "teegarden" else "spherex/frames/") + s
                      for s in SPHEREX_FIELDS}

# WISE slug -> (physical group, resolver, RA offset of field centre from the target in deg)
WISE_FIELDS = {
    "barnard": ("barnard", "wise/metadata/barnard-sesame.xml", 0.0),
    "barnard_control": ("barnard", "wise/metadata/barnard-sesame.xml", 0.4),
    "wise0855": ("wise0855", "wise/metadata/wise0855-sesame.xml", 0.0),
    "wise0855_control": ("wise0855", "wise/metadata/wise0855-sesame.xml", 0.4),
    "luhman16": ("luhman16", "wise/metadata/luhman16-sesame.xml", 0.0),
    "epsilon_indi_b": ("epsilon_indi_b", "wise/metadata/epsilon_indi_b-sesame.xml", 0.0),
    "wolf359": ("wolf359", "wise/metadata/wolf359-sesame.xml", 0.0),
    "lalande21185": ("lalande21185", "wise/metadata/lalande21185-sesame.xml", 0.0),
    "61cyg_a": ("61cyg_a", "wise/metadata/61cyg_a-sesame.xml", 0.0),
    "teegarden": ("teegarden", "wise/metadata/teegarden-sesame.xml", 0.0),
}

NEOSSAT_GROUPS = {
    # folder: (physical group, Horizons id, id_type, class)
    "comet_12P": ("neossat_12P", "12P", "designation", "comet"),
    "comet_2021S3": ("neossat_2021S3", "C/2021 S3", "designation", "comet"),
    "neowise": ("neossat_neowise", "C/2020 F3", "designation", "comet"),
    "phaethon": ("neossat_phaethon", "3200", "smallbody", "asteroid"),
    "asteroid_1999KW4": ("neossat_1999KW4", "66391", "smallbody", "asteroid"),
    "control_TCrB": ("neossat_TCrB", None, None, "star_field"),
    "control_tabby": ("neossat_tabby", None, None, "star_field"),
}

COMET_3I = {"folder": "spherex/comet-frames/3iatlas", "group": "3iatlas", "horizons_id": "C/2025 N1"}

HORIZONS_OBSERVER = {"spherex": "500@-163182", "wise": "500@-163", "neossat": "500@-139089"}
HORIZONS_SPACECRAFT_ID = {"spherex": "-163182", "wise": "-163", "neossat": "-139089"}

# Five folds by physical group, balanced by object class (brown dwarf, variable,
# galaxy, high proper motion star, solar-system mover). Default benchmark:
# test = fold 0, validation = fold 1, train = folds 2-4.
FOLDS = {
    0: ["wise0855", "r_draconis", "m31", "wolf359", "neossat_neowise"],
    1: ["epsilon_indi_b", "rw_aurigae", "m101", "lalande21185", "neossat_phaethon"],
    2: ["luhman16", "m51", "61cyg_a", "neossat_2021S3", "neossat_tabby"],
    3: ["barnard", "proxima", "neossat_TCrB", "neossat_1999KW4"],
    # 12P (all frames fail registration: smeared stars) is kept out of the test fold
    4: ["teegarden", "ross128", "3iatlas", "neossat_12P"],
}
DEFAULT_SPLIT = {"test": [0], "val": [1], "train": [2, 3, 4]}

# ---------------------------------------------------------------------------
# Common sky grids (north up, east left, TAN projection)
# ---------------------------------------------------------------------------
GRID = {
    "spherex": {"npix": 128, "scale_arcsec": 6.15},   # native SPHEREx pixel ~6.13"
    "wise": {"npix": 128, "scale_arcsec": 2.75},      # native WISE pixel ~2.75"
    "neossat": {"npix": 256, "scale_arcsec": 3.0},    # native NEOSSat pixel ~2.97"
    "comet": {"npix": 128, "scale_arcsec": 6.15},
}
CATALOG_RADIUS_DEG = 0.17   # covers the SPHEREx grid half-diagonal (0.156 deg)

# ---------------------------------------------------------------------------
# Pixel flag policy. "hard" bits make a pixel invalid (value unusable). "artifact"
# bits are kept as an input channel: the value is real but may be contaminated,
# and a model should learn to discount it. Pixels flagged SOURCE are never masked.
# ---------------------------------------------------------------------------
SPHEREX_FLAG_NAMES = {0: "TRANSIENT", 1: "OVERFLOW", 2: "SUR_ERROR", 6: "NONFUNC", 7: "DICHROIC",
                      9: "MISSING_DATA", 10: "HOT", 11: "COLD", 12: "FULLSAMPLE", 14: "PHANMISS",
                      15: "NONLINEAR", 17: "PERSIST", 19: "OUTLIER", 20: "CROSSTALK", 21: "SOURCE",
                      22: "GHOST", 23: "GHOST_FPA", 24: "GHOST_EXT", 25: "STREAK", 26: "BLOOM",
                      27: "SNOWBALL", 28: "HALO"}
SPHEREX_HARD_BITS = [1, 2, 6, 9, 10, 11, 15]
# OUTLIER fires on real moving objects: it flags the 3I/ATLAS comet in every frame
# where the comet is bright. It stays an input channel and does not veto a mover.
SPHEREX_MOVER_TOLERATED_BITS = [19]
SPHEREX_ARTIFACT_BITS = [0, 7, 14, 17, 19, 20, 22, 23, 24, 25, 26, 27, 28]

# NEOWISE frame mask (Explanatory Supplement, Table 7). Hard set = the bits the
# pipeline already NaNs (1048154) plus bit 30 (inadvertent NaN).
WISE_HARD_BITS = [1, 3, 4, 6] + list(range(9, 20)) + [30]
WISE_ARTIFACT_BITS = [21, 26, 28]   # transient bad pixel, unreliable nonlinearity, replaced spike

def bitmask(bits):
    m = 0
    for b in bits:
        m |= 1 << b
    return m

# ---------------------------------------------------------------------------
# Frame quality gates
# ---------------------------------------------------------------------------
QC = {
    "min_valid_fraction": 0.50,        # of the grid
    "min_center_valid_fraction": 0.80,  # central 32x32 pixels, where the target sits
    "min_reference_stars": 5,          # for registration refinement
    # Registration is judged by the uncertainty of the applied shift
    # (1.2533 * scatter / sqrt(n_ref)). Per-star scatter grows on bright structured
    # backgrounds (M31 disk) without the shift being wrong, so it only has a loose cap.
    "max_registration_error_px": 0.10,
    "max_registration_rms_px": 1.0,
    "max_registration_shift_px": 3.0,  # larger WCS corrections are treated as failures
    "max_noise_ratio": 3.0,            # sky noise vs the median of the same detector/band in the field
}

# Photometry apertures in grid pixels: (radius, annulus inner, annulus outer)
APERTURE = {"spherex": (2.0, 5.0, 8.0), "wise": (2.5, 7.0, 11.0), "neossat": (3.0, 8.0, 12.0)}
DETECTION_SNR = 5.0
MOVER_MIN_PX = 1.0   # total displacement within a stack that counts as "moving"


# ---------------------------------------------------------------------------
# Extra SPHEREx fields downloaded by s00_acquire_more.py: the fastest-moving Gaia
# stars. A star that already has a WISE field keeps that field's group, so both
# missions' data of one object stay in one fold; new groups are spread over folds.
# ---------------------------------------------------------------------------
import json as _json

EXTRA_FIELDS = {}
_extra = CACHE / "raw_extra" / "fields.json"
if _extra.exists():
    EXTRA_FIELDS = {k: v for k, v in _json.loads(_extra.read_text()).items() if v.get("n_frames_downloaded", 0) >= 12}
    _known = {"1872046609345556480": "61cyg_a", "1872046574983497216": "61cyg_a",
              "762815470562110464": "lalande21185", "3864972938605115520": "wolf359"}
    _assigned = {g for gs in FOLDS.values() for g in gs}
    _new = []
    for _slug in sorted(EXTRA_FIELDS):
        _f = EXTRA_FIELDS[_slug]
        _f["group"] = _known.get(_f["source_id"], _f["group"])
        SPHEREX_FIELDS[_slug] = (_f["group"], None)
        SPHEREX_FRAME_DIRS[_slug] = str(CACHE / "raw_extra" / "spherex" / _slug)
        if _f["group"] not in _assigned and _f["group"] not in _new:
            _new.append(_f["group"])
    for _i, _g in enumerate(_new):
        FOLDS[_i % 5].append(_g)
