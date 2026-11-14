"""Restore the exact trained morphology checkpoint in a GitHub checkout."""
from pathlib import Path
import hashlib
import json


def restore():
    folder = Path(__file__).resolve().parent
    manifest = json.loads((folder / "checkpoint.json").read_text())
    checkpoint = b"".join((folder / "parts" / name).read_bytes() for name in manifest["parts"])
    if hashlib.sha256(checkpoint).hexdigest() != manifest["sha256"]:
        raise ValueError("Morphology checkpoint checksum mismatch")
    (folder / "model.joblib").write_bytes(checkpoint)


if __name__ == "__main__":
    restore()
