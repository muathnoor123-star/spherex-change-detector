"""Recover the original v1.3.0 ZIP from GitHub Release assets."""
import argparse
import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile

SHA256 = "e7dde2f15d429e3db28e63cac3005f2d9c4c193b275db56538864e2808bdbf79"


def join(folder):
    parts = [folder / f"planetx-v13.part{i:02}" for i in range(1, 40)]
    missing = [p.name for p in parts if not p.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    target = folder / "planetx-dataset-v1.3.0.zip"
    if target.exists():
        with target.open("rb") as src:
            if hashlib.file_digest(src, "sha256").hexdigest() == SHA256:
                return target
        raise FileExistsError(target)
    digest = hashlib.sha256()
    with NamedTemporaryFile(dir=folder, delete=False) as dst:
        temp = Path(dst.name)
    try:
        with temp.open("wb") as dst:
            for part in parts:
                with part.open("rb") as src:
                    while block := src.read(8 << 20):
                        digest.update(block)
                        dst.write(block)
        if digest.hexdigest() != SHA256:
            raise ValueError("Checksum mismatch. Check all 39 downloads.")
        temp.replace(target)
        return target
    finally:
        temp.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    print(join(parser.parse_args().folder))
