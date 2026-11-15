"""Restore exact v1.3 files using the v1.2 GitHub Release ZIP and the checked-in overlay."""
import argparse
import hashlib
from pathlib import Path
import shutil
import zipfile


def restore(source, output):
    overlay = Path(__file__).resolve().parent / "dataset_overlay"
    manifest = overlay / "MANIFEST.sha256"
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(source) as archive:
        manifests = [n for n in archive.namelist() if n.rsplit("/", 1)[-1] == "MANIFEST.sha256"]
        if len(manifests) != 1:
            raise ValueError("Expected the original v1.2 dataset ZIP")
        prefix = manifests[0].removesuffix("MANIFEST.sha256")
        for line in manifest.read_text().splitlines():
            digest, name = line.split(None, 1)
            target = (output / name).resolve()
            if not target.is_relative_to(output.resolve()):
                raise ValueError(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            changed = overlay / name
            with (changed.open("rb") if changed.exists() else archive.open(prefix + name)) as src:
                with target.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
            with target.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                    raise ValueError(f"Checksum mismatch for {name}")
    shutil.copyfile(manifest, output / manifest.name)
    print(f"Verified {len(manifest.read_text().splitlines())} files in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    restore(args.source, args.output)
