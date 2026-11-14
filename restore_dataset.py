"""Restore the original dataset ZIP from the two GitHub Release assets."""
import hashlib
import shutil
from pathlib import Path
import argparse

def restore(folder):
    output = folder/"planetx-dataset-v1.2.0.zip"
    expected = "ffa6013c79e561d767cdfa6750551c0434944563aaac67ef7230de21fefd3aa7"
    parts = [folder/(output.name + suffix) for suffix in (".part1", ".part2")]
    for part in parts:
        if not part.is_file():
            raise FileNotFoundError(part)
    with output.open("xb") as dst:
        for part in parts:
            with part.open("rb") as src:
                shutil.copyfileobj(src,dst,8*1024*1024)
    with output.open("rb") as src:
        if hashlib.file_digest(src,"sha256").hexdigest() != expected:
            raise ValueError(f"Checksum mismatch. Inspect {output} and the downloaded parts.")
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder",type=Path)
    print(restore(parser.parse_args().folder))
