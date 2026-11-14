"""Restore the original dataset ZIP from the GitHub Release assets."""
import hashlib
import shutil
from pathlib import Path
import argparse
import zipfile

def restore(folder):
    output = folder/"planetx-dataset-v1.2.0.zip"
    expected = "ffa6013c79e561d767cdfa6750551c0434944563aaac67ef7230de21fefd3aa7"
    parts = sorted(folder.glob("data-[12]-[0-9][0-9].zip"))
    names = [f"data-{i}-{j:02}.zip" for i in (1,2) for j in range(10)]
    if [part.name for part in parts] != names:
        raise FileNotFoundError("Download all twenty ZIP assets into one directory.")
    with output.open("xb") as dst:
        for part in parts:
            with zipfile.ZipFile(part) as container:
                with container.open("chunk-" + part.stem.removeprefix("data-")) as src:
                    shutil.copyfileobj(src,dst,8*1024*1024)
    with output.open("rb") as src:
        if hashlib.file_digest(src,"sha256").hexdigest() != expected:
            raise ValueError(f"Checksum mismatch. Inspect {output} and the downloaded parts.")
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder",type=Path)
    print(restore(parser.parse_args().folder))
