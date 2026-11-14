"""Reassemble the trained checkpoint distributed with the experiment."""
from pathlib import Path
import hashlib

folder = Path(__file__).resolve().parent
parts = [folder / "parts" / f"{i:02}.bin" for i in range(19)]
data = b"".join(part.read_bytes() for part in parts)
if hashlib.sha256(data).hexdigest() != "bee1429dd108766c8ce45ee0870c9aa25668b285f4115bbf333c1e9f07ddbffa":
    raise ValueError("Checkpoint checksum mismatch")
(folder / "selected.joblib").write_bytes(data)
