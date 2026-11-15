"""Verify downloaded release integrity: python tools/verify_release.py [dataset_root]."""
import hashlib
import sys
from pathlib import Path

def main():
    root=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parents[1]
    count=0
    for line in (root/'MANIFEST.sha256').read_text().splitlines():
        expected,relative=line.split(None,1)
        p=root/relative.strip()
        if not p.is_file(): raise RuntimeError(f'Missing file: {relative}')
        h=hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''): h.update(block)
        if h.hexdigest()!=expected: raise RuntimeError(f'Hash mismatch: {relative}')
        count+=1
    print(f'PASS: {count} files match the release manifest.')
if __name__=='__main__': main()
