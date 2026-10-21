"""Download the ECGFounder 1-lead checkpoint (MIT licence) from Hugging Face.

    python download_weights.py            # -> weights/1_lead_ECGFounder.pth (~370 MB)

The weights are git-ignored; never commit them. Source:
https://huggingface.co/PKUDigitalHealth/ECGFounder (ungated). The SHA-256 is
printed so a run can be tied to an exact file; record it in RESULTS.md.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
URL = "https://huggingface.co/PKUDigitalHealth/ECGFounder/resolve/main/{name}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="1_lead_ECGFounder.pth")
    ap.add_argument("--out-dir", type=Path, default=HERE / "weights")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dest = args.out_dir / args.name
    if not dest.exists():
        tmp = dest.with_suffix(".part")
        print(f"downloading {URL.format(name=args.name)}", file=sys.stderr)
        with urllib.request.urlopen(URL.format(name=args.name)) as r, tmp.open("wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        tmp.rename(dest)
    print(f"{dest}  {dest.stat().st_size / 1e6:.1f} MB  sha256 {sha256(dest)}")


if __name__ == "__main__":
    main()
