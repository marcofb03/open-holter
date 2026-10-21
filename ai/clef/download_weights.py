"""Download the CLEF-Medium checkpoint from Zenodo (record 17572734).

    python download_weights.py            # -> weights/clef_medium.ckpt (368 MB)

The file is git-ignored; never commit it. Zenodo states no licence for the
weights (the code is Clear BSD; the model was pre-trained on MIMIC-IV-ECG), so
this repository only downloads them for evaluation and never redistributes
them or anything fine-tuned from them. The MD5 is checked against Zenodo's
listing and the SHA-256 printed so a run can be tied to an exact file.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request

from clef import DEFAULT_CKPT

URL = "https://zenodo.org/records/17572734/files/clef_medium.ckpt?download=1"
MD5 = "781eed16d190ef97733994cd2d01d1ff"  # Zenodo file listing


def digest(path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    DEFAULT_CKPT.parent.mkdir(parents=True, exist_ok=True)
    if not DEFAULT_CKPT.exists():
        tmp = DEFAULT_CKPT.with_suffix(".part")
        with urllib.request.urlopen(URL) as r, tmp.open("wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        if digest(tmp, "md5") != MD5:
            tmp.unlink()
            raise SystemExit("MD5 mismatch: download corrupted or the Zenodo file changed")
        tmp.rename(DEFAULT_CKPT)
    print(f"{DEFAULT_CKPT}  {DEFAULT_CKPT.stat().st_size / 1e6:.1f} MB  sha256 {digest(DEFAULT_CKPT, 'sha256')}")


if __name__ == "__main__":
    main()
