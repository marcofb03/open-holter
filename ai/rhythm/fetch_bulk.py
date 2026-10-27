"""Download and unpack the two sources that come as one archive (run once where prep_seg.py runs).

    python fetch_bulk.py iridia-af sph          # -> $OH_LOCAL_DATA/iridia-af, $OH_LOCAL_DATA/sph

iridia-af  Zenodo 8405941 (CC BY 4.0): iridia-af-metadata-v1.0.1.csv + records zip (10.5 GB)
sph        figshare collection 5779802 (records CC0, paper CC BY 4.0): metadata.csv + records.tar.gz (2.3 GB)
Both are resumable (curl -C -) and unpacked into records/ next to the metadata. The archive is
deleted after a complete unpack. VTaC, Challenge 2015 and Long-Term ST are only on physionet.org
(~9 KB/s per connection): see TRANSFORMER_PLAN.md for the parallel background fetch.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import extra_sources as xs  # noqa: E402

ZENODO = "https://zenodo.org/api/records/8405941"
FIGSHARE_ARTICLES = {"metadata": 17912441, "records": 17912444}


def curl(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-L", "--retry", "8", "--retry-delay", "15", "-C", "-", "-o", str(dst), url], check=True)


def figshare_files(article: int) -> list[dict]:
    with urllib.request.urlopen(f"https://api.figshare.com/v2/articles/{article}", timeout=60) as r:
        return json.load(r)["files"]


def iridia() -> None:
    root = xs.local("iridia-af")
    with urllib.request.urlopen(ZENODO, timeout=60) as r:
        files = json.load(r)["files"]
    for f in files:
        dst = root / f["key"]
        if f["key"].endswith(".csv"):
            curl(f["links"]["self"], dst)
        elif not (root / "records").exists():
            curl(f["links"]["self"], dst)
            with zipfile.ZipFile(dst) as z:
                z.extractall(root)
            dst.unlink()
    print(f"iridia-af -> {root}")


def sph() -> None:
    root = xs.local("sph")
    for f in figshare_files(FIGSHARE_ARTICLES["metadata"]):
        curl(f["download_url"], root / f["name"])
    if not (root / "records").exists():
        for f in figshare_files(FIGSHARE_ARTICLES["records"]):
            dst = root / f["name"]
            curl(f["download_url"], dst)
            with tarfile.open(dst) as t:
                t.extractall(root, filter="data")
            dst.unlink()
    print(f"sph -> {root}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="+", choices=["iridia-af", "sph"])
    for w in ap.parse_args().what:
        {"iridia-af": iridia, "sph": sph}[w]()


if __name__ == "__main__":
    main()
