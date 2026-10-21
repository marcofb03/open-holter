"""Download PhysioNet databases used for validation into data/<db>/.

Usage (from analysis/):  python tools/fetch_physionet.py mitdb svdb incartdb

Only the files needed for evaluation are fetched (.hea, .dat, .atr). The
official PhysioNet server is tried first; if it cannot be reached, the
PhysioNet open-data mirror on AWS S3 is used (same files). Data are written
under data/<db>/, which is git-ignored: never commit them.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VERSIONS = {"mitdb": "1.0.0", "svdb": "1.0.0", "incartdb": "1.0.0"}
MIRRORS = [
    "https://physionet.org/files/{db}/{ver}/",
    "https://physionet-open.s3.amazonaws.com/{db}/{ver}/",
]
EXTENSIONS = (".hea", ".dat", ".atr")


def _get(url: str, timeout: float = 60) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def _base_url(db: str) -> str:
    for m in MIRRORS:
        url = m.format(db=db, ver=VERSIONS[db])
        try:
            _get(url + "RECORDS", timeout=20)
            return url
        except OSError:
            continue
    raise RuntimeError(f"Cannot reach any PhysioNet mirror for {db}")


def fetch(db: str, records: list[str] | None = None, out: Path | None = None) -> Path:
    """Download (if missing) the given records of `db`; return the folder."""
    out = out or REPO / "data" / db
    out.mkdir(parents=True, exist_ok=True)
    need = [r for r in (records or [None])]  # None = whole database
    if records and all((out / f"{r}{e}").exists() for r in records for e in EXTENSIONS):
        return out
    base = _base_url(db)
    if records is None or None in need:
        records = _get(base + "RECORDS").decode().split()
    for r in records:
        for e in EXTENSIONS:
            dst = out / f"{r}{e}"
            if dst.exists() and dst.stat().st_size > 0:
                continue
            tmp = dst.with_suffix(dst.suffix + ".part")
            tmp.write_bytes(_get(base + f"{r}{e}"))
            tmp.rename(dst)
        print(f"  {db}/{r}", flush=True)
    return out


def main() -> None:
    dbs = sys.argv[1:] or list(VERSIONS)
    for db in dbs:
        print(f"Fetching {db} ...", flush=True)
        fetch(db)


if __name__ == "__main__":
    main()
