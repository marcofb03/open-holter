"""Download PhysioNet databases used for validation into data/<db>/.

Usage (from analysis/):
    python tools/fetch_physionet.py mitdb svdb incartdb
    python tools/fetch_physionet.py --aws ltafdb afdb ltdb vfdb cudb nstdb cpsc2021 shdb-af

Only the files needed for evaluation are fetched (.hea, .dat and the
annotation files listed in ANNOTATIONS). The official PhysioNet server is tried
first; if it cannot be reached, the PhysioNet open-data mirror on AWS S3 is
used (same files). `--aws` (or PHYSIONET_MIRROR=aws) tries the S3 mirror first,
which is much faster for the multi-GB long-term databases. Record names may
contain a sub-folder (cpsc2021: Training_set_I/data_0_1); it is kept. A record
without a signal file (afdb 00735, 03665) is skipped with a note. Data are
written under data/<db>/, which is git-ignored: never commit them.

Icentia11k (11,000 patients, 1.1 TB) is not fetched whole: see
ai/benchmark/datasets.py (a few patients for the benchmark) and
ai/finetune/ (streamed patient by patient on the cluster).
"""

from __future__ import annotations

import argparse
import http.client
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VERSIONS = {
    "mitdb": "1.0.0", "svdb": "1.0.0", "incartdb": "1.0.0",
    "ltafdb": "1.0.0", "afdb": "1.0.0", "ltdb": "1.0.0", "vfdb": "1.0.0", "cudb": "1.0.0",
    "nstdb": "1.0.0", "cpsc2021": "1.0.0", "shdb-af": "1.0.1",
    "icentia11k-continuous-ecg": "1.0",
}
PHYSIONET = "https://physionet.org/files/{db}/{ver}/"
AWS = "https://physionet-open.s3.amazonaws.com/{db}/{ver}/"
# Annotation files per database (default: reference annotations only).
ANNOTATIONS = {"ltafdb": (".atr", ".qrs"), "afdb": (".atr", ".qrs"), "shdb-af": (".atr", ".qrs")}
EXTENSIONS = (".hea", ".dat", ".atr")
NO_RECORDS_FILE = {"shdb-af"}


def _get(url: str, timeout: float = 60, tries: int = 5) -> bytes:
    for k in range(tries):  # long transfers are sometimes cut: retry with backoff
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError:
            raise
        except (OSError, http.client.HTTPException):
            if k == tries - 1:
                raise
            time.sleep(2 ** (k + 1))
    raise AssertionError


def _mirrors(aws_first: bool) -> list[str]:
    return [AWS, PHYSIONET] if aws_first else [PHYSIONET, AWS]


def _base_url(db: str, aws_first: bool = False) -> str:
    for m in _mirrors(aws_first):
        url = m.format(db=db, ver=VERSIONS[db])
        try:
            _get(url + ("RECORDS" if db not in NO_RECORDS_FILE else "README.md"), timeout=20, tries=2)
            return url
        except OSError:
            continue
    raise RuntimeError(f"Cannot reach any PhysioNet mirror for {db}")


def _s3_records(db: str) -> bytes:
    """Record names from the S3 listing (for databases without a RECORDS file)."""
    import re
    import urllib.parse

    prefix, keys, token = f"{db}/{VERSIONS[db]}/", [], None
    while True:
        q = f"?list-type=2&prefix={prefix}" + (f"&continuation-token={urllib.parse.quote(token)}" if token else "")
        x = _get("https://physionet-open.s3.amazonaws.com/" + q).decode()
        keys += re.findall(r"<Key>([^<]*)</Key>", x)
        m = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", x)
        if not m:
            break
        token = m.group(1)
    names = sorted(k[len(prefix):-4] for k in keys if k.endswith(".hea"))
    return ("\n".join(names) + "\n").encode()


def extensions(db: str) -> tuple[str, ...]:
    return (".hea", ".dat") + ANNOTATIONS.get(db, (".atr",))


def fetch(db: str, records: list[str] | None = None, out: Path | None = None,
          aws_first: bool | None = None) -> Path:
    """Download (if missing) the given records of `db`; return the folder."""
    if aws_first is None:
        aws_first = os.environ.get("PHYSIONET_MIRROR", "").lower() == "aws"
    out = out or REPO / "data" / db
    out.mkdir(parents=True, exist_ok=True)
    exts = extensions(db)
    if records and all((out / f"{r}{e}").exists() for r in records for e in exts):
        return out
    base = _base_url(db, aws_first)
    rec_file = None
    if records is None:
        rec_file = _s3_records(db) if db in NO_RECORDS_FILE else _get(base + "RECORDS")
        records = rec_file.decode().split()
    for r in records:
        for e in exts:
            dst = out / f"{r}{e}"
            if dst.exists() and dst.stat().st_size > 0:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                data = _get(base + f"{r}{e}", timeout=600)
            except urllib.error.HTTPError as err:
                if err.code == 404:  # afdb: records without a signal file; cpsc2021: RECORDS lists 11 absent ones
                    print(f"  {db}/{r}{e}: not on the server, skipped", flush=True)
                    if e == ".hea":
                        break
                    continue
                raise
            tmp = dst.with_suffix(dst.suffix + ".part")
            tmp.write_bytes(data)
            tmp.rename(dst)
        print(f"  {db}/{r}", flush=True)
    if rec_file is not None:  # written last: its presence means the download is complete
        (out / "RECORDS").write_bytes(rec_file)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("db", nargs="*", default=["mitdb", "svdb", "incartdb"], choices=sorted(VERSIONS))
    ap.add_argument("--aws", action="store_true", help="try the AWS S3 mirror first")
    args = ap.parse_args()
    for db in args.db:
        if db == "icentia11k-continuous-ecg":
            sys.exit("Icentia11k is 1.1 TB: see ai/benchmark/datasets.py or ai/finetune/ instead")
        print(f"Fetching {db} ...", flush=True)
        fetch(db, aws_first=args.aws or None)


if __name__ == "__main__":
    main()
