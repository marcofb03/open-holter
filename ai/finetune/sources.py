"""Training sources: patient lists, patient-level splits, and readers that stream raw files.

Every reader downloads one patient's raw files into a temporary folder (on the
cluster: node-local $TMPDIR), turns them into H10-like records and deletes the
raw files before the next patient, so the shared file system only ever holds
the finished shards.

Splits (patient level; "benchmark" policy, the default, keeps every patient of
the benchmark's held-out split out of training, so ai/benchmark/ stays a fair
test of the fine-tuned models):
    icentia11k  patients 0-9,999: train, every 20th (id % 20 == 0) val;
                10,000-10,999: test (heartKIT's own test patients, and the
                benchmark's Icentia11k sample is drawn from them)
    ltafdb, cpsc2021, shdb-af, afdb   benchmark "tune" patients: train (1 in 10 val);
                benchmark "held" patients: test. With --policy all they are split
                80/10/10 by patient hash (only for a final model, after evaluation).
    ptb-xl      strat_fold 1-8 train, 9 val, 10 test (the authors' recommendation);
                leads I and II of each ECG as two examples of the same patient
    cinc2017    80/10/10 by record hash (records are short and anonymous)
    cpsc2020    A01-A08 train, A09 val, A10 test (one patient per record)
"""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import shutil
import sys
import urllib.error
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "ai" / "benchmark"), str(REPO / "analysis"), str(REPO / "analysis" / "tools")]
import datasets as bds  # noqa: E402
import fetch_physionet as fp  # noqa: E402
import h10  # noqa: E402

S3 = "https://physionet-open.s3.amazonaws.com/"
CPSC2020_URL = "https://opensz.oss-cn-beijing.aliyuncs.com/ICBEB2020/file/TrainingSet.zip"  # reachable 2026-10-21
BENCH_DBS = ("ltafdb", "cpsc2021", "shdb-af", "afdb")
SOURCES = ("icentia11k", *BENCH_DBS, "ptb-xl", "cinc2017", "cpsc2020")


def _get(url: str) -> bytes:
    return fp._get(url, timeout=600)


def _hash10(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % 10


def _three_way(key: str) -> str:
    h = _hash10(key)
    return "test" if h == 0 else "val" if h == 1 else "train"


# --------------------------------------------------------------------------
# Patient lists: [(patient, split, payload)]
# --------------------------------------------------------------------------

def patients(source: str, policy: str = "benchmark") -> list[tuple[str, str, object]]:
    if source == "icentia11k":
        return [(f"p{i:05d}", "test" if i >= bds.HEARTKIT_TRAIN_MAX else "val" if i % 20 == 0 else "train", None)
                for i in range(11_000)]
    if source in BENCH_DBS:
        base = fp.AWS.format(db=source, ver=fp.VERSIONS[source])
        recs = (fp._s3_records(source) if source in fp.NO_RECORDS_FILE else _get(base + "RECORDS")).decode().split()
        by_pt: dict[str, list[str]] = {}
        for r in recs:
            by_pt.setdefault(bds.patient_of(source, r), []).append(r)
        out = []
        for p, rs in sorted(by_pt.items()):
            if policy == "all":
                split = _three_way(f"{source}:{p}")
            else:
                rule = bds.DBS[source]["split"]
                bsplit = bds.third(source, p) if rule == "third" else rule
                split = "test" if bsplit == "held" else ("val" if _hash10(f"val:{source}:{p}") == 0 else "train")
            out.append((p, split, rs))
        return out
    if source == "ptb-xl":
        rows = list(csv.DictReader(io.StringIO(_get(S3 + "ptb-xl/1.0.3/ptbxl_database.csv").decode())))
        by_pt: dict = {}
        for r in rows:
            by_pt.setdefault(r["patient_id"], []).append(r)
        out = []
        for p, rs in by_pt.items():
            fold = int(rs[0]["strat_fold"])
            out.append((p.split(".")[0], "test" if fold == 10 else "val" if fold == 9 else "train", rs))
        return sorted(out)
    if source == "cinc2017":
        ref = _get(S3 + "challenge-2017/1.0.0/REFERENCE-v3.csv").decode().split()
        return [(r.split(",")[0], _three_way(f"cinc2017:{r.split(',')[0]}"), r.split(",")[1]) for r in ref]
    if source == "cpsc2020":
        return [(f"A{i:02d}", "test" if i == 10 else "val" if i == 9 else "train", None) for i in range(1, 11)]
    raise ValueError(source)


# --------------------------------------------------------------------------
# Readers: patient -> [(label namespace, raw int16 uV at the source rate, fs, reference)]
# --------------------------------------------------------------------------

def read_patient(source: str, patient: str, payload, tmp: Path) -> list[tuple[str, np.ndarray, float, dict]]:
    import wfdb

    d = tmp / f"{source}_{patient}"
    d.mkdir(parents=True, exist_ok=True)
    try:
        if source == "icentia11k":
            names = _get(f"{bds.ICENTIA_URL}{patient[:3]}/{patient}/RECORDS").decode().split()

            def get(s):  # a few segments lack a file on the server (e.g. p00008): skip those
                try:
                    bds.fetch_icentia(f"{patient}/{s}", out=d)
                    return s
                except urllib.error.HTTPError as err:
                    if err.code != 404:
                        raise
                    print(f"  {patient}/{s}: missing on the server, skipped", file=sys.stderr)
                    return None
            with ThreadPoolExecutor(8) as ex:
                ok = [s for s in ex.map(get, names) if s]
            return [_wfdb("icentia11k", d / patient / s) for s in ok]
        if source in BENCH_DBS:
            out = []
            for r in payload:
                fp.fetch(source, [r], out=d, aws_first=True)
                if (d / f"{r}.dat").exists():
                    out.append(_wfdb(source, d / r))
            return out
        if source == "ptb-xl":
            out = []
            for row in payload:
                p = d / row["filename_hr"].split("/")[-1]
                for e in (".hea", ".dat"):
                    p.with_suffix(e).write_bytes(_get(f"{S3}ptb-xl/1.0.3/{row['filename_hr']}{e}"))
                rec = wfdb.rdrecord(str(p))
                ref = ptbxl_reference(row["scp_codes"], rec.sig_len / rec.fs)
                for lead in ("I", "II"):
                    out.append(("ptb-xl", h10.raw16(rec.p_signal[:, rec.sig_name.index(lead)]), rec.fs, ref))
            return out
        if source == "cinc2017":
            p = d / patient
            for e in (".hea", ".mat"):
                p.with_suffix(e).write_bytes(_get(f"{S3}challenge-2017/1.0.0/training/{patient[:3]}/{patient}{e}"))
            rec = wfdb.rdrecord(str(p))
            return [("cinc2017", h10.raw16(rec.p_signal[:, 0]), rec.fs, {"cinc_label": payload,
                     "dur_s": rec.sig_len / rec.fs, "has_beats": False, "has_rhythm": False, "rhythm": [],
                     "beats_t": None, "beats_c": None})]
        if source == "cpsc2020":
            return [cpsc2020_record(patient, tmp)]
        raise ValueError(source)
    finally:
        shutil.rmtree(d, ignore_errors=True)  # raw files never outlive their patient


def _wfdb(db: str, path: Path) -> tuple[str, np.ndarray, float, dict]:
    import wfdb

    h = wfdb.rdheader(str(path))
    lead = bds.DBS[db]["lead"]
    ch = h.sig_name.index(lead) if isinstance(lead, str) and lead in h.sig_name else (lead if isinstance(lead, int) else 0)
    rec = wfdb.rdrecord(str(path), channels=[ch])
    ref = bds.reference_from_ann(db, wfdb.rdann(str(path), "atr"), rec.fs, rec.sig_len)
    return db, h10.raw16(rec.p_signal[:, 0]), rec.fs, ref


PTBXL_MAP = {"af": {"AFIB", "AFLT"}, "pvc": {"PVC", "BIGU", "TRIGU"}, "pac": {"PAC"}, "svt": {"SVTAC", "PSVT"}}


def ptbxl_reference(scp: str, dur: float) -> dict:
    """PTB-XL: statement-level labels for the whole 10 s ECG (no beat annotations)."""
    codes = set(ast.literal_eval(scp))
    return {"ptbxl": {k: int(bool(codes & v)) for k, v in PTBXL_MAP.items()}, "dur_s": dur,
            "has_beats": False, "has_rhythm": False, "rhythm": [], "beats_t": None, "beats_c": None}


def cpsc2020_record(patient: str, tmp: Path) -> tuple[str, np.ndarray, float, dict]:
    """CPSC2020: 24 h single-lead Holter, 400 Hz, S and V beat positions (N beats not annotated:
    taken from the analysis/ detector; detected beats within 150 ms of an S/V mark get its class)."""
    import scipy.io as sio
    from ecgpipe.beats import analyse_beats

    z = tmp / "cpsc2020.zip"
    if not z.exists():
        z.write_bytes(_get(CPSC2020_URL))
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
        dat = next(n for n in names if n.endswith(f"data/{patient}.mat"))
        ref = next(n for n in names if n.endswith(f"ref/R{patient[1:]}.mat"))
        ecg = sio.loadmat(io.BytesIO(zf.read(dat)), squeeze_me=True)["ecg"].astype(float)
        r = sio.loadmat(io.BytesIO(zf.read(ref)), squeeze_me=True, struct_as_record=False)["ref"]
    s_t, v_t = np.atleast_1d(r.S_ref) / 400.0, np.atleast_1d(r.V_ref) / 400.0
    _, b = analyse_beats(h10.to_h10(ecg, 400.0) / 1000.0, h10.FS)  # N beats from the phone's detector
    bt = b.idx / h10.FS
    bc = np.full(len(bt), "N", "<U1")
    marks = np.concatenate([s_t, v_t])
    kind = np.array(["S"] * len(s_t) + ["V"] * len(v_t))
    extra_t, extra_c = [], []
    for t, k in zip(marks, kind):
        j = np.searchsorted(bt, t)
        cand = [i for i in (j - 1, j) if 0 <= i < len(bt)]
        i = min(cand, key=lambda i: abs(bt[i] - t)) if cand else None
        if i is not None and abs(bt[i] - t) <= 0.15:
            bc[i] = k
        else:
            extra_t.append(t)
            extra_c.append(k)
    t_all = np.concatenate([bt, extra_t])
    order = np.argsort(t_all)
    return "cpsc2020", h10.raw16(ecg), 400.0, {"dur_s": len(ecg) / 400.0, "has_beats": True, "has_rhythm": False, "rhythm": [],
                             "beats_t": t_all[order], "beats_c": np.concatenate([bc, extra_c]).astype("<U1")[order]}


def labels_override(ns: str, ref: dict, y: np.ndarray) -> np.ndarray:
    """Statement-level labels (PTB-XL, CinC 2017) for every window of the record."""
    y = y.copy()
    T = h10.WINDOW_TARGETS
    if ns == "ptb-xl":
        for k, v in ref["ptbxl"].items():
            y[:, T.index(k)] = v
    elif ns == "cinc2017":
        lab = ref["cinc_label"]
        y[:, T.index("af")] = 1 if lab == "A" else (0 if lab in "NO" else -1)
        y[:, T.index("noise")] = 1 if lab == "~" else 0
    return y


def noise_arrays() -> dict[str, np.ndarray]:
    """NSTDB bw / em / ma, first channel, 130 Hz uV (float32; polyphase only, the front end comes later)."""
    import tempfile

    import wfdb

    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        fp.fetch("nstdb", ["bw", "em", "ma"], out=Path(tmp))
        for k in ("bw", "em", "ma"):
            rec = wfdb.rdrecord(str(Path(tmp) / k), channels=[0])
            out[k] = h10._poly(1000.0 * rec.p_signal[:, 0], rec.fs, h10.FS).astype(np.float32)
    return out

