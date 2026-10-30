"""Readers for the open sources added for the rhythm transformer (../TRANSFORMER_PLAN.md).

Same contract as ai/finetune/sources.py:
    patients(source) -> [(patient, split, payload)]
    read_patient(source, patient, payload, tmp) -> [(namespace, raw16 int16 uV, fs, ref)]
ref is the uniform reference of ai/benchmark/datasets.py (dur_s, beats_t, beats_c, rhythm,
has_beats, has_rhythm) plus, where a source has them:
    "statements"  {target: 1 / 0 / -1} for the whole record (SPH, Chapman-Shaoxing-Ningbo)
    "alarm"       {"t": onset s, "target": "vt" | "pause" | "brady" | "tachy", "true": bool}
                  (VTaC, PhysioNet/CinC Challenge 2015)
    "quality"     [(t0, t1, cls)] consensus signal quality, cls 1 good / 2 QRS only / 3 unreadable (BUT QDB)

Where the raw files come from:
    S3 mirror (streamed per patient): edb, sddb, butqdb, chapman
    a local copy (OH_LOCAL_DATA, default data/): vtac, challenge-2015, ltstdb (physionet.org
        only, ~9 KB/s per connection: fetched once in the background, see TRANSFORMER_PLAN.md),
        iridia-af (Zenodo zip) and sph (figshare tar.gz): fetch_bulk.py
Splits: VTaC keeps its own train/val/test; every other source 80/10/10 by patient
(sha256 of "<source>:<patient>"), fixed before any training. None of these sources is in
the long-term benchmark, so its held-out patients stay untouched.
"""

from __future__ import annotations

import csv
import io
import os
import re
import shutil
import sys
import urllib.error
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
AI = HERE.parent
sys.path[:0] = [str(AI / "finetune"), str(AI / "benchmark"), str(AI.parent / "analysis" / "tools")]
import datasets as bds  # noqa: E402
import fetch_physionet as fp  # noqa: E402
import findings as F  # noqa: E402
import h10  # noqa: E402
import sources as fsrc  # noqa: E402

S3 = "https://physionet-open.s3.amazonaws.com/"
LOCAL = Path(os.environ.get("OH_LOCAL_DATA", AI.parent / "data"))
SOURCES = ("ltstdb", "edb", "sddb", "vtac", "challenge-2015", "butqdb", "iridia-af", "sph", "chapman")
VERSION = {"ltstdb": "1.0.0", "edb": "1.0.0", "sddb": "1.0.0", "butqdb": "1.0.0", "chapman": "1.0.0"}
S3_DIR = {"edb": "edb/1.0.0", "sddb": "sddb/1.0.0", "butqdb": "butqdb/1.0.0", "chapman": "ecg-arrhythmia/1.0.0"}

# WFDB Holter databases with reviewed beat labels: register them with the benchmark's reference
# builder (process-local; the benchmark itself never lists them). SDDB: VT/VF episodes from the
# rhythm marks; records without a reviewed .atr keep only the VF onset from the header.
for _db in ("ltstdb", "edb", "sddb"):
    bds.DBS.setdefault(_db, dict(lead=0, beats=True, rhythm=True, split="train"))
F.VT_RHYTHM_DBS.add("sddb")

ALARM_2015 = {"Asystole": "pause", "Bradycardia": "brady", "Tachycardia": "tachy",  # header names (checked)
              "Ventricular_Tachycardia": "vt", "Ventricular_Flutter_Fib": "vt"}
ALARM_T = 300.0  # VTaC and Challenge 2015: the alarm goes off 5 min into the record

# Record-level statements -> segment targets (1 present, 0 absent: the statements are exhaustive
# for these rhythms; vt / pause are never stated, so they stay unknown).
SPH_MAP = {"af": {"50", "51"}, "svt": {"54"}, "pvc": {"60"}, "pac": {"30", "36"},
           "avb": {"83", "84", "85", "86", "87", "88"}}
CHAPMAN_MAP = {"af": {"164889003", "164890007"}, "svt": {"426761007", "713422000", "233896004", "233897008"},
               "pvc": {"17338001", "11157007"}, "pac": {"284470004", "251173003", "251164006"},
               "avb": {"195042002", "54016002", "28189009", "27885002"}}
CHAPMAN_AMBIGUOUS_AVB = "233917008"  # "atrioventricular block" without a degree: avb unknown


def local(source: str) -> Path:
    return LOCAL / {"challenge-2015": "challenge-2015", "iridia-af": "iridia-af", "sph": "sph"}.get(source, source)


def records_dir(source: str) -> Path:
    """IRIDIA-AF / SPH record folder: `records/`, or the folder the release zip unpacks to
    (IRIDIA-AF v1.0.1: `iridia-af-records-v1.0.1/`)."""
    root = local(source)
    if (root / "records").is_dir():
        return root / "records"
    found = sorted(p for p in root.glob(f"{source}-records*") if p.is_dir())
    return found[-1] if found else root / "records"


def _get(url: str) -> bytes:
    return fp._get(url, timeout=600)


def _s3(source: str, rel: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(_get(f"{S3}{S3_DIR[source]}/{rel}"))
    return dst


def _split(source: str, patient: str) -> str:
    return fsrc._three_way(f"{source}:{patient}")


def _empty_ref(dur: float) -> dict:
    return {"dur_s": dur, "beats_t": None, "beats_c": None, "rhythm": [], "has_beats": False, "has_rhythm": False}


def pick_lead(names: list[str], prefer=("II", "MLII", "ECG", "ECG1", "I", "V5", "V4", "MLIII")) -> int | None:
    """Index of the preferred ECG lead; None if the record has no ECG lead at all."""
    up = [n.strip().upper() for n in names]
    for p in prefer:
        if p in up:
            return up.index(p)
    for i, n in enumerate(up):  # any ECG-looking channel, never pulse / pressure / respiration
        if n not in {"PLETH", "ABP", "ART", "RESP", "CVP", "PAP", "ICP"} and not n.startswith(("PLETH", "ABP", "RESP")):
            return i
    return None


# --------------------------------------------------------------------------
# Patient lists
# --------------------------------------------------------------------------

def patients(source: str) -> list[tuple[str, str, object]]:
    if source in ("edb", "sddb"):
        recs = _get(f"{S3}{S3_DIR[source]}/RECORDS").decode().split()
        return [(r, _split(source, r), None) for r in recs]
    if source == "ltstdb":
        recs = (local(source) / "RECORDS").read_text().split()
        return [(r, _split(source, r), None) for r in recs]
    if source == "butqdb":
        recs = _get(f"{S3}{S3_DIR[source]}/RECORDS").decode().split()
        by: dict[str, set[str]] = {}
        for r in recs:  # lines like "100001/100001_ECG" (and _ACC); first three digits: the subject
            rec = r.strip("/").split("/")[0]
            by.setdefault(rec[:3], set()).add(rec)
        return [(p, _split(source, p), sorted(rs)) for p, rs in sorted(by.items())]
    if source == "vtac":
        rows = list(csv.DictReader(io.StringIO((local(source) / "benchmark_data_split.csv").read_text())))
        lab = {r["event"]: r["decision"] for r in csv.DictReader(io.StringIO((local(source) / "event_labels.csv").read_text()))}
        by: dict[str, list] = {}
        split = {}
        for r in rows:
            by.setdefault(r["record"], []).append((r["event"], lab.get(r["event"])))
            split[r["record"]] = {"train": "train", "val": "val", "test": "test"}[r["split"]]
        return [(p, split[p], evs) for p, evs in sorted(by.items())]
    if source == "challenge-2015":
        # the RECORDS list sits in training/ or, depending on how the zip was unpacked, next to it
        lst = local(source) / "training" / "RECORDS"
        recs = (lst if lst.exists() else local(source) / "RECORDS").read_text().split()
        return [(r, _split(source, r), None) for r in recs]
    if source == "iridia-af":
        rows = list(csv.DictReader(io.StringIO(next(local(source).glob("iridia-af-metadata*.csv")).read_text())))
        by = {}
        for r in rows:
            by.setdefault(r["patient_id"], []).append(r["record_id"])
        return [(p, _split(source, p), rs) for p, rs in sorted(by.items())]
    if source == "sph":
        rows = list(csv.DictReader(io.StringIO((local(source) / "metadata.csv").read_text())))
        by = {}
        for r in rows:
            by.setdefault(r["Patient_ID"], []).append((r["ECG_ID"], r["AHA_Code"]))
        return [(p, _split(source, p), rs) for p, rs in sorted(by.items())]
    if source == "chapman":
        recs = _get(f"{S3}{S3_DIR[source]}/RECORDS").decode().split()
        # RECORDS lists 452 folders of ~100 ECGs (WFDBRecords/01/010/); no patient ids are published,
        # so a folder is the unit of the split and of a preprocessing job
        return [(f.strip("/"), _split(source, f.strip("/")), None) for f in recs]
    raise ValueError(source)


# --------------------------------------------------------------------------
# Parsers (pure functions, tested offline)
# --------------------------------------------------------------------------

def parse_2015_header(text: str) -> tuple[str | None, bool | None]:
    """Challenge 2015 .hea comments: '#Asystole' and '#True alarm' / '#False alarm'."""
    lines = [ln[1:].strip() for ln in text.splitlines() if ln.startswith("#")]
    kind = next((ln for ln in lines if ln in ALARM_2015), None)
    truth = next((ln.lower().startswith("true") for ln in lines if ln.lower().endswith("alarm")), None)
    return kind, truth


def butqdb_quality(csv_text: str, fs: float) -> list[tuple[float, float, int]]:
    """Consensus columns (10-12) of a BUT QDB *_ANN.csv: [(t0 s, t1 s, class)], class 0 dropped."""
    out = []
    for row in csv.reader(io.StringIO(csv_text)):
        if len(row) < 12 or not row[9].strip():
            continue
        a, b, c = int(float(row[9])), int(float(row[10])), int(float(row[11]))
        if c in (1, 2, 3):
            out.append(((a - 1) / fs, b / fs, c))
    return sorted(out)


def statements(codes: set[str], mapping: dict[str, set[str]], ambiguous_avb: str | None = None) -> dict[str, int]:
    st = {k: int(bool(codes & v)) for k, v in mapping.items()}
    if ambiguous_avb and ambiguous_avb in codes and not st["avb"]:
        st["avb"] = -1
    return st


def sph_codes(aha: str) -> set[str]:
    """'50+346;60' -> {'50', '60'} (primary statements; '+nnn' modifiers dropped)."""
    return {c.split("+")[0].strip() for c in aha.split(";") if c.strip()}


def chapman_codes(header: str) -> set[str]:
    for ln in header.splitlines():
        if ln.startswith("#Dx:"):
            return {c.strip() for c in ln[4:].split(",") if c.strip()}
    return set()


def vf_onset(header: str) -> float | None:
    """SDDB header comment '#vfon: 07:54:33' -> seconds from the record start (start time on line 1)."""
    first = header.splitlines()[0].split()
    t0 = first[4] if len(first) > 4 else None
    for ln in header.splitlines():
        if ln.lower().startswith("#vfon:"):
            def sec(s):
                h, m, x = (float(v) for v in s.split(":"))
                return h * 3600 + m * 60 + x
            dt = sec(ln.split(":", 1)[1].strip()) - (sec(t0) if t0 else 0.0)
            return dt % 86400
    return None


# --------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------

def _rec_lead(path: Path, prefer=None):
    import wfdb

    h = wfdb.rdheader(str(path))
    ch = pick_lead(h.sig_name, prefer) if prefer else pick_lead(h.sig_name)
    if ch is None:
        return None, None, h
    rec = wfdb.rdrecord(str(path), channels=[ch])
    return h10.raw16(rec.p_signal[:, 0]), float(rec.fs), h


def read_patient(source: str, patient: str, payload, tmp: Path) -> list[tuple[str, np.ndarray, float, dict]]:
    import wfdb

    d = tmp / f"{source}_{patient.replace('/', '_')}"
    d.mkdir(parents=True, exist_ok=True)
    try:
        if source in ("edb", "sddb", "ltstdb"):
            if source == "ltstdb":
                base = local(source) / patient
            else:
                for e in (".hea", ".dat"):
                    _s3(source, patient + e, d / (patient + e))
                try:
                    _s3(source, patient + ".atr", d / (patient + ".atr"))
                except urllib.error.HTTPError as err:
                    if err.code != 404:
                        raise
                base = d / patient
            x16, fs, h = _rec_lead(base, prefer=("MLII", "ECG", "II", "V4", "V5", "MLIII", "V1"))
            if x16 is None:
                return []
            if base.with_suffix(".atr").exists():
                ref = bds.reference_from_ann(source, wfdb.rdann(str(base), "atr"), fs, len(x16))
            else:  # SDDB without reviewed beats: VF from the onset in the header
                ref = _empty_ref(len(x16) / fs)
                on = vf_onset(base.with_suffix(".hea").read_text()) if source == "sddb" else None
                if on is not None and on < ref["dur_s"]:
                    ref.update(rhythm=[(0.0, on, "N"), (on, ref["dur_s"], "VF")], has_rhythm=True)
            return [(source, x16, fs, ref)]
        if source == "vtac":
            out = []
            for event, decision in payload:
                if decision not in ("True", "False"):
                    continue
                x16, fs, _ = _rec_lead(local(source) / "waveforms" / patient / event)
                if x16 is None:
                    continue
                ref = _empty_ref(len(x16) / fs) | {"alarm": {"t": ALARM_T, "target": "vt", "true": decision == "True"}}
                out.append(("vtac", x16, fs, ref))
            return out
        if source == "challenge-2015":
            base = local(source) / "training" / patient
            kind, truth = parse_2015_header(base.with_suffix(".hea").read_text())
            x16, fs, _ = _rec_lead(base)
            if x16 is None or kind is None or truth is None:
                return []
            return [("challenge-2015", x16, fs,
                     _empty_ref(len(x16) / fs) | {"alarm": {"t": ALARM_T, "target": ALARM_2015[kind], "true": truth}})]
        if source == "butqdb":
            out = []
            for r in payload:
                for e in ("_ECG.hea", "_ECG.dat", "_ANN.csv"):
                    _s3(source, f"{r}/{r}{e}", d / f"{r}{e}")
                x16, fs, _ = _rec_lead(d / f"{r}_ECG")
                q = butqdb_quality((d / f"{r}_ANN.csv").read_text(), fs)
                ref = _empty_ref(len(x16) / fs) | {"quality": q, "rhythm": [(a, b, "NOISE") for a, b, c in q if c == 3]}
                out.append(("butqdb", x16, fs, ref))
            return out
        if source == "iridia-af":
            import h5py

            out = []
            for rec in payload:
                folder = records_dir(source) / rec
                labels = list(csv.DictReader(io.StringIO(next(folder.glob("*ecg_labels.csv")).read_text())))
                for k, f in enumerate(sorted(folder.glob("*_ecg_*.h5"))):
                    with h5py.File(f, "r") as hf:
                        a = np.asarray(hf[list(hf.keys())[0]][:], np.float64)
                    lead = a[:, 0] if a.ndim == 2 and a.shape[0] > a.shape[1] else (a[0] if a.ndim == 2 else a)
                    fs = 200.0
                    dur = len(lead) / fs
                    af = []
                    for r in labels:
                        s0, s1 = int(r["start_file_index"]), int(r["end_file_index"])
                        if s0 <= k <= s1:
                            t0 = int(r["start_qrs_index"]) / fs if s0 == k else 0.0
                            t1 = int(r["end_qrs_index"]) / fs if s1 == k else dur
                            af.append((t0, t1, "AF"))
                    ref = _empty_ref(dur) | {"rhythm": sorted(af), "has_rhythm": True}
                    out.append(("iridia-af", h10.raw16(_to_mv(lead)), fs, ref))
            return out
        if source == "sph":
            import h5py

            out = []
            for ecg_id, aha in payload:
                with h5py.File(records_dir(source) / f"{ecg_id}.h5", "r") as hf:
                    a = np.asarray(hf[list(hf.keys())[0]][:], np.float64)
                lead = a[1] if a.shape[0] == 12 else a[:, 1]  # lead II
                fs = 500.0
                ref = _empty_ref(len(lead) / fs) | {"statements": statements(sph_codes(aha), SPH_MAP)}
                out.append(("sph", h10.raw16(_to_mv(lead)), fs, ref))
            return out
        if source == "chapman":
            out = []
            listing = _get(f"{S3}?list-type=2&prefix={S3_DIR[source]}/{patient}/").decode()
            names = sorted({k.rsplit("/", 1)[1].rsplit(".", 1)[0] for k in re.findall(r"<Key>([^<]+\.hea)</Key>", listing)})
            for nm in names:
                for e in (".hea", ".mat"):
                    _s3(source, f"{patient}/{nm}{e}", d / f"{nm}{e}")
                x16, fs, _ = _rec_lead(d / nm, prefer=("II", "I"))
                if x16 is None:
                    continue
                st = statements(chapman_codes((d / f"{nm}.hea").read_text()), CHAPMAN_MAP, CHAPMAN_AMBIGUOUS_AVB)
                out.append(("chapman", x16, fs, _empty_ref(len(x16) / fs) | {"statements": st}))
            return out
        raise ValueError(source)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _to_mv(x: np.ndarray) -> np.ndarray:
    """Bulk h5 files state no unit: values of ECG size in µV (or raw ADC counts) are scaled to mV.
    A QRS is ~0.5-3 mV, so a robust amplitude above 50 cannot be mV."""
    x = np.nan_to_num(np.asarray(x, np.float64))
    amp = np.percentile(np.abs(x - np.median(x)), 99)
    return x / 1000.0 if amp > 50 else x
