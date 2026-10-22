"""Databases of the long-term benchmark: records, patients, tuning/held-out split, references.

Every record is reduced to one lead and a uniform reference:
    beats     (t_s, AAMI class N/S/V/F/Q) when the database has reviewed beat labels
    rhythm    [(t0, t1, code)] with code in AF, SVT, VT, VF, NOISE, N (anything else)
    flags     which findings the annotations can confirm or refute (see FINDINGS in
              findings.py): a database without beat labels cannot say whether a
              detected V run is false, so VT is "not assessable" there

Split (patient level, fixed before any system was run; never changed after):
    tune   MIT-BIH (all 44 non-paced), SVDB, INCART (the previous benchmark's
           development and held-out sets), plus one third of the patients of
           LTAFDB, CPSC2021, VFDB, CUDB and of the Icentia11k sample, chosen by
           sha256("<db>:<patient>") mod 3 == 0
    held   the other two thirds of those, and all of AFDB, LTDB and SHDB-AF
           (external databases with no tuning share at all)
    noise  NSTDB: MIT-BIH 118 and 119 with electrode-motion noise at 24 ... -6 dB
           (PhysioNet's own mixes) plus muscle-artefact mixes made the same way;
           reported as a stress test only (118 and 119 are tuning patients)

Icentia11k: heartKIT's beat model was trained on patients 0-9,999 (heartkit/
datasets/icentia11k.py: get_train_patient_ids = patient_ids[:10_000]; the
rest, 10,000-10,999, is its test set). The sample is 60 patients drawn with a
fixed seed from 10,000-10,999 only, so no system has seen them; the first 10
segments (~70 min each, ~11.7 h) of each patient are used.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / "data"
sys.path[:0] = [str(REPO / "analysis" / "tools")]

AAMI = {**{s: "N" for s in "NLRejnB"}, **{s: "S" for s in "AaJS"}, **{s: "V" for s in "VE"},
        "F": "F", **{s: "Q" for s in "/fQ"}}
PACED = {"102", "104", "107", "217"}
MITDB_DS1 = {"101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122", "124",
             "201", "203", "205", "207", "208", "209", "215", "220", "223", "230"}

# lead: channel name (or index) closest to a chest-strap / lead II view.
# beats: reviewed beat labels; rhythm: rhythm labels (True / "some": per record).
DBS: dict[str, dict] = {
    "mitdb": dict(lead="MLII", beats=True, rhythm=True, split="tune"),
    "svdb": dict(lead="ECG1", beats=True, rhythm=False, split="tune"),
    "incartdb": dict(lead="II", beats=True, rhythm="some", split="tune"),
    "ltafdb": dict(lead=0, beats=True, rhythm=True, split="third"),
    "afdb": dict(lead="ECG1", beats=False, rhythm=True, split="held"),
    "ltdb": dict(lead="ECG1", beats=True, rhythm=False, split="held"),
    "vfdb": dict(lead=0, beats=False, rhythm=True, split="third"),
    "cudb": dict(lead=0, beats=False, rhythm=True, split="third"),
    "cpsc2021": dict(lead="II", beats=False, rhythm=True, split="third"),
    "shdb-af": dict(lead="ECG1", beats=False, rhythm=True, split="held"),
    "icentia11k": dict(lead=0, beats=True, rhythm=True, split="third"),
}
LONG_DBS = {"ltafdb", "afdb", "ltdb", "shdb-af", "icentia11k"}  # >= 10 h per patient
ICENTIA_URL = "https://physionet-open.s3.amazonaws.com/icentia11k-continuous-ecg/1.0/"
ICENTIA_N, ICENTIA_SEGMENTS, ICENTIA_SEED = 60, 10, 20261021
HEARTKIT_TRAIN_MAX = 10_000  # heartKIT trained on Icentia11k patients 0..9,999

SVT_CODES = {"SVTA", "AT", "PAT"}         # NOD only in SHDB-AF (intranodal tachycardia)
AF_CODES = {"AFIB", "AFL", "WPWAF", "AF"}
VT_CODES = {"VT"}
VF_CODES = {"VFL", "VF", "VFIB"}


@dataclass(frozen=True)
class Rec:
    db: str
    rec: str        # path below data/<db>/ without extension (or Icentia "p10xxx/p10xxx_s00")
    patient: str
    split: str      # tune / held / noise

    @property
    def key(self) -> str:
        return f"{self.db}/{self.rec}"


def third(db: str, patient: str) -> str:
    h = int(hashlib.sha256(f"{db}:{patient}".encode()).hexdigest(), 16)
    return "tune" if h % 3 == 0 else "held"


def patient_of(db: str, rec: str) -> str:
    if db == "cpsc2021":            # Training_set_I/data_<patient>_<record>
        return rec.split("/")[-1].split("_")[1]
    if db == "icentia11k":
        return rec.split("/")[0]
    return rec


def icentia_patients() -> list[str]:
    rng = np.random.default_rng(ICENTIA_SEED)
    ids = rng.choice(np.arange(HEARTKIT_TRAIN_MAX, 11_000), ICENTIA_N, replace=False)
    return [f"p{i:05d}" for i in sorted(ids)]


def _local_records(db: str) -> list[str]:
    d = DATA / db
    if not (d / "RECORDS").exists():
        from fetch_physionet import fetch

        fetch(db, out=d, aws_first=True)
    names = [r.strip() for r in (d / "RECORDS").read_text().split() if r.strip()]
    # a signal file, or a front-end cache (frontend.py --delete-raw removes the signal once cached);
    # afdb 00735 / 03665 have neither (no signal on PhysioNet)
    cached = DATA / "work" / "bench2" / db
    return [r for r in names if (d / f"{r}.dat").exists() or (cached / f"{r}.pkl").exists()]


NST_SNR = ("24", "18", "12", "06", "00", "_6")


def nstdb_records() -> list[Rec]:
    """PhysioNet's electrode-motion mixes (118e24 ... 119e_6) and our muscle-artefact
    mixes of the same records ('m' instead of 'e', made by nst_mix)."""
    return [Rec("nstdb", f"{p}{k}{snr}", p, "noise") for k in "em" for p in ("118", "119") for snr in NST_SNR]


def records(dbs: list[str] | None = None) -> list[Rec]:
    out = []
    for db in dbs or list(DBS) + ["nstdb"]:
        if db == "nstdb":
            out += nstdb_records()
            continue
        if db == "icentia11k":
            for p in icentia_patients():
                for s in range(ICENTIA_SEGMENTS):
                    out.append(Rec(db, f"{p}/{p}_s{s:02d}", p, third(db, p)))
            continue
        names = _local_records(db)
        if db == "mitdb":
            names = [r for r in names if r not in PACED]
        if db == "shdb-af":  # 98 of 128 recordings have rhythm annotations
            names = [r for r in names if (DATA / db / f"{r}.atr").exists()]
        rule = DBS[db]["split"]
        for r in names:
            p = patient_of(db, r)
            out.append(Rec(db, r, p, third(db, p) if rule == "third" else rule))
    return out


def subset(r: Rec) -> str:
    """Reporting group: database, MIT-BIH split into DS1/DS2 as before."""
    if r.db == "mitdb":
        return "mitdb-DS1" if r.rec in MITDB_DS1 else "mitdb-DS2"
    return r.db


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------

def fetch_icentia(rec: str, out: Path | None = None) -> Path:
    """Download one Icentia11k segment (.hea/.dat/.atr, ~2.2 MB) if missing."""
    out = out or DATA / "icentia11k"
    p = rec.split("/")[0]
    for e in (".hea", ".atr", ".dat"):
        dst = out / f"{rec}{e}"
        if dst.exists() and dst.stat().st_size:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        url = f"{ICENTIA_URL}{p[:3]}/{rec}{e}"
        with urllib.request.urlopen(url, timeout=300) as resp:
            data = resp.read()
        tmp = dst.with_suffix(e + ".part")
        tmp.write_bytes(data)
        tmp.rename(dst)
    return out / rec


def path_of(r: Rec) -> Path:
    if r.db == "icentia11k":
        return fetch_icentia(r.rec)
    return DATA / r.db / r.rec


def read_signal(r: Rec) -> tuple[np.ndarray, float, str]:
    import wfdb

    p = path_of(r)
    h = wfdb.rdheader(str(p))
    lead = DBS[r.db]["lead"]
    ch = h.sig_name.index(lead) if isinstance(lead, str) and lead in h.sig_name else (lead if isinstance(lead, int) else 0)
    rec = wfdb.rdrecord(str(p), channels=[ch])
    return np.nan_to_num(rec.p_signal[:, 0]), float(rec.fs), h.sig_name[ch]


def nst_mix(clean: np.ndarray, noise: np.ndarray, fs: float, beat_samples: np.ndarray,
            snr_db: float) -> np.ndarray:
    """Add NSTDB noise as nstdbgen does: from minute 5, 2 min on / 2 min off, scaled so
    that 10 log10(S / N) = snr_db with S = mean squared QRS peak-to-peak amplitude / 8
    and N = mean squared noise (both after removing the mean)."""
    from scipy.signal import butter, sosfiltfilt

    hp = sosfiltfilt(butter(2, 0.5, "highpass", fs=fs, output="sos"), clean)
    h = int(0.05 * fs)
    pp = [np.ptp(hp[max(b - h, 0): b + h]) for b in beat_samples if h < b < len(hp) - h]
    s = np.mean(np.square(pp)) / 8
    nz = np.resize(noise - noise.mean(), len(clean))
    gain = np.sqrt(s / (np.mean(nz ** 2) * 10 ** (snr_db / 10)))
    on = np.zeros(len(clean), bool)
    for a in range(int(300 * fs), len(clean), int(240 * fs)):
        on[a: a + int(120 * fs)] = True
    return clean + gain * nz * on


# --------------------------------------------------------------------------
# Reference annotations
# --------------------------------------------------------------------------

def _code(db: str, aux: str) -> str:
    c = aux.strip("\x00").strip().lstrip("(").upper()
    if c in AF_CODES:
        return "AF"
    if c in SVT_CODES or (db == "shdb-af" and c == "NOD"):
        return "SVT"
    if c in VT_CODES:
        return "VT"
    if c in VF_CODES:
        return "VF"
    if c.startswith("NOISE"):
        return "NOISE"
    return "N"


def reference(r: Rec, fs: float, n_samples: int) -> dict:
    """Uniform reference of one record (times in s)."""
    import wfdb

    return reference_from_ann(r.db, wfdb.rdann(str(path_of(r)), "atr"), fs, n_samples)


def reference_from_ann(db: str, ann, fs: float, n_samples: int) -> dict:
    """Uniform reference from a wfdb annotation of database db (see DBS)."""
    spec = DBS[db]
    sym = np.asarray(ann.symbol)
    t = np.asarray(ann.sample) / fs
    aux = [a.strip("\x00").strip() if a else "" for a in ann.aux_note]
    dur = n_samples / fs

    # Rhythm change points: any annotation whose aux starts with "(" (PhysioNet "+"
    # rhythm marks; SHDB-AF puts them on '"' comments), ")" closes (Icentia).
    marks = [(ti, _code(db, a)) for ti, a in zip(t, aux) if a.startswith("(")]
    marks += [(ti, "N") for ti, a in zip(t, aux) if a == ")"]
    marks.sort()
    rhythm = []
    for i, (t0, c) in enumerate(marks):
        t1 = marks[i + 1][0] if i + 1 < len(marks) else dur
        if t1 > t0:
            rhythm.append((float(t0), float(t1), c))
    # VF episodes marked '[' ... ']' (MIT-BIH 207, CUDB)
    for k in np.where(sym == "[")[0]:
        ends = np.where((sym == "]") & (np.arange(len(sym)) > k))[0]
        rhythm.append((float(t[k]), float(t[ends[0]]) if ends.size else dur, "VF"))
    # Signal-quality marks '~' (MIT-BIH format): subtype != 0 = some channel noisy/unreadable,
    # 0 = clean again. Beats are often not annotated in between, so these spans must not
    # create reference pauses (findings.py) - LTDB has several.
    start = None
    sub = np.asarray(ann.subtype) if ann.subtype is not None else np.zeros(len(sym), int)
    for k in np.where(sym == "~")[0]:
        if sub[k] != 0 and start is None:
            start = t[k]
        elif sub[k] == 0 and start is not None:
            rhythm.append((float(start), float(t[k]), "NOISE"))
            start = None
    if start is not None:
        rhythm.append((float(start), dur, "NOISE"))
    has_rhythm = bool(marks) if spec["rhythm"] == "some" else bool(spec["rhythm"])

    beats_t = beats_c = None
    if spec["beats"]:
        isb = np.isin(sym, list(AAMI))
        beats_t, beats_c = t[isb], np.array([AAMI[s] for s in sym[isb]], dtype="<U1")
    return {"dur_s": dur, "beats_t": beats_t, "beats_c": beats_c, "rhythm": sorted(rhythm),
            "has_beats": beats_t is not None, "has_rhythm": has_rhythm}

