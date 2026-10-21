import numpy as np
import pytest

from ecgpipe.io import (load_accel_csv, load_ecglogger_csv, load_hr_csv, split_on_gaps,
                        write_ecglogger_csv)

T0_NS = 1_700_000_000_000_000_000


def _rows(n=1300, fs=130.0, gap_at=None, gap_s=5.0):
    """(timestamp_ns, mV) for n samples; optional gap after sample gap_at."""
    t = T0_NS + (np.arange(n) / fs * 1e9).astype(np.int64)
    if gap_at is not None:
        t[gap_at:] += int(gap_s * 1e9)
    v = np.sin(np.arange(n) / 10.0)
    return t, v


def _write(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_without_header(tmp_path):
    t, v = _rows()
    p = _write(tmp_path / "a.csv", [f"{a},{b:.4f},,," for a, b in zip(t, v)])
    rec = load_ecglogger_csv(p)
    assert len(rec.ecg) == len(t)
    assert rec.fs == pytest.approx(130.0, rel=1e-3)
    assert rec.t[0] == pytest.approx(T0_NS / 1e9)
    np.testing.assert_allclose(rec.ecg, np.round(v, 4))


def test_with_header(tmp_path):
    t, v = _rows()
    lines = ["timestamp_ns,ecg_mV,pulse,rr_ms,marker"] + [f"{a},{b:.4f},,," for a, b in zip(t, v)]
    rec = load_ecglogger_csv(_write(tmp_path / "b.csv", lines))
    assert len(rec.ecg) == len(t)


def test_semicolon_and_decimal_comma(tmp_path):
    t, v = _rows()
    lines = ["timestamp_ns;ecg_mV;pulse;rr_ms;marker"]
    lines += [f"{a};{b:.4f};;;".replace(".", ",") for a, b in zip(t, v)]
    rec = load_ecglogger_csv(_write(tmp_path / "c.csv", lines))
    assert len(rec.ecg) == len(t)
    np.testing.assert_allclose(rec.ecg, np.round(v, 4))


def test_sparse_pulse_rr_marker(tmp_path):
    t, v = _rows()
    lines = []
    for i, (a, b) in enumerate(zip(t, v)):
        pulse = "72" if i % 130 == 0 else ""
        rr = "833" if i % 130 == 0 else ""
        mark = "1" if i == 500 else ""
        lines.append(f"{a},{b:.4f},{pulse},{rr},{mark}")
    # A row with only a marker (no ECG value) must not add a sample.
    lines.insert(1100, f"{t[1099] + 1},,,,1")
    rec = load_ecglogger_csv(_write(tmp_path / "d.csv", lines))
    assert len(rec.ecg) == len(t)
    assert len(rec.device_rr) == 10
    assert all(r == 833 for _, r in rec.device_rr)
    assert len(rec.markers) == 2


def test_marker_deduplication(tmp_path):
    t, v = _rows(n=1300 * 2)
    # One press written on several consecutive rows, a second press 10 s later.
    marked = set(range(100, 110)) | {1400}
    lines = [f"{a},{b:.4f},,,{'1' if i in marked else ''}" for i, (a, b) in enumerate(zip(t, v))]
    rec = load_ecglogger_csv(_write(tmp_path / "e.csv", lines))
    assert len(rec.markers) == 2
    assert rec.markers[0] == pytest.approx(t[100] / 1e9)


def test_gap_splitting(tmp_path):
    t, v = _rows(n=130 * 60, gap_at=130 * 30)
    rec = load_ecglogger_csv(_write(tmp_path / "f.csv", [f"{a},{b:.4f},,,{'1' if i == 130 * 40 else ''}"
                                                         for i, (a, b) in enumerate(zip(t, v))]))
    assert rec.fs == pytest.approx(130.0, rel=1e-3)   # the gap does not distort fs
    segs = split_on_gaps(rec)
    assert [len(s.ecg) for s in segs] == [130 * 30, 130 * 30]
    assert segs[0].markers == [] and len(segs[1].markers) == 1


def test_gap_splitting_drops_short_fragments(tmp_path):
    t, v = _rows(n=130 * 40, gap_at=130 * 35)
    rec = load_ecglogger_csv(_write(tmp_path / "g.csv", [f"{a},{b:.4f},,," for a, b in zip(t, v)]))
    segs = split_on_gaps(rec)
    assert len(segs) == 1 and len(segs[0].ecg) == 130 * 35


def test_roundtrip_writer(tmp_path):
    t = 1.7e9 + np.arange(1300) / 130.0
    v = np.cos(np.arange(1300) / 7.0)
    p = tmp_path / "h.csv"
    write_ecglogger_csv(p, t, v, markers=[t[650]])
    rec = load_ecglogger_csv(p)
    np.testing.assert_allclose(rec.ecg, v, atol=1e-4)
    assert rec.markers == [pytest.approx(t[650], abs=1e-6)]


def test_empty_file_raises(tmp_path):
    with pytest.raises(ValueError):
        load_ecglogger_csv(_write(tmp_path / "i.csv", ["timestamp_ns,ecg_mV"]))


def test_side_files(tmp_path):
    acc = _write(tmp_path / "acc.csv", ["timestamp_ns,x_mG,y_mG,z_mG",
                                        f"{T0_NS + 40_000_000},1,2,1000", f"{T0_NS},0,0,1000"])
    a = load_accel_csv(acc)
    assert a.xyz.shape == (2, 3) and a.t[0] < a.t[1]          # sorted by time
    hr = _write(tmp_path / "hr.csv", ["timestamp_ns,hr_bpm,rr_ms,contact,contact_supported",
                                      f"{T0_NS},70,857,true,true",
                                      f"{T0_NS + 10**9},71,840 845,false,true",
                                      f"{T0_NS + 2 * 10**9},,,1,1"])
    h = load_hr_csv(hr)
    assert list(h.contact) == [True, False, True]
    assert h.contact_supported.all()
    assert [r for _, r in h.rr] == [857, 840, 845]
    assert np.isnan(h.hr[2])
