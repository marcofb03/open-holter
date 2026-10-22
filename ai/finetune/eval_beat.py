"""Score trained beat models on the benchmark, exactly where heartKIT is scored.

    python eval_beat.py RUN [RUN ...] [--tag-names aug noaug]   # RUN: a train.py run folder (best.pt)

For every benchmark record with beat labels (MIT-BIH, SVDB, INCART, LTAFDB, LTDB,
the Icentia11k sample) the model classifies the beats found by the shared
detector (the rules' beat positions from ai/benchmark/frontend.py), on the
benchmark's 130 Hz CSV-round-trip signal passed through the nominal H10-like
front end (h10.front_end), i.e. the deterministic val/test path. Its V and S
probabilities then replace heartKIT's in the router's rows, and the heartKIT
family of candidates (V runs from p(V) >= t; PVC burden) is selected on the
tuning split and scored on the held-out split, by the same rule as the router.
This is the augmentation ablation of FINETUNING_PLAN.md: run it on a model
trained with and one trained without --no-augment for the same epochs.

Writes results/ablation.md and results/ablation.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / "benchmark"
sys.path[:0] = [str(HERE), str(BENCH)]
import datasets as ds  # noqa: E402
import findings as F  # noqa: E402
import frontend as fe  # noqa: E402
import h10  # noqa: E402
import models  # noqa: E402
import router as R  # noqa: E402

BEAT_DBS = ["mitdb", "svdb", "incartdb", "ltafdb", "ltdb", "icentia11k"]
LABEL = np.array(["N", "S", "V", "Q"])


_NETS: dict = {}


def _init(runs: list[str], names: list[str]) -> None:
    import torch

    torch.set_num_threads(1)
    for run, name in zip(runs, names):
        ck = torch.load(Path(run) / "best.pt", map_location="cpu", weights_only=True)
        net = models.build("beat")
        net.load_state_dict(ck["model"])
        _NETS[name] = net.eval()


def record_probs(job: tuple[ds.Rec, np.ndarray]) -> str:
    """Beat probabilities of every model for one record (signal read and converted once), cached."""
    import torch

    r, idx = job
    outs = {n: fe.CACHE / "beatnet" / n / r.db / f"{r.rec}.npy" for n in _NETS}
    if all(o.exists() for o in outs.values()):
        return r.key
    x, fs, _ = ds.read_signal(r)
    ecg = h10.front_end(fe.to_h10(x, fs) * 1000.0)  # benchmark signal -> H10-like uV
    for name, net in _NETS.items():
        p = []
        for i in range(0, len(idx), 4096):
            fr = models.preprocess("beat", h10.frames_130(ecg, idx[i:i + 4096]))
            with torch.inference_mode():
                p.append(torch.softmax(net(torch.from_numpy(fr)), 1).numpy())
        p = np.concatenate(p) if p else np.zeros((0, models.N_BEAT))
        outs[name].parent.mkdir(parents=True, exist_ok=True)
        np.save(outs[name], p.astype(np.float16))
    if r.db == "icentia11k":
        for e in (".hea", ".dat", ".atr"):
            (ds.DATA / "icentia11k" / f"{r.rec}{e}").unlink(missing_ok=True)
    return r.key


def best_single(ev: R.Evaluator, f: str, idx: dict) -> dict:
    cands = [c for c in R.candidates(f) if c.mode == "atom" and c.parts[0].fam == "heartkit"]
    scored = [(c, ev.score(f, c, idx["tune"])) for c in cands]
    b = max(R._num(s["Se"], 0.0) for _, s in scored)
    c, s = min(scored, key=lambda x: R.rank_key(x[1], x[0], b))
    return {"cand": c.label(), "tune": s, "held": ev.score(f, c, idx["held"]), "long": ev.score(f, c, idx["long"])}


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--tag-names", nargs="+", default=None)
    ap.add_argument("--out", type=Path, default=HERE / "results")
    ap.add_argument("-j", "--jobs", type=int, default=4)
    args = ap.parse_args()
    names = args.tag_names or [p.name for p in args.runs]
    recs = [r for r in ds.records(BEAT_DBS) if fe.cache_path(r).exists()]
    rows = [R.prepare(r) for r in recs]
    idx = {"tune": [i for i, x in enumerate(rows) if x["split"] == "tune"],
           "held": [i for i, x in enumerate(rows) if x["split"] == "held"]}
    idx["long"] = [i for i in idx["held"] if rows[i]["long"]]
    res = {"heartKIT (published)": {f: best_single(R.Evaluator(rows), f, idx) for f in ("vt", "pvc")}}
    jobs = [(r, np.round(row["bt"] * fe.FS).astype(np.int64)) for r, row in zip(recs, rows)]
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(args.jobs, initializer=_init, initargs=([str(p) for p in args.runs], names)) as ex:
        for k, key in enumerate(ex.map(record_probs, jobs, chunksize=4)):
            if k % 100 == 0:
                print(f"  {k}/{len(jobs)} {key}", file=sys.stderr, flush=True)
    for run, name in zip(args.runs, names):
        sub = []
        for r, row in zip(recs, rows):
            p = np.load(fe.CACHE / "beatnet" / name / r.db / f"{r.rec}.npy").astype(np.float32)
            sub.append({**row, "pV": p[:, 2], "pS": p[:, 1], "hk_lab": LABEL[p.argmax(1)]})
        res[name] = {f: best_single(R.Evaluator(sub), f, idx) for f in ("vt", "pvc")}
        res[name]["val"] = torch.load(run / "best.pt", map_location="cpu", weights_only=True).get("val")
        print(name, json.dumps({f: res[name][f]["held"]["Se"] for f in ("vt", "pvc")}), file=sys.stderr)
    args.out.mkdir(exist_ok=True)
    (args.out / "ablation.json").write_text(json.dumps(res, indent=1, default=float))
    L = ["<!-- generated by ai/finetune/eval_beat.py -->", "",
         "| beat model | VT choice (tuning) | VT held-out per-patient Se [95 % CI] | VT false / day (long) | "
         "PVC choice (tuning) | PVC held-out Se [95 % CI] | PVC PPV | PVC false / day (long) |",
         "|---|---|---:|---:|---|---:|---:|---:|"]
    for name, d in res.items():
        v, p = d["vt"], d["pvc"]
        L.append(f"| {name} | {v['cand']} | {R.se_cell(v['held'])} | {R.fmt(v['long']['false_per_day'], 1)} | "
                 f"{p['cand']} | {R.se_cell(p['held'])} | {R.fmt(p['held']['PPV'], 2)} | "
                 f"{R.fmt(p['long']['false_per_day'], 2)} |")
    (args.out / "ablation.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
