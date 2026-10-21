"""Download the heartKIT BEAT-3-EFF-SM and ARR-2-EFF-SM models (BSD-3-Clause, Ambiq).

    python download_models.py      # -> weights/<model>/{model.tflite,configuration.json}

Source: the heartKIT model zoo on AWS S3 (see https://github.com/AmbiqAI/heartkit).
The files are git-ignored; never commit them. SHA-256 sums are printed so a run
can be tied to exact files ("latest" can change).
"""

from __future__ import annotations

import hashlib
import urllib.request

from hk import MODELS, WEIGHTS

BASE = "https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/heartkit/{task}/{name}/latest/{file}"
TASK = {"beat": "beat", "rhythm": "rhythm"}


def main() -> None:
    for task, name in MODELS.items():
        d = WEIGHTS / name
        d.mkdir(parents=True, exist_ok=True)
        for f in ("model.tflite", "configuration.json"):
            dest = d / f
            if not dest.exists():
                with urllib.request.urlopen(BASE.format(task=TASK[task], name=name, file=f)) as r:
                    dest.write_bytes(r.read())
            print(f"{dest}  {dest.stat().st_size / 1e3:.0f} kB  sha256 {hashlib.sha256(dest.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
