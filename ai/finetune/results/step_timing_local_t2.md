CPU: Intel(R) Xeon(R) Processor @ 2.80GHz, 2 threads, torch 2.14.1+cpu

| model | batch | fp32 s/step | fp32 examples/s | bf16 s/step | bf16 examples/s | data pipeline ms/example (1 thread) |
|---|---:|---:|---:|---:|---:|---:|
| beat | 512 | 2.88 | 178.1 | 2.59 | 197.6 | 0.10 |
| ecgfounder | 64 | 8.69 | 7.4 | 13.78 | 4.6 | 1.47 |
| clef | 64 | 7.92 | 8.1 | 13.56 | 4.7 | 1.31 |
