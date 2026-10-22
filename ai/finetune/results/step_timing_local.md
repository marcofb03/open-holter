CPU: Intel(R) Xeon(R) Processor @ 2.80GHz, 4 threads, torch 2.14.1+cpu

| model | batch | fp32 s/step | fp32 examples/s | bf16 s/step | bf16 examples/s | data pipeline ms/example (1 thread) |
|---|---:|---:|---:|---:|---:|---:|
| beat | 512 | 1.80 | 284.5 | 1.57 | 326.9 | 0.09 |
| ecgfounder | 64 | 4.42 | 14.5 | 7.44 | 8.6 | 1.45 |
| clef | 64 | 4.92 | 13.0 | 8.05 | 8.0 | 1.22 |
