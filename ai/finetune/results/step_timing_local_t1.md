CPU: Intel(R) Xeon(R) Processor @ 2.80GHz, 1 threads, torch 2.14.1+cpu

| model | batch | fp32 s/step | fp32 examples/s | bf16 s/step | bf16 examples/s | data pipeline ms/example (1 thread) |
|---|---:|---:|---:|---:|---:|---:|
| beat | 512 | 4.50 | 113.8 | 4.20 | 121.9 | 0.11 |
| ecgfounder | 64 | 14.95 | 4.3 | 24.86 | 2.6 | 1.56 |
| clef | 64 | 14.37 | 4.5 | 24.83 | 2.6 | 1.17 |
