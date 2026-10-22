CPU: AMD EPYC 9654 96-Core Processor, 24 threads, torch 2.14.1+cpu

| model | batch | fp32 s/step | fp32 examples/s | bf16 s/step | bf16 examples/s | data pipeline ms/example (1 thread) |
|---|---:|---:|---:|---:|---:|---:|
| ecgfounder | 64 | 0.54 | 118.2 | 0.36 | 178.1 | 0.82 |
| beat | 512 | 0.58 | 875.2 | 0.54 | 955.4 | 0.04 |
