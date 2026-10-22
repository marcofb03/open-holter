CPU: AMD EPYC 9654 96-Core Processor, 48 threads, torch 2.14.1+cpu

| model | batch | fp32 s/step | fp32 examples/s | bf16 s/step | bf16 examples/s | data pipeline ms/example (1 thread) |
|---|---:|---:|---:|---:|---:|---:|
| beat | 512 | 1.48 | 346.4 | 4.42 | 115.9 | 0.09 |
| ecgfounder | 64 | 0.56 | 113.3 | 0.35 | 183.1 | 1.21 |
| clef | 64 | 0.55 | 115.6 | 0.36 | 180.1 | 0.42 |
