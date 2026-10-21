Parameters 30.8 M, 1.14 GMAC per 10 s window.

| model | size MB | max abs Δp vs torch fp32 | mean abs Δp |
|---|---:|---:|---:|
| onnx fp32 | 123 | 0.0002 | 0.0001 |
| onnx int8 dynamic | 31 | 0.0015 | 0.0005 |
| onnx int8 static | 32 | 0.0037 | 0.0010 |

Δp: probability difference on the focus labels, synthetic windows.

CPU latency (Intel(R) Xeon(R) Processor @ 2.10GHz; torch 2.14.1+cu130, onnxruntime 1.31.0):

| runtime | threads | batch | ms / window | core-min per day (8,640 windows) |
|---|---:|---:|---:|---:|
| torch fp32 | 1 | 1 | 51.3 | 7.4 |
| onnx fp32 | 1 | 1 | 47.5 | 6.8 |
| onnx int8 dynamic | 1 | 1 | 138.4 | 19.9 |
| onnx int8 static | 1 | 1 | 24.9 | 3.6 |
| torch fp32 | 1 | 32 | 34.0 | 4.9 |
| onnx fp32 | 1 | 32 | 45.4 | 6.5 |
| onnx int8 dynamic | 1 | 32 | 117.9 | 17.0 |
| onnx int8 static | 1 | 32 | 21.8 | 3.1 |
| torch fp32 | 4 | 1 | 46.0 | 26.5 |
| onnx fp32 | 4 | 1 | 27.7 | 16.0 |
| onnx int8 dynamic | 4 | 1 | 57.6 | 33.1 |
| onnx int8 static | 4 | 1 | 19.1 | 11.0 |
| torch fp32 | 4 | 32 | 12.0 | 6.9 |
| onnx fp32 | 4 | 32 | 14.3 | 8.2 |
| onnx int8 dynamic | 4 | 32 | 53.4 | 30.8 |
| onnx int8 static | 4 | 32 | 12.3 | 7.1 |

Preprocessing (filters + z-score): 8.4 ms / window.
core-min per day = ms/window x 8,640 x threads (an upper bound on CPU time).
