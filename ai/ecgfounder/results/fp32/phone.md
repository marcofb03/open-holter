Parameters 30.8 M, 1.14 GMAC per 10 s window; checkpoint file 370 MB.

| model | size MB | max abs Δp vs torch fp32 | mean abs Δp |
|---|---:|---:|---:|
| onnx fp32 | 123 | 0.0000 | 0.0000 |
| onnx int8 dynamic | 31 | 0.7630 | 0.0408 |
| onnx int8 static | 32 | 0.7586 | 0.0346 |

Δp: probability difference on the focus labels, MIT-BIH (calibration DS1, test DS2, H10-like) windows.

CPU latency (Intel(R) Xeon(R) Processor @ 2.10GHz; torch 2.14.1+cu130, onnxruntime 1.31.0):

| runtime | threads | batch | ms / window | core-min per day (8,640 windows) |
|---|---:|---:|---:|---:|
| torch fp32 | 1 | 1 | 61.7 | 8.9 |
| onnx fp32 | 1 | 1 | 47.9 | 6.9 |
| onnx int8 dynamic | 1 | 1 | 159.2 | 22.9 |
| onnx int8 static | 1 | 1 | 18.8 | 2.7 |
| torch fp32 | 1 | 32 | 34.7 | 5.0 |
| onnx fp32 | 1 | 32 | 40.7 | 5.9 |
| onnx int8 dynamic | 1 | 32 | 148.2 | 21.3 |
| onnx int8 static | 1 | 32 | 19.5 | 2.8 |
| torch fp32 | 4 | 1 | 37.5 | 21.6 |
| onnx fp32 | 4 | 1 | 20.6 | 11.9 |
| onnx int8 dynamic | 4 | 1 | 65.2 | 37.5 |
| onnx int8 static | 4 | 1 | 17.7 | 10.2 |
| torch fp32 | 4 | 32 | 10.7 | 6.2 |
| onnx fp32 | 4 | 32 | 14.5 | 8.4 |
| onnx int8 dynamic | 4 | 32 | 70.6 | 40.7 |
| onnx int8 static | 4 | 32 | 9.0 | 5.2 |

Preprocessing (filters + z-score): 10.9 ms / window.
core-min per day = ms/window x 8,640 x threads (an upper bound on CPU time).
