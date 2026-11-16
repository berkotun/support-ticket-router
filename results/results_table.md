# Results

Test set: official CLINC150 (plus) test set, 5500 messages (4500 in-scope, 1000 out-of-scope). Typo test: the same messages with keyboard typos (src/typos.py, seed 42).

Hardware: CPU: Intel(R) Xeon(R) CPU @ 2.00GHz | torch threads: 1 | Python 3.13.16 | torch 2.14.1+cu130 | Linux 6.6.159+

Latency: batch size 1, mean over 200 messages after 10 warm-up runs, end to end (normalization + spelling correction + model).

| Model                                                        | Precision    |   In-scope accuracy |   Out-of-scope recall |   Routing accuracy |   In-scope acc. (typo test) |   OOS recall (typo test) |   Threshold tau (from val) |   Size (MB) |   Latency mean (ms) |   Latency p95 (ms) |
|:-------------------------------------------------------------|:-------------|--------------------:|----------------------:|-------------------:|----------------------------:|-------------------------:|---------------------------:|------------:|--------------------:|-------------------:|
| tf-idf + LogReg (classical baseline)                         | FP32         |              0.8529 |                 0.905 |             0.8758 |                      0.8129 |                    0.916 |                       0.65 |        18.7 |               11.28 |              15.2  |
| fastText avg + FFN (own neural net)                          | FP32         |              0.798  |                 0.862 |             0.8324 |                      0.7429 |                    0.865 |                       0.85 |        40.4 |                0.26 |               0.34 |
| BERT-base fine-tuned (Transformer, teacher)                  | FP32         |              0.9067 |                 0.896 |             0.9104 |                      0.8413 |                    0.898 |                       0.85 |       418.8 |              104.39 |             127.97 |
| BERT-base fine-tuned (Transformer, teacher) + INT8 quantized | INT8 dynamic |              0.9131 |                 0.776 |             0.9067 |                      0.8558 |                    0.806 |                       0.35 |       173.9 |               26.46 |              34.42 |
| DistilBERT distilled (student)                               | FP32         |              0.9118 |                 0.863 |             0.9113 |                      0.8456 |                    0.866 |                       0.8  |       256.5 |               52.57 |              66.01 |
| DistilBERT distilled (student) + INT8 quantized              | INT8 dynamic |              0.8971 |                 0.873 |             0.9009 |                      0.8269 |                    0.887 |                       0.75 |       133.1 |               13.72 |              17.9  |
| tf-idf + LogReg, no spell correction (ablation)              | FP32         |              0.8836 |                 0.876 |             0.9047 |                      0.7671 |                    0.89  |                       0.5  |        18.7 |               15.96 |              20.24 |
