# Results

Threshold: 95th percentile of validation-normal scores (target false-positive rate 5%). Mean ± std over folds and seeds.

## Protocol: `within_recording`

| Model | ROC-AUC | Test FPR (target 0.05) | Detection rate |
|---|---|---|---|
| RMSBaseline | 1.000 ± 0.000 | 0.049 ± 0.000 | 1.000 ± 0.000 |
| IsolationForest | 1.000 ± 0.000 | 0.039 ± 0.012 | 1.000 ± 0.000 |
| OneClassSVM | 1.000 ± 0.000 | 0.099 ± 0.000 | 1.000 ± 0.000 |
| SpectralAutoencoder | 1.000 ± 0.000 | 0.053 ± 0.003 | 1.000 ± 0.000 |

Detection rate per fault (type + diameter in mils):

| model               |   IR007 |   B007 |   OR007 |   IR014 |   B014 |   OR014 |   IR021 |   B021 |   OR021 |
|:--------------------|--------:|-------:|--------:|--------:|-------:|--------:|--------:|-------:|--------:|
| RMSBaseline         |       1 |      1 |       1 |       1 |      1 |       1 |       1 |      1 |       1 |
| IsolationForest     |       1 |      1 |       1 |       1 |      1 |       1 |       1 |      1 |       1 |
| OneClassSVM         |       1 |      1 |       1 |       1 |      1 |       1 |       1 |      1 |       1 |
| SpectralAutoencoder |       1 |      1 |       1 |       1 |      1 |       1 |       1 |      1 |       1 |

## Protocol: `cross_load`

| Model | ROC-AUC | Test FPR (target 0.05) | Detection rate |
|---|---|---|---|
| RMSBaseline | 1.000 ± 0.000 | 0.251 ± 0.432 | 1.000 ± 0.000 |
| IsolationForest | 1.000 ± 0.000 | 0.287 ± 0.412 | 1.000 ± 0.000 |
| OneClassSVM | 1.000 ± 0.000 | 0.424 ± 0.362 | 1.000 ± 0.000 |
| SpectralAutoencoder | 1.000 ± 0.000 | 0.877 ± 0.133 | 1.000 ± 0.000 |

Detection rate per fault (type + diameter in mils):

| model               |   IR007 |   B007 |   OR007 |   IR014 |   B014 |   OR014 |   IR021 |   B021 |   OR021 |
|:--------------------|--------:|-------:|--------:|--------:|-------:|--------:|--------:|-------:|--------:|
| RMSBaseline         |       1 |      1 |       1 |       1 |      1 |       1 |       1 |      1 |       1 |
| IsolationForest     |       1 |      1 |       1 |       1 |      1 |       1 |       1 |      1 |       1 |
| OneClassSVM         |       1 |      1 |       1 |       1 |      1 |       1 |       1 |      1 |       1 |
| SpectralAutoencoder |       1 |      1 |       1 |       1 |      1 |       1 |       1 |      1 |       1 |

## Per held-out load (`cross_load`)

|                                        |   roc_auc |   test_fpr |   detection_rate |
|:---------------------------------------|----------:|-----------:|-----------------:|
| ('RMSBaseline', 'held_out_L0')         |     1     |      1     |                1 |
| ('IsolationForest', 'held_out_L0')     |     1     |      1     |                1 |
| ('OneClassSVM', 'held_out_L0')         |     1     |      1     |                1 |
| ('SpectralAutoencoder', 'held_out_L0') |     1     |      1     |                1 |
| ('RMSBaseline', 'held_out_L1')         |     1     |      0.004 |                1 |
| ('IsolationForest', 'held_out_L1')     |     0.999 |      0.059 |                1 |
| ('OneClassSVM', 'held_out_L1')         |     1     |      0.449 |                1 |
| ('SpectralAutoencoder', 'held_out_L1') |     1     |      0.956 |                1 |
| ('RMSBaseline', 'held_out_L2')         |     1     |      0     |                1 |
| ('IsolationForest', 'held_out_L2')     |     1     |      0.023 |                1 |
| ('OneClassSVM', 'held_out_L2')         |     1     |      0.055 |                1 |
| ('SpectralAutoencoder', 'held_out_L2') |     1     |      0.66  |                1 |
| ('RMSBaseline', 'held_out_L3')         |     1     |      0     |                1 |
| ('IsolationForest', 'held_out_L3')     |     1     |      0.068 |                1 |
| ('OneClassSVM', 'held_out_L3')         |     1     |      0.19  |                1 |
| ('SpectralAutoencoder', 'held_out_L3') |     1     |      0.892 |                1 |
