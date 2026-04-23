# 2D Load Test After Kafka Migration

Test setup:

- endpoint: `POST /api/v1/analyze/2d`
- target URL: `http://localhost:8000/api/v1/analyze/2d`
- test image: `data/train/14_002_5_0016.bmp`
- `pixel_spacing_mm = 1.0`
- `confidence_threshold = 0.25`
- queue backend: `Kafka`

## Current results

| RPS | Requests | Mean response, s | Min, s | Max, s | Success rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.2 | 4 | 1.627 | 1.484 | 1.802 | 4/4 |
| 0.5 | 4 | 1.521 | 1.290 | 1.680 | 4/4 |
| 1.0 | 4 | 1.477 | 1.312 | 1.606 | 4/4 |
| 2.0 | 8 | 1.498 | 1.227 | 1.656 | 8/8 |
| 4.0 | 8 | 1.460 | 1.344 | 1.666 | 8/8 |
| 6.0 | 8 | 1.515 | 1.466 | 1.561 | 8/8 |

## Comparison With The Pre-Kafka Baseline

| RPS | Mean before Kafka, s | Mean after Kafka, s | Delta, s |
| --- | ---: | ---: | ---: |
| 0.2 | 1.671 | 1.627 | -0.044 |
| 0.5 | 1.475 | 1.521 | +0.046 |
| 1.0 | 1.440 | 1.477 | +0.037 |
| 2.0 | 1.409 | 1.498 | +0.089 |
| 4.0 | 1.445 | 1.460 | +0.015 |
| 6.0 | n/a | 1.515 | n/a |

## Short conclusion

The 2D contour remained stable after the migration from `Redis` to `Kafka`: all `36/36` requests completed successfully. The average response time stayed in the `1.46-1.63 s` range across the tested load levels, and no failures were observed up to `6 RPS`.
