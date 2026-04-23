## 2D cluster load test after scaling to two nodes

### Test setup

- Target endpoint: `POST /api/v1/analyze/2d`
- Input image: `data/train/14_002_5_0016.bmp`
- Cluster topology: `api-gateway + 2 x 2D nodes`
- Balancing mode: gateway round-robin over `APP_2D_NODE_URLS`
- Requests per scenario: `8`
- Compared modes:
  - `render_artifacts=true`
  - `render_artifacts=false`
- Recorded resources:
  - `stenosis-api`
  - `stenosis-twod-node-1`
  - `stenosis-twod-node-2`

### Table 1. Latency and balancing results

| RPS | Render | Mean, s | Min, s | Max, s | Success | Node 1 | Node 2 |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| 1.0 | on  | 1.313 | 1.245 | 1.454 | 8/8 | 4 | 4 |
| 2.0 | on  | 1.428 | 1.232 | 1.641 | 8/8 | 4 | 4 |
| 4.0 | on  | 1.466 | 1.381 | 1.576 | 8/8 | 4 | 4 |
| 6.0 | on  | 2.032 | 1.809 | 2.273 | 8/8 | 4 | 4 |
| 1.0 | off | 1.059 | 0.980 | 1.176 | 8/8 | 4 | 4 |
| 2.0 | off | 1.115 | 1.043 | 1.186 | 8/8 | 4 | 4 |
| 4.0 | off | 1.197 | 1.027 | 1.272 | 8/8 | 4 | 4 |
| 6.0 | off | 1.599 | 1.350 | 1.954 | 8/8 | 4 | 4 |

### Table 2. Average resource consumption

| RPS | Render | API CPU, % | API RAM, MiB | Node 1 CPU, % | Node 1 RAM, MiB | Node 2 CPU, % | Node 2 RAM, MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0 | on  | 1.137 | 75.577 | 77.953 | 1051.648 | 56.240 | 1081.685 |
| 2.0 | on  | 1.545 | 77.245 | 157.835 | 1054.720 | 318.975 | 1089.024 |
| 4.0 | on  | 2.335 | 79.635 | 338.910 | 1081.344 | 335.355 | 1092.096 |
| 6.0 | on  | 3.725 | 79.990 | 608.090 | 1068.544 | 569.690 | 1100.800 |
| 1.0 | off | 0.870 | 77.777 | 94.870 | 1038.336 | 189.167 | 1073.493 |
| 2.0 | off | 1.630 | 75.095 | 163.100 | 1038.336 | 329.055 | 1093.120 |
| 4.0 | off | 5.770 | 74.410 | 675.690 | 1100.800 | 676.800 | 1098.752 |
| 6.0 | off | 2.610 | 76.380 | 684.380 | 1101.824 | 1019.840 | 1108.992 |

### Table 3. Peak resource consumption

| RPS | Render | API CPU peak, % | API RAM peak, MiB | Node 1 CPU peak, % | Node 1 RAM peak, MiB | Node 2 CPU peak, % | Node 2 RAM peak, MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0 | on  | 1.420 | 76.090 | 218.670 | 1051.648 | 116.850 | 1083.392 |
| 2.0 | on  | 2.400 | 77.560 | 315.540 | 1058.816 | 324.490 | 1091.584 |
| 4.0 | on  | 4.520 | 79.740 | 677.650 | 1106.944 | 670.530 | 1101.824 |
| 6.0 | on  | 7.270 | 80.480 | 1216.040 | 1080.320 | 1139.210 | 1111.040 |
| 1.0 | off | 1.470 | 79.090 | 248.710 | 1038.336 | 320.900 | 1076.224 |
| 2.0 | off | 2.740 | 75.690 | 326.060 | 1038.336 | 333.910 | 1095.680 |
| 4.0 | off | 5.770 | 74.410 | 675.690 | 1100.800 | 676.800 | 1098.752 |
| 6.0 | off | 2.610 | 76.380 | 684.380 | 1101.824 | 1019.840 | 1108.992 |

### Short conclusions for the thesis

1. Scaling the 2D contour to two independent nodes achieved stable horizontal balancing: in all scenarios the requests were distributed evenly, `4` requests per node out of `8`.
2. All requests completed successfully in every scenario, which gives `100%` availability on the tested range from `1` to `6 RPS`.
3. Disabling artifact rendering reduced the mean response time at every load level:
   - `1 RPS`: from `1.313 s` to `1.059 s`, about `19.3%` faster
   - `2 RPS`: from `1.428 s` to `1.115 s`, about `21.9%` faster
   - `4 RPS`: from `1.466 s` to `1.197 s`, about `18.3%` faster
   - `6 RPS`: from `2.032 s` to `1.599 s`, about `21.3%` faster
4. The API gateway itself remained lightweight, while the main resource consumption was concentrated on the two 2D inference nodes, which confirms the correctness of the chosen service decomposition.
5. The obtained results show that the `render_artifacts` flag is practically useful: for interactive demonstrations it can stay enabled, and for load-sensitive or batch processing scenarios it is more efficient to disable rendering.

### Raw result files

- `load_test_2d_cluster_render_on.json`
- `load_test_2d_cluster_render_off.json`
