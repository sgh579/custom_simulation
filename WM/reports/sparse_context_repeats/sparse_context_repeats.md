# Sparse-Context WM Repeated Evaluation

Validation-selected thresholds are tuned independently for each method, ratio, and random sparse-context seed.

## Delta CI

| Method | Ratio | Mean Aggregate Dice Delta | Mean Sample Dice Delta | 95% CI | Seeds |
|---|---:|---:|---:|---:|---:|
| world_curve_unet | 0.25 | +0.013014 | +0.009708 | [+0.004467, +0.015141] | 5 |
| world_unet | 0.25 | +0.007407 | -0.002126 | [-0.007966, +0.003852] | 5 |
| world_curve_unet | 0.5 | +0.048515 | +0.051042 | [+0.046947, +0.055014] | 5 |
| world_unet | 0.5 | +0.053092 | +0.042152 | [+0.037141, +0.047355] | 5 |
| world_curve_unet | 0.75 | +0.050285 | +0.052940 | [+0.049036, +0.056817] | 5 |
| world_unet | 0.75 | +0.061658 | +0.053129 | [+0.048387, +0.057982] | 5 |

## Per-Seed Metrics

| Method | Ratio | Seed | Test Dice | Threshold |
|---|---:|---:|---:|---:|
| sparse_unet | 0.25 | 20260701 | 0.650650 | 0.675 |
| world_unet | 0.25 | 20260701 | 0.658234 | 0.775 |
| world_curve_unet | 0.25 | 20260701 | 0.664688 | 0.425 |
| sparse_unet | 0.25 | 20260702 | 0.648555 | 0.675 |
| world_unet | 0.25 | 20260702 | 0.655612 | 0.775 |
| world_curve_unet | 0.25 | 20260702 | 0.660409 | 0.450 |
| sparse_unet | 0.25 | 20260703 | 0.647884 | 0.650 |
| world_unet | 0.25 | 20260703 | 0.653972 | 0.800 |
| world_curve_unet | 0.25 | 20260703 | 0.662387 | 0.500 |
| sparse_unet | 0.25 | 20260704 | 0.647839 | 0.650 |
| world_unet | 0.25 | 20260704 | 0.661566 | 0.775 |
| world_curve_unet | 0.25 | 20260704 | 0.660479 | 0.425 |
| sparse_unet | 0.25 | 20260705 | 0.650284 | 0.675 |
| world_unet | 0.25 | 20260705 | 0.652865 | 0.775 |
| world_curve_unet | 0.25 | 20260705 | 0.662319 | 0.475 |
| sparse_unet | 0.5 | 20260701 | 0.684080 | 0.650 |
| world_unet | 0.5 | 20260701 | 0.735761 | 0.675 |
| world_curve_unet | 0.5 | 20260701 | 0.731405 | 0.625 |
| sparse_unet | 0.5 | 20260702 | 0.680832 | 0.650 |
| world_unet | 0.5 | 20260702 | 0.736550 | 0.675 |
| world_curve_unet | 0.5 | 20260702 | 0.729692 | 0.625 |
| sparse_unet | 0.5 | 20260703 | 0.681781 | 0.700 |
| world_unet | 0.5 | 20260703 | 0.736677 | 0.675 |
| world_curve_unet | 0.5 | 20260703 | 0.732235 | 0.600 |
| sparse_unet | 0.5 | 20260704 | 0.681770 | 0.675 |
| world_unet | 0.5 | 20260704 | 0.736277 | 0.675 |
| world_curve_unet | 0.5 | 20260704 | 0.729816 | 0.600 |
| sparse_unet | 0.5 | 20260705 | 0.685645 | 0.675 |
| world_unet | 0.5 | 20260705 | 0.734302 | 0.675 |
| world_curve_unet | 0.5 | 20260705 | 0.733534 | 0.600 |
| sparse_unet | 0.75 | 20260701 | 0.694029 | 0.625 |
| world_unet | 0.75 | 20260701 | 0.754285 | 0.650 |
| world_curve_unet | 0.75 | 20260701 | 0.742077 | 0.600 |
| sparse_unet | 0.75 | 20260702 | 0.692925 | 0.625 |
| world_unet | 0.75 | 20260702 | 0.756369 | 0.675 |
| world_curve_unet | 0.75 | 20260702 | 0.743061 | 0.575 |
| sparse_unet | 0.75 | 20260703 | 0.691798 | 0.600 |
| world_unet | 0.75 | 20260703 | 0.755848 | 0.650 |
| world_curve_unet | 0.75 | 20260703 | 0.743862 | 0.550 |
| sparse_unet | 0.75 | 20260704 | 0.693757 | 0.600 |
| world_unet | 0.75 | 20260704 | 0.755113 | 0.675 |
| world_curve_unet | 0.75 | 20260704 | 0.744170 | 0.575 |
| sparse_unet | 0.75 | 20260705 | 0.693500 | 0.625 |
| world_unet | 0.75 | 20260705 | 0.752684 | 0.650 |
| world_curve_unet | 0.75 | 20260705 | 0.744263 | 0.575 |
