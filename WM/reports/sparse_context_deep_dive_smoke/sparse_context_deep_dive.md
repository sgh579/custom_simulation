# Sparse-Context Deep Dive

All rows use 128x128 scan-area GT. Thresholds are selected on validation for each method, sparse ratio, pattern, and seed.

## Ratio-Pattern Delta Summary

| Pattern | Ratio | Method | Mean Test Dice | Delta vs No-World | Sample Delta 95% CI |
|---|---:|---|---:|---:|---:|
| random | 0.5 | world_curve_unet | 0.734368 | +0.047296 | [+0.041257, +0.059004] |
| random | 0.5 | world_unet | 0.739219 | +0.052147 | [+0.031325, +0.051197] |

## No-World Baseline Means

| Pattern | Ratio | Sparse UNet Mean Test Dice |
|---|---:|---:|
| random | 0.5 | 0.687072 |

## Area-Bin Delta Summary

| Pattern | Ratio | Area Bin | Method | Mean Delta vs No-World |
|---|---:|---|---|---:|
| random | 0.5 | large | world_curve_unet | +0.047252 |
| random | 0.5 | large | world_unet | +0.078237 |
| random | 0.5 | medium | world_curve_unet | +0.039970 |
| random | 0.5 | medium | world_unet | +0.024324 |
| random | 0.5 | small | world_curve_unet | +0.061780 |
| random | 0.5 | small | world_unet | +0.024085 |
