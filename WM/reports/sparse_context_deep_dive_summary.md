# Sparse-Context Deep-Dive Summary

Generated: 2026-07-01. Data: Newton random-trajectory synthetic palpation, 1000 train / 400 val / 400 test, 20x20 scan grid, 128x128 scan-area GT.

This report uses matched sparse training: `sparse_unet`, `world_unet`, and `world_curve_unet` are all fine-tuned with `seg_context_min=0.25`, `seg_context_max=1.0`. Action-world pretraining is reused from the existing 20260622 run; active sensing is not included.

## Matched Sparse Training: Fixed-Threshold Ratio Sweep
| Observed Ratio | No-World Sparse | WM Curve | Curve Gain | WM Latent | Latent Gain |
| --- | --- | --- | --- | --- | --- |
| 0.1 | 0.451985 | 0.573383 | +0.121398 | 0.573910 | +0.121925 |
| 0.25 | 0.580140 | 0.687019 | +0.106880 | 0.717424 | +0.137285 |
| 0.5 | 0.687030 | 0.719825 | +0.032795 | 0.763548 | +0.076518 |
| 0.75 | 0.707725 | 0.729012 | +0.021287 | 0.780487 | +0.072762 |
| 0.9 | 0.711123 | 0.730257 | +0.019134 | 0.784904 | +0.073781 |

## Practical Sparse Patterns: Mean Test Dice Over 3 Seeds
| Pattern | Ratio | No-World | Best WM | WM Dice | Agg Gain | Sample 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| random | 0.1 | 0.449191 | world_curve_unet | 0.578058 | +0.128867 | [+0.114935, +0.135993] |
| random | 0.25 | 0.595427 | world_unet | 0.712668 | +0.117242 | [+0.109064, +0.127445] |
| random | 0.5 | 0.689405 | world_unet | 0.764419 | +0.075014 | [+0.057024, +0.070344] |
| random | 0.75 | 0.707407 | world_unet | 0.774994 | +0.067587 | [+0.041437, +0.052551] |
| random | 0.9 | 0.709114 | world_unet | 0.779125 | +0.070011 | [+0.039535, +0.050415] |
| space_filling | 0.1 | 0.540637 | world_unet | 0.633175 | +0.092538 | [+0.077762, +0.095037] |
| space_filling | 0.25 | 0.653196 | world_unet | 0.735533 | +0.082337 | [+0.073002, +0.087154] |
| space_filling | 0.5 | 0.696460 | world_unet | 0.764310 | +0.067850 | [+0.047271, +0.058547] |
| space_filling | 0.75 | 0.707309 | world_unet | 0.774312 | +0.067003 | [+0.039917, +0.050552] |
| space_filling | 0.9 | 0.710029 | world_unet | 0.778334 | +0.068305 | [+0.039062, +0.049593] |
| raster_lines | 0.1 | 0.213384 | world_curve_unet | 0.285541 | +0.072157 | [+0.047657, +0.069246] |
| raster_lines | 0.25 | 0.587813 | world_curve_unet | 0.666203 | +0.078390 | [+0.074469, +0.091183] |
| raster_lines | 0.5 | 0.698703 | world_unet | 0.763552 | +0.064849 | [+0.042650, +0.054690] |
| raster_lines | 0.75 | 0.704530 | world_unet | 0.774754 | +0.070224 | [+0.041072, +0.051538] |
| raster_lines | 0.9 | 0.706842 | world_unet | 0.777457 | +0.070615 | [+0.040426, +0.051234] |

## Hard Missingness Patterns
| Pattern | Ratio | No-World | Best WM | WM Dice | Agg Gain | Sample 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| local_block | 0.1 | 0.199589 | world_curve_unet | 0.280169 | +0.080580 | [+0.050409, +0.071538] |
| local_block | 0.25 | 0.259563 | world_unet | 0.468441 | +0.208879 | [+0.120217, +0.147950] |
| local_block | 0.5 | 0.404767 | world_unet | 0.682968 | +0.278201 | [+0.227996, +0.252124] |
| local_block | 0.75 | 0.640372 | world_unet | 0.760286 | +0.119914 | [+0.083434, +0.099136] |
| local_block | 0.9 | 0.707780 | world_unet | 0.777794 | +0.070013 | [+0.035732, +0.047020] |
| missing_block | 0.1 | 0.165067 | world_curve_unet | 0.177537 | +0.012470 | [+0.005284, +0.025996] |
| missing_block | 0.25 | 0.279416 | world_curve_unet | 0.413134 | +0.133718 | [+0.127617, +0.151792] |
| missing_block | 0.5 | 0.385906 | world_curve_unet | 0.603940 | +0.218034 | [+0.221326, +0.247104] |
| missing_block | 0.75 | 0.467901 | world_curve_unet | 0.653510 | +0.185610 | [+0.202691, +0.229130] |
| missing_block | 0.9 | 0.624384 | world_unet | 0.749352 | +0.124968 | [+0.125894, +0.145417] |

## Area-Bin Gain: WM Latent vs No-World
| Pattern | Ratio | Small Masks | Medium Masks | Large Masks |
| --- | --- | --- | --- | --- |
| random | 0.25 | +0.161150 | +0.086296 | +0.102139 |
| random | 0.5 | +0.042988 | +0.048511 | +0.100637 |
| random | 0.75 | +0.002367 | +0.034417 | +0.108166 |
| space_filling | 0.25 | +0.097775 | +0.067960 | +0.072680 |
| space_filling | 0.5 | +0.023078 | +0.041806 | +0.096469 |
| space_filling | 0.75 | -0.001364 | +0.033780 | +0.107908 |
| raster_lines | 0.25 | +0.074605 | +0.035688 | +0.064438 |
| raster_lines | 0.5 | +0.017987 | +0.037271 | +0.094218 |
| raster_lines | 0.75 | -0.001580 | +0.033222 | +0.111756 |

## Interpretation

- Matched sparse training turns sparse-context into the strongest WM result: `world_unet` gains +0.075, +0.068, +0.070 Dice over no-world sparse on random 50/75/90% observed contexts, and +0.082/+0.068/+0.067 on space-filling 25/50/75%.
- The improvement is not limited to random missingness. Corrected raster-line contexts still show strong gains at 25/50/75/90% observed ratios.
- Hard missingness reveals the most WM-specific behavior: local-block and missing-block patterns create large blind regions where the no-world sparse baseline collapses, while action-conditioned WM recovers much more of the shape.
- The curve-aux variant is most useful for missing-block completion; the latent/action WM variant is usually strongest for random, space-filling, raster-lines, and local-block contexts.
- For the paper, the clean main claim should be sparse-context completion under matched training, with active sensing reserved for future work.

## Files

- `sparse_matched_025_100/leaderboard.csv`
- `sparse_matched_025_100/world_model_metrics.csv`
- `sparse_context_deep_dive_matched_025_100/sparse_context_deep_metrics.csv`
- `sparse_context_deep_dive_matched_025_100/sparse_context_deep_delta_ci.csv`
- `sparse_context_deep_dive_matched_025_100/sparse_context_deep_area_bins.csv`
- `sparse_context_deep_dive_matched_025_100/sparse_context_deep_dive.md`
