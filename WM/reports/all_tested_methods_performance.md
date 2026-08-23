# All Tested WM / JEPA / 3D Projection Methods

Generated: 2026-07-01

Data contract: Newton random-trajectory synthetic palpation split, 1000 train / 400 val / 400 test, 20x20 scan grid with 20 press steps. Unless otherwise stated, GT is the 128x128 scan-area mask, not the global phantom-area mask.

Metric note: 2D rows report test Dice at the validation-selected threshold. Sparse fixed-threshold rows are separately labeled. 3D rows report both 2D projection Dice and voxel Dice; projection Dice is the metric closest to the 2D segmentation target.

## Full-Context 2D Segmentation
| Family | Method | Test Dice | Delta vs CNN32 | Val Best | Thr | Aux | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| score fusion | r128_cnn32_plus_temporal_cnn_world_unet_fusion | 0.787268 | +0.017820 | 0.798207 | 0.550000 |  | alpha_world=0.55 |
| score fusion | r128_cnn32_plus_temporal_cnn_world_curve_unet_fusion | 0.774143 | +0.004696 | 0.802017 | 0.500000 |  | alpha_world=0.5 |
| 2D baseline | r128_wrench_temporal_cnn32_unet | 0.769447 | +0.000000 | 0.777026 | 0.700000 |  |  |
| 2D baseline | r128_wrench_temporal_gru32_unet | 0.767595 | -0.001853 | 0.770418 | 0.600000 |  |  |
| 2D baseline | r128_wrench_temporal_attention32_unet | 0.765195 | -0.004253 | 0.773780 | 0.650000 |  |  |
| action WM direct | r128_wrench_temporal_cnn_world_unet | 0.763675 | -0.005772 | 0.779587 | 0.650000 | curve_mse=0.705917 |  |
| 2D baseline | r128_wrench_temporal_multiscale32_unet | 0.761481 | -0.007966 | 0.778203 | 0.650000 |  |  |
| 2D baseline | r128_wrench_unet_aug_focal | 0.758670 | -0.010777 | 0.772094 | 0.650000 |  |  |
| action WM direct | r128_wrench_temporal_cnn_world_curve_unet | 0.745888 | -0.023559 | 0.787827 | 0.500000 | curve_mse=0.110572 |  |
| ViT direct | r128_wrench_vit_unet | 0.738122 | -0.031326 | 0.751884 | 0.650000 |  |  |
| 2D baseline | r128_wrench_unet | 0.736724 | -0.032724 | 0.766556 | 0.650000 |  |  |
| JEPA/VISReg direct | r128_wrench_lejepa_vit_unet_m50 | 0.733767 | -0.035681 | 0.742690 | 0.550000 | latent_mse=0.075352 |  |
| JEPA/VISReg direct | r128_wrench_visreg_vit_unet_m50 | 0.726168 | -0.043279 | 0.757599 | 0.500000 | latent_mse=0.075865 |  |
| JEPA/VISReg direct | r128_wrench_jepa_vit_unet_m50 | 0.722743 | -0.046705 | 0.751425 | 0.500000 | latent_mse=0.006915 |  |
| 2D baseline | r128_wrench_temporal_cnn16_unet | 0.718338 | -0.051110 | 0.753388 | 0.650000 |  |  |
| no-world sparse-trained | r128_wrench_temporal_cnn_sparse_unet | 0.697361 | -0.072086 | 0.720061 | 0.550000 | curve_mse=0.513876 | trained with sparse context; no WM pretrain |
| early sparse WM pilot | r128_wrench_world_with_curve_decoder | 0.519426 | -0.250022 | 0.570203 | 0.550000 | curve_mse=0.077129 |  |

## Sparse-Context Repeated Evaluation
| Observed Ratio | Method | Mean Test Dice | Aggregate Delta vs No-World | Sample-Level Delta CI |
| --- | --- | --- | --- | --- |
| 0.25 | sparse_unet | 0.649042 |  |  |
| 0.25 | world_curve_unet | 0.662056 | 0.013014 | sample mean delta 0.009708, 95% CI [0.004467, 0.015141] |
| 0.25 | world_unet | 0.656450 | 0.007407 | sample mean delta -0.002126, 95% CI [-0.007966, 0.003852] |
| 0.50 | sparse_unet | 0.682822 |  |  |
| 0.50 | world_curve_unet | 0.731337 | 0.048515 | sample mean delta 0.051042, 95% CI [0.046947, 0.055014] |
| 0.50 | world_unet | 0.735914 | 0.053092 | sample mean delta 0.042152, 95% CI [0.037141, 0.047355] |
| 0.75 | sparse_unet | 0.693202 |  |  |
| 0.75 | world_curve_unet | 0.743487 | 0.050285 | sample mean delta 0.052940, 95% CI [0.049036, 0.056817] |
| 0.75 | world_unet | 0.754860 | 0.061658 | sample mean delta 0.053129, 95% CI [0.048387, 0.057982] |

## Sparse-Context Fixed-Threshold Test
| Observed Ratio | Method | Test Dice @ 0.5 Thr | Delta vs No-World | Note |
| --- | --- | --- | --- | --- |
| 0.10 | r128_wrench_temporal_cnn_sparse_unet | 0.555118 |  | no-world baseline |
| 0.25 | r128_wrench_temporal_cnn_sparse_unet | 0.648360 |  | no-world baseline |
| 0.25 | r128_wrench_temporal_cnn_world_curve_unet | 0.664635 | +0.016276 | action-conditioned WM |
| 0.25 | r128_wrench_temporal_cnn_world_unet | 0.653095 | +0.004735 | action-conditioned WM |
| 0.50 | r128_wrench_temporal_cnn_sparse_unet | 0.683289 |  | no-world baseline |
| 0.50 | r128_wrench_temporal_cnn_world_curve_unet | 0.733619 | +0.050330 | action-conditioned WM |
| 0.50 | r128_wrench_temporal_cnn_world_unet | 0.736903 | +0.053615 | action-conditioned WM |
| 0.75 | r128_wrench_temporal_cnn_sparse_unet | 0.694862 |  | no-world baseline |
| 0.75 | r128_wrench_temporal_cnn_world_curve_unet | 0.745727 | +0.050865 | action-conditioned WM |
| 0.75 | r128_wrench_temporal_cnn_world_unet | 0.754124 | +0.059262 | action-conditioned WM |

## 3D Volume to 2D Projection
| Family | Method | Projection Dice | Delta vs 3D Direct | Voxel Dice | Thr | Alpha |
| --- | --- | --- | --- | --- | --- | --- |
| 3D score fusion | v32x64x64_supervised_plus_world_curve_volume_fusion | 0.758536 | +0.008288 | 0.547261 | 0.600000 | 0.65 |
| 3D depth-aware fusion | v32x64x64_supervised_plus_world_curve_depthaware_fusion | 0.758536 | +0.008288 | 0.547261 | 0.600000 | 0.65 |
| 3D score fusion | v32x64x64_supervised_plus_world_volume_fusion | 0.754456 | +0.004208 | 0.542046 | 0.450000 | 0.55 |
| 3D depth-aware fusion | v32x64x64_supervised_plus_world_volume_depthaware_fusion | 0.751659 | +0.001411 | 0.551125 | 0.500000 | 0.5 |
| 3D direct | v32x64x64_wrench_temporal_cnn_volume | 0.750248 | +0.000000 | 0.542382 | 0.500000 |  |
| 3D WM direct | v32x64x64_wrench_temporal_cnn_world_curve_volume | 0.749322 | -0.000926 | 0.532237 | 0.750000 |  |
| 3D depth-aware direct | v32x64x64_wrench_temporal_cnn_volume | 0.746898 | -0.003350 | 0.536142 | 0.400000 |  |
| 3D WM direct | v32x64x64_wrench_temporal_cnn_world_volume | 0.741435 | -0.008813 | 0.526624 | 0.550000 |  |
| 3D regularized direct | v32x64x64_wrench_temporal_cnn_volume | 0.726097 | -0.024150 | 0.502329 | 0.400000 |  |
| 3D depth-aware direct | v32x64x64_wrench_temporal_cnn_world_curve_volume | 0.719163 | -0.031084 | 0.506628 | 0.550000 |  |
| 3D depth-aware direct | v32x64x64_wrench_temporal_cnn_world_volume | 0.718376 | -0.031872 | 0.486460 | 0.650000 |  |

## Compact Takeaway

- Best full-context 2D score: `r128_cnn32_plus_temporal_cnn_world_unet_fusion`, 0.787268, +0.017820 over CNN32.
- Best sparse-context repeated result: `world_unet` at 0.75 observed ratio, mean Dice 0.754860, aggregate delta +0.061658 vs no-world sparse baseline.
- Most publishable 5-point result: sparse-context setting at ratios 0.5 and 0.75; repeated sparse-mask evaluation keeps the improvement around +0.05 Dice with positive sample-level CIs.
- Direct JEPA/LeJEPA/VISReg did not beat the temporal CNN full-context baseline; they are better treated as negative/diagnostic ablations unless reformulated as action-conditioned future-prediction world models.
- 3D projection fusion helps slightly over direct 3D prediction, but it does not beat the best full-context 2D fusion; its stronger role is mechanistic support rather than the main SOTA claim.

## Source Files

- `raw_leaderboards/runs/nonlinear_trajectory_20x_repeats10_seed20260618/highres_wrench_temporal_variants/leaderboard.csv`
- `raw_leaderboards/runs/wrench_temporal_cnn_world_20260622/leaderboard.csv`
- `raw_leaderboards/runs/wrench_world_model_sweep_20260622/leaderboard.csv`
- `raw_leaderboards/runs/wrench_world_fusion_20260622/leaderboard.csv`
- `raw_leaderboards/WM/runs/jepa_visreg_core_20260701/leaderboard.csv`
- `raw_leaderboards/WM/runs/sparse_baseline_core_20260701/leaderboard.csv`
- `raw_leaderboards/runs/wrench_world_model_sparse_context_20260622/leaderboard.csv`
- `raw_leaderboards/runs/wrench_temporal_cnn_world_20260622/world_model_metrics.csv`
- `raw_leaderboards/WM/runs/sparse_baseline_core_20260701/world_model_metrics.csv`
- `sparse_context_repeats/sparse_context_repeated_metrics.csv`
- `sparse_context_repeats/sparse_context_delta_ci.csv`
- `raw_leaderboards/runs/wrench_3d_volume_prediction_20260622/leaderboard.csv`
- `raw_leaderboards/runs/wrench_3d_volume_prediction_regularized_20260622/leaderboard.csv`
- `raw_leaderboards/runs/wrench_3d_volume_world_20260622/leaderboard.csv`
- `raw_leaderboards/runs/wrench_3d_volume_fusion_20260622/leaderboard.csv`
- `raw_leaderboards/runs/wrench_3d_volume_depth_aware_train_20260622/leaderboard.csv`
- `raw_leaderboards/runs/wrench_3d_volume_depth_aware_fusion_20260622/leaderboard.csv`
