# WM / JEPA / VISReg Palpation Results

Generated: 2026-07-01 22:39:33

## Summary Table

| Task | Method | Value | Note |
|---|---:|---:|---|
| full_context_2d | `temporal_cnn32_unet` | 0.769447 | baseline |
| full_context_2d | `r128_cnn32_plus_temporal_cnn_world_unet_fusion` | 0.787268 | delta_vs_temporal_cnn32=+0.017820 alpha_world=0.55 |
| full_context_2d | `r128_cnn32_plus_temporal_cnn_world_curve_unet_fusion` | 0.774143 | delta_vs_temporal_cnn32=+0.004696 alpha_world=0.5 |
| sparse_context_2d | `r128_wrench_temporal_cnn_world_curve_unet@0.25` | 0.664635 | fixed_threshold=0.5 |
| sparse_context_2d | `r128_wrench_temporal_cnn_world_curve_unet@0.5` | 0.733619 | fixed_threshold=0.5 |
| sparse_context_2d | `r128_wrench_temporal_cnn_world_curve_unet@0.75` | 0.745727 | fixed_threshold=0.5 |
| sparse_context_2d | `r128_wrench_temporal_cnn_world_unet@0.25` | 0.653095 | fixed_threshold=0.5 |
| sparse_context_2d | `r128_wrench_temporal_cnn_world_unet@0.5` | 0.736903 | fixed_threshold=0.5 |
| sparse_context_2d | `r128_wrench_temporal_cnn_world_unet@0.75` | 0.754124 | fixed_threshold=0.5 |
| sparse_context_2d | `r128_wrench_temporal_cnn_sparse_unet@0.1` | 0.555118 | fixed_threshold=0.5 |
| sparse_context_2d | `r128_wrench_temporal_cnn_sparse_unet@0.25` | 0.648360 | fixed_threshold=0.5 |
| sparse_context_2d | `r128_wrench_temporal_cnn_sparse_unet@0.5` | 0.683289 | fixed_threshold=0.5 |
| sparse_context_2d | `r128_wrench_temporal_cnn_sparse_unet@0.75` | 0.694862 | fixed_threshold=0.5 |
| sparse_context_delta | `r128_wrench_temporal_cnn_world_curve_unet@0.25 vs sparse_unet` | 0.016276 | world=0.664635 baseline=0.648360 |
| sparse_context_delta | `r128_wrench_temporal_cnn_world_unet@0.25 vs sparse_unet` | 0.004735 | world=0.653095 baseline=0.648360 |
| sparse_context_delta | `r128_wrench_temporal_cnn_world_curve_unet@0.5 vs sparse_unet` | 0.050330 | world=0.733619 baseline=0.683289 |
| sparse_context_delta | `r128_wrench_temporal_cnn_world_unet@0.5 vs sparse_unet` | 0.053615 | world=0.736903 baseline=0.683289 |
| sparse_context_delta | `r128_wrench_temporal_cnn_world_curve_unet@0.75 vs sparse_unet` | 0.050865 | world=0.745727 baseline=0.694862 |
| sparse_context_delta | `r128_wrench_temporal_cnn_world_unet@0.75 vs sparse_unet` | 0.059262 | world=0.754124 baseline=0.694862 |
| sparse_context_repeated_ci | `world_curve_unet@0.25 vs sparse_unet` | 0.013014 | sample_mean_delta=0.009708 sample_bootstrap_95ci=[0.004467,0.015141] seeds=5 |
| sparse_context_repeated_ci | `world_unet@0.25 vs sparse_unet` | 0.007407 | sample_mean_delta=-0.002126 sample_bootstrap_95ci=[-0.007966,0.003852] seeds=5 |
| sparse_context_repeated_ci | `world_curve_unet@0.5 vs sparse_unet` | 0.048515 | sample_mean_delta=0.051042 sample_bootstrap_95ci=[0.046947,0.055014] seeds=5 |
| sparse_context_repeated_ci | `world_unet@0.5 vs sparse_unet` | 0.053092 | sample_mean_delta=0.042152 sample_bootstrap_95ci=[0.037141,0.047355] seeds=5 |
| sparse_context_repeated_ci | `world_curve_unet@0.75 vs sparse_unet` | 0.050285 | sample_mean_delta=0.052940 sample_bootstrap_95ci=[0.049036,0.056817] seeds=5 |
| sparse_context_repeated_ci | `world_unet@0.75 vs sparse_unet` | 0.061658 | sample_mean_delta=0.053129 sample_bootstrap_95ci=[0.048387,0.057982] seeds=5 |
| 3d_to_2d_projection | `v32x64x64_wrench_temporal_cnn_volume` | 0.746898 | projection Dice from predicted volume |
| 3d_to_2d_projection | `v32x64x64_wrench_temporal_cnn_world_curve_volume` | 0.719163 | projection Dice from predicted volume |
| 3d_to_2d_projection | `v32x64x64_wrench_temporal_cnn_world_volume` | 0.718376 | projection Dice from predicted volume |
| 3d_to_2d_projection | `v32x64x64_supervised_plus_world_volume_depthaware_fusion` | 0.751659 | projection Dice from predicted volume |
| 3d_to_2d_projection | `v32x64x64_supervised_plus_world_curve_depthaware_fusion` | 0.758536 | projection Dice from predicted volume |
| jepa_visreg_2d | `r128_wrench_lejepa_vit_unet_m50` | 0.733767 | heldout_latent_mse=0.0753520369529724 |
| jepa_visreg_2d | `r128_wrench_visreg_vit_unet_m50` | 0.726168 | heldout_latent_mse=0.07586548075079919 |
| jepa_visreg_2d | `r128_wrench_jepa_vit_unet_m50` | 0.722743 | heldout_latent_mse=0.0069145749555900695 |

## Current Interpretation

- Full-context 2D WM fusion is useful but modest; it should be treated as supporting evidence, not the main 5-point claim.
- The main WM task should be sparse-context palpation: same object and split, fewer observed probe locations, full-mask prediction.
- The sparse-context delta rows are the key decision gate: keep the WM contribution only if the action-conditioned world objective beats the no-world sparse baseline by at least 0.05 Dice on validation-selected test metrics.
- The 3D-to-2D projection rows support the mechanistic story that latent 3D occupancy contains useful segmentation signal, even when full voxel Dice remains harder than projected Dice.
