# Two-stage LESS port for synthetic palpation V2

This experiment preserves the released LESS training contract rather than only
borrowing its local-particle terminology.

1. Stage 1 assigns V2 force curves to fixed particles by Euclidean distance in
   physical XY coordinates. It holds out one local force curve and reconstructs
   it from the other neighbouring curves without using segmentation labels.
2. The best representation checkpoint is loaded and frozen.
3. Independent 2D and 3D transposed-convolution decoders map every frozen
   particle to a local patch. Patches are added on the output canvas.

The default radius is 13.6 mm. The V2 scan spacing is approximately 7.895 mm,
so an interior particle receives the centre, four axial neighbours, and four
diagonal neighbours.

## Resolution modes

The default remains the completed low-resolution contract:

- 20 x 20 particle grid;
- 20 x 20 2D output and 16 x 20 x 20 3D output;
- 5 x 5 local decoder patch.

Passing `--output-size 128` keeps the same 20 x 20 observations and frozen
Stage 1 representation but expands the task decoders to:

- 128 x 128 2D output and 16 x 128 x 128 3D output;
- 32 x 32 local patch for each particle;
- the same 1/4 local-patch-to-canvas ratio as 5/20 and the released 32/128
  LESS setting;
- analytic 128 x 128 scan-area labels using the same grid-centre convention as
  the Grid U-Net baseline. The extent comes from the 20 x 20 palpation `xy`
  coordinates, not the wider full-phantom `label_xy` grid.

Particle centres are mapped to their nearest 128 x 128 grid positions. Patch
overlap is summed, including at boundaries, just as in the original 20 x 20
implementation. Stage 1 is resolution-independent, so a completed Stage 1
checkpoint can be reused and only the two high-resolution decoders need to be
trained.

For the full V2 comparison, use a new output directory so the completed
20 x 20 run is not overwritten. Decoder batch size 16 was checked on lab204's
32 GB RTX 5090; the 3D training step peaks at about 19.4 GiB before the frozen
representations and labels are made resident:

```bash
python ops/less_v2_two_stage/train_less_v2_two_stage.py \
  --data-root /home/guoheng/custom_simulation/data/palpation_fixed_depth_36mm_40step_ecoflex0010_ref55k_100pctmodulus_pm20pct_linear_2000train_800val_800test_20260702 \
  --output-dir /home/guoheng/custom_simulation/runs/less_v2_two_stage_full_128_20260819_s9400 \
  --train-samples 2000 --val-samples 800 --test-samples 800 \
  --pretrain-epochs 1000 --decoder-epochs 1000 \
  --batch-size 32 --decoder-batch-size 16 \
  --output-size 128 \
  --stage1-checkpoint /home/guoheng/custom_simulation/runs/less_v2_two_stage_full_20260813_s9400/stage1_representation/best_representation.pt
```

The run manifest records the target and decoder output shapes, patch size,
label policy, and whether the 128 x 128 Grid U-Net target convention is active.
