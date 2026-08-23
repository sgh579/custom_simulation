# Sparse-Context Deep Dive

All rows use 128x128 scan-area GT. Thresholds are selected on validation for each method, sparse ratio, pattern, and seed.

## Ratio-Pattern Delta Summary

| Pattern | Ratio | Method | Mean Test Dice | Delta vs No-World | Sample Delta 95% CI |
|---|---:|---|---:|---:|---:|
| local_block | 0.1 | world_curve_unet | 0.280169 | +0.080580 | [+0.050409, +0.071538] |
| local_block | 0.1 | world_unet | 0.278456 | +0.078867 | [+0.014185, +0.038400] |
| local_block | 0.25 | world_curve_unet | 0.412708 | +0.153145 | [+0.116943, +0.138725] |
| local_block | 0.25 | world_unet | 0.468441 | +0.208879 | [+0.120217, +0.147950] |
| local_block | 0.5 | world_curve_unet | 0.645361 | +0.240594 | [+0.216129, +0.236252] |
| local_block | 0.5 | world_unet | 0.682968 | +0.278201 | [+0.227996, +0.252124] |
| local_block | 0.75 | world_curve_unet | 0.715869 | +0.075497 | [+0.060103, +0.074517] |
| local_block | 0.75 | world_unet | 0.760286 | +0.119914 | [+0.083434, +0.099136] |
| local_block | 0.9 | world_curve_unet | 0.728878 | +0.021097 | [+0.009493, +0.018315] |
| local_block | 0.9 | world_unet | 0.777794 | +0.070013 | [+0.035732, +0.047020] |
| missing_block | 0.1 | world_curve_unet | 0.177537 | +0.012470 | [+0.005284, +0.025996] |
| missing_block | 0.1 | world_unet | 0.142327 | -0.022741 | [-0.026161, -0.008637] |
| missing_block | 0.25 | world_curve_unet | 0.413134 | +0.133718 | [+0.127617, +0.151792] |
| missing_block | 0.25 | world_unet | 0.342494 | +0.063078 | [+0.054225, +0.080104] |
| missing_block | 0.5 | world_curve_unet | 0.603940 | +0.218034 | [+0.221326, +0.247104] |
| missing_block | 0.5 | world_unet | 0.538296 | +0.152390 | [+0.148569, +0.176266] |
| missing_block | 0.75 | world_curve_unet | 0.653510 | +0.185610 | [+0.202691, +0.229130] |
| missing_block | 0.75 | world_unet | 0.642834 | +0.174933 | [+0.181744, +0.211744] |
| missing_block | 0.9 | world_curve_unet | 0.714860 | +0.090476 | [+0.109415, +0.129432] |
| missing_block | 0.9 | world_unet | 0.749352 | +0.124968 | [+0.125894, +0.145417] |
| random | 0.1 | world_curve_unet | 0.578058 | +0.128867 | [+0.114935, +0.135993] |
| random | 0.1 | world_unet | 0.572608 | +0.123417 | [+0.098125, +0.119574] |
| random | 0.25 | world_curve_unet | 0.690060 | +0.094633 | [+0.095892, +0.114205] |
| random | 0.25 | world_unet | 0.712668 | +0.117242 | [+0.109064, +0.127445] |
| random | 0.5 | world_curve_unet | 0.722141 | +0.032736 | [+0.028989, +0.042871] |
| random | 0.5 | world_unet | 0.764419 | +0.075014 | [+0.057024, +0.070344] |
| random | 0.75 | world_curve_unet | 0.729310 | +0.021903 | [+0.013442, +0.023659] |
| random | 0.75 | world_unet | 0.774994 | +0.067587 | [+0.041437, +0.052551] |
| random | 0.9 | world_curve_unet | 0.730749 | +0.021635 | [+0.012032, +0.021229] |
| random | 0.9 | world_unet | 0.779125 | +0.070011 | [+0.039535, +0.050415] |
| raster_lines | 0.1 | world_curve_unet | 0.285541 | +0.072157 | [+0.047657, +0.069246] |
| raster_lines | 0.1 | world_unet | 0.252900 | +0.039517 | [-0.003955, +0.018988] |
| raster_lines | 0.25 | world_curve_unet | 0.666203 | +0.078390 | [+0.074469, +0.091183] |
| raster_lines | 0.25 | world_unet | 0.655922 | +0.068109 | [+0.049660, +0.067243] |
| raster_lines | 0.5 | world_curve_unet | 0.725797 | +0.027094 | [+0.019008, +0.030335] |
| raster_lines | 0.5 | world_unet | 0.763552 | +0.064849 | [+0.042650, +0.054690] |
| raster_lines | 0.75 | world_curve_unet | 0.728673 | +0.024143 | [+0.012997, +0.022635] |
| raster_lines | 0.75 | world_unet | 0.774754 | +0.070224 | [+0.041072, +0.051538] |
| raster_lines | 0.9 | world_curve_unet | 0.728805 | +0.021964 | [+0.012311, +0.021960] |
| raster_lines | 0.9 | world_unet | 0.777457 | +0.070615 | [+0.040426, +0.051234] |
| space_filling | 0.1 | world_curve_unet | 0.630589 | +0.089952 | [+0.085375, +0.103281] |
| space_filling | 0.1 | world_unet | 0.633175 | +0.092538 | [+0.077762, +0.095037] |
| space_filling | 0.25 | world_curve_unet | 0.707232 | +0.054036 | [+0.052328, +0.068351] |
| space_filling | 0.25 | world_unet | 0.735533 | +0.082337 | [+0.073002, +0.087154] |
| space_filling | 0.5 | world_curve_unet | 0.723986 | +0.027527 | [+0.020540, +0.031705] |
| space_filling | 0.5 | world_unet | 0.764310 | +0.067850 | [+0.047271, +0.058547] |
| space_filling | 0.75 | world_curve_unet | 0.728827 | +0.021518 | [+0.011997, +0.021555] |
| space_filling | 0.75 | world_unet | 0.774312 | +0.067003 | [+0.039917, +0.050552] |
| space_filling | 0.9 | world_curve_unet | 0.730455 | +0.020426 | [+0.010781, +0.020060] |
| space_filling | 0.9 | world_unet | 0.778334 | +0.068305 | [+0.039062, +0.049593] |

## No-World Baseline Means

| Pattern | Ratio | Sparse UNet Mean Test Dice |
|---|---:|---:|
| local_block | 0.1 | 0.199589 |
| local_block | 0.25 | 0.259563 |
| local_block | 0.5 | 0.404767 |
| local_block | 0.75 | 0.640372 |
| local_block | 0.9 | 0.707780 |
| missing_block | 0.1 | 0.165067 |
| missing_block | 0.25 | 0.279416 |
| missing_block | 0.5 | 0.385906 |
| missing_block | 0.75 | 0.467901 |
| missing_block | 0.9 | 0.624384 |
| random | 0.1 | 0.449191 |
| random | 0.25 | 0.595427 |
| random | 0.5 | 0.689405 |
| random | 0.75 | 0.707407 |
| random | 0.9 | 0.709114 |
| raster_lines | 0.1 | 0.213384 |
| raster_lines | 0.25 | 0.587813 |
| raster_lines | 0.5 | 0.698703 |
| raster_lines | 0.75 | 0.704530 |
| raster_lines | 0.9 | 0.706842 |
| space_filling | 0.1 | 0.540637 |
| space_filling | 0.25 | 0.653196 |
| space_filling | 0.5 | 0.696460 |
| space_filling | 0.75 | 0.707309 |
| space_filling | 0.9 | 0.710029 |

## Area-Bin Delta Summary

| Pattern | Ratio | Area Bin | Method | Mean Delta vs No-World |
|---|---:|---|---|---:|
| local_block | 0.1 | large | world_curve_unet | +0.039977 |
| local_block | 0.1 | large | world_unet | +0.003037 |
| local_block | 0.1 | medium | world_curve_unet | +0.087201 |
| local_block | 0.1 | medium | world_unet | +0.036567 |
| local_block | 0.1 | small | world_curve_unet | +0.056399 |
| local_block | 0.1 | small | world_unet | +0.038286 |
| local_block | 0.25 | large | world_curve_unet | +0.066692 |
| local_block | 0.25 | large | world_unet | +0.105330 |
| local_block | 0.25 | medium | world_curve_unet | +0.162493 |
| local_block | 0.25 | medium | world_unet | +0.145728 |
| local_block | 0.25 | small | world_curve_unet | +0.153090 |
| local_block | 0.25 | small | world_unet | +0.148585 |
| local_block | 0.5 | large | world_curve_unet | +0.189359 |
| local_block | 0.5 | large | world_unet | +0.256075 |
| local_block | 0.5 | medium | world_curve_unet | +0.211069 |
| local_block | 0.5 | medium | world_unet | +0.214615 |
| local_block | 0.5 | small | world_curve_unet | +0.273351 |
| local_block | 0.5 | small | world_unet | +0.246971 |
| local_block | 0.75 | large | world_curve_unet | +0.067770 |
| local_block | 0.75 | large | world_unet | +0.148720 |
| local_block | 0.75 | medium | world_curve_unet | +0.046841 |
| local_block | 0.75 | medium | world_unet | +0.075569 |
| local_block | 0.75 | small | world_curve_unet | +0.085476 |
| local_block | 0.75 | small | world_unet | +0.052154 |
| local_block | 0.9 | large | world_curve_unet | +0.034903 |
| local_block | 0.9 | large | world_unet | +0.117176 |
| local_block | 0.9 | medium | world_curve_unet | -0.003655 |
| local_block | 0.9 | medium | world_unet | +0.027489 |
| local_block | 0.9 | small | world_curve_unet | +0.010308 |
| local_block | 0.9 | small | world_unet | -0.015223 |
| missing_block | 0.1 | large | world_curve_unet | -0.033432 |
| missing_block | 0.1 | large | world_unet | -0.080106 |
| missing_block | 0.1 | medium | world_curve_unet | +0.038427 |
| missing_block | 0.1 | medium | world_unet | -0.003009 |
| missing_block | 0.1 | small | world_curve_unet | +0.039294 |
| missing_block | 0.1 | small | world_unet | +0.026185 |
| missing_block | 0.25 | large | world_curve_unet | +0.058851 |
| missing_block | 0.25 | large | world_unet | -0.068238 |
| missing_block | 0.25 | medium | world_curve_unet | +0.137923 |
| missing_block | 0.25 | medium | world_unet | +0.045230 |
| missing_block | 0.25 | small | world_curve_unet | +0.215811 |
| missing_block | 0.25 | small | world_unet | +0.213692 |
| missing_block | 0.5 | large | world_curve_unet | +0.121035 |
| missing_block | 0.5 | large | world_unet | +0.046622 |
| missing_block | 0.5 | medium | world_curve_unet | +0.206257 |
| missing_block | 0.5 | medium | world_unet | +0.131496 |
| missing_block | 0.5 | small | world_curve_unet | +0.364289 |
| missing_block | 0.5 | small | world_unet | +0.299525 |
| missing_block | 0.75 | large | world_curve_unet | +0.093031 |
| missing_block | 0.75 | large | world_unet | +0.086304 |
| missing_block | 0.75 | medium | world_curve_unet | +0.170853 |
| missing_block | 0.75 | medium | world_unet | +0.158550 |
| missing_block | 0.75 | small | world_curve_unet | +0.372180 |
| missing_block | 0.75 | small | world_unet | +0.335386 |
| missing_block | 0.9 | large | world_curve_unet | +0.059761 |
| missing_block | 0.9 | large | world_unet | +0.121885 |
| missing_block | 0.9 | medium | world_curve_unet | +0.072733 |
| missing_block | 0.9 | medium | world_unet | +0.092372 |
| missing_block | 0.9 | small | world_curve_unet | +0.216832 |
| missing_block | 0.9 | small | world_unet | +0.187901 |
| random | 0.1 | large | world_curve_unet | +0.078295 |
| random | 0.1 | large | world_unet | +0.074405 |
| random | 0.1 | medium | world_curve_unet | +0.088955 |
| random | 0.1 | medium | world_unet | +0.082399 |
| random | 0.1 | small | world_curve_unet | +0.204330 |
| random | 0.1 | small | world_unet | +0.167281 |
| random | 0.25 | large | world_curve_unet | +0.064297 |
| random | 0.25 | large | world_unet | +0.102139 |
| random | 0.25 | medium | world_curve_unet | +0.056852 |
| random | 0.25 | medium | world_unet | +0.086296 |
| random | 0.25 | small | world_curve_unet | +0.187856 |
| random | 0.25 | small | world_unet | +0.161150 |
| random | 0.5 | large | world_curve_unet | +0.032337 |
| random | 0.5 | large | world_unet | +0.100637 |
| random | 0.5 | medium | world_curve_unet | +0.011869 |
| random | 0.5 | medium | world_unet | +0.048511 |
| random | 0.5 | small | world_curve_unet | +0.060764 |
| random | 0.5 | small | world_unet | +0.042988 |
| random | 0.75 | large | world_curve_unet | +0.031654 |
| random | 0.75 | large | world_unet | +0.108166 |
| random | 0.75 | medium | world_curve_unet | +0.002498 |
| random | 0.75 | medium | world_unet | +0.034417 |
| random | 0.75 | small | world_curve_unet | +0.021430 |
| random | 0.75 | small | world_unet | +0.002367 |
| random | 0.9 | large | world_curve_unet | +0.034136 |
| random | 0.9 | large | world_unet | +0.115786 |
| random | 0.9 | medium | world_curve_unet | +0.002001 |
| random | 0.9 | medium | world_unet | +0.032696 |
| random | 0.9 | small | world_curve_unet | +0.013709 |
| random | 0.9 | small | world_unet | -0.008381 |
| raster_lines | 0.1 | large | world_curve_unet | +0.016928 |
| raster_lines | 0.1 | large | world_unet | -0.049393 |
| raster_lines | 0.1 | medium | world_curve_unet | +0.079109 |
| raster_lines | 0.1 | medium | world_unet | +0.015810 |
| raster_lines | 0.1 | small | world_curve_unet | +0.077184 |
| raster_lines | 0.1 | small | world_unet | +0.051696 |
| raster_lines | 0.25 | large | world_curve_unet | +0.065100 |
| raster_lines | 0.25 | large | world_unet | +0.064438 |
| raster_lines | 0.25 | medium | world_curve_unet | +0.044792 |
| raster_lines | 0.25 | medium | world_unet | +0.035688 |
| raster_lines | 0.25 | small | world_curve_unet | +0.133879 |
| raster_lines | 0.25 | small | world_unet | +0.074605 |
| raster_lines | 0.5 | large | world_curve_unet | +0.033608 |
| raster_lines | 0.5 | large | world_unet | +0.094218 |
| raster_lines | 0.5 | medium | world_curve_unet | +0.007166 |
| raster_lines | 0.5 | medium | world_unet | +0.037271 |
| raster_lines | 0.5 | small | world_curve_unet | +0.032901 |
| raster_lines | 0.5 | small | world_unet | +0.017987 |
| raster_lines | 0.75 | large | world_curve_unet | +0.036972 |
| raster_lines | 0.75 | large | world_unet | +0.111756 |
| raster_lines | 0.75 | medium | world_curve_unet | +0.000395 |
| raster_lines | 0.75 | medium | world_unet | +0.033222 |
| raster_lines | 0.75 | small | world_curve_unet | +0.016283 |
| raster_lines | 0.75 | small | world_unet | -0.001580 |
| raster_lines | 0.9 | large | world_curve_unet | +0.032751 |
| raster_lines | 0.9 | large | world_unet | +0.113358 |
| raster_lines | 0.9 | medium | world_curve_unet | +0.000673 |
| raster_lines | 0.9 | medium | world_unet | +0.032242 |
| raster_lines | 0.9 | small | world_curve_unet | +0.017508 |
| raster_lines | 0.9 | small | world_unet | -0.003221 |
| space_filling | 0.1 | large | world_curve_unet | +0.044930 |
| space_filling | 0.1 | large | world_unet | +0.048074 |
| space_filling | 0.1 | medium | world_curve_unet | +0.058288 |
| space_filling | 0.1 | medium | world_unet | +0.079672 |
| space_filling | 0.1 | small | world_curve_unet | +0.173387 |
| space_filling | 0.1 | small | world_unet | +0.128210 |
| space_filling | 0.25 | large | world_curve_unet | +0.036062 |
| space_filling | 0.25 | large | world_unet | +0.072680 |
| space_filling | 0.25 | medium | world_curve_unet | +0.027340 |
| space_filling | 0.25 | medium | world_unet | +0.067960 |
| space_filling | 0.25 | small | world_curve_unet | +0.113266 |
| space_filling | 0.25 | small | world_unet | +0.097775 |
| space_filling | 0.5 | large | world_curve_unet | +0.032187 |
| space_filling | 0.5 | large | world_unet | +0.096469 |
| space_filling | 0.5 | medium | world_curve_unet | +0.005643 |
| space_filling | 0.5 | medium | world_unet | +0.041806 |
| space_filling | 0.5 | small | world_curve_unet | +0.039412 |
| space_filling | 0.5 | small | world_unet | +0.023078 |
| space_filling | 0.75 | large | world_curve_unet | +0.033443 |
| space_filling | 0.75 | large | world_unet | +0.107908 |
| space_filling | 0.75 | medium | world_curve_unet | +0.001328 |
| space_filling | 0.75 | medium | world_unet | +0.033780 |
| space_filling | 0.75 | small | world_curve_unet | +0.015517 |
| space_filling | 0.75 | small | world_unet | -0.001364 |
| space_filling | 0.9 | large | world_curve_unet | +0.033864 |
| space_filling | 0.9 | large | world_unet | +0.112289 |
| space_filling | 0.9 | medium | world_curve_unet | -0.002049 |
| space_filling | 0.9 | medium | world_unet | +0.030690 |
| space_filling | 0.9 | small | world_curve_unet | +0.014326 |
| space_filling | 0.9 | small | world_unet | -0.004957 |
