# Sparse-Context Deep Dive

All rows use 128x128 scan-area GT. Thresholds are selected on validation for each method, sparse ratio, pattern, and seed.

## Ratio-Pattern Delta Summary

| Pattern | Ratio | Method | Mean Test Dice | Delta vs No-World | Sample Delta 95% CI |
|---|---:|---|---:|---:|---:|
| local_block | 0.1 | world_curve_unet | 0.211570 | -0.012482 | [-0.055364, -0.038022] |
| local_block | 0.1 | world_unet | 0.209631 | -0.014421 | [-0.042906, -0.021953] |
| local_block | 0.25 | world_curve_unet | 0.350555 | -0.022424 | [-0.088325, -0.072127] |
| local_block | 0.25 | world_unet | 0.299453 | -0.073526 | [-0.110359, -0.090830] |
| local_block | 0.5 | world_curve_unet | 0.605030 | +0.016133 | [-0.020891, -0.006140] |
| local_block | 0.5 | world_unet | 0.569854 | -0.019043 | [-0.042984, -0.027140] |
| local_block | 0.75 | world_curve_unet | 0.726493 | +0.042183 | [+0.034183, +0.045012] |
| local_block | 0.75 | world_unet | 0.732433 | +0.048124 | [+0.029512, +0.043313] |
| local_block | 0.9 | world_curve_unet | 0.744522 | +0.047195 | [+0.043565, +0.053756] |
| local_block | 0.9 | world_unet | 0.757990 | +0.060664 | [+0.046506, +0.058921] |
| missing_block | 0.1 | world_curve_unet | 0.137508 | -0.038550 | [-0.059844, -0.043462] |
| missing_block | 0.1 | world_unet | 0.196633 | +0.020575 | [-0.007883, +0.013247] |
| missing_block | 0.25 | world_curve_unet | 0.287677 | -0.049725 | [-0.084240, -0.067512] |
| missing_block | 0.25 | world_unet | 0.276685 | -0.060718 | [-0.092398, -0.074132] |
| missing_block | 0.5 | world_curve_unet | 0.515893 | +0.024359 | [-0.004227, +0.013846] |
| missing_block | 0.5 | world_unet | 0.490438 | -0.001096 | [-0.030048, -0.010444] |
| missing_block | 0.75 | world_curve_unet | 0.630412 | +0.060404 | [+0.051501, +0.072609] |
| missing_block | 0.75 | world_unet | 0.613472 | +0.043464 | [+0.028439, +0.049793] |
| missing_block | 0.9 | world_curve_unet | 0.717416 | +0.047395 | [+0.050401, +0.063706] |
| missing_block | 0.9 | world_unet | 0.727301 | +0.057280 | [+0.047976, +0.062626] |
| random | 0.1 | world_curve_unet | 0.418750 | -0.136547 | [-0.170436, -0.152042] |
| random | 0.1 | world_unet | 0.429449 | -0.125848 | [-0.144392, -0.127407] |
| random | 0.25 | world_curve_unet | 0.663769 | +0.012139 | [+0.001839, +0.015694] |
| random | 0.25 | world_unet | 0.661080 | +0.009450 | [-0.009566, +0.005781] |
| random | 0.5 | world_curve_unet | 0.733973 | +0.049131 | [+0.046842, +0.057135] |
| random | 0.5 | world_unet | 0.738628 | +0.053785 | [+0.038702, +0.050981] |
| random | 0.75 | world_curve_unet | 0.744386 | +0.051058 | [+0.048769, +0.058831] |
| random | 0.75 | world_unet | 0.754730 | +0.061403 | [+0.045594, +0.058309] |
| random | 0.9 | world_curve_unet | 0.744916 | +0.048839 | [+0.047326, +0.057063] |
| random | 0.9 | world_unet | 0.760734 | +0.064656 | [+0.051932, +0.064045] |
| raster_lines | 0.1 | world_curve_unet | 0.219252 | -0.055486 | [-0.091388, -0.075218] |
| raster_lines | 0.1 | world_unet | 0.193494 | -0.081244 | [-0.103822, -0.083446] |
| raster_lines | 0.25 | world_curve_unet | 0.685499 | +0.042568 | [+0.023800, +0.036912] |
| raster_lines | 0.25 | world_unet | 0.684738 | +0.041807 | [+0.023332, +0.035195] |
| raster_lines | 0.5 | world_curve_unet | 0.738472 | +0.050722 | [+0.050566, +0.059577] |
| raster_lines | 0.5 | world_unet | 0.747577 | +0.059828 | [+0.046672, +0.058282] |
| raster_lines | 0.75 | world_curve_unet | 0.657161 | +0.027545 | [-0.008405, +0.010099] |
| raster_lines | 0.75 | world_unet | 0.663796 | +0.034180 | [+0.014937, +0.034667] |
| raster_lines | 0.9 | world_curve_unet | 0.738729 | +0.044648 | [+0.039747, +0.050343] |
| raster_lines | 0.9 | world_unet | 0.753105 | +0.059023 | [+0.043664, +0.056484] |
| space_filling | 0.1 | world_curve_unet | 0.516471 | -0.089748 | [-0.121834, -0.103822] |
| space_filling | 0.1 | world_unet | 0.518859 | -0.087359 | [-0.107523, -0.091372] |
| space_filling | 0.25 | world_curve_unet | 0.701649 | +0.034192 | [+0.030707, +0.043644] |
| space_filling | 0.25 | world_unet | 0.705750 | +0.038293 | [+0.029141, +0.042239] |
| space_filling | 0.5 | world_curve_unet | 0.735388 | +0.049234 | [+0.046120, +0.056296] |
| space_filling | 0.5 | world_unet | 0.741279 | +0.055125 | [+0.040487, +0.052838] |
| space_filling | 0.75 | world_curve_unet | 0.743338 | +0.049350 | [+0.046564, +0.056119] |
| space_filling | 0.75 | world_unet | 0.755045 | +0.061057 | [+0.046462, +0.058762] |
| space_filling | 0.9 | world_curve_unet | 0.745413 | +0.049151 | [+0.047411, +0.057481] |
| space_filling | 0.9 | world_unet | 0.760943 | +0.064682 | [+0.051936, +0.063843] |

## No-World Baseline Means

| Pattern | Ratio | Sparse UNet Mean Test Dice |
|---|---:|---:|
| local_block | 0.1 | 0.224052 |
| local_block | 0.25 | 0.372979 |
| local_block | 0.5 | 0.588897 |
| local_block | 0.75 | 0.684310 |
| local_block | 0.9 | 0.697326 |
| missing_block | 0.1 | 0.176058 |
| missing_block | 0.25 | 0.337403 |
| missing_block | 0.5 | 0.491534 |
| missing_block | 0.75 | 0.570008 |
| missing_block | 0.9 | 0.670021 |
| random | 0.1 | 0.555297 |
| random | 0.25 | 0.651630 |
| random | 0.5 | 0.684842 |
| random | 0.75 | 0.693327 |
| random | 0.9 | 0.696077 |
| raster_lines | 0.1 | 0.274738 |
| raster_lines | 0.25 | 0.642931 |
| raster_lines | 0.5 | 0.687750 |
| raster_lines | 0.75 | 0.629616 |
| raster_lines | 0.9 | 0.694081 |
| space_filling | 0.1 | 0.606219 |
| space_filling | 0.25 | 0.667457 |
| space_filling | 0.5 | 0.686154 |
| space_filling | 0.75 | 0.693989 |
| space_filling | 0.9 | 0.696262 |

## Area-Bin Delta Summary

| Pattern | Ratio | Area Bin | Method | Mean Delta vs No-World |
|---|---:|---|---|---:|
| local_block | 0.1 | large | world_curve_unet | +0.019616 |
| local_block | 0.1 | large | world_unet | +0.073953 |
| local_block | 0.1 | medium | world_curve_unet | -0.064513 |
| local_block | 0.1 | medium | world_unet | -0.065222 |
| local_block | 0.1 | small | world_curve_unet | -0.091648 |
| local_block | 0.1 | small | world_unet | -0.100062 |
| local_block | 0.25 | large | world_curve_unet | -0.040274 |
| local_block | 0.25 | large | world_unet | -0.079792 |
| local_block | 0.25 | medium | world_curve_unet | -0.103441 |
| local_block | 0.25 | medium | world_unet | -0.115598 |
| local_block | 0.25 | small | world_curve_unet | -0.094418 |
| local_block | 0.25 | small | world_unet | -0.105736 |
| local_block | 0.5 | large | world_curve_unet | +0.019461 |
| local_block | 0.5 | large | world_unet | +0.009863 |
| local_block | 0.5 | medium | world_curve_unet | -0.044142 |
| local_block | 0.5 | medium | world_unet | -0.079132 |
| local_block | 0.5 | small | world_curve_unet | -0.015812 |
| local_block | 0.5 | small | world_unet | -0.035182 |
| local_block | 0.75 | large | world_curve_unet | +0.047702 |
| local_block | 0.75 | large | world_unet | +0.079340 |
| local_block | 0.75 | medium | world_curve_unet | +0.021343 |
| local_block | 0.75 | medium | world_unet | +0.007467 |
| local_block | 0.75 | small | world_curve_unet | +0.049968 |
| local_block | 0.75 | small | world_unet | +0.023209 |
| local_block | 0.9 | large | world_curve_unet | +0.054004 |
| local_block | 0.9 | large | world_unet | +0.092008 |
| local_block | 0.9 | medium | world_curve_unet | +0.023955 |
| local_block | 0.9 | medium | world_unet | +0.018559 |
| local_block | 0.9 | small | world_curve_unet | +0.066721 |
| local_block | 0.9 | small | world_unet | +0.047173 |
| missing_block | 0.1 | large | world_curve_unet | -0.003246 |
| missing_block | 0.1 | large | world_unet | +0.147428 |
| missing_block | 0.1 | medium | world_curve_unet | -0.069067 |
| missing_block | 0.1 | medium | world_unet | -0.030105 |
| missing_block | 0.1 | small | world_curve_unet | -0.080609 |
| missing_block | 0.1 | small | world_unet | -0.100216 |
| missing_block | 0.25 | large | world_curve_unet | -0.047151 |
| missing_block | 0.25 | large | world_unet | +0.009126 |
| missing_block | 0.25 | medium | world_curve_unet | -0.104501 |
| missing_block | 0.25 | medium | world_unet | -0.113579 |
| missing_block | 0.25 | small | world_curve_unet | -0.076964 |
| missing_block | 0.25 | small | world_unet | -0.142350 |
| missing_block | 0.5 | large | world_curve_unet | -0.010940 |
| missing_block | 0.5 | large | world_unet | +0.017682 |
| missing_block | 0.5 | medium | world_curve_unet | -0.025968 |
| missing_block | 0.5 | medium | world_unet | -0.032405 |
| missing_block | 0.5 | small | world_curve_unet | +0.048268 |
| missing_block | 0.5 | small | world_unet | -0.043483 |
| missing_block | 0.75 | large | world_curve_unet | +0.019524 |
| missing_block | 0.75 | large | world_unet | +0.036063 |
| missing_block | 0.75 | medium | world_curve_unet | +0.011341 |
| missing_block | 0.75 | medium | world_unet | -0.007334 |
| missing_block | 0.75 | small | world_curve_unet | +0.148699 |
| missing_block | 0.75 | small | world_unet | +0.086032 |
| missing_block | 0.9 | large | world_curve_unet | +0.040051 |
| missing_block | 0.9 | large | world_unet | +0.080165 |
| missing_block | 0.9 | medium | world_curve_unet | +0.029490 |
| missing_block | 0.9 | medium | world_unet | +0.019317 |
| missing_block | 0.9 | small | world_curve_unet | +0.097964 |
| missing_block | 0.9 | small | world_unet | +0.065841 |
| random | 0.1 | large | world_curve_unet | -0.130049 |
| random | 0.1 | large | world_unet | -0.069957 |
| random | 0.1 | medium | world_curve_unet | -0.167824 |
| random | 0.1 | medium | world_unet | -0.144633 |
| random | 0.1 | small | world_curve_unet | -0.184292 |
| random | 0.1 | small | world_unet | -0.187870 |
| random | 0.25 | large | world_curve_unet | +0.005332 |
| random | 0.25 | large | world_unet | +0.033606 |
| random | 0.25 | medium | world_curve_unet | +0.008134 |
| random | 0.25 | medium | world_unet | -0.004974 |
| random | 0.25 | small | world_curve_unet | +0.011237 |
| random | 0.25 | small | world_unet | -0.031678 |
| random | 0.5 | large | world_curve_unet | +0.051170 |
| random | 0.5 | large | world_unet | +0.079593 |
| random | 0.5 | medium | world_curve_unet | +0.039960 |
| random | 0.5 | medium | world_unet | +0.025375 |
| random | 0.5 | small | world_curve_unet | +0.064568 |
| random | 0.5 | small | world_unet | +0.030432 |
| random | 0.75 | large | world_curve_unet | +0.058479 |
| random | 0.75 | large | world_unet | +0.094082 |
| random | 0.75 | medium | world_curve_unet | +0.031438 |
| random | 0.75 | medium | world_unet | +0.020837 |
| random | 0.75 | small | world_curve_unet | +0.069960 |
| random | 0.75 | small | world_unet | +0.041552 |
| random | 0.9 | large | world_curve_unet | +0.056146 |
| random | 0.9 | large | world_unet | +0.095848 |
| random | 0.9 | medium | world_curve_unet | +0.028321 |
| random | 0.9 | medium | world_unet | +0.022519 |
| random | 0.9 | small | world_curve_unet | +0.070582 |
| random | 0.9 | small | world_unet | +0.054779 |
| raster_lines | 0.1 | large | world_curve_unet | -0.041118 |
| raster_lines | 0.1 | large | world_unet | -0.023349 |
| raster_lines | 0.1 | medium | world_curve_unet | -0.059653 |
| raster_lines | 0.1 | medium | world_unet | -0.048785 |
| raster_lines | 0.1 | small | world_curve_unet | -0.144391 |
| raster_lines | 0.1 | small | world_unet | -0.199504 |
| raster_lines | 0.25 | large | world_curve_unet | +0.032463 |
| raster_lines | 0.25 | large | world_unet | +0.056829 |
| raster_lines | 0.25 | medium | world_curve_unet | +0.045272 |
| raster_lines | 0.25 | medium | world_unet | +0.027556 |
| raster_lines | 0.25 | small | world_curve_unet | +0.014962 |
| raster_lines | 0.25 | small | world_unet | +0.006450 |
| raster_lines | 0.5 | large | world_curve_unet | +0.055019 |
| raster_lines | 0.5 | large | world_unet | +0.085531 |
| raster_lines | 0.5 | medium | world_curve_unet | +0.042645 |
| raster_lines | 0.5 | medium | world_unet | +0.034679 |
| raster_lines | 0.5 | small | world_curve_unet | +0.066644 |
| raster_lines | 0.5 | small | world_unet | +0.038264 |
| raster_lines | 0.75 | large | world_curve_unet | +0.033162 |
| raster_lines | 0.75 | large | world_unet | +0.044200 |
| raster_lines | 0.75 | medium | world_curve_unet | -0.043475 |
| raster_lines | 0.75 | medium | world_unet | -0.026742 |
| raster_lines | 0.75 | small | world_curve_unet | +0.011285 |
| raster_lines | 0.75 | small | world_unet | +0.053466 |
| raster_lines | 0.9 | large | world_curve_unet | +0.055717 |
| raster_lines | 0.9 | large | world_unet | +0.098144 |
| raster_lines | 0.9 | medium | world_curve_unet | +0.016108 |
| raster_lines | 0.9 | medium | world_unet | +0.009297 |
| raster_lines | 0.9 | small | world_curve_unet | +0.062026 |
| raster_lines | 0.9 | small | world_unet | +0.043192 |
| space_filling | 0.1 | large | world_curve_unet | -0.099156 |
| space_filling | 0.1 | large | world_unet | -0.033872 |
| space_filling | 0.1 | medium | world_curve_unet | -0.113927 |
| space_filling | 0.1 | medium | world_unet | -0.118875 |
| space_filling | 0.1 | small | world_curve_unet | -0.124238 |
| space_filling | 0.1 | small | world_unet | -0.143656 |
| space_filling | 0.25 | large | world_curve_unet | +0.021102 |
| space_filling | 0.25 | large | world_unet | +0.049147 |
| space_filling | 0.25 | medium | world_curve_unet | +0.046212 |
| space_filling | 0.25 | medium | world_unet | +0.028906 |
| space_filling | 0.25 | small | world_curve_unet | +0.043457 |
| space_filling | 0.25 | small | world_unet | +0.029901 |
| space_filling | 0.5 | large | world_curve_unet | +0.052586 |
| space_filling | 0.5 | large | world_unet | +0.080291 |
| space_filling | 0.5 | medium | world_curve_unet | +0.039276 |
| space_filling | 0.5 | medium | world_unet | +0.027067 |
| space_filling | 0.5 | small | world_curve_unet | +0.061390 |
| space_filling | 0.5 | small | world_unet | +0.033832 |
| space_filling | 0.75 | large | world_curve_unet | +0.057879 |
| space_filling | 0.75 | large | world_unet | +0.092783 |
| space_filling | 0.75 | medium | world_curve_unet | +0.030191 |
| space_filling | 0.75 | medium | world_unet | +0.022619 |
| space_filling | 0.75 | small | world_curve_unet | +0.065466 |
| space_filling | 0.75 | small | world_unet | +0.043158 |
| space_filling | 0.9 | large | world_curve_unet | +0.056730 |
| space_filling | 0.9 | large | world_unet | +0.096314 |
| space_filling | 0.9 | medium | world_curve_unet | +0.027854 |
| space_filling | 0.9 | medium | world_unet | +0.023022 |
| space_filling | 0.9 | small | world_curve_unet | +0.071144 |
| space_filling | 0.9 | small | world_unet | +0.054702 |
