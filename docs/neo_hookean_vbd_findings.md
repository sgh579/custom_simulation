# Neo-Hookean VBD Benchmark Findings

Date: 2026-06-03

## Purpose

The benchmark series tested whether the current Newton/VBD soft-mesh pipeline,
using a Neo-Hookean-like material assignment plus penalty contact, can generate
smooth and physically plausible F-z trajectories for hard-inclusion synthetic
palpation data.

The target curve family is monotone and smooth, with increasing slope as
indentation grows and local slope changes caused by a recognizable stiff
inclusion.

## Runs

Primary generated artifacts were written under:

```text
runs/perf_curve_center_sphere/
runs/hard_inclusion_fz_dataset/
```

The center-sphere performance sweep includes 13 completed runs, from tiny smoke
cases through an hour-scale 96x96x32 mesh run:

```text
runs/perf_curve_center_sphere/benchmark_summary.csv
runs/perf_curve_center_sphere/benchmark_perf_curves.png
runs/perf_curve_center_sphere/all_run_fz_curves.png
```

Generated run artifacts are intentionally not tracked by Git.

## Findings

The continuous Neo-Hookean material model can in principle support smooth,
monotone, convex indentation curves. In this implementation, however, the saved
F-z values are instantaneous penalty-contact reaction estimates after finite VBD
iterations, not exact derivatives of a converged quasi-static energy.

The benchmark curves showed persistent local force decreases and contact-driven
roughness. Increasing mesh size and solver work reduced some noise but did not
make the trajectories reliable enough for synthetic training data at practical
cost.

Representative non-monotonicity from the center-sphere sweep:

```text
h01 negative_step_fraction = 0.089
h02 negative_step_fraction = 0.065
h03 negative_step_fraction = 0.051
```

The hour-scale h03 run still retained non-monotone steps, despite using a
96x96x32 mesh, 15x15 scan grid, 96 press steps, 8 substeps, and 16 VBD
iterations.

## Conclusion

The current Neo-Hookean/VBD/penalty-contact pipeline should not be used as the
production generator for synthetic F-z training data without a major change in
the force extraction or quasi-static solve strategy.

The committed code preserves the useful experiment infrastructure:

- date-prefixed run directories,
- sample and dataset resource metadata,
- disk/GPU usage recording,
- hard-inclusion F-z dataset generation,
- centered-sphere performance benchmarking support.
