from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class PhantomConfig:
    """Geometry and discretization of the soft phantom."""

    size_x: float = 0.18
    size_y: float = 0.18
    height: float = 0.08
    cells_x: int = 32
    cells_y: int = 32
    cells_z: int = 12
    density: float = 500.0
    particle_radius: float = 0.004

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass
class MaterialConfig:
    """Base material parameters and contact material for Newton/VBD."""

    k_mu: float = 2.0e5
    k_lambda: float = 2.0e5
    k_damp: float = 1.0e-4
    soft_contact_ke: float = 2.0e6
    soft_contact_kd: float = 1.0e-7
    soft_contact_mu: float = 0.5
    probe_contact_mu: float = 0.8
    lump_stiffness_min: float = 2.0
    lump_stiffness_max: float = 8.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class ScanConfig:
    """Grid scan and probe motion settings."""

    grid_h: int = 9
    grid_w: int = 9
    edge_margin: float = 0.015
    probe_radius: float = 0.012
    max_indentation: float = 0.018
    press_steps: int = 16
    sim_substeps_per_depth: int = 3
    sim_dt: float = 1.0 / 600.0
    vbd_iterations: int = 5
    soft_contact_margin: float = 0.004
    preload_gap: float = 0.0005
    reset_between_points: bool = True

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)

    def x_values(self, phantom: PhantomConfig) -> list[float]:
        lo = -0.5 * phantom.size_x + self.edge_margin
        hi = 0.5 * phantom.size_x - self.edge_margin
        return _linspace(lo, hi, self.grid_w)

    def y_values(self, phantom: PhantomConfig) -> list[float]:
        lo = -0.5 * phantom.size_y + self.edge_margin
        hi = 0.5 * phantom.size_y - self.edge_margin
        return _linspace(lo, hi, self.grid_h)

    def indentation_values(self) -> list[float]:
        return _linspace(0.0, self.max_indentation, self.press_steps)


def _linspace(start: float, stop: float, count: int) -> list[float]:
    if count <= 1:
        return [0.5 * (start + stop)]
    step = (stop - start) / float(count - 1)
    return [start + step * i for i in range(count)]
