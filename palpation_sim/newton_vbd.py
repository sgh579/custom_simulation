from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import MaterialConfig, PhantomConfig, ScanConfig
from .phantom import (
    LumpSpec,
    create_structured_tet_mesh,
    lumps_to_json,
    mask_for_scan_grid,
    material_arrays_for_lumps,
    normalize_lumps,
)
from .workflow import DEFAULT_NEWTON_ROOT, REQUIRED_NEWTON_DEVICE, require_runtime_environment


class NewtonUnavailableError(RuntimeError):
    """Raised when Newton/Warp cannot be imported from the requested environment."""


class NewtonVBDPalpationSimulator:
    """Newton/VBD phantom palpation simulator with a kinematic spherical probe."""

    def __init__(
        self,
        phantom: PhantomConfig,
        material: MaterialConfig,
        scan: ScanConfig,
        *,
        newton_root: str | Path | None = None,
        device: str | None = REQUIRED_NEWTON_DEVICE,
    ) -> None:
        self.phantom = phantom
        self.material = material
        self.scan = scan
        require_runtime_environment(require_newton=True, newton_root=newton_root)
        self.newton, self.wp, self.SolverVBD = _import_newton(newton_root)
        self.device = _resolve_device(self.wp, device)

    def run_sample(self, lumps: LumpSpec | Sequence[LumpSpec]) -> dict[str, np.ndarray | str]:
        lump_list = normalize_lumps(lumps)
        mesh = create_structured_tet_mesh(self.phantom)
        k_mu, k_lambda, k_damp, tet_lump_mask, tet_lump_id = material_arrays_for_lumps(mesh, self.material, lump_list)
        model, probe_body, probe_shape = self._build_model(mesh, k_mu, k_lambda, k_damp)

        wp = self.wp
        newton = self.newton
        scan = self.scan
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()

        initial_particle_q = wp.clone(state_0.particle_q)
        initial_body_q = wp.clone(state_0.body_q)

        collision_pipeline = newton.CollisionPipeline(model, soft_contact_margin=scan.soft_contact_margin)
        contacts = collision_pipeline.contacts()
        solver = self.SolverVBD(
            model,
            iterations=scan.vbd_iterations,
            integrate_with_external_rigid_solver=True,
            particle_enable_self_contact=False,
            particle_collision_detection_interval=-1,
        )

        xs = np.asarray(scan.x_values(self.phantom), dtype=np.float32)
        ys = np.asarray(scan.y_values(self.phantom), dtype=np.float32)
        depths = np.asarray(scan.indentation_values(), dtype=np.float32)
        h, w, t = scan.grid_h, scan.grid_w, scan.press_steps

        presses = np.zeros((h, w, t, 2), dtype=np.float32)
        probe_pose = np.zeros((h, w, t, 7), dtype=np.float32)
        indentation = np.broadcast_to(depths, (h, w, t)).copy().astype(np.float32)
        fz = np.zeros((h, w, t), dtype=np.float32)
        contact_features = np.zeros((h, w, t, 5), dtype=np.float32)

        for row, y in enumerate(ys):
            for col, x in enumerate(xs):
                if scan.reset_between_points:
                    _reset_state(wp, state_0, state_1, initial_particle_q, initial_body_q)

                previous_z = self.phantom.height + scan.probe_radius + scan.preload_gap
                substeps = max(int(scan.sim_substeps_per_depth), 1)
                for step, depth in enumerate(depths):
                    target_z = self.phantom.height + scan.probe_radius + scan.preload_gap - float(depth)

                    for substep in range(substeps):
                        alpha_0 = float(substep) / float(substeps)
                        alpha_1 = float(substep + 1) / float(substeps)
                        z_0 = previous_z + (target_z - previous_z) * alpha_0
                        z_1 = previous_z + (target_z - previous_z) * alpha_1
                        vz = (z_1 - z_0) / max(scan.sim_dt, 1e-9)
                        state_0.clear_forces()
                        state_1.clear_forces()
                        _copy_particle_state(state_0, state_1)
                        _set_probe_kinematic_pose(wp, model, state_0, probe_body, float(x), float(y), z_0, vz)
                        _set_probe_kinematic_pose(wp, model, state_1, probe_body, float(x), float(y), z_1, vz)
                        if hasattr(solver, "rebuild_bvh"):
                            solver.rebuild_bvh(state_0)
                        collision_pipeline.collide(state_1, contacts)
                        solver.step(state_0, state_1, control, contacts, scan.sim_dt)
                        state_0, state_1 = state_1, state_0

                    _set_probe_kinematic_pose(wp, model, state_0, probe_body, float(x), float(y), target_z, 0.0)
                    collision_pipeline.collide(state_0, contacts)
                    force_z, patch = _estimate_probe_reaction_z(model, state_0, contacts, solver, probe_shape, self.material)

                    presses[row, col, step, 0] = depth
                    presses[row, col, step, 1] = force_z
                    fz[row, col, step] = force_z
                    probe_pose[row, col, step] = np.asarray([x, y, target_z, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                    contact_features[row, col, step] = patch
                    previous_z = target_z

        nonlinearity_ratio = _nonlinearity_ratio_map(indentation, fz)
        xy_grid = np.stack(np.meshgrid(xs, ys), axis=-1).astype(np.float32)
        first_lump_json = json.dumps(lump_list[0].to_dict(self.phantom)) if lump_list else "{}"
        return {
            "presses": presses,
            "mask": mask_for_scan_grid(scan, self.phantom, lump_list),
            "xy": xy_grid,
            "probe_pose": probe_pose,
            "indentation_depth": indentation,
            "fz": fz,
            "contact_features": contact_features,
            "nonlinearity_ratio": nonlinearity_ratio,
            "tet_lump_mask": tet_lump_mask,
            "tet_lump_id": tet_lump_id,
            "lump_json": first_lump_json,
            "lumps_json": lumps_to_json(lump_list, self.phantom),
            "num_lumps": np.asarray(len(lump_list), dtype=np.int32),
            "backend": np.asarray("newton_vbd"),
        }

    def _build_model(
        self,
        mesh: Any,
        k_mu: np.ndarray,
        k_lambda: np.ndarray,
        k_damp: np.ndarray,
    ) -> tuple[Any, int, int]:
        newton = self.newton
        wp = self.wp
        builder = newton.ModelBuilder(gravity=0.0)

        vertices = [wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in mesh.vertices]
        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=vertices,
            indices=mesh.flat_indices,
            density=self.phantom.density,
            k_mu=k_mu,
            k_lambda=k_lambda,
            k_damp=k_damp,
            particle_radius=self.phantom.particle_radius,
            label="phantom",
        )

        probe_body = builder.add_body(
            xform=wp.transform(
                wp.vec3(0.0, 0.0, self.phantom.height + self.scan.probe_radius + self.scan.preload_gap),
                wp.quat_identity(),
            ),
            label="kinematic_probe",
            is_kinematic=True,
        )
        shape_cfg = newton.ModelBuilder.ShapeConfig()
        shape_cfg.density = 0.0
        shape_cfg.ke = self.material.soft_contact_ke
        shape_cfg.kd = self.material.soft_contact_kd
        shape_cfg.mu = self.material.probe_contact_mu
        probe_shape = builder.add_shape_sphere(
            probe_body,
            radius=self.scan.probe_radius,
            cfg=shape_cfg,
            label="kinematic_probe_sphere",
        )

        builder.color()
        model = builder.finalize(device=self.device, requires_grad=False) if self.device else builder.finalize(requires_grad=False)
        model.soft_contact_ke = self.material.soft_contact_ke
        model.soft_contact_kd = self.material.soft_contact_kd
        model.soft_contact_mu = self.material.soft_contact_mu
        model.shape_material_ke.fill_(self.material.soft_contact_ke)
        model.shape_material_kd.fill_(self.material.soft_contact_kd)
        model.shape_material_mu.fill_(self.material.probe_contact_mu)
        _fix_bottom_particles(wp, model, mesh.bottom_vertex_mask)
        return model, probe_body, probe_shape


def _import_newton(newton_root: str | Path | None) -> tuple[Any, Any, Any]:
    if newton_root is not None:
        root = Path(newton_root).expanduser()
        if root != DEFAULT_NEWTON_ROOT:
            raise NewtonUnavailableError(f"Newton root is pinned to {DEFAULT_NEWTON_ROOT}; got {root}")
        sys.path.insert(0, str(root))
    else:
        default_root = DEFAULT_NEWTON_ROOT
        if default_root.exists():
            sys.path.insert(0, str(default_root))
    try:
        import newton  # type: ignore[import-not-found]
        import warp as wp  # type: ignore[import-not-found]
        from newton.solvers import SolverVBD  # type: ignore[import-not-found]
    except ImportError as exc:
        raise NewtonUnavailableError(
            "Cannot import Newton/Warp. This workflow expects conda env 'palpation' "
            f"and Newton at {DEFAULT_NEWTON_ROOT}."
        ) from exc
    return newton, wp, SolverVBD


def _resolve_device(wp: Any, device: str | None) -> str:
    if device is None or str(device).strip() == "":
        requested = REQUIRED_NEWTON_DEVICE
    else:
        requested = str(device).strip()
    if requested.lower() == "auto":
        raise RuntimeError(
            "Device 'auto' is disabled for Newton/VBD simulation. "
            f"Use the pinned GPU device '{REQUIRED_NEWTON_DEVICE}'."
        )
    if requested == "cuda":
        requested = REQUIRED_NEWTON_DEVICE
    if requested != REQUIRED_NEWTON_DEVICE:
        raise RuntimeError(
            f"Newton/VBD simulation is pinned to '{REQUIRED_NEWTON_DEVICE}' in this workflow; got '{requested}'."
        )
    if not requested.startswith("cuda"):
        raise RuntimeError(
            f"Newton/VBD simulation is GPU-only in this workflow; got device '{requested}'. "
            f"Use '{REQUIRED_NEWTON_DEVICE}'."
        )
    try:
        devices = [str(candidate) for candidate in wp.get_devices()]
    except Exception as exc:
        raise RuntimeError(f"Cannot query Warp devices while requiring '{requested}'.") from exc
    if requested not in devices:
        raise RuntimeError(f"Required Warp CUDA device '{requested}' is not visible. Visible devices: {devices}")
    return requested


def _fix_bottom_particles(wp: Any, model: Any, bottom_mask: np.ndarray) -> None:
    mass = model.particle_mass.numpy()
    inv_mass = model.particle_inv_mass.numpy()
    mass[bottom_mask] = 0.0
    inv_mass[bottom_mask] = 0.0
    model.particle_mass.assign(wp.array(mass, dtype=wp.float32, device=model.device))
    model.particle_inv_mass.assign(wp.array(inv_mass, dtype=wp.float32, device=model.device))


def _reset_state(wp: Any, state_0: Any, state_1: Any, initial_particle_q: Any, initial_body_q: Any) -> None:
    state_0.particle_q.assign(initial_particle_q)
    state_1.particle_q.assign(initial_particle_q)
    state_0.particle_qd.zero_()
    state_1.particle_qd.zero_()
    state_0.body_q.assign(initial_body_q)
    state_1.body_q.assign(initial_body_q)
    state_0.body_qd.zero_()
    state_1.body_qd.zero_()


def _copy_particle_state(state_in: Any, state_out: Any) -> None:
    state_out.particle_q.assign(state_in.particle_q)
    state_out.particle_qd.assign(state_in.particle_qd)


def _set_probe_kinematic_pose(
    wp: Any,
    model: Any,
    state: Any,
    body_index: int,
    x: float,
    y: float,
    z: float,
    vz: float,
) -> None:
    body_q = state.body_q.numpy()
    body_q[body_index] = np.asarray([x, y, z, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    state.body_q = wp.array(body_q, dtype=wp.transform, device=model.device)

    body_qd = state.body_qd.numpy()
    body_qd[body_index] = np.asarray([0.0, 0.0, vz, 0.0, 0.0, 0.0], dtype=np.float32)
    state.body_qd = wp.array(body_qd, dtype=wp.spatial_vector, device=model.device)


def _estimate_probe_reaction_z(
    model: Any,
    state: Any,
    contacts: Any,
    solver: Any,
    probe_shape: int,
    material: MaterialConfig,
) -> tuple[float, np.ndarray]:
    count = int(contacts.soft_contact_count.numpy()[0])
    if count <= 0:
        return 0.0, np.zeros(5, dtype=np.float32)

    shape = contacts.soft_contact_shape.numpy()[:count]
    keep = shape == probe_shape
    if not np.any(keep):
        return 0.0, np.zeros(5, dtype=np.float32)

    idx = np.nonzero(keep)[0]
    particles = contacts.soft_contact_particle.numpy()[:count][idx].astype(np.int64)
    body_pos = contacts.soft_contact_body_pos.numpy()[:count][idx].astype(np.float32)
    normals = contacts.soft_contact_normal.numpy()[:count][idx].astype(np.float32)
    particle_q = state.particle_q.numpy()[particles].astype(np.float32)
    particle_radius = model.particle_radius.numpy()[particles].astype(np.float32)
    shape_body = model.shape_body.numpy()[probe_shape]
    body_q = state.body_q.numpy()[shape_body].astype(np.float32)
    bx = _transform_points(body_q, body_pos)

    penetration = -(np.einsum("ij,ij->i", normals, particle_q - bx) - particle_radius)
    penetration = np.maximum(penetration, 0.0).astype(np.float32)
    if hasattr(solver, "body_particle_contact_penalty_k") and solver.body_particle_contact_penalty_k.shape[0] >= count:
        ke = solver.body_particle_contact_penalty_k.numpy()[:count][idx].astype(np.float32)
    else:
        shape_ke = float(model.shape_material_ke.numpy()[probe_shape])
        ke = np.full(idx.shape[0], 0.5 * (material.soft_contact_ke + shape_ke), dtype=np.float32)

    particle_force = normals * (penetration * ke)[:, None]
    reaction_z = float(-np.sum(particle_force[:, 2]))
    patch = np.asarray(
        [
            float(idx.shape[0]),
            float(np.mean(penetration)) if penetration.size else 0.0,
            float(np.max(penetration)) if penetration.size else 0.0,
            float(np.mean(bx[:, 2])) if bx.size else 0.0,
            float(np.mean(normals[:, 2])) if normals.size else 0.0,
        ],
        dtype=np.float32,
    )
    return max(reaction_z, 0.0), patch


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    translation = transform[:3]
    quat = transform[3:7]
    return _quat_rotate(quat, points) + translation


def _quat_rotate(quat: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    q_xyz = quat[:3]
    q_w = float(quat[3])
    uv = np.cross(q_xyz, vectors)
    uuv = np.cross(q_xyz, uv)
    return vectors + 2.0 * (q_w * uv + uuv)


def _nonlinearity_ratio_map(depth: np.ndarray, fz: np.ndarray) -> np.ndarray:
    ratios = np.zeros(fz.shape[:2], dtype=np.float32)
    for row in range(fz.shape[0]):
        for col in range(fz.shape[1]):
            z = depth[row, col].astype(np.float64)
            f = fz[row, col].astype(np.float64)
            early = _segment_slope(z, f, 0.10, 0.35)
            late = _segment_slope(z, f, 0.65, 0.90)
            ratios[row, col] = np.float32(late / early) if early > 1.0e-12 else np.float32(0.0)
    return ratios


def _segment_slope(z: np.ndarray, f: np.ndarray, lo: float, hi: float) -> float:
    if z.size < 2:
        return 0.0
    span = float(np.max(z) - np.min(z))
    if span <= 1.0e-12:
        return 0.0
    keep = (z >= float(np.min(z)) + lo * span) & (z <= float(np.min(z)) + hi * span)
    if int(np.count_nonzero(keep)) < 2:
        return 0.0
    zz = z[keep].astype(np.float64)
    ff = f[keep].astype(np.float64)
    zc = zz - float(np.mean(zz))
    denom = float(np.sum(zc * zc))
    if denom <= 1.0e-18:
        return 0.0
    return float(np.sum(zc * (ff - float(np.mean(ff)))) / denom)
