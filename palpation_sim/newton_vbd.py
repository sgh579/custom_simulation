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
        device: str | None = None,
    ) -> None:
        self.phantom = phantom
        self.material = material
        self.scan = scan
        self.newton, self.wp, self.SolverVBD = _import_newton(newton_root)
        self.device = device

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
                for step, depth in enumerate(depths):
                    z = self.phantom.height + scan.probe_radius + scan.preload_gap - float(depth)
                    vz = (z - previous_z) / max(scan.sim_dt, 1e-9)

                    for _ in range(scan.sim_substeps_per_depth):
                        state_0.clear_forces()
                        state_1.clear_forces()
                        _set_probe_kinematic_pose(wp, model, state_0, probe_body, float(x), float(y), z, vz)
                        _set_probe_kinematic_pose(wp, model, state_1, probe_body, float(x), float(y), z, vz)
                        if hasattr(solver, "rebuild_bvh"):
                            solver.rebuild_bvh(state_0)
                        collision_pipeline.collide(state_0, contacts)
                        solver.step(state_0, state_1, control, contacts, scan.sim_dt)
                        state_0, state_1 = state_1, state_0

                    _set_probe_kinematic_pose(wp, model, state_0, probe_body, float(x), float(y), z, 0.0)
                    collision_pipeline.collide(state_0, contacts)
                    force_z, patch = _estimate_probe_reaction_z(model, state_0, contacts, solver, probe_shape, self.material)

                    presses[row, col, step, 0] = depth
                    presses[row, col, step, 1] = force_z
                    fz[row, col, step] = force_z
                    probe_pose[row, col, step] = np.asarray([x, y, z, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                    contact_features[row, col, step] = patch
                    previous_z = z

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
        sys.path.insert(0, str(Path(newton_root).expanduser()))
    else:
        default_root = Path("/home/goodmansun/newton")
        if default_root.exists():
            sys.path.insert(0, str(default_root))
    try:
        import newton  # type: ignore[import-not-found]
        import warp as wp  # type: ignore[import-not-found]
        from newton.solvers import SolverVBD  # type: ignore[import-not-found]
    except ImportError as exc:
        raise NewtonUnavailableError(
            "Cannot import Newton/Warp. Run with /home/goodmansun/newton/.venv/bin/python "
            "or pass --newton-root to the dataset generator."
        ) from exc
    return newton, wp, SolverVBD


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
