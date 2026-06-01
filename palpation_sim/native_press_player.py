from __future__ import annotations

from pathlib import Path

import numpy as np

from .native_data import load_phantom_scan_material_lumps, load_sample_arrays, resolve_sample_or_metadata
from .visual_geometry import (
    LUMP_COLORS,
    analytic_lump_polydata,
    deformed_body_points,
    deformed_surface_points,
    phantom_box_polydata,
    require_pyvista,
    scan_points_polydata,
    top_surface_polydata,
)
from .vtk_exports import tet_grid_from_sample

MeshStyle = str


def run_press_player(
    selector: Path,
    *,
    data_dir: Path | None = None,
    surface_resolution: int = 96,
    start_row: int | None = None,
    start_col: int | None = None,
    mesh_style: MeshStyle = "continuous",
    tet_stride: int = 64,
    vertex_stride: int = 16,
) -> None:
    deps = _require_player_deps()
    QtCore = deps["QtCore"]
    QtWidgets = deps["QtWidgets"]

    sample_path, metadata_path = resolve_sample_or_metadata(selector, data_dir)
    if sample_path is None:
        raise SystemExit("The native press player needs a .npz sample.")

    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QtWidgets.QApplication([])

    window_class = _make_press_player_window_class(deps)
    window = window_class(
        deps,
        sample_path=sample_path,
        metadata_path=metadata_path,
        surface_resolution=surface_resolution,
        start_row=start_row,
        start_col=start_col,
        mesh_style=mesh_style,
        tet_stride=tet_stride,
        vertex_stride=vertex_stride,
    )
    window.resize(1360, 860)
    window.show()
    if owns_app:
        app.exec()


def _require_player_deps() -> dict[str, object]:
    pv = require_pyvista()
    try:
        from PySide6 import QtCore, QtWidgets
        from pyvistaqt import QtInteractor
        import pyqtgraph as pg
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The native press player requires PySide6, pyvistaqt, and pyqtgraph. "
            "Recreate the environment from environment.yml."
        ) from exc
    return {"pv": pv, "QtCore": QtCore, "QtWidgets": QtWidgets, "QtInteractor": QtInteractor, "pg": pg}


def _make_press_player_window_class(deps: dict[str, object]):
    QtWidgets = deps["QtWidgets"]

    class PressPlayerWindow(_PressPlayerWindowMixin, QtWidgets.QMainWindow):  # type: ignore[attr-defined,misc]
        pass

    return PressPlayerWindow


class _PressPlayerWindowMixin:
    def __init__(
        self,
        deps: dict[str, object],
        *,
        sample_path: Path,
        metadata_path: Path | None,
        surface_resolution: int,
        start_row: int | None,
        start_col: int | None,
        mesh_style: MeshStyle,
        tet_stride: int,
        vertex_stride: int,
    ) -> None:
        super().__init__()
        self.deps = deps
        self.pv = deps["pv"]
        self.QtCore = deps["QtCore"]
        self.QtWidgets = deps["QtWidgets"]
        self.QtInteractor = deps["QtInteractor"]
        self.pg = deps["pg"]
        self.mesh_style = _normalize_mesh_style(mesh_style)
        self.tet_stride = max(int(tet_stride), 1)
        self.vertex_stride = max(int(vertex_stride), 1)
        self.fem_deform_targets: list[tuple[object, np.ndarray]] = []

        self.sample_path = sample_path
        self.metadata_path = metadata_path
        self.phantom, self.scan, _material, self.lumps, _metadata = load_phantom_scan_material_lumps(
            sample_path=sample_path,
            metadata_path=metadata_path,
        )
        keys = {"xy", "indentation_depth", "fz", "mask", "nonlinearity_ratio"}
        if self._uses_discrete_mesh:
            keys |= {"mesh_vertices", "mesh_tets", "tet_lump_id", "tet_lump_mask"}
        arrays = load_sample_arrays(sample_path, keys)
        self.xy = np.asarray(arrays["xy"], dtype=np.float32)
        self.indentation = np.asarray(arrays["indentation_depth"], dtype=np.float32)
        self.fz = np.asarray(arrays["fz"], dtype=np.float32)
        self.mask = np.asarray(arrays["mask"], dtype=np.float32) if "mask" in arrays else None
        self.mesh_arrays = {
            key: arrays[key]
            for key in ("mesh_vertices", "mesh_tets", "tet_lump_id", "tet_lump_mask")
            if key in arrays
        }
        self.rows, self.cols, self.steps = self.fz.shape
        peak = np.nanmax(self.fz, axis=-1)
        default_row, default_col = (int(v) for v in np.unravel_index(np.nanargmax(peak), peak.shape))
        self.row = _clamp_index(default_row if start_row is None else start_row, self.rows)
        self.col = _clamp_index(default_col if start_col is None else start_col, self.cols)
        self.step = 0
        self._updating_controls = False

        self.setWindowTitle(f"Native press player - {sample_path.name} [{self.mesh_style}]")
        self._build_ui()
        self._build_scene(surface_resolution)
        self._update_curve()
        self._update_frame()

        self.timer = self.QtCore.QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._advance)

    def _build_ui(self) -> None:
        QtCore = self.QtCore
        QtWidgets = self.QtWidgets
        pg = self.pg

        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(splitter)
        self.setCentralWidget(central)

        self.plotter = self.QtInteractor(parent=splitter)
        splitter.addWidget(self.plotter.interactor)

        side = QtWidgets.QWidget()
        side_layout = QtWidgets.QVBoxLayout(side)
        side_layout.setContentsMargins(12, 12, 12, 12)
        side_layout.setSpacing(10)
        splitter.addWidget(side)
        splitter.setSizes([920, 420])

        self.info = QtWidgets.QLabel()
        self.info.setWordWrap(True)
        side_layout.addWidget(self.info)

        form = QtWidgets.QGridLayout()
        side_layout.addLayout(form)
        form.addWidget(QtWidgets.QLabel("row"), 0, 0)
        self.row_spin = QtWidgets.QSpinBox()
        self.row_spin.setRange(0, self.rows - 1)
        self.row_spin.setValue(self.row)
        form.addWidget(self.row_spin, 0, 1)
        form.addWidget(QtWidgets.QLabel("col"), 0, 2)
        self.col_spin = QtWidgets.QSpinBox()
        self.col_spin.setRange(0, self.cols - 1)
        self.col_spin.setValue(self.col)
        form.addWidget(self.col_spin, 0, 3)

        self.play_button = QtWidgets.QPushButton("Play")
        self.play_button.setCheckable(True)
        form.addWidget(self.play_button, 1, 0)
        form.addWidget(QtWidgets.QLabel("speed"), 1, 1)
        self.speed_spin = QtWidgets.QSpinBox()
        self.speed_spin.setRange(1, 12)
        self.speed_spin.setValue(1)
        form.addWidget(self.speed_spin, 1, 2)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, self.steps - 1)
        side_layout.addWidget(self.slider)

        self.chart = pg.PlotWidget()
        self.chart.setBackground("#f7f4ee")
        self.chart.setLabel("bottom", "indentation", units="mm")
        self.chart.setLabel("left", "Fz", units="N")
        self.chart.showGrid(x=True, y=True, alpha=0.28)
        side_layout.addWidget(self.chart, stretch=1)
        self.curve_item = self.chart.plot([], [], pen=pg.mkPen("#176d8f", width=2.2))
        self.marker_item = self.chart.plot([], [], pen=None, symbol="o", symbolSize=10, symbolBrush="#d84b2a")

        self.surface_toggle = QtWidgets.QCheckBox("surface")
        self.surface_toggle.setChecked(True)
        self.lump_toggle = QtWidgets.QCheckBox("analytic lumps")
        self.lump_toggle.setChecked(True)
        self.scan_toggle = QtWidgets.QCheckBox("scan map")
        self.scan_toggle.setChecked(True)
        self.probe_toggle = QtWidgets.QCheckBox("probe")
        self.probe_toggle.setChecked(True)
        toggle_row = QtWidgets.QHBoxLayout()
        toggle_row.addWidget(self.surface_toggle)
        toggle_row.addWidget(self.lump_toggle)
        toggle_row.addWidget(self.scan_toggle)
        toggle_row.addWidget(self.probe_toggle)
        side_layout.addLayout(toggle_row)

        if self._uses_discrete_mesh:
            self.normal_tissue_toggle = QtWidgets.QCheckBox("normal tissue")
            self.normal_tissue_toggle.setChecked(True)
            self.lump_tet_toggle = QtWidgets.QCheckBox("lump tet surfaces")
            self.lump_tet_toggle.setChecked(True)
            self.tet_wire_toggle = QtWidgets.QCheckBox(f"tet wire / {self.tet_stride}")
            self.tet_wire_toggle.setChecked(True)
            self.vertex_toggle = QtWidgets.QCheckBox(f"vertices / {self.vertex_stride}")
            self.vertex_toggle.setChecked(False)
            fem_row = QtWidgets.QHBoxLayout()
            fem_row.addWidget(self.normal_tissue_toggle)
            fem_row.addWidget(self.lump_tet_toggle)
            fem_row.addWidget(self.tet_wire_toggle)
            fem_row.addWidget(self.vertex_toggle)
            side_layout.addLayout(fem_row)
        else:
            self.normal_tissue_toggle = None
            self.lump_tet_toggle = None
            self.tet_wire_toggle = None
            self.vertex_toggle = None

        self.row_spin.valueChanged.connect(self._press_changed)
        self.col_spin.valueChanged.connect(self._press_changed)
        self.slider.valueChanged.connect(self._step_changed)
        self.play_button.toggled.connect(self._play_toggled)
        self.surface_toggle.toggled.connect(lambda checked: self._set_actor_visible(self.surface_actor, checked))
        self.lump_toggle.toggled.connect(lambda checked: self._set_actor_visible(self.lump_actors, checked))
        self.scan_toggle.toggled.connect(lambda checked: self._set_actor_visible(self.scan_actor, checked))
        self.probe_toggle.toggled.connect(lambda checked: self._set_actor_visible(self.probe_actor, checked))
        if self._uses_discrete_mesh:
            self.normal_tissue_toggle.toggled.connect(
                lambda checked: self._set_actor_visible(self.normal_tissue_actor, checked)
            )
            self.lump_tet_toggle.toggled.connect(lambda checked: self._set_actor_visible(self.lump_tet_actors, checked))
            self.tet_wire_toggle.toggled.connect(lambda checked: self._set_actor_visible(self.tet_wire_actor, checked))
            self.vertex_toggle.toggled.connect(lambda checked: self._set_actor_visible(self.vertex_actor, checked))

    def _build_scene(self, surface_resolution: int) -> None:
        pv = self.pv
        self.plotter.set_background("#101316")
        self.phantom_box_actor = self.plotter.add_mesh(
            phantom_box_polydata(self.phantom),
            color="#90c9ea",
            opacity=0.10,
            show_edges=True,
            edge_color="#a9c8d8",
        )
        self.surface = None
        self.surface_base_points = None
        self.surface_actor = None
        if self._uses_continuous_surface:
            self.surface = top_surface_polydata(self.phantom, resolution=surface_resolution)
            self.surface_base_points = np.asarray(self.surface.points, dtype=np.float32).copy()
            self.surface_actor = self.plotter.add_mesh(
                self.surface,
                scalars="visual_z_displacement_m",
                cmap="viridis",
                clim=(0.0, max(float(np.nanmax(self.indentation)), 1e-9)),
                opacity=0.78,
                smooth_shading=True,
                show_scalar_bar=True,
                scalar_bar_args={"title": "visual z displacement [m]"},
            )
        else:
            self.surface_toggle.setChecked(False)
            self.surface_toggle.setEnabled(False)

        self.lump_actors = []
        if self._uses_analytic_lumps:
            for idx, lump in enumerate(self.lumps):
                mesh = analytic_lump_polydata(lump, resolution=44)
                actor = self.plotter.add_mesh(
                    mesh,
                    color=LUMP_COLORS[idx % len(LUMP_COLORS)],
                    opacity=0.56,
                    smooth_shading=True,
                )
                self.lump_actors.append(actor)
        else:
            self.lump_toggle.setChecked(False)

        peak = np.nanmax(self.fz, axis=-1)
        scan_cloud = scan_points_polydata(self.xy, peak, z=0.001)
        scan_cloud["peak_fz_n"] = peak.reshape(-1)
        if "value" in scan_cloud.point_data:
            del scan_cloud.point_data["value"]
        self.scan_actor = self.plotter.add_mesh(
            scan_cloud,
            scalars="peak_fz_n",
            cmap="magma",
            render_points_as_spheres=True,
            point_size=9,
            opacity=0.88,
            scalar_bar_args={"title": "peak Fz [N]"},
        )

        self.probe = pv.Sphere(
            radius=float(self.scan.probe_radius),
            center=(0.0, 0.0, 0.0),
            theta_resolution=40,
            phi_resolution=20,
        )
        self.probe_base_points = np.asarray(self.probe.points, dtype=np.float32).copy()
        self.probe_actor = self.plotter.add_mesh(self.probe, color="#f1eee6", smooth_shading=True, specular=0.25)

        self.normal_tissue_actor = None
        self.lump_tet_actors = []
        self.tet_wire_actor = None
        self.vertex_actor = None
        if self._uses_discrete_mesh:
            self._add_discrete_fem_layers()

        radius = max(self.phantom.size_x, self.phantom.size_y, self.phantom.height)
        self.plotter.camera_position = [
            (radius * 0.95, -radius * 1.35, radius * 0.9),
            (0.0, 0.0, self.phantom.height * 0.45),
            (0.0, 0.0, 1.0),
        ]
        self.plotter.add_axes()

    def _press_changed(self) -> None:
        if self._updating_controls:
            return
        self.row = int(self.row_spin.value())
        self.col = int(self.col_spin.value())
        self.step = min(self.step, self.steps - 1)
        self._update_curve()
        self._update_frame()

    def _step_changed(self, value: int) -> None:
        if self._updating_controls:
            return
        self.step = int(value)
        self._update_frame()

    def _play_toggled(self, checked: bool) -> None:
        self.play_button.setText("Pause" if checked else "Play")
        if checked:
            self.timer.start()
        else:
            self.timer.stop()

    def _advance(self) -> None:
        self.step = (self.step + int(self.speed_spin.value())) % self.steps
        self._sync_step_control()
        self._update_frame()

    def _update_curve(self) -> None:
        depth_mm = self.indentation[self.row, self.col] * 1000.0
        force = self.fz[self.row, self.col]
        self.curve_item.setData(depth_mm, force)
        self.chart.setTitle(f"press r{self.row:03d} c{self.col:03d}")

    def _update_frame(self) -> None:
        x = float(self.xy[self.row, self.col, 0])
        y = float(self.xy[self.row, self.col, 1])
        depth = float(self.indentation[self.row, self.col, self.step])
        force = float(self.fz[self.row, self.col, self.step])
        if self.surface is not None and self.surface_base_points is not None:
            points, displacement = deformed_surface_points(
                self.surface_base_points,
                self.phantom,
                self.scan,
                x=x,
                y=y,
                depth=depth,
            )
            self.surface.points = points
            self.surface["visual_z_displacement_m"] = displacement
            self.surface.Modified()
        for mesh, base_points in self.fem_deform_targets:
            points, _displacement = deformed_body_points(
                base_points,
                self.phantom,
                self.scan,
                x=x,
                y=y,
                depth=depth,
            )
            mesh.points = points
            mesh.Modified()

        probe_z = float(self.phantom.height + self.scan.probe_radius + self.scan.preload_gap - depth)
        self.probe.points = self.probe_base_points + np.asarray([x, y, probe_z], dtype=np.float32)
        self.probe.Modified()

        self.marker_item.setData([depth * 1000.0], [force])
        self.info.setText(
            f"{self.sample_path.name}\n"
            f"grid {self.rows} x {self.cols}, step {self.step + 1} / {self.steps}\n"
            f"x={x:.5f} m, y={y:.5f} m\n"
            f"indentation={depth:.6f} m, Fz={force:.6g} N"
        )
        self._sync_step_control()
        self.plotter.update()

    def _add_discrete_fem_layers(self) -> None:
        if "mesh_vertices" not in self.mesh_arrays or "mesh_tets" not in self.mesh_arrays:
            self.info.setText(f"{self.sample_path.name}\nNo mesh_vertices/mesh_tets arrays found for discrete mode.")
            return
        vertices = np.asarray(self.mesh_arrays["mesh_vertices"], dtype=np.float32)
        tets = np.asarray(self.mesh_arrays["mesh_tets"], dtype=np.int64)
        tet_lump_id = np.asarray(self.mesh_arrays.get("tet_lump_id", np.full(tets.shape[0], -1)), dtype=np.int32)
        self.fem_grid = tet_grid_from_sample(self.mesh_arrays, self.lumps)

        normal_surface = self.fem_grid.extract_surface(algorithm="dataset_surface")
        normal_surface["surface_z_m"] = np.asarray(normal_surface.points[:, 2], dtype=np.float32)
        self.normal_tissue_actor = self.plotter.add_mesh(
            normal_surface,
            color="#8dbfd5",
            opacity=0.24,
            smooth_shading=True,
        )
        self._track_fem_deformation(normal_surface)

        for idx, _lump in enumerate(self.lumps):
            cell_ids = np.flatnonzero(tet_lump_id == idx)
            if cell_ids.size == 0:
                continue
            surface = self.fem_grid.extract_cells(cell_ids).extract_surface(algorithm="dataset_surface")
            actor = self.plotter.add_mesh(
                surface,
                color=LUMP_COLORS[idx % len(LUMP_COLORS)],
                opacity=0.68,
                show_edges=True,
                edge_color="#101316",
                line_width=1,
                smooth_shading=False,
            )
            self.lump_tet_actors.append(actor)
            self._track_fem_deformation(surface)

        sampled_grid = _sample_tet_grid(self.pv, vertices, tets, tet_lump_id, self.tet_stride)
        self.tet_wire_actor = self.plotter.add_mesh(
            sampled_grid,
            style="wireframe",
            color="#d8e0e3",
            opacity=0.16,
            line_width=1,
        )
        self._track_fem_deformation(sampled_grid)

        vertex_cloud = self.pv.PolyData(vertices[:: self.vertex_stride].copy())
        self.vertex_actor = self.plotter.add_mesh(
            vertex_cloud,
            color="#f2efe4",
            render_points_as_spheres=False,
            point_size=2,
            opacity=0.62,
        )
        self.vertex_actor.SetVisibility(False)
        self._track_fem_deformation(vertex_cloud)

    def _track_fem_deformation(self, mesh: object) -> None:
        self.fem_deform_targets.append((mesh, np.asarray(mesh.points, dtype=np.float32).copy()))

    @property
    def _uses_discrete_mesh(self) -> bool:
        return self.mesh_style in {"discrete", "both"}

    @property
    def _uses_continuous_surface(self) -> bool:
        return self.mesh_style in {"continuous", "both"}

    @property
    def _uses_analytic_lumps(self) -> bool:
        return self.mesh_style in {"continuous", "both"}

    def _sync_step_control(self) -> None:
        self._updating_controls = True
        self.slider.setValue(self.step)
        self._updating_controls = False

    def _set_actor_visible(self, actors, visible: bool) -> None:
        if isinstance(actors, list):
            for actor in actors:
                actor.SetVisibility(bool(visible))
        elif actors is not None:
            actors.SetVisibility(bool(visible))
        self.plotter.update()


def _clamp_index(value: int, count: int) -> int:
    return max(0, min(int(value), max(int(count) - 1, 0)))


def _normalize_mesh_style(value: MeshStyle) -> MeshStyle:
    style = str(value).strip().lower()
    if style not in {"continuous", "discrete", "both"}:
        raise ValueError("--mesh-style must be one of: continuous, discrete, both")
    return style


def _sample_tet_grid(pv: object, vertices: np.ndarray, tets: np.ndarray, tet_lump_id: np.ndarray, stride: int):
    selected_tets = np.asarray(tets[:: max(int(stride), 1)], dtype=np.int64)
    selected_lump_id = np.asarray(tet_lump_id[:: max(int(stride), 1)], dtype=np.int32)
    if selected_tets.size == 0:
        return pv.UnstructuredGrid()
    unique_vertices, inverse = np.unique(selected_tets.reshape(-1), return_inverse=True)
    compact_tets = inverse.reshape(-1, 4)
    cell_sizes = np.full((compact_tets.shape[0], 1), 4, dtype=np.int64)
    cells = np.hstack([cell_sizes, compact_tets]).reshape(-1)
    celltypes = np.full(compact_tets.shape[0], int(pv.CellType.TETRA), dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, celltypes, vertices[unique_vertices].copy())
    grid.cell_data["lump_id"] = selected_lump_id
    grid.cell_data["is_lump"] = (selected_lump_id >= 0).astype(np.uint8)
    return grid
