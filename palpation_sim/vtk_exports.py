from __future__ import annotations

import html
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import PhantomConfig, ScanConfig
from .native_data import load_phantom_scan_material_lumps, load_sample_arrays
from .phantom import LumpSpec
from .visual_geometry import (
    analytic_lump_polydata,
    deformed_surface_points,
    require_pyvista,
    scan_points_polydata,
    top_surface_polydata,
)


def write_sample_vtk_bundle(
    sample_path: Path,
    out_dir: Path,
    *,
    metadata_path: Path | None = None,
    include_tet_mesh: bool = True,
    extract_surface: bool = False,
    include_scan_points: bool = True,
    analytic_lump_resolution: int = 56,
    timeseries_press: tuple[int, int] | None = None,
    timeseries_stride: int = 1,
    timeseries_surface_resolution: int = 96,
) -> dict[str, Path | list[Path]]:
    """Export a palpation sample to native VTK/ParaView-friendly files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    phantom, scan, _material, lumps, _metadata = load_phantom_scan_material_lumps(
        sample_path=sample_path,
        metadata_path=metadata_path,
    )
    outputs: dict[str, Path | list[Path]] = {}

    sample_keys = {"xy", "fz", "mask", "nonlinearity_ratio", "indentation_depth"}
    if include_tet_mesh or extract_surface:
        sample_keys |= {"mesh_vertices", "mesh_tets", "tet_lump_id", "tet_lump_mask"}
    sample = load_sample_arrays(sample_path, sample_keys)

    if include_tet_mesh or extract_surface:
        grid = tet_grid_from_sample(sample, lumps)
    else:
        grid = None

    if include_tet_mesh and grid is not None:
        mesh_path = out_dir / f"{sample_path.stem}_tet_mesh.vtu"
        _save_vtk_xml(grid, mesh_path)
        outputs["tet_mesh"] = mesh_path
    if extract_surface and grid is not None:
        surface_path = out_dir / f"{sample_path.stem}_tet_surface.vtp"
        _save_vtk_xml(grid.extract_surface(), surface_path)
        outputs["tet_surface"] = surface_path

    if include_scan_points and "xy" in sample and "fz" in sample:
        scan_path = out_dir / f"{sample_path.stem}_scan_points.vtp"
        write_scan_points_vtp(scan_path, np.asarray(sample["xy"]), sample)
        outputs["scan_points"] = scan_path

    if lumps:
        lump_dir = out_dir / "analytic_lumps"
        outputs["analytic_lumps"] = _write_lump_surfaces(lump_dir, lumps, resolution=analytic_lump_resolution)

    if timeseries_press is not None:
        row, col = timeseries_press
        if "xy" not in sample or "indentation_depth" not in sample or "fz" not in sample:
            raise ValueError("Timeseries export requires xy, indentation_depth, and fz arrays.")
        pvd_path = write_press_surface_timeseries(
            out_dir / f"{sample_path.stem}_press_r{row:03d}_c{col:03d}.pvd",
            phantom,
            scan,
            np.asarray(sample["xy"], dtype=np.float32),
            np.asarray(sample["indentation_depth"], dtype=np.float32),
            np.asarray(sample["fz"], dtype=np.float32),
            row=row,
            col=col,
            stride=timeseries_stride,
            surface_resolution=timeseries_surface_resolution,
        )
        outputs["press_timeseries"] = pvd_path

    return outputs


def tet_grid_from_sample(sample: dict[str, np.ndarray], lumps: Sequence[LumpSpec] = ()):
    pv = require_pyvista()
    if "mesh_vertices" not in sample or "mesh_tets" not in sample:
        raise ValueError("Sample does not contain mesh_vertices and mesh_tets arrays.")
    vertices = np.asarray(sample["mesh_vertices"], dtype=np.float32)
    tets = np.asarray(sample["mesh_tets"], dtype=np.int64)
    cell_sizes = np.full((tets.shape[0], 1), 4, dtype=np.int64)
    cells = np.hstack([cell_sizes, tets]).reshape(-1)
    celltypes = np.full(tets.shape[0], int(pv.CellType.TETRA), dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, celltypes, vertices)
    if "tet_lump_id" in sample:
        lump_id = np.asarray(sample["tet_lump_id"], dtype=np.int32)
        grid.cell_data["lump_id"] = lump_id
        if lumps:
            stiffness = np.ones(lump_id.shape[0], dtype=np.float32)
            for idx, lump in enumerate(lumps):
                stiffness[lump_id == idx] = float(lump.stiffness_multiplier)
            grid.cell_data["stiffness_multiplier"] = stiffness
    if "tet_lump_mask" in sample:
        grid.cell_data["is_lump"] = np.asarray(sample["tet_lump_mask"], dtype=np.uint8)
    return grid


def write_scan_points_vtp(path: Path, xy: np.ndarray, sample: dict[str, np.ndarray]) -> Path:
    values = None
    if "fz" in sample:
        values = np.nanmax(np.asarray(sample["fz"], dtype=np.float32), axis=-1)
    cloud = scan_points_polydata(xy, values, z=0.0)
    if values is not None:
        cloud["peak_fz_n"] = values.reshape(-1)
        if "value" in cloud.point_data:
            del cloud.point_data["value"]
    if "mask" in sample:
        cloud["mask"] = np.asarray(sample["mask"], dtype=np.float32).reshape(-1)
    if "nonlinearity_ratio" in sample:
        cloud["nonlinearity_ratio"] = np.asarray(sample["nonlinearity_ratio"], dtype=np.float32).reshape(-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    _save_vtk_xml(cloud, path)
    return path


def write_press_surface_timeseries(
    pvd_path: Path,
    phantom: PhantomConfig,
    scan: ScanConfig,
    xy: np.ndarray,
    indentation: np.ndarray,
    fz: np.ndarray,
    *,
    row: int,
    col: int,
    stride: int = 1,
    surface_resolution: int = 96,
) -> Path:
    pvd_path.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = pvd_path.with_suffix("")
    frame_dir.mkdir(parents=True, exist_ok=True)
    base_mesh = top_surface_polydata(phantom, resolution=surface_resolution)
    base_points = np.asarray(base_mesh.points, dtype=np.float32).copy()
    x = float(xy[row, col, 0])
    y = float(xy[row, col, 1])
    steps = indentation.shape[-1]
    stride = max(int(stride), 1)
    entries: list[tuple[float, Path]] = []
    for step in range(0, steps, stride):
        depth = float(indentation[row, col, step])
        points, displacement = deformed_surface_points(base_points, phantom, scan, x=x, y=y, depth=depth)
        mesh = base_mesh.copy(deep=True)
        mesh.points = points
        mesh["visual_z_displacement_m"] = displacement
        mesh.field_data["row_col_step"] = np.asarray([row, col, step], dtype=np.int32)
        mesh.field_data["indentation_depth_m"] = np.asarray([depth], dtype=np.float32)
        mesh.field_data["force_z_n"] = np.asarray([float(fz[row, col, step])], dtype=np.float32)
        frame_path = frame_dir / f"surface_step_{step:04d}.vtp"
        _save_vtk_xml(mesh, frame_path)
        entries.append((float(step), frame_path.relative_to(pvd_path.parent)))
    _write_pvd(pvd_path, entries)
    return pvd_path


def _write_lump_surfaces(out_dir: Path, lumps: Sequence[LumpSpec], *, resolution: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for idx, lump in enumerate(lumps):
        mesh = analytic_lump_polydata(lump, resolution=resolution)
        mesh["lump_id"] = np.full(mesh.n_points, idx, dtype=np.int32)
        path = out_dir / f"lump_{idx:02d}_{lump.shape}_analytic.vtp"
        _save_vtk_xml(mesh, path)
        paths.append(path)
    return paths


def _save_vtk_xml(dataset: object, path: Path) -> None:
    """Write VTK XML in binary mode with zlib compression when possible."""
    try:
        import vtk

        suffix = path.suffix.lower()
        if suffix == ".vtu":
            writer = vtk.vtkXMLUnstructuredGridWriter()
        elif suffix == ".vtp":
            writer = vtk.vtkXMLPolyDataWriter()
        else:
            raise ValueError(f"Unsupported VTK XML suffix: {path.suffix}")
        writer.SetFileName(str(path))
        writer.SetInputData(dataset)
        writer.SetDataModeToBinary()
        if hasattr(writer, "SetCompressorTypeToZLib"):
            writer.SetCompressorTypeToZLib()
        if writer.Write() != 1:
            raise RuntimeError(f"VTK writer failed for {path}")
    except Exception:
        dataset.save(path, binary=True)  # type: ignore[attr-defined]


def _write_pvd(path: Path, entries: Sequence[tuple[float, Path]]) -> None:
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    for timestep, file_path in entries:
        lines.append(
            f'    <DataSet timestep="{timestep:.9g}" group="" part="0" file="{html.escape(file_path.as_posix())}"/>'
        )
    lines.extend(["  </Collection>", "</VTKFile>", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
