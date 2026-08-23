#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate step-by-step Ecoflex pouring manuals for fabrication phantoms.")
    parser.add_argument("batch_dir", type=Path, help="Directory containing random_fabrication_manifest.json.")
    parser.add_argument("--out-dir-name", type=str, default="manuals")
    parser.add_argument("--group-tolerance-mm", type=float, default=1.0)
    parser.add_argument("--extra-mix-fraction", type=float, default=0.10)
    parser.add_argument("--zip", action="store_true", help="Refresh the parent batch zip after writing manuals.")
    args = parser.parse_args()

    batch_dir = args.batch_dir.expanduser().resolve()
    manifest_path = batch_dir / "random_fabrication_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out_dir = batch_dir / args.out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_rows: list[dict[str, object]] = []
    manual_index: list[dict[str, object]] = []
    for item in manifest["configurations"]:
        config_path = Path(item["config_json"])
        config = json.loads(config_path.read_text(encoding="utf-8"))
        plan = build_pouring_plan(
            config,
            group_tolerance_mm=float(args.group_tolerance_mm),
            extra_mix_fraction=float(args.extra_mix_fraction),
        )
        config_id = str(config["config_id"])
        csv_path = out_dir / f"{config_id}_pouring_steps.csv"
        md_path = out_dir / f"{config_id}_manual.md"
        csv_path.write_text(_rows_to_csv_text(plan["steps"]), encoding="utf-8")
        md_path.write_text(_manual_markdown(config, plan, csv_path=csv_path), encoding="utf-8")
        for row in plan["steps"]:
            batch_rows.append({"config_id": config_id, **row})
        manual_index.append(
            {
                "config_id": config_id,
                "lump_count": config["lump_count"],
                "manual_md": str(md_path),
                "steps_csv": str(csv_path),
                "estimated_total_ecoflex_ml": plan["total_ecoflex_ml"],
                "suggested_total_mix_ml": plan["suggested_total_mix_ml"],
                "cut_steps": plan["cut_step_count"],
            }
        )
        print(f"wrote manual for {config_id}: {md_path}", flush=True)

    (out_dir / "all_pouring_steps.csv").write_text(_rows_to_csv_text(batch_rows), encoding="utf-8")
    (out_dir / "manual_index.csv").write_text(_rows_to_csv_text(manual_index), encoding="utf-8")
    (out_dir / "README.md").write_text(_batch_readme(manifest, manual_index), encoding="utf-8")

    if args.zip:
        zip_path = batch_dir.with_suffix(".zip")
        _zip_dir(batch_dir, zip_path)
        print(f"refreshed zip -> {zip_path}", flush=True)
    print(f"Wrote manuals to {out_dir}")


def build_pouring_plan(config: dict, *, group_tolerance_mm: float, extra_mix_fraction: float) -> dict[str, object]:
    params = config["fabrication_params"]
    cavity_area = float(params["inner_size_mm"]) ** 2
    ecoflex_bottom_z = float(params["ecoflex_bottom_z_mm"])
    ecoflex_top_z = float(params["ecoflex_top_z_mm"])
    cast_height = float(params["phantom_height_mm"])
    lumps = [dict(lump, _manual_index=idx) for idx, lump in enumerate(config["lumps"])]
    lumps = sorted(lumps, key=lambda lump: float(lump["center_z_mm_from_mold_bottom"]))
    groups = _group_lumps_by_center_z(lumps, tolerance_mm=max(float(group_tolerance_mm), 0.0))

    current_z = ecoflex_bottom_z
    current_volume = fill_volume_ml(current_z, lumps=lumps, cavity_area_mm2=cavity_area, bottom_z_mm=ecoflex_bottom_z)
    steps: list[dict[str, object]] = []
    cumulative_mix_ml = 0.0

    step_number = 1
    for group in groups:
        target_z = max(float(lump["center_z_mm_from_mold_bottom"]) for lump in group)
        if target_z <= current_z + 1e-6:
            continue
        target_volume = fill_volume_ml(target_z, lumps=lumps, cavity_area_mm2=cavity_area, bottom_z_mm=ecoflex_bottom_z)
        increment = max(target_volume - current_volume, 0.0)
        suggested = increment * (1.0 + float(extra_mix_fraction))
        cumulative_mix_ml += suggested
        cut_names = ", ".join(_lump_name(lump) for lump in group)
        steps.append(
            {
                "step": step_number,
                "target": "半浸没lump",
                "target_level_z_mm_from_mold_bottom": _round(target_z),
                "target_level_mm_above_ecoflex_bottom": _round(target_z - ecoflex_bottom_z),
                "pour_increment_ml": _round(increment),
                "suggested_mix_ml_with_extra": _round(suggested),
                "cumulative_ecoflex_ml": _round(target_volume),
                "cumulative_suggested_mix_ml": _round(cumulative_mix_ml),
                "support_action": f"剪除支撑: {cut_names}",
                "lumps_to_cut": cut_names,
                "notes": "倒到目标液面后暂停，待该层能固定lump，再在支撑杆被后续液面淹没前剪除对应支撑。",
            }
        )
        current_z = target_z
        current_volume = target_volume
        step_number += 1

    if current_z < ecoflex_top_z - 1e-6:
        target_volume = fill_volume_ml(ecoflex_top_z, lumps=lumps, cavity_area_mm2=cavity_area, bottom_z_mm=ecoflex_bottom_z)
        increment = max(target_volume - current_volume, 0.0)
        suggested = increment * (1.0 + float(extra_mix_fraction))
        cumulative_mix_ml += suggested
        steps.append(
            {
                "step": step_number,
                "target": "最终补满",
                "target_level_z_mm_from_mold_bottom": _round(ecoflex_top_z),
                "target_level_mm_above_ecoflex_bottom": _round(cast_height),
                "pour_increment_ml": _round(increment),
                "suggested_mix_ml_with_extra": _round(suggested),
                "cumulative_ecoflex_ml": _round(target_volume),
                "cumulative_suggested_mix_ml": _round(cumulative_mix_ml),
                "support_action": "不需要剪支撑；所有lump支撑应已在前面步骤剪除。",
                "lumps_to_cut": "",
                "notes": "补到最终phantom高度并固化。",
            }
        )

    final_volume = fill_volume_ml(ecoflex_top_z, lumps=lumps, cavity_area_mm2=cavity_area, bottom_z_mm=ecoflex_bottom_z)
    return {
        "steps": steps,
        "total_ecoflex_ml": _round(final_volume),
        "suggested_total_mix_ml": _round(sum(float(row["suggested_mix_ml_with_extra"]) for row in steps)),
        "cut_step_count": sum(1 for row in steps if row["lumps_to_cut"]),
        "assumptions": {
            "cavity_area_mm2": cavity_area,
            "ecoflex_bottom_z_mm": ecoflex_bottom_z,
            "ecoflex_top_z_mm": ecoflex_top_z,
            "cast_height_mm": cast_height,
            "extra_mix_fraction": float(extra_mix_fraction),
            "group_tolerance_mm": float(group_tolerance_mm),
            "support_volume_policy": "Ignored because supports should be cut before they are submerged by later pours.",
        },
    }


def fill_volume_ml(level_z_mm: float, *, lumps: list[dict], cavity_area_mm2: float, bottom_z_mm: float) -> float:
    liquid_height = max(float(level_z_mm) - float(bottom_z_mm), 0.0)
    gross_mm3 = float(cavity_area_mm2) * liquid_height
    displaced_mm3 = sum(submerged_lump_volume_mm3(lump, level_z_mm) for lump in lumps)
    return max((gross_mm3 - displaced_mm3) / 1000.0, 0.0)


def submerged_lump_volume_mm3(lump: dict, level_z_mm: float) -> float:
    shape = str(lump["shape"])
    zc = float(lump["center_z_mm_from_mold_bottom"])
    rx, ry, rz = (float(value) for value in lump["radii_mm"])
    q = float(level_z_mm) - zc
    if shape in {"sphere", "ellipsoid"}:
        if q <= -rz:
            return 0.0
        total = 4.0 / 3.0 * math.pi * rx * ry * rz
        if q >= rz:
            return total
        t = q / rz
        fraction = (2.0 + 3.0 * t - t**3) / 4.0
        return total * fraction
    if shape == "cylinder":
        if q <= -rz:
            return 0.0
        total = math.pi * rx * ry * 2.0 * rz
        if q >= rz:
            return total
        return total * ((q + rz) / (2.0 * rz))
    if shape == "capsule":
        radius = rx
        half_axis = rz
        return _capsule_submerged_volume_mm3(q, radius=radius, half_axis=half_axis)
    if shape == "box":
        if q <= -rz:
            return 0.0
        total = 8.0 * rx * ry * rz
        if q >= rz:
            return total
        return total * ((q + rz) / (2.0 * rz))
    raise ValueError(f"Unsupported shape: {shape}")


def _capsule_submerged_volume_mm3(q: float, *, radius: float, half_axis: float) -> float:
    r = float(radius)
    a = max(float(half_axis), 0.0)
    total = math.pi * r * r * (2.0 * a) + 4.0 / 3.0 * math.pi * r**3
    if q <= -a - r:
        return 0.0
    if q >= a + r:
        return total
    lower_hemi = 2.0 / 3.0 * math.pi * r**3
    cyl_area = math.pi * r * r
    if q < -a:
        h = q - (-a - r)
        return _sphere_cap_volume_mm3(h, r)
    if q <= a:
        return lower_hemi + cyl_area * (q + a)
    upper_h = q - (a - r)
    return lower_hemi + cyl_area * (2.0 * a) + _sphere_cap_volume_mm3(upper_h, r)


def _sphere_cap_volume_mm3(height: float, radius: float) -> float:
    h = min(max(float(height), 0.0), 2.0 * float(radius))
    r = float(radius)
    return math.pi * h * h * (r - h / 3.0)


def _group_lumps_by_center_z(lumps: list[dict], *, tolerance_mm: float) -> list[list[dict]]:
    groups: list[list[dict]] = []
    for lump in lumps:
        z = float(lump["center_z_mm_from_mold_bottom"])
        if groups and z - max(float(item["center_z_mm_from_mold_bottom"]) for item in groups[-1]) <= tolerance_mm:
            groups[-1].append(lump)
        else:
            groups.append([lump])
    return groups


def _manual_markdown(config: dict, plan: dict, *, csv_path: Path) -> str:
    params = config["fabrication_params"]
    hanging = config["hanging_layout"]
    lines = [
        f"# {config['config_id']} 制作流程手册",
        "",
        "## 几何与总体用量",
        "",
        f"- 成型phantom高度: {params['phantom_height_mm']} mm",
        f"- Ecoflex成型区间: 从模具底部量 z={params['ecoflex_bottom_z_mm']}..{params['ecoflex_top_z_mm']} mm",
        f"- 内腔截面: {params['inner_size_mm']} x {params['inner_size_mm']} mm",
        f"- 顶部吊架横杆中心: z={hanging['top_rod_center_z_mm']} mm",
        f"- 估算最终Ecoflex体积: {plan['total_ecoflex_ml']} mL",
        f"- 建议总配胶量: {plan['suggested_total_mix_ml']} mL",
        "",
        "## Lump列表",
        "",
        "| Lump | 形状 | 中心z, 从模具底部量 (mm) | 顶部z (mm) | 底部z (mm) | 操作目标 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for idx, lump in enumerate(config["lumps"]):
        lines.append(
            "| "
            + " | ".join(
                [
                    _lump_name(lump, idx=idx),
                    str(lump["shape"]),
                    str(_round(lump["center_z_mm_from_mold_bottom"])),
                    str(_round(lump["top_z_mm_from_mold_bottom"])),
                    str(_round(lump["bottom_z_mm_from_mold_bottom"])),
                    f"倒到 z={_round(lump['center_z_mm_from_mold_bottom'])}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 倒胶与剪支撑步骤",
            "",
            "| 步骤 | 目标液面z, 从模具底部量 (mm) | 高于Ecoflex底部 (mm) | 本步倒入 (mL) | 建议本步配胶量 (mL) | 操作 |",
            "|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in plan["steps"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["step"]),
                    str(row["target_level_z_mm_from_mold_bottom"]),
                    str(row["target_level_mm_above_ecoflex_bottom"]),
                    str(row["pour_increment_ml"]),
                    str(row["suggested_mix_ml_with_extra"]),
                    str(row["support_action"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 操作说明",
            "",
            "1. 将吊顶整体装入wall顶部定位槽，确认所有lump都位于底部30 mm成型区内。",
            "2. 在模具或临时标尺上标出表格中的目标液面z高度；z均从模具底部向上量。",
            "3. 每一步按表格体积倒入Ecoflex，直到液面到达目标z高度。",
            "4. 等该层材料足以固定表中列出的lump后，用钳子尽量贴近lump剪断对应支撑。",
            "5. 所有lump支撑剪除后，最后补胶到z=35 mm并完成固化。",
            "",
            "## 估算假设",
            "",
            "- 体积单位为mL，按80 x 80 mm内腔计算，并扣除了已被lump占据的体积。",
            "- 支撑杆体积没有扣除；工艺假设是在支撑杆被后续液面淹没前就剪除。",
            "- 中心高度相差不超过1 mm的lump会合并为同一层；较低的lump会略超过半浸没。",
            "- 建议配胶量默认比估算倒入量多10%，可按实际量杯/针筒残留调整。",
            f"- 机器可读步骤表: `{csv_path.name}`",
            "",
        ]
    )
    return "\n".join(lines)


def _batch_readme(manifest: dict, manual_index: list[dict[str, object]]) -> str:
    return "\n".join(
        [
            "# 制作流程手册",
            "",
            "每个 `*_manual.md` 文件对应一个phantom，包含目标液面高度、估算Ecoflex体积和剪支撑操作。",
            "",
            f"- 配置数量: {len(manual_index)}",
            f"- 成型phantom高度: {manifest['fabrication_params']['phantom_height_mm']} mm",
            f"- 顶部吊架横杆中心: {manifest['hanging_layout']['top_rod_center_z_mm']} mm",
            "- 体积单位: mL，已扣除lump淹没部分的排液体积。",
            "",
        ]
    )


def _lump_name(lump: dict, idx: int | None = None) -> str:
    if idx is None:
        idx = int(lump.get("_manual_index", lump.get("lump_index", -1)))
    prefix = f"L{idx}" if idx is not None and idx >= 0 else "L?"
    return f"{prefix}-{str(lump['shape']).upper()}"


def _rows_to_csv_text(rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""
    import io

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _round(value: object, ndigits: int = 2) -> float:
    return round(float(value), ndigits)


def _zip_dir(src_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src_dir.parent))


if __name__ == "__main__":
    main()
