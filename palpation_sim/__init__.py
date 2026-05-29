"""Synthetic palpation data generation and validation U-Net utilities."""

from .config import MaterialConfig, PhantomConfig, ScanConfig
from .phantom import LumpSpec, create_structured_tet_mesh, sample_lump, sample_lumps

__all__ = [
    "LumpSpec",
    "MaterialConfig",
    "PhantomConfig",
    "ScanConfig",
    "create_structured_tet_mesh",
    "sample_lump",
    "sample_lumps",
]
