"""Parametric MuJoCo MJCF model builder for DM100 3D."""

from __future__ import annotations

from dm100_3d.mjcf.builder import (
    FIN_NAMES,
    FOOT_BODY_MASS_KG,
    FOOT_RADIUS_M,
    LEG_NAMES,
    build_dm100_mjcf_xml,
    compile_dm100_model,
)

__all__ = [
    "FIN_NAMES",
    "FOOT_BODY_MASS_KG",
    "FOOT_RADIUS_M",
    "LEG_NAMES",
    "build_dm100_mjcf_xml",
    "compile_dm100_model",
]
