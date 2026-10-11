"""Regression: full 6x6 vs ANBA4 on real blade sections with UD caps + sandwich cores (IEA22 55-86 m,
se_blade_v2 61 m) and an ellipse with 50 mm carbon / glass caps, on identical meshes and materials.

Before the ANBA shear-chain port the chordwise shear K00 was 1e2-1e4x too low here (the solvability
correction of the shear chain was missing, the unsolvable RHS part was projected away). Also checks
exact similarity scaling: K/s^2 (forces) and K/s^4 (moments) on the same mesh scaled by s.
"""
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("dolfinx")

from b3_secfem import SectionInput, solve, write_quad_xdmf
from b3_secfem.config import RegionMat
from b3_secfem.materials import OrthotropicMaterial

DATA = Path(__file__).parent / "data" / "anba_shear"
CASES = sorted(p.name.removesuffix(".json.gz") for p in DATA.glob("*.json.gz"))


def _load(name):
    return json.load(gzip.open(DATA / f"{name}.json.gz", "rt"))


def _solve(d, tmp_path, scale=1.0):
    regs = {int(k): RegionMat(material=OrthotropicMaterial(**r["material"]), beta_deg=r["beta_deg"],
                              alpha_deg=r["alpha_deg"]) for k, r in d["regions"].items()}
    p = tmp_path / f"s{scale}.xdmf"
    write_quad_xdmf(str(p), np.asarray(d["points"]) * scale, np.asarray(d["quads"]),
                    cell_tags=np.asarray(d["tags"], np.int32))
    return np.asarray(solve(SectionInput(mesh_path=p, region_materials=regs, linear_solver="lu")).K)


@pytest.mark.parametrize("name", CASES)
def test_6x6_matches_anba(name, tmp_path):
    d = _load(name)
    K, Ka = _solve(d, tmp_path), np.asarray(d["K_anba"])
    rel = np.abs(K - Ka) / np.sqrt(np.outer(np.diag(Ka), np.diag(Ka)))
    assert rel[0, 0] < 0.02, (K[0, 0], Ka[0, 0])
    assert rel.max() < 0.01, (rel.max(), np.unravel_index(rel.argmax(), rel.shape))


@pytest.mark.parametrize("name", ["iea22_z0058.182", "ellipse_carbon"])
def test_similarity_scaling(name, tmp_path):
    d = _load(name)
    K1 = _solve(d, tmp_path)
    for s in (0.25, 4.0):
        D = np.sqrt([s**2] * 3 + [s**4] * 3)
        Ks = _solve(d, tmp_path, s) / np.outer(D, D)
        assert np.abs(Ks - K1).max() / np.abs(np.diag(K1)).max() < 1e-8
