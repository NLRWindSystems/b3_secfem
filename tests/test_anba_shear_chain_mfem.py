"""mfem backend shear chain (ANBA correction step): single-region fully anisotropic solid (beta=61, alpha=-27,
E2 > E1, G13 << G12), where the old chain had K00 at 0.14x. mfem must match fenicsx (same Q2 discretisation)
and ANBA4 (P2 triangles; 2.9 % residual from the element type on this extreme material)."""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("dolfinx")
pytest.importorskip("mfem.ser")

from b3_secfem import SectionInput, solve, write_quad_xdmf
from b3_secfem.config import RegionMat
from b3_secfem.materials import OrthotropicMaterial

D = json.loads(
    (
        Path(__file__).parent / "data" / "anba_mfem" / "general_beta_alpha_solid.json"
    ).read_text()
)


def test_mfem_matches_fenicsx_and_anba(tmp_path):
    r = D["region"]
    v = {
        k: r["material"][k]
        for k in ("E1", "E2", "E3", "G12", "G13", "G23", "nu12", "nu13", "nu23", "rho")
    }
    regs = {
        1: RegionMat(
            material=OrthotropicMaterial(**v),
            beta_deg=r["beta_deg"],
            alpha_deg=r["alpha_deg"],
        )
    }
    p = tmp_path / "s.xdmf"
    write_quad_xdmf(str(p), np.asarray(D["points"]), np.asarray(D["quads"]))
    K = {
        b: np.asarray(
            solve(
                SectionInput(
                    mesh_path=p, region_materials=regs, backend=b, linear_solver="lu"
                )
            ).K
        )
        for b in ("fenicsx", "mfem")
    }
    Ka = np.asarray(D["K_anba"])
    den = np.sqrt(np.outer(np.diag(Ka), np.diag(Ka)))
    assert (np.abs(K["mfem"] - K["fenicsx"]) / den).max() < 1e-6
    assert (np.abs(K["mfem"] - Ka) / den).max() < 0.04
