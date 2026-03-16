from pathlib import Path

import wtec.workflow.topology_pipeline as topology_pipeline
from wtec.workflow.topology_pipeline import TopologyPipeline


def test_run_hr_generation_reuses_configured_hr_for_current_structure(tmp_path: Path, monkeypatch) -> None:
    hr = tmp_path / "TaP_hr.dat"
    hr.write_text("dummy")
    win = tmp_path / "TaP.win"
    win.write_text("dummy")
    monkeypatch.setattr(
        topology_pipeline.ClusterConfig,
        "from_env",
        staticmethod(lambda: (_ for _ in ()).throw(AssertionError("cluster config should not be used"))),
    )

    pipe = TopologyPipeline(
        hr,
        run_dir=tmp_path / "run",
        cfg={"material": "TaP", "hr_dat_path": str(hr)},
    )
    rows = [
        {
            "status": "pending_variant",
            "is_pristine": True,
            "variant_id": "current_structure",
            "hr_dat_path": str(hr),
            "win_path": None,
        }
    ]

    out = pipe._run_hr_generation(rows=rows, topo_cfg={}, thicknesses=[6])
    assert out[0]["status"] == "ready"
    assert out[0]["reason"] == "configured_hr_dat_path"
    assert out[0]["hr_dat_path"] == str(hr.resolve())
    assert out[0]["win_path"] == str(win.resolve())


def test_reuse_configured_hr_rejects_defect_variants(tmp_path: Path) -> None:
    hr = tmp_path / "TaP_hr.dat"
    hr.write_text("dummy")

    pipe = TopologyPipeline(
        hr,
        run_dir=tmp_path / "run",
        cfg={"material": "TaP", "hr_dat_path": str(hr)},
    )
    rows = [
        {
            "status": "pending_variant",
            "is_pristine": False,
            "variant_id": "vacancy_o_0",
        }
    ]

    assert pipe._can_reuse_configured_hr_for_current_structure(rows) is False
