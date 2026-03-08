from ase import Atoms

from wtec.qe.inputs import QEInputGenerator
from wtec.siesta.inputs import SiestaInputGenerator
from wtec.siesta.runner import SiestaPipeline
from wtec.workflow.dft_pipeline import DFTPipeline


def _si_atom() -> Atoms:
    return Atoms(
        symbols=["Si"],
        positions=[[0.0, 0.0, 0.0]],
        cell=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        pbc=[True, True, True],
    )


def test_qe_input_emits_d3_keywords() -> None:
    gen = QEInputGenerator(
        atoms=_si_atom(),
        material_name="CoSi",
        pseudopots={"Si": "Si.UPF"},
        dispersion_enabled=True,
        dispersion_method="d3",
        qe_vdw_corr="grimme-d3",
        qe_dftd3_version=4,
        qe_dftd3_threebody=True,
    )
    text = gen.scf()
    assert "vdw_corr = 'grimme-d3'" in text
    assert "dftd3_version = 4" in text
    assert "dftd3_threebody = .true." in text


def test_qe_input_skips_d3_when_disabled() -> None:
    gen = QEInputGenerator(
        atoms=_si_atom(),
        material_name="CoSi",
        pseudopots={"Si": "Si.UPF"},
        dispersion_enabled=False,
        dispersion_method="none",
    )
    text = gen.scf()
    assert "vdw_corr" not in text
    assert "dftd3_version" not in text
    assert "dftd3_threebody" not in text


def test_siesta_input_emits_d3_keywords(tmp_path) -> None:
    gen = SiestaInputGenerator(
        atoms=_si_atom(),
        material_name="CoSi",
        pseudo_dir="/pseudo",
        pseudopots={"Si": "Si.psml"},
        basis_profile="default",
        dispersion_enabled=True,
        dispersion_method="d3",
        siesta_dftd3_use_xc_defaults=True,
    )
    out = tmp_path / "scf.fdf"
    gen.scf(out)
    text = out.read_text()
    assert "DFTD3             true" in text
    assert "DFTD3.UseXCDefaults   true" in text


def test_siesta_input_skips_d3_when_disabled(tmp_path) -> None:
    gen = SiestaInputGenerator(
        atoms=_si_atom(),
        material_name="CoSi",
        pseudo_dir="/pseudo",
        pseudopots={"Si": "Si.psml"},
        basis_profile="default",
        dispersion_enabled=False,
        dispersion_method="none",
    )
    out = tmp_path / "scf.fdf"
    gen.scf(out)
    text = out.read_text()
    assert "DFTD3" not in text


def test_siesta_input_emits_scf_tuning_controls(tmp_path) -> None:
    gen = SiestaInputGenerator(
        atoms=_si_atom(),
        material_name="CoSi",
        pseudo_dir="/pseudo",
        pseudopots={"Si": "Si.psml"},
        basis_profile="default",
        dm_mixing_weight=0.2,
        dm_number_pulay=6,
        electronic_temperature_k=250.0,
        max_scf_iterations=123,
    )
    out = tmp_path / "scf.fdf"
    gen.scf(out)
    text = out.read_text()
    assert "DM.MixingWeight   0.200000" in text
    assert "DM.NumberPulay    6" in text
    assert "ElectronicTemperature  250 K" in text
    assert "MaxSCFIterations  123" in text


def test_siesta_pipeline_resolves_queue_scoped_factorization_defaults(tmp_path) -> None:
    class DummyJM:
        def resolve_queue(self, queue, fallback_order=None):  # noqa: ANN001, ANN002
            return "g3"

    pipe = SiestaPipeline(
        _si_atom(),
        "CoSi",
        DummyJM(),
        run_dir=tmp_path,
        remote_base="/remote",
        n_nodes=1,
        n_cores_per_node=32,
        n_cores_by_queue={"g3": 32},
        factorization_defaults={
            "g3_32": {
                "mpi_np_scf": 16,
                "omp_threads_scf": 2,
            }
        },
    )
    mpi_np, omp_threads = pipe._resolve_stage_parallel("scf")
    assert mpi_np == 16
    assert omp_threads == 2


def test_dft_pipeline_run_scf_returns_meta(tmp_path) -> None:
    class DummyJM:
        def submit_and_wait(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return {"job_id": "12345", "status": "DONE"}

    pipe = DFTPipeline(
        _si_atom(),
        "CoSi",
        DummyJM(),
        run_dir=tmp_path,
        remote_base="/remote",
    )
    pipe._resolved_pseudopots_map = lambda: {"Si": "Si.UPF"}  # type: ignore[method-assign]
    pipe._recommended_nbnd = lambda: 8  # type: ignore[method-assign]
    pipe._preflight_step = lambda required_commands, local_inputs: None  # type: ignore[method-assign]
    pipe._resolved_queue_name = lambda: "g1"  # type: ignore[method-assign]
    pipe._resolved_cores_per_node = lambda: 32  # type: ignore[method-assign]
    meta = pipe.run_scf()
    assert isinstance(meta, dict)
    assert meta.get("job_id") == "12345"
