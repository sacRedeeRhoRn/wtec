from wtec.cluster.mpi import MPIConfig, build_command


def test_build_command_emits_explicit_bind_none() -> None:
    cmd = build_command("wannier90.x", extra_args="TiS", mpi=MPIConfig(n_cores=1, bind_to="none"))
    assert cmd.startswith("mpirun -np 1 --bind-to none wannier90.x")
