from wtec.cluster.pbs import PBSJobConfig, generate_script, wannier90_script


def test_generate_script_uses_select_resources_by_default() -> None:
    script = generate_script(
        PBSJobConfig(
            job_name="demo",
            n_nodes=2,
            n_cores_per_node=32,
        ),
        ["mpirun -np 64 demo.x"],
    )
    assert "#PBS -l select=2:ncpus=32:mpiprocs=32:ompthreads=1" in script
    assert "#PBS -l nodes=2:ppn=32" not in script


def test_generate_script_supports_single_rank_threaded_layout() -> None:
    script = generate_script(
        PBSJobConfig(
            job_name="demo",
            n_nodes=1,
            n_cores_per_node=64,
            mpi_procs_per_node=1,
            omp_threads=64,
        ),
        ["mpirun -np 1 python3 demo.py"],
    )
    assert "#PBS -l select=1:ncpus=64:mpiprocs=1:ompthreads=64" in script


def test_generate_script_accepts_custom_log_paths() -> None:
    script = generate_script(
        PBSJobConfig(
            job_name="demo",
            n_nodes=1,
            n_cores_per_node=32,
            stdout_path="/tmp/demo/stdout_demo.log",
            runtime_log_path="/tmp/demo/wtec_job_demo.log",
            work_dir="/tmp/demo",
        ),
        ["mpirun -np 32 demo.x"],
    )
    assert "#PBS -o /tmp/demo/stdout_demo.log" in script
    assert "exec > >(tee -a /tmp/demo/wtec_job_demo.log) 2>&1" in script


def test_wannier90_script_uses_mixed_layout_for_pw2wan_and_final_wannier() -> None:
    script = wannier90_script(
        "TiS",
        work_dir="/tmp/demo",
        n_nodes=1,
        n_cores_per_node=64,
        queue="g4",
    )
    assert "#PBS -l select=1:ncpus=64:mpiprocs=64:ompthreads=1" in script
    assert "env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 mpirun -np 1 --bind-to none wannier90.x  -pp TiS" in script or "env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 mpirun -np 1 --bind-to none wannier90.x -pp TiS" in script
    assert "-pp TiS" in script
    assert "env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 mpirun -np 64 --bind-to core pw2wannier90.x  -nk 64 < TiS.pw2wan.in > TiS.pw2wan.out" in script or "env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 mpirun -np 64 --bind-to core pw2wannier90.x -nk 64 < TiS.pw2wan.in > TiS.pw2wan.out" in script
    assert f"env OMP_NUM_THREADS=64 MKL_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64 NUMEXPR_NUM_THREADS=64 mpirun -np 1 --bind-to none wannier90.x  TiS" in script or "env OMP_NUM_THREADS=64 MKL_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64 NUMEXPR_NUM_THREADS=64 mpirun -np 1 --bind-to none wannier90.x TiS" in script


def test_wannier90_restart_script_uses_threaded_single_rank_layout() -> None:
    script = wannier90_script(
        "TiS",
        work_dir="/tmp/demo",
        n_nodes=1,
        n_cores_per_node=64,
        queue="g4",
        restart_only=True,
    )
    assert "#PBS -l select=1:ncpus=64:mpiprocs=1:ompthreads=64" in script
    assert "pw2wannier90.x" not in script
    assert "-pp TiS" not in script
    assert "mpirun -np 1 --bind-to none wannier90.x TiS" in script or "mpirun -np 1 --bind-to none wannier90.x  TiS" in script
