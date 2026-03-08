"""SIESTA cluster pipeline: SCF -> NSCF -> Wannier90."""

from __future__ import annotations

import math
import re
import shlex
from pathlib import Path
from typing import Any

from wtec.cluster.mpi import MPIConfig, build_command
from wtec.cluster.pbs import PBSJobConfig, generate_script
from wtec.config.materials import get_material
from wtec.siesta.inputs import SiestaInputGenerator
from wtec.siesta.wannier_bridge import prepare_wannier_bridge
from wtec.wannier.convergence import assert_wannier_converged
from wtec.wannier.inputs import generate_win


class SiestaPipeline:
    """Orchestrate SIESTA+Wannier90 stages via qsub/mpirun."""

    def __init__(
        self,
        atoms,
        material: str,
        job_manager,
        *,
        run_dir: Path,
        remote_base: str,
        n_nodes: int = 1,
        n_cores_per_node: int = 32,
        n_cores_by_queue: dict[str, int] | None = None,
        queue: str | None = None,
        queue_priority: list[str] | None = None,
        modules: list[str] | None = None,
        bin_dirs: list[str] | None = None,
        kpoints_scf: tuple[int, int, int] = (8, 8, 8),
        kpoints_nscf: tuple[int, int, int] = (12, 12, 12),
        pseudo_dir: str = "/pseudo",
        basis_profile: str = "",
        wannier_interface: str = "sisl",
        spin_orbit: bool = True,
        include_pao_basis: bool = True,
        mpi_np_scf: int = 0,
        mpi_np_nscf: int = 0,
        mpi_np_wannier: int = 0,
        omp_threads_scf: int = 0,
        omp_threads_nscf: int = 0,
        omp_threads_wannier: int = 0,
        factorization_defaults: dict[str, Any] | None = None,
        dm_mixing_weight: float = 0.10,
        dm_number_pulay: int = 8,
        electronic_temperature_k: float = 300.0,
        max_scf_iterations: int = 200,
        dispersion_cfg: dict[str, Any] | None = None,
        walltime_scf: str = "12:00:00",
        walltime_nscf: str = "12:00:00",
        walltime_wan: str = "04:00:00",
        omp_threads: int | None = None,
        live_log: bool = False,
        log_poll_interval: int = 30,
        stale_log_seconds: int = 300,
    ) -> None:
        self.atoms = atoms
        self.material = str(material)
        self.preset = get_material(self.material)
        self.jm = job_manager
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.remote_base = str(remote_base).rstrip("/")
        self.n_nodes = int(n_nodes)
        self.n_cpn = int(n_cores_per_node)
        self.n_cpn_by_queue = dict(n_cores_by_queue or {})
        self.queue = queue
        self.queue_priority = list(queue_priority or ["g4", "g3", "g2", "g1"])
        self.modules = list(modules or [])
        self.bin_dirs = list(dict.fromkeys([p for p in (bin_dirs or []) if p]))
        self.kpoints_scf = tuple(int(v) for v in kpoints_scf)
        self.kpoints_nscf = tuple(int(v) for v in kpoints_nscf)
        self.pseudo_dir = str(pseudo_dir)
        self.basis_profile = str(basis_profile or self.preset.siesta_basis_profile or "default")
        self.wannier_interface = str(wannier_interface or "sisl").strip().lower()
        self.spin_orbit = bool(spin_orbit)
        self.include_pao_basis = bool(include_pao_basis)
        self.mpi_np_scf = int(mpi_np_scf)
        self.mpi_np_nscf = int(mpi_np_nscf)
        self.mpi_np_wannier = int(mpi_np_wannier)
        self.omp_threads_scf = int(omp_threads_scf)
        self.omp_threads_nscf = int(omp_threads_nscf)
        self.omp_threads_wannier = int(omp_threads_wannier)
        self.factorization_defaults = (
            dict(factorization_defaults) if isinstance(factorization_defaults, dict) else {}
        )
        self.dm_mixing_weight = float(dm_mixing_weight)
        self.dm_number_pulay = int(dm_number_pulay)
        self.electronic_temperature_k = float(electronic_temperature_k)
        self.max_scf_iterations = int(max_scf_iterations)
        disp = dispersion_cfg if isinstance(dispersion_cfg, dict) else {}
        method = str(disp.get("method", "d3")).strip().lower() or "d3"
        self.dispersion_enabled = bool(disp.get("enabled", True)) and method != "none"
        self.dispersion_method = method
        self.siesta_dftd3_use_xc_defaults = bool(
            disp.get(
                "siesta_dftd3_use_xc_defaults",
                disp.get("siesta_dftd3_use_xc_functional", True),
            )
        )
        self.walltime_scf = str(walltime_scf)
        self.walltime_nscf = str(walltime_nscf)
        self.walltime_wan = str(walltime_wan)
        self.omp_threads = omp_threads
        self.live_log = bool(live_log)
        self.log_poll_interval = int(log_poll_interval) if int(log_poll_interval) > 0 else 30
        self.stale_log_seconds = int(stale_log_seconds) if int(stale_log_seconds) > 0 else 300
        self._resolved_queue: str | None = None
        self._resolved_cores: int | None = None
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def run_scf(self) -> dict[str, Any]:
        gen = self._input_generator()
        scf_in = self.run_dir / f"{self.material}.scf.fdf"
        scf_out = self.run_dir / f"{self.material}.scf.out"
        gen.scf(scf_in)
        remote_dir = self._remote_dir()
        self._preflight(["qsub", "qstat", "mpirun", "siesta"], [scf_in])
        queue = self._resolved_queue_name()
        cores_per_node = self._resolved_cores_per_node()
        mpi_np, omp_threads = self._resolve_stage_parallel("scf")
        mpi = MPIConfig(n_cores=mpi_np)
        raw_cmd = build_command(
            "siesta",
            input_file=scf_in.name,
            output_file=scf_out.name,
            mpi=mpi,
        )
        cmd = self._timed_command(raw_cmd)
        cmds = [*self._pseudo_stage_commands(), cmd]
        script = generate_script(
            PBSJobConfig(
                job_name=f"scf_{self.material}"[:15],
                n_nodes=self.n_nodes,
                n_cores_per_node=cores_per_node,
                walltime=self.walltime_scf,
                memory_gb=self._estimate_memory_gb(),
                queue=queue,
                work_dir=remote_dir,
                modules=self.modules,
                env_vars=self._runtime_env(stage_threads=omp_threads),
            ),
            cmds,
        )
        meta = self.jm.submit_and_wait(
            script,
            remote_dir=remote_dir,
            local_dir=self.run_dir,
            retrieve_patterns=[scf_out.name, f"scf_{self.material}.log"],
            script_name=f"scf_{self.material}.pbs",
            stage_files=[scf_in],
            expected_local_outputs=[scf_out.name],
            queue_used=queue,
            poll_interval=self.log_poll_interval,
            live_log=self.live_log,
            live_files=[scf_out.name, f"scf_{self.material}.log"],
            stale_log_seconds=self.stale_log_seconds,
        )
        meta["runtime_sec"] = self._extract_stage_runtime(
            self.run_dir / f"scf_{self.material}.log"
        )
        meta["mpi_np"] = int(mpi_np)
        meta["omp_threads"] = int(omp_threads)
        return meta

    def run_nscf(self, _fermi_ev: float | None = None) -> dict[str, Any]:
        gen = self._input_generator()
        nscf_in = self.run_dir / f"{self.material}.nscf.fdf"
        nscf_out = self.run_dir / f"{self.material}.nscf.out"
        gen.nscf(nscf_in)
        remote_dir = self._remote_dir()
        self._preflight(["qsub", "qstat", "mpirun", "siesta"], [nscf_in])
        queue = self._resolved_queue_name()
        cores_per_node = self._resolved_cores_per_node()
        mpi_np, omp_threads = self._resolve_stage_parallel("nscf")
        mpi = MPIConfig(n_cores=mpi_np)
        raw_cmd = build_command(
            "siesta",
            input_file=nscf_in.name,
            output_file=nscf_out.name,
            mpi=mpi,
        )
        cmd = self._timed_command(raw_cmd)
        cmds = [*self._pseudo_stage_commands(), cmd]
        script = generate_script(
            PBSJobConfig(
                job_name=f"nscf_{self.material}"[:15],
                n_nodes=self.n_nodes,
                n_cores_per_node=cores_per_node,
                walltime=self.walltime_nscf,
                memory_gb=self._estimate_memory_gb(),
                queue=queue,
                work_dir=remote_dir,
                modules=self.modules,
                env_vars=self._runtime_env(stage_threads=omp_threads),
            ),
            cmds,
        )
        meta = self.jm.submit_and_wait(
            script,
            remote_dir=remote_dir,
            local_dir=self.run_dir,
            retrieve_patterns=[nscf_out.name, f"nscf_{self.material}.log", "*.amn", "*.mmn", "*.eig"],
            script_name=f"nscf_{self.material}.pbs",
            stage_files=[nscf_in],
            expected_local_outputs=[nscf_out.name],
            queue_used=queue,
            poll_interval=self.log_poll_interval,
            live_log=self.live_log,
            live_files=[nscf_out.name, f"nscf_{self.material}.log"],
            stale_log_seconds=self.stale_log_seconds,
        )
        meta["runtime_sec"] = self._extract_stage_runtime(
            self.run_dir / f"nscf_{self.material}.log"
        )
        meta["mpi_np"] = int(mpi_np)
        meta["omp_threads"] = int(omp_threads)
        return meta

    def run_wannier(
        self,
        fermi_energy: float,
        *,
        dis_win_override: tuple[float, float] | None = None,
        dis_froz_win_override: tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        # generate Wannier win from existing helper
        win = self.run_dir / f"{self.material}.win"
        dis_win = (
            tuple(float(v) for v in dis_win_override)
            if dis_win_override is not None
            else self.preset.dis_win
        )
        dis_froz_win = (
            tuple(float(v) for v in dis_froz_win_override)
            if dis_froz_win_override is not None
            else self.preset.dis_froz_win
        )
        generate_win(
            self.atoms,
            self.material,
            num_bands=max(int(self.preset.num_bands), 2 * int(self.preset.num_wann)),
            fermi_energy=float(fermi_energy),
            kpoints=self.kpoints_nscf,
            dis_win=dis_win,
            dis_froz_win=dis_froz_win,
            spinors=True,
            outfile=win,
        )
        bridge_info = prepare_wannier_bridge(
            run_dir=self.run_dir,
            seedname=self.material,
            interface=self.wannier_interface,
        )
        remote_dir = self._remote_dir()
        self._preflight(["qsub", "qstat", "mpirun", "wannier90.x"], [win])
        queue = self._resolved_queue_name()
        cores_per_node = self._resolved_cores_per_node()
        mpi_np, omp_threads = self._resolve_stage_parallel("wannier")
        mpi = MPIConfig(n_cores=mpi_np)
        cmds = [
            build_command("wannier90.x", extra_args=f"-pp {self.material}", mpi=mpi),
            build_command("wannier90.x", extra_args=f"{self.material}", mpi=mpi),
        ]
        script = generate_script(
            PBSJobConfig(
                job_name=f"wan_{self.material}"[:15],
                n_nodes=self.n_nodes,
                n_cores_per_node=cores_per_node,
                walltime=self.walltime_wan,
                memory_gb=self._estimate_memory_gb(),
                queue=queue,
                work_dir=remote_dir,
                modules=self.modules,
                env_vars=self._runtime_env(stage_threads=omp_threads),
            ),
            cmds,
        )
        meta = self.jm.submit_and_wait(
            script,
            remote_dir=remote_dir,
            local_dir=self.run_dir,
            retrieve_patterns=[
                f"{self.material}.wout",
                f"{self.material}_hr.dat",
                f"wan_{self.material}.log",
            ],
            script_name=f"wan_{self.material}.pbs",
            stage_files=[win],
            expected_local_outputs=[f"{self.material}.wout", f"{self.material}_hr.dat"],
            queue_used=queue,
            poll_interval=self.log_poll_interval,
            live_log=self.live_log,
            live_files=[f"{self.material}.wout", f"wan_{self.material}.log"],
            stale_log_seconds=self.stale_log_seconds,
        )
        meta["runtime_sec"] = self._extract_stage_runtime(
            self.run_dir / f"wan_{self.material}.log"
        )
        meta["mpi_np"] = int(mpi_np)
        meta["omp_threads"] = int(omp_threads)
        assert_wannier_converged(
            wout_path=self.run_dir / f"{self.material}.wout",
            win_path=win,
        )
        meta["bridge"] = bridge_info
        return meta

    def run_full(self) -> Path:
        from wtec.siesta.parser import parse_fermi_energy

        self.run_scf()
        fermi_ev = parse_fermi_energy(self.run_dir / f"{self.material}.scf.out")
        self.run_nscf(fermi_ev)
        self.run_wannier(fermi_ev)
        hr = self.run_dir / f"{self.material}_hr.dat"
        if not hr.exists():
            raise FileNotFoundError(f"_hr.dat not found after SIESTA/Wannier stage: {hr}")
        return hr

    def _input_generator(self) -> SiestaInputGenerator:
        return SiestaInputGenerator(
            atoms=self.atoms,
            material_name=self.material,
            # Read pseudopotentials from local workdir for robust MPI node access.
            pseudo_dir=".",
            pseudopots=self.preset.siesta_pseudopots,
            basis_profile=self.basis_profile,
            kpoints_scf=self.kpoints_scf,
            kpoints_nscf=self.kpoints_nscf,
            spin_orbit=self.spin_orbit,
            include_pao_basis=self.include_pao_basis,
            num_bands=max(int(self.preset.num_bands), 2 * int(self.preset.num_wann)),
            dispersion_enabled=self.dispersion_enabled,
            dispersion_method=self.dispersion_method,
            siesta_dftd3_use_xc_defaults=self.siesta_dftd3_use_xc_defaults,
            dm_mixing_weight=self.dm_mixing_weight,
            dm_number_pulay=self.dm_number_pulay,
            electronic_temperature_k=self.electronic_temperature_k,
            max_scf_iterations=self.max_scf_iterations,
        )

    def _pseudo_stage_commands(self) -> list[str]:
        cmds: list[str] = []
        pseudo_dir_q = shlex.quote(str(self.pseudo_dir))
        for pp in sorted(set(self.preset.siesta_pseudopots.values())):
            pp_q = shlex.quote(str(pp))
            cmds.append(f"cp -f {pseudo_dir_q}/{pp_q} ./{pp_q}")
        return cmds

    def _remote_dir(self) -> str:
        return f"{self.remote_base}/{self.material}"

    def _resolved_queue_name(self) -> str:
        if self._resolved_queue is None:
            self._resolved_queue = self.jm.resolve_queue(
                self.queue,
                fallback_order=self.queue_priority,
            )
        return self._resolved_queue

    def _resolved_cores_per_node(self) -> int:
        if self._resolved_cores is None:
            queue = self._resolved_queue_name()
            mapped = self.n_cpn_by_queue.get(queue) or self.n_cpn_by_queue.get(queue.lower())
            self._resolved_cores = int(mapped) if mapped else int(self.n_cpn)
        return self._resolved_cores

    def _runtime_env(self, *, stage_threads: int | None = None) -> dict[str, str]:
        env: dict[str, str] = {}
        if self.bin_dirs:
            env["PATH"] = ":".join(self.bin_dirs) + ":$PATH"
        thr = stage_threads if stage_threads is not None else self.omp_threads
        if thr is not None:
            t = str(int(thr))
            env["OMP_NUM_THREADS"] = t
            env["MKL_NUM_THREADS"] = t
            env["OPENBLAS_NUM_THREADS"] = t
            env["NUMEXPR_NUM_THREADS"] = t
        return env

    @staticmethod
    def _timed_command(command: str) -> str:
        payload = (
            "SECONDS=0; "
            + command
            + "; rc=$?; "
            "echo WTEC_STAGE_RUNTIME_SEC=$SECONDS; "
            "exit $rc"
        )
        # Use non-login shell to avoid cluster profile scripts changing CWD.
        return f"bash -c {shlex.quote(payload)}"

    @staticmethod
    def _extract_stage_runtime(log_path: Path) -> float | None:
        if not log_path.exists() or log_path.stat().st_size == 0:
            return None
        text = log_path.read_text(errors="ignore")
        matches = re.findall(r"WTEC_STAGE_RUNTIME_SEC=([0-9]+(?:\\.[0-9]+)?)", text)
        if not matches:
            return None
        try:
            return float(matches[-1])
        except Exception:
            return None

    def _queue_factorization_defaults(self) -> dict[str, Any]:
        raw = self.factorization_defaults if isinstance(self.factorization_defaults, dict) else {}
        if not raw:
            return {}
        queue = self._resolved_queue_name().strip().lower()
        cores = int(self._resolved_cores_per_node())
        key = f"{queue}_{cores}"
        scoped = raw.get(key)
        if not isinstance(scoped, dict):
            # Legacy/support path for nested {queue: {cores: {...}}}
            qtbl = raw.get(queue)
            if isinstance(qtbl, dict):
                scoped = qtbl.get(str(cores)) or qtbl.get(int(cores))
        return dict(scoped) if isinstance(scoped, dict) else {}

    def _resolve_stage_parallel(self, stage: str) -> tuple[int, int]:
        queue = self._resolved_queue_name()
        cores_per_node = self._resolved_cores_per_node()
        total_alloc = max(1, int(self.n_nodes) * int(cores_per_node))
        scoped = self._queue_factorization_defaults()
        st = stage.strip().lower()
        if st == "scf":
            mpi_np = int(self.mpi_np_scf)
            omp_thr = int(self.omp_threads_scf)
        elif st == "nscf":
            mpi_np = int(self.mpi_np_nscf)
            omp_thr = int(self.omp_threads_nscf)
        else:
            mpi_np = int(self.mpi_np_wannier)
            omp_thr = int(self.omp_threads_wannier)

        if mpi_np <= 0:
            scoped_np = scoped.get(f"mpi_np_{st}")
            if scoped_np is not None:
                mpi_np = int(scoped_np)
        if mpi_np <= 0:
            mpi_np = total_alloc
        if mpi_np > total_alloc:
            raise ValueError(
                f"SIESTA {st} mpi_np={mpi_np} exceeds allocated cores={total_alloc} "
                f"(queue={queue}, nodes={self.n_nodes}, ppn={cores_per_node})."
            )
        if omp_thr <= 0:
            scoped_thr = scoped.get(f"omp_threads_{st}")
            if scoped_thr is not None:
                omp_thr = int(scoped_thr)
        if omp_thr <= 0:
            if self.omp_threads is not None and int(self.omp_threads) > 0:
                omp_thr = int(self.omp_threads)
            else:
                omp_thr = max(1, total_alloc // mpi_np)
        if mpi_np * omp_thr > total_alloc:
            raise ValueError(
                f"SIESTA {st} oversubscribed: mpi_np={mpi_np}, omp_threads={omp_thr}, "
                f"allocated={total_alloc} (queue={queue})."
            )
        return int(mpi_np), int(omp_thr)

    def _estimate_memory_gb(self) -> int:
        """Conservative O(N_orb^2)-scaled memory estimate for SIESTA jobs."""
        # Crude per-atom orbital budget for spinor basis.
        orbs_per_atom = {
            "Ta": 20,
            "Nb": 20,
            "Co": 18,
            "P": 8,
            "Si": 8,
            "O": 8,
        }
        n_orb = 0
        for sym in self.atoms.get_chemical_symbols():
            n_orb += int(orbs_per_atom.get(sym, 10))
        n_orb = max(64, int(n_orb))
        # Complex dense-like scaling + safety margin.
        bytes_est = 16.0 * float(n_orb) * float(n_orb) * 3.0
        gb = max(4, int(math.ceil(bytes_est / (1024.0**3))) + 2)
        return gb

    def _preflight(self, required_commands: list[str], local_inputs: list[Path]) -> None:
        self.jm.ensure_remote_commands(
            required_commands,
            modules=self.modules,
            bin_dirs=self.bin_dirs,
        )
        self.jm.ensure_remote_mpi_binaries(
            [cmd for cmd in required_commands if cmd not in {"qsub", "qstat"}],
            modules=self.modules,
            bin_dirs=self.bin_dirs,
        )
        self.jm.ensure_remote_files(
            self.pseudo_dir,
            sorted(set(self.preset.siesta_pseudopots.values())),
        )
        for p in local_inputs:
            if not Path(p).exists() or Path(p).stat().st_size == 0:
                raise FileNotFoundError(f"Missing required local SIESTA input file: {p}")
