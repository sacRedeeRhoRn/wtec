"""VASP cluster pipeline: SCF -> NSCF -> Wannier90."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from wtec.cluster.mpi import MPIConfig, build_command
from wtec.cluster.pbs import PBSJobConfig, generate_script
from wtec.config.materials import get_material
from wtec.vasp.inputs import VaspInputGenerator
from wtec.wannier.convergence import (
    assert_wannier_converged,
    assert_wannier_topology_from_files,
)
from wtec.wannier.inputs import generate_win


class VaspPipeline:
    """Orchestrate VASP + Wannier90 stages via qsub/mpirun."""

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
        executable: str = "vasp_std",
        encut_ev: float = 520.0,
        ediff: float = 1e-6,
        ismear: int = 0,
        sigma: float = 0.05,
        disable_symmetry: bool = True,
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
        self.executable = str(executable or "vasp_std").strip() or "vasp_std"
        self.encut_ev = float(encut_ev)
        self.ediff = float(ediff)
        self.ismear = int(ismear)
        self.sigma = float(sigma)
        self.disable_symmetry = bool(disable_symmetry)
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
        files = gen.write_scf_inputs(self.run_dir)
        scf_out = self.run_dir / f"{self.material}.scf.out"

        remote_dir = self._remote_dir()
        self._preflight(["qsub", "qstat", "mpirun", self.executable], list(files.values()))
        queue = self._resolved_queue_name()
        cores_per_node = self._resolved_cores_per_node()
        mpi = MPIConfig(n_cores=self.n_nodes * cores_per_node)

        cmds = [
            "cp INCAR.scf INCAR",
            "cp KPOINTS.scf KPOINTS",
            self._potcar_build_command(species_order=gen.species_order()),
            build_command(self.executable, output_file=scf_out.name, mpi=mpi),
            "if [ -f OUTCAR ]; then cp OUTCAR OUTCAR.scf; fi",
        ]
        script = generate_script(
            PBSJobConfig(
                job_name=f"scf_{self.material}"[:15],
                n_nodes=self.n_nodes,
                n_cores_per_node=cores_per_node,
                walltime=self.walltime_scf,
                queue=queue,
                work_dir=remote_dir,
                modules=self.modules,
                env_vars=self._runtime_env(),
            ),
            cmds,
        )
        return self.jm.submit_and_wait(
            script,
            remote_dir=remote_dir,
            local_dir=self.run_dir,
            retrieve_patterns=[scf_out.name, "OUTCAR", "OUTCAR.scf", f"scf_{self.material}.log"],
            script_name=f"scf_{self.material}.pbs",
            stage_files=list(files.values()),
            expected_local_outputs=[scf_out.name],
            queue_used=queue,
            poll_interval=self.log_poll_interval,
            live_log=self.live_log,
            live_files=[scf_out.name, f"scf_{self.material}.log"],
            stale_log_seconds=self.stale_log_seconds,
        )

    def run_nscf(self, _fermi_ev: float | None = None) -> dict[str, Any]:
        gen = self._input_generator()
        files = gen.write_nscf_inputs(self.run_dir)
        nscf_out = self.run_dir / f"{self.material}.nscf.out"

        remote_dir = self._remote_dir()
        self._preflight(["qsub", "qstat", "mpirun", self.executable], list(files.values()))
        queue = self._resolved_queue_name()
        cores_per_node = self._resolved_cores_per_node()
        mpi = MPIConfig(n_cores=self.n_nodes * cores_per_node)

        cmds = [
            "cp INCAR.nscf INCAR",
            "cp KPOINTS.nscf KPOINTS",
            self._potcar_build_command(species_order=gen.species_order()),
            build_command(self.executable, output_file=nscf_out.name, mpi=mpi),
            "if [ -f OUTCAR ]; then cp OUTCAR OUTCAR.nscf; fi",
        ]
        script = generate_script(
            PBSJobConfig(
                job_name=f"nscf_{self.material}"[:15],
                n_nodes=self.n_nodes,
                n_cores_per_node=cores_per_node,
                walltime=self.walltime_nscf,
                queue=queue,
                work_dir=remote_dir,
                modules=self.modules,
                env_vars=self._runtime_env(),
            ),
            cmds,
        )
        return self.jm.submit_and_wait(
            script,
            remote_dir=remote_dir,
            local_dir=self.run_dir,
            retrieve_patterns=[nscf_out.name, "OUTCAR", "OUTCAR.nscf", f"nscf_{self.material}.log"],
            script_name=f"nscf_{self.material}.pbs",
            stage_files=list(files.values()),
            expected_local_outputs=[nscf_out.name],
            queue_used=queue,
            poll_interval=self.log_poll_interval,
            live_log=self.live_log,
            live_files=[nscf_out.name, f"nscf_{self.material}.log"],
            stale_log_seconds=self.stale_log_seconds,
        )

    def run_wannier(
        self,
        fermi_energy: float,
        *,
        dis_win_override: tuple[float, float] | None = None,
        dis_froz_win_override: tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        gen = self._input_generator()
        files = gen.write_wannier_inputs(self.run_dir)

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

        remote_dir = self._remote_dir()
        self._preflight(
            ["qsub", "qstat", "mpirun", self.executable, "wannier90.x"],
            [*list(files.values()), win],
        )
        queue = self._resolved_queue_name()
        cores_per_node = self._resolved_cores_per_node()
        mpi = MPIConfig(n_cores=self.n_nodes * cores_per_node)

        cmds = [
            "cp INCAR.wannier INCAR",
            "cp KPOINTS.nscf KPOINTS",
            self._potcar_build_command(species_order=gen.species_order()),
            build_command(self.executable, output_file=f"{self.material}.wannier.vasp.out", mpi=mpi),
            build_command("wannier90.x", extra_args=f"-pp {self.material}", mpi=mpi),
            build_command("wannier90.x", extra_args=f"{self.material}", mpi=mpi),
        ]
        script = generate_script(
            PBSJobConfig(
                job_name=f"wan_{self.material}"[:15],
                n_nodes=self.n_nodes,
                n_cores_per_node=cores_per_node,
                walltime=self.walltime_wan,
                queue=queue,
                work_dir=remote_dir,
                modules=self.modules,
                env_vars=self._runtime_env(),
            ),
            cmds,
        )
        meta = self.jm.submit_and_wait(
            script,
            remote_dir=remote_dir,
            local_dir=self.run_dir,
            retrieve_patterns=[
                f"{self.material}.wannier.vasp.out",
                f"{self.material}.wout",
                f"{self.material}_hr.dat",
                f"wan_{self.material}.log",
            ],
            script_name=f"wan_{self.material}.pbs",
            stage_files=[*list(files.values()), win],
            expected_local_outputs=[f"{self.material}.wout", f"{self.material}_hr.dat"],
            queue_used=queue,
            poll_interval=self.log_poll_interval,
            live_log=self.live_log,
            live_files=[f"{self.material}.wout", f"wan_{self.material}.log"],
            stale_log_seconds=self.stale_log_seconds,
        )
        assert_wannier_converged(
            wout_path=self.run_dir / f"{self.material}.wout",
            win_path=win,
        )
        assert_wannier_topology_from_files(
            hr_dat_path=self.run_dir / f"{self.material}_hr.dat",
            win_path=win,
            material_class=getattr(self.preset, "material_class", "generic"),
        )
        return meta

    def run_full(self) -> Path:
        from wtec.vasp.parser import parse_fermi_energy

        self.run_scf()
        fermi_ev = parse_fermi_energy(self.run_dir / f"{self.material}.scf.out")
        self.run_nscf(fermi_ev)
        self.run_wannier(fermi_ev)
        hr = self.run_dir / f"{self.material}_hr.dat"
        if not hr.exists():
            raise FileNotFoundError(f"_hr.dat not found after VASP/Wannier stage: {hr}")
        return hr

    def _input_generator(self) -> VaspInputGenerator:
        return VaspInputGenerator(
            atoms=self.atoms,
            material_name=self.material,
            kpoints_scf=self.kpoints_scf,
            kpoints_nscf=self.kpoints_nscf,
            encut_ev=self.encut_ev,
            ediff=self.ediff,
            ismear=self.ismear,
            sigma=self.sigma,
            spin_orbit=True,
            disable_symmetry=self.disable_symmetry,
            num_bands=max(int(self.preset.num_bands), 2 * int(self.preset.num_wann)),
        )

    def _runtime_env(self) -> dict[str, str]:
        env: dict[str, str] = {}
        if self.omp_threads is not None:
            t = str(int(self.omp_threads))
            env["OMP_NUM_THREADS"] = t
            env["MKL_NUM_THREADS"] = t
            env["OPENBLAS_NUM_THREADS"] = t
            env["NUMEXPR_NUM_THREADS"] = t
        return env

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

    def _potcar_map(self) -> dict[str, str]:
        mapping = dict(getattr(self.preset, "vasp_potcars", {}) or {})
        if not mapping:
            for sym in sorted(set(self.atoms.get_chemical_symbols())):
                mapping[sym] = f"{sym}/POTCAR"
        return mapping

    def _potcar_build_command(self, *, species_order: list[str]) -> str:
        mapping = self._potcar_map()
        potcars: list[str] = []
        for sym in species_order:
            rel = mapping.get(sym)
            if not rel:
                raise ValueError(
                    f"Missing VASP POTCAR mapping for species {sym!r} in preset {self.material}."
                )
            potcars.append(f"{self.pseudo_dir.rstrip('/')}/{rel.lstrip('/')}" )
        cat_args = " ".join(shlex.quote(p) for p in potcars)
        return f"cat {cat_args} > POTCAR"

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
            sorted(set(self._potcar_map().values())),
        )
        for p in local_inputs:
            pp = Path(p)
            if not pp.exists() or pp.stat().st_size == 0:
                raise FileNotFoundError(f"Missing required local VASP input file: {pp}")
