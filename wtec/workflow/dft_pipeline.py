"""DFT pipeline: SCF → NSCF → pw2wannier90 → Wannier90."""

from __future__ import annotations

import math
import posixpath
import re
import shlex
from pathlib import Path
from typing import Iterable

from wtec.config.materials import get_material
from wtec.cluster.pbs import qe_scf_script, wannier90_script, PBSJobConfig, generate_script
from wtec.cluster.mpi import MPIConfig, build_command
from wtec.qe.inputs import QEInputGenerator
from wtec.qe.pw2wan import generate as pw2wan_generate
from wtec.wannier.inputs import generate_win
from wtec.wannier.convergence import (
    assert_wannier_converged,
    assert_wannier_topology_from_files,
)


class DFTPipeline:
    """Orchestrate the full DFT → Wannier90 pipeline on a cluster."""

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
        n_pool: int = 4,
        modules: list[str] | None = None,
        bin_dirs: list[str] | None = None,
        kpoints_scf: tuple = (8, 8, 8),
        kpoints_nscf: tuple = (12, 12, 12),
        queue: str | None = None,
        queue_priority: list[str] | None = None,
        walltime_scf: str = "12:00:00",
        walltime_nscf: str = "12:00:00",
        walltime_wan: str = "4:00:00",
        pseudo_dir: str = "/pseudo",
        omp_threads: int | None = None,
        qe_noncolin: bool = True,
        qe_lspinorb: bool = True,
        qe_disable_symmetry: bool = False,
        dispersion_cfg: dict | None = None,
        live_log: bool = False,
        log_poll_interval: int = 30,
        stale_log_seconds: int = 300,
    ) -> None:
        self.atoms = atoms
        self.material = material
        self.preset = get_material(material)
        self.jm = job_manager
        self.run_dir = Path(run_dir)
        self.remote_base = remote_base.rstrip("/")
        self.n_nodes = n_nodes
        self.n_cpn = n_cores_per_node
        self.n_cpn_by_queue = n_cores_by_queue or {}
        self.n_pool = n_pool
        self.modules = modules or []
        self.bin_dirs = list(dict.fromkeys([p for p in (bin_dirs or []) if p]))
        self.kpoints_scf = kpoints_scf
        self.kpoints_nscf = kpoints_nscf
        self.queue = queue
        self.queue_priority = queue_priority or ["g4", "g3", "g2", "g1"]
        self.walltime_scf = walltime_scf
        self.walltime_nscf = walltime_nscf
        self.walltime_wan = walltime_wan
        self.pseudo_dir = pseudo_dir
        self.omp_threads = omp_threads
        self.qe_noncolin = bool(qe_noncolin)
        self.qe_lspinorb = bool(qe_lspinorb)
        self.qe_disable_symmetry = bool(qe_disable_symmetry)
        disp = dispersion_cfg if isinstance(dispersion_cfg, dict) else {}
        method = str(disp.get("method", "d3")).strip().lower() or "d3"
        self.dispersion_enabled = bool(disp.get("enabled", True)) and method != "none"
        self.dispersion_method = method
        self.qe_vdw_corr = str(disp.get("qe_vdw_corr", "grimme-d3")).strip() or "grimme-d3"
        self.qe_dftd3_version = int(disp.get("qe_dftd3_version", 4))
        self.qe_dftd3_threebody = bool(disp.get("qe_dftd3_threebody", True))
        self.live_log = bool(live_log)
        self.log_poll_interval = int(log_poll_interval) if int(log_poll_interval) > 0 else 30
        self.stale_log_seconds = int(stale_log_seconds) if int(stale_log_seconds) > 0 else 300
        self._resolved_queue: str | None = None
        self._resolved_cores: int | None = None
        self._resolved_pseudopots: dict[str, str] | None = None
        self._pseudo_dir_listing: list[str] | None = None
        self._pseudo_valence_cache: dict[str, float | None] = {}
        self.run_dir.mkdir(parents=True, exist_ok=True)

    # ── public methods ────────────────────────────────────────────────────────

    def run_scf(self) -> dict:
        """Run pw.x SCF calculation."""
        nbnd = self._recommended_nbnd()
        print(
            f"[DFTPipeline] qe_lspinorb={self.qe_lspinorb} nbnd={nbnd} "
            f"pseudopotentials={self._resolved_pseudopots_map()}"
        )
        gen = self._input_generator(calculation="scf")
        scf_in = self.run_dir / f"{self.material}.scf.in"
        gen.scf(outfile=scf_in)
        remote_dir = self._remote_dir()
        self._preflight_step(
            required_commands=["qsub", "qstat", "mpirun", "pw.x"],
            local_inputs=[scf_in],
        )
        queue_used = self._resolved_queue_name()
        cores_per_node = self._resolved_cores_per_node()

        script = qe_scf_script(
            self.material,
            work_dir=remote_dir,
            n_nodes=self.n_nodes,
            n_cores_per_node=cores_per_node,
            n_pool=self.n_pool,
            walltime=self.walltime_scf,
            queue=queue_used,
            modules=self.modules,
            env_vars=self._runtime_env_vars(),
        )
        meta = self.jm.submit_and_wait(
            script,
            remote_dir=remote_dir,
            local_dir=self.run_dir,
            retrieve_patterns=[
                f"{self.material}.scf.out",
                f"scf_{self.material}.log",
                "*.xml",
            ],
            script_name=f"scf_{self.material}.pbs",
            stage_files=[scf_in],
            expected_local_outputs=[f"{self.material}.scf.out"],
            queue_used=queue_used,
            poll_interval=self.log_poll_interval,
            live_log=self.live_log,
            live_files=[f"{self.material}.scf.out", f"scf_{self.material}.log"],
            stale_log_seconds=self.stale_log_seconds,
        )
        return meta

    def run_nscf(self, fermi_energy: float) -> dict:
        """Run pw.x NSCF calculation."""
        gen = self._input_generator(calculation="nscf")
        nscf_in = self.run_dir / f"{self.material}.nscf.in"
        gen.nscf(outfile=nscf_in)
        remote_dir = self._remote_dir()
        self._preflight_step(
            required_commands=["qsub", "qstat", "mpirun", "pw.x"],
            local_inputs=[nscf_in],
        )
        queue_used = self._resolved_queue_name()
        cores_per_node = self._resolved_cores_per_node()

        mpi = MPIConfig(n_cores=self.n_nodes * cores_per_node, n_pool=self.n_pool)
        cmd = build_command(
            "pw.x",
            input_file=f"{self.material}.nscf.in",
            output_file=f"{self.material}.nscf.out",
            mpi=mpi,
        )
        cfg = PBSJobConfig(
            f"nscf_{self.material}",
            n_nodes=self.n_nodes,
            n_cores_per_node=cores_per_node,
            walltime=self.walltime_nscf,
            queue=queue_used,
            work_dir=remote_dir,
            modules=self.modules,
            env_vars=self._runtime_env_vars(),
        )
        script = generate_script(cfg, [cmd])
        meta = self.jm.submit_and_wait(
            script,
            remote_dir=remote_dir,
            local_dir=self.run_dir,
            retrieve_patterns=[
                f"{self.material}.nscf.out",
                f"nscf_{self.material}.log",
            ],
            script_name=f"nscf_{self.material}.pbs",
            stage_files=[nscf_in],
            expected_local_outputs=[f"{self.material}.nscf.out"],
            queue_used=queue_used,
            poll_interval=self.log_poll_interval,
            live_log=self.live_log,
            live_files=[f"{self.material}.nscf.out", f"nscf_{self.material}.log"],
            stale_log_seconds=self.stale_log_seconds,
        )
        return meta

    def run_wannier(
        self,
        fermi_energy: float,
        *,
        dis_win_override: tuple[float, float] | None = None,
        dis_froz_win_override: tuple[float, float] | None = None,
    ) -> dict:
        """Run pw2wannier90 + wannier90."""
        # pw2wannier90 input
        pw2wan_in = self.run_dir / f"{self.material}.pw2wan.in"
        pw2wan_generate(
            prefix=self.material,
            outdir="./out",
            seedname=self.material,
            outfile=pw2wan_in,
        )

        # .win file
        win_file = self.run_dir / f"{self.material}.win"
        # Keep Wannier's num_bands consistent with QE SCF/NSCF nbnd setting.
        nbnd = self._recommended_nbnd()
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
        current_dis_win = dis_win
        # Generate initial .win before preflight file checks.
        generate_win(
            self.atoms,
            self.material,
            num_bands=nbnd,
            fermi_energy=fermi_energy,
            kpoints=self.kpoints_nscf,
            dis_win=current_dis_win,
            dis_froz_win=dis_froz_win,
            spinors=self.qe_noncolin,
            outfile=win_file,
        )
        remote_dir = self._remote_dir()
        self._preflight_step(
            required_commands=[
                "qsub",
                "qstat",
                "mpirun",
                "pw2wannier90.x",
                "wannier90.x",
            ],
            local_inputs=[pw2wan_in, win_file],
        )
        queue_used = self._resolved_queue_name()
        cores_per_node = self._resolved_cores_per_node()

        script = wannier90_script(
            self.material,
            work_dir=remote_dir,
            n_nodes=self.n_nodes,
            n_cores_per_node=cores_per_node,
            walltime=self.walltime_wan,
            queue=queue_used,
            modules=self.modules,
            env_vars=self._runtime_env_vars(),
        )

        attempts = 0
        max_attempts = 2
        while True:
            attempts += 1
            generate_win(
                self.atoms,
                self.material,
                num_bands=nbnd,
                fermi_energy=fermi_energy,
                kpoints=self.kpoints_nscf,
                dis_win=current_dis_win,
                dis_froz_win=dis_froz_win,
                # Keep Wannier spinor mode consistent with QE noncollinear/SOC mode.
                spinors=self.qe_noncolin,
                outfile=win_file,
            )
            try:
                meta = self.jm.submit_and_wait(
                    script,
                    remote_dir=remote_dir,
                    local_dir=self.run_dir,
                    retrieve_patterns=[
                        f"{self.material}_hr.dat",
                        f"{self.material}.wout",
                        f"w90_{self.material}.log",
                        f"{self.material}.pw2wan.out",
                    ],
                    script_name=f"wan_{self.material}.pbs",
                    stage_files=[pw2wan_in, win_file],
                    expected_local_outputs=[f"{self.material}_hr.dat", f"{self.material}.wout"],
                    queue_used=queue_used,
                    poll_interval=self.log_poll_interval,
                    live_log=self.live_log,
                    live_files=[
                        f"{self.material}.pw2wan.out",
                        f"{self.material}.wout",
                        f"w90_{self.material}.log",
                    ],
                    stale_log_seconds=self.stale_log_seconds,
                )
                break
            except RuntimeError as exc:
                if attempts >= max_attempts:
                    raise
                if not self._is_wannier_dis_window_failure(remote_dir):
                    raise
                low, high = float(current_dis_win[0]), float(current_dis_win[1])
                widened = (min(low, -20.0), max(high, 20.0))
                if widened == current_dis_win:
                    raise
                print(
                    "[DFTPipeline] Wannier dis_win too narrow for some k-points; "
                    f"retrying with dis_win={widened} (previous={current_dis_win})."
                )
                current_dis_win = widened
                continue

        assert_wannier_converged(
            wout_path=self.run_dir / f"{self.material}.wout",
            win_path=win_file,
        )
        assert_wannier_topology_from_files(
            hr_dat_path=self.run_dir / f"{self.material}_hr.dat",
            win_path=win_file,
            material_class=getattr(self.preset, "material_class", "generic"),
        )
        return meta

    def _is_wannier_dis_window_failure(self, remote_dir: str) -> bool:
        """Detect dis_windows under-population failure in remote wannier logs."""
        needle = "dis_windows: Energy window contains fewer states than number of target WFs"
        wout = posixpath.join(remote_dir, f"{self.material}.wout")
        cmd = (
            "bash -lc "
            + shlex.quote(
                "for f in "
                + shlex.quote(wout)
                + " "
                + shlex.quote(posixpath.join(remote_dir, f"{self.material}.node_*.werr"))
                + "; do "
                "if [ -f \"$f\" ] && grep -Fq "
                + shlex.quote(needle)
                + " \"$f\"; then echo yes; exit 0; fi; "
                "done; echo no"
            )
        )
        rc, stdout, _ = self.jm._ssh.run(cmd, check=False)
        if rc != 0:
            return False
        return stdout.strip().splitlines()[-1:].pop() == "yes"

    def run_full(self) -> Path:
        """Run full SCF → NSCF → Wannier90 pipeline.

        Returns
        -------
        Path
            Path to the retrieved *_hr.dat file.
        """
        from wtec.qe.parser import parse_fermi_energy

        print(f"[DFTPipeline] Running SCF for {self.material}...")
        self.run_scf()

        scf_out = self.run_dir / f"{self.material}.scf.out"
        fermi_ev = parse_fermi_energy(scf_out)
        print(f"[DFTPipeline] Fermi energy: {fermi_ev:.4f} eV")

        print("[DFTPipeline] Running NSCF...")
        self.run_nscf(fermi_ev)

        print("[DFTPipeline] Running Wannier90...")
        self.run_wannier(fermi_ev)

        hr_dat = self.run_dir / f"{self.material}_hr.dat"
        if not hr_dat.exists():
            raise FileNotFoundError(f"_hr.dat not found after Wannier90 run: {hr_dat}")
        print(f"[DFTPipeline] Done. TB Hamiltonian at {hr_dat}")
        return hr_dat

    # ── helpers ───────────────────────────────────────────────────────────────

    def _input_generator(self, calculation: str = "scf") -> QEInputGenerator:
        pseudopots = self._resolved_pseudopots_map()
        nbnd = self._recommended_nbnd()
        return QEInputGenerator(
            atoms=self.atoms,
            material_name=self.material,
            pseudo_dir=self.pseudo_dir,
            outdir="./out",
            ecutwfc=80.0,
            ecutrho=640.0,
            kpoints_scf=self.kpoints_scf,
            kpoints_nscf=self.kpoints_nscf,
            pseudopots=pseudopots,
            nbnd=nbnd,
            noncolin=self.qe_noncolin,
            lspinorb=self.qe_lspinorb,
            dispersion_enabled=self.dispersion_enabled,
            dispersion_method=self.dispersion_method,
            qe_vdw_corr=self.qe_vdw_corr,
            qe_dftd3_version=self.qe_dftd3_version,
            qe_dftd3_threebody=self.qe_dftd3_threebody,
            disable_symmetry=self.qe_disable_symmetry,
        )

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
            mapped = (
                self.n_cpn_by_queue.get(queue)
                or self.n_cpn_by_queue.get(queue.lower())
            )
            if mapped is None:
                self._resolved_cores = self.n_cpn
            else:
                self._resolved_cores = int(mapped)
        return self._resolved_cores

    def _recommended_nbnd(self) -> int:
        """Heuristic nbnd scaling for slab systems (prevents 'too few bands')."""
        nat = len(self.atoms)
        # Bulk presets are too small for oxide-capped slabs; scale with atom count.
        nbnd = max(int(self.preset.num_bands), int(4 * max(1, nat)))
        counts: dict[str, int] = {}
        for sym in self.atoms.get_chemical_symbols():
            counts[sym] = counts.get(sym, 0) + 1
        pseudo_map = self._resolved_pseudopots_map()
        total_valence = 0.0
        valence_complete = True
        for sym, count in counts.items():
            fname = pseudo_map.get(sym)
            if not fname:
                valence_complete = False
                break
            z_val = self._pseudo_z_valence(fname)
            if z_val is None:
                valence_complete = False
                break
            total_valence += float(count) * float(z_val)

        if valence_complete and total_valence > 0.0:
            # In QE noncollinear+SOC mode, each band carries one electron.
            occupied_min = (
                int(math.ceil(total_valence))
                if self.qe_noncolin
                else int(math.ceil(total_valence / 2.0))
            )
            nbnd = max(nbnd, occupied_min + max(16, int(0.2 * occupied_min)))
        if nbnd % 2 != 0:
            nbnd += 1
        return nbnd

    def _pseudo_z_valence(self, filename: str) -> float | None:
        cached = self._pseudo_valence_cache.get(filename)
        if filename in self._pseudo_valence_cache:
            return cached
        target = f"{self.pseudo_dir.rstrip('/')}/{filename}"
        cmd = "bash -lc " + shlex.quote(f"grep -i -m1 'z_valence' {shlex.quote(target)}")
        rc, stdout, _ = self.jm._ssh.run(cmd, check=False)
        if rc != 0 or not stdout.strip():
            self._pseudo_valence_cache[filename] = None
            return None
        line = stdout.strip().splitlines()[0]
        m = re.search(
            r"z_valence[^0-9+.\-]*([0-9]+(?:\.[0-9]*)?(?:[Ee][+-]?[0-9]+)?)",
            line,
            flags=re.IGNORECASE,
        )
        if not m:
            self._pseudo_valence_cache[filename] = None
            return None
        try:
            val = float(m.group(1))
        except Exception:
            self._pseudo_valence_cache[filename] = None
            return None
        self._pseudo_valence_cache[filename] = val
        return val

    def _required_pseudopotential_files(self) -> list[str]:
        mapping = self._resolved_pseudopots_map()
        return list(dict.fromkeys(mapping.values()))

    def _resolved_pseudopots_map(self) -> dict[str, str]:
        if self._resolved_pseudopots is not None:
            return dict(self._resolved_pseudopots)

        symbols = list(dict.fromkeys(self.atoms.get_chemical_symbols()))
        resolved: dict[str, str] = {}
        for sym in symbols:
            preset_name = str(self.preset.pseudopots.get(sym, "")).strip()
            if self.qe_lspinorb:
                resolved[sym] = self._discover_remote_pseudo_filename(
                    sym,
                    preferred_name=(preset_name or None),
                    prefer_rel=True,
                )
            elif preset_name:
                resolved[sym] = preset_name
            else:
                resolved[sym] = self._discover_remote_pseudo_filename(sym)

        self._resolved_pseudopots = dict(resolved)
        return resolved

    def _discover_remote_pseudo_filename(
        self,
        symbol: str,
        *,
        preferred_name: str | None = None,
        prefer_rel: bool = False,
    ) -> str:
        files = self._list_remote_pseudo_files()
        if not files:
            raise RuntimeError(
                f"No .UPF files found in pseudo_dir: {self.pseudo_dir}. "
                "Run `wtec init` to prepare pseudopotentials."
            )

        sym = symbol.lower()
        preferred = (preferred_name or "").strip().lower()

        def is_symbol_match(fname: str) -> bool:
            if not fname.startswith(sym):
                return False
            if len(fname) == len(sym):
                return True
            return fname[len(sym)] in {".", "-", "_"}

        def score(name: str) -> int:
            s = name.lower()
            if not s.endswith(".upf"):
                return -1
            if not is_symbol_match(s):
                return -1
            base = 100
            if preferred and s == preferred:
                base += 30
            is_rel = ".rel-" in s
            if prefer_rel:
                base += 60 if is_rel else -80
            elif is_rel:
                base += 5
            if "pbe" in s:
                base += 10
            if "psl" in s:
                base += 6
            if "kjpaw" in s or "rrkjus" in s:
                base += 4
            return base

        ranked = sorted(
            ((score(name), name) for name in files),
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )
        best_score, best_name = ranked[0]
        if best_score < 0:
            raise RuntimeError(
                f"No matching pseudopotential found for element {symbol!r} in {self.pseudo_dir}."
            )
        return best_name

    def _list_remote_pseudo_files(self) -> list[str]:
        if self._pseudo_dir_listing is not None:
            return list(self._pseudo_dir_listing)

        # Keep this POSIX-only because remote cluster execution is Linux.
        quoted = shlex.quote(self.pseudo_dir.rstrip("/"))
        cmd = (
            "bash -lc "
            + shlex.quote(
                f"find {quoted} -maxdepth 1 -type f -iname '*.upf' -printf '%f\\n' 2>/dev/null"
            )
        )
        rc, stdout, _ = self.jm._ssh.run(cmd, check=False)
        if rc != 0 and not stdout.strip():
            self._pseudo_dir_listing = []
            return []
        files = [line.strip() for line in stdout.splitlines() if line.strip()]
        self._pseudo_dir_listing = files
        return list(files)

    def _preflight_step(
        self,
        *,
        required_commands: list[str],
        local_inputs: Iterable[Path],
    ) -> None:
        self.jm.ensure_remote_commands(
            required_commands,
            modules=self.modules,
            bin_dirs=self.bin_dirs,
        )
        mpi_targets = [
            c for c in required_commands
            if c.endswith(".x") and c not in {"qsub", "qstat", "mpirun"}
        ]
        if mpi_targets:
            self.jm.ensure_remote_mpi_binaries(
                mpi_targets,
                modules=self.modules,
                bin_dirs=self.bin_dirs,
            )
        self.jm.ensure_remote_files(
            self.pseudo_dir,
            self._required_pseudopotential_files(),
        )
        for p in local_inputs:
            if not p.exists():
                raise FileNotFoundError(f"Required local input file missing: {p}")
            if p.stat().st_size == 0:
                raise ValueError(f"Required local input file is empty: {p}")

    def _runtime_env_vars(self) -> dict[str, str]:
        env: dict[str, str] = {}
        if self.bin_dirs:
            env["PATH"] = ":".join(self.bin_dirs) + ":$PATH"
        if self.omp_threads is not None:
            val = str(self.omp_threads)
            env.update(
                {
                    "OMP_NUM_THREADS": val,
                    "MKL_NUM_THREADS": val,
                    "OPENBLAS_NUM_THREADS": val,
                    "NUMEXPR_NUM_THREADS": val,
                }
            )
        return env
