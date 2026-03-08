"""Topology deviation workflow (thickness x defect variants)."""

from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import json
import shlex
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from wtec.cluster.mpi import MPIConfig, build_command
from wtec.cluster.pbs import PBSJobConfig, generate_script
from wtec.config.cluster import ClusterConfig
from wtec.topology.arc_scan import compute_arc_connectivity
from wtec.topology.deviation import build_result
from wtec.topology.evaluator import evaluate_topology_point
from wtec.topology.variant_discovery import discover_variants
from wtec.wannier.delta_h import apply_delta_h_to_hr_file, load_delta_h_artifact
from wtec.wannier.model import WannierTBModel


def _clip01(x: float | None) -> float | None:
    if x is None:
        return None
    v = float(x)
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


class TopologyPipeline:
    """Run topology-deviation analysis for thickness x defect grid."""

    def __init__(
        self,
        hr_dat_path: str | Path,
        *,
        run_dir: str | Path,
        cfg: dict[str, Any],
        transport_results: dict | None = None,
        fermi_ev_checkpoint: float | None = None,
    ) -> None:
        self.hr_dat_path = Path(hr_dat_path).expanduser().resolve()
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.cfg = cfg
        self.transport_results = transport_results or {}
        self.fermi_ev_checkpoint = fermi_ev_checkpoint
        self.topo_dir = self.run_dir / "topology"
        self.topo_dir.mkdir(parents=True, exist_ok=True)

    def _topology_cfg(self) -> dict[str, Any]:
        user = self.cfg.get("topology", {})
        if not isinstance(user, dict):
            user = {}
        out = {
            "enabled": True,
            "backend": "qsub",
            "execution_mode": "single_batch",
            "strict_qsub": False,
            "failure_policy": "strict",
            "max_concurrent_point_jobs": 1,
            "max_concurrent_variant_dft_jobs": 1,
            "point_resource_profile": {
                "node_phase_mpi_np": None,
                "node_phase_threads": 1,
                "arc_phase_mpi_np": 1,
                "arc_phase_threads": None,
            },
            "walltime_per_point": "00:30:00",
            "w0_only": True,
            "transport_axis_primary": "x",
            "transport_axis_aux": "z",
            "n_layers_y": 4,
            "n_layers_x": 4,
            "coarse_kmesh": [20, 20, 20],
            "refine_kmesh": [5, 5, 5],
            "newton_max_iter": 50,
            "allow_partial": True,
            "score": {"w_topo": 0.70, "w_transport": 0.30},
            "variant_discovery_glob": "slab_outputs/**/*.generated.meta.json",
            "fermi_ev": None,
            "arc_engine": "siesta_slab_ldos",
            "arc_allow_proxy_fallback": False,
            "node_method": "wannierberri_flux",
            "siesta_slab_ldos_autogen": "kwant_proxy",
            "hr_scope": "per_variant",
            "caveat_reuse_global_hr_dat": False,
            "delta_h_artifact_path": None,
            "hr_grid": {
                "thickness_mapping": "middle_layer_scale",
                "middle_layer_role": "active",
                "reference_thickness_uc": None,
                "reuse_successful_points": True,
                "max_parallel_hr_points": 1,
            },
            "transport_probe": {
                "enabled": True,
                "n_ensemble": 1,
                "disorder_strength": 0.0,
                "energy_shift_ev": 0.0,
                "thickness_axis": "z",
            },
            "gap_threshold_ev": 0.05,
            "max_candidates": 64,
            "dedup_tol": 0.04,
            "tiering": {
                "mode": "single",
                "refine_top_k_per_thickness": 2,
                "always_include_pristine": True,
                "selection_metric": "S_total",
                "screen": {
                    "arc_engine": "siesta_slab_ldos",
                    "node_method": "wannierberri_flux",
                    "coarse_kmesh": [10, 10, 10],
                    "refine_kmesh": [3, 3, 3],
                    "newton_max_iter": 20,
                    "max_candidates": 24,
                },
            },
        }
        out.update(user)
        hr_grid = out.get("hr_grid", {})
        if not isinstance(hr_grid, dict):
            hr_grid = {}
        hr_default = {
            "thickness_mapping": "middle_layer_scale",
            "middle_layer_role": "active",
            "reference_thickness_uc": None,
            "reuse_successful_points": True,
            "max_parallel_hr_points": 1,
        }
        hr_default.update(hr_grid)
        out["hr_grid"] = hr_default
        tprobe = out.get("transport_probe", {})
        if not isinstance(tprobe, dict):
            tprobe = {}
        tprobe_default = {
            "enabled": True,
            "n_ensemble": 1,
            "disorder_strength": 0.0,
            "energy_shift_ev": 0.0,
            "thickness_axis": "z",
        }
        tprobe_default.update(tprobe)
        out["transport_probe"] = tprobe_default
        score = out.get("score", {})
        if not isinstance(score, dict):
            score = {}
        sc = {"w_topo": 0.70, "w_transport": 0.30}
        sc.update(score)
        out["score"] = sc
        tiering = out.get("tiering", {})
        if not isinstance(tiering, dict):
            tiering = {}
        tier_default = {
            "mode": "single",
            "refine_top_k_per_thickness": 2,
            "always_include_pristine": True,
            "selection_metric": "S_total",
            "screen": {
                "arc_engine": "siesta_slab_ldos",
                "node_method": "wannierberri_flux",
                "coarse_kmesh": [10, 10, 10],
                "refine_kmesh": [3, 3, 3],
                "newton_max_iter": 20,
                "max_candidates": 24,
            },
        }
        merged_tiering = dict(tier_default)
        merged_tiering.update(tiering)
        screen = merged_tiering.get("screen", {})
        if not isinstance(screen, dict):
            screen = {}
        screen_default = dict(tier_default["screen"])
        screen_default.update(screen)
        merged_tiering["screen"] = screen_default
        out["tiering"] = merged_tiering
        out["siesta_slab_ldos_autogen"] = (
            str(out.get("siesta_slab_ldos_autogen", "kwant_proxy")).strip().lower()
            or "kwant_proxy"
        )
        return out

    def _dft_dispersion_cfg(self) -> dict[str, Any]:
        flat = self.cfg.get("dft_dispersion", {})
        nested = self.cfg.get("dft", {}).get("dispersion") if isinstance(self.cfg.get("dft"), dict) else {}
        out: dict[str, Any] = {
            "enabled": True,
            "method": "d3",
            "qe_vdw_corr": "grimme-d3",
            "qe_dftd3_version": 4,
            "qe_dftd3_threebody": True,
            "siesta_dftd3_use_xc_defaults": True,
        }
        if isinstance(nested, dict):
            out.update(nested)
        if isinstance(flat, dict):
            out.update(flat)
        return out

    def _dft_siesta_cfg(self) -> dict[str, Any]:
        flat = self.cfg.get("dft_siesta", {})
        nested = self.cfg.get("dft", {}).get("siesta") if isinstance(self.cfg.get("dft"), dict) else {}
        out: dict[str, Any] = {}
        if isinstance(nested, dict):
            out.update(nested)
        if isinstance(flat, dict):
            out.update(flat)
        return out

    def _dft_abacus_cfg(self) -> dict[str, Any]:
        flat = self.cfg.get("dft_abacus", {})
        nested = self.cfg.get("dft", {}).get("abacus") if isinstance(self.cfg.get("dft"), dict) else {}
        out: dict[str, Any] = {}
        if isinstance(nested, dict):
            out.update(nested)
        if isinstance(flat, dict):
            out.update(flat)
        return out

    def _resolve_fermi_ev(self, topo_cfg: dict[str, Any]) -> float | None:
        if self.fermi_ev_checkpoint is not None:
            return float(self.fermi_ev_checkpoint)
        fcfg = topo_cfg.get("fermi_ev")
        if fcfg is not None:
            return float(fcfg)
        # Optional parse from local SCF output near hr_dat.
        scf_out = self.hr_dat_path.with_name(f"{self.cfg.get('material', 'material')}.scf.out")
        if scf_out.exists():
            try:
                from wtec.qe.parser import parse_fermi_energy

                return float(parse_fermi_energy(scf_out))
            except Exception:
                return None
        return None

    def _worker_source_zip(self) -> Path:
        """Create a lightweight zip of local `wtec` package for remote workers."""
        bundle = self.topo_dir / "wtec_src.zip"
        pkg_dir = Path(__file__).resolve().parents[1]  # .../wtec/wtec
        with zipfile.ZipFile(bundle, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in pkg_dir.rglob("*.py"):
                rel = p.relative_to(pkg_dir)
                zf.write(p, arcname=str(Path("wtec") / rel))
        return bundle

    def _remote_run_base(self, cluster_cfg: ClusterConfig) -> str:
        """Resolve per-run remote base directory.

        Priority:
        1) explicit `cfg["remote_workdir"]` (run-local override)
        2) cluster default root + run_name
        """
        cfg_remote = str(self.cfg.get("remote_workdir", "")).strip()
        if cfg_remote:
            return cfg_remote.rstrip("/")
        run_name = str(self.cfg.get("name", "run")).strip() or "run"
        return f"{cluster_cfg.remote_workdir.rstrip('/')}/{run_name}"

    def _thicknesses(self) -> list[int]:
        t_cfg = self.cfg.get("thicknesses")
        if isinstance(t_cfg, list) and t_cfg:
            return [int(x) for x in t_cfg]
        # fallback from transport result keys
        ts = self.transport_results.get("thickness_scan", {})
        if isinstance(ts, dict) and ts:
            first = next(iter(ts.values()))
            vals = first.get("thickness_uc")
            if vals is not None:
                return [int(x) for x in vals]
        return [4, 6, 8, 10]

    def _variants(self, topo_cfg: dict[str, Any]) -> list[dict[str, Any]]:
        variants = discover_variants(
            structure_file=self.cfg.get("structure_file"),
            run_dir=self.run_dir,
            glob_pattern=str(topo_cfg.get("variant_discovery_glob", "slab_outputs/**/*.generated.meta.json")),
        )
        # Optional explicit hr mapping by variant_id.
        hr_map = topo_cfg.get("variant_hr_map", {})
        if not isinstance(hr_map, dict):
            hr_map = {}
        for v in variants:
            mapped = hr_map.get(v["variant_id"])
            if mapped:
                v["hr_dat_path"] = str(Path(mapped).expanduser().resolve())
            else:
                v["hr_dat_path"] = str(self.hr_dat_path)
        return variants

    def _hr_scope(self, topo_cfg: dict[str, Any]) -> str:
        scope = str(topo_cfg.get("hr_scope", "per_variant")).strip().lower()
        if scope != "per_variant":
            raise ValueError(
                "topology.hr_scope must be 'per_variant'. "
                f"Unsupported value: {scope!r}"
            )
        return scope

    def _variant_dft_engine(self, topo_cfg: dict[str, Any]) -> str:
        """Resolve DFT engine used for per-variant HR generation."""
        mode_raw = self.cfg.get("dft_mode")
        if mode_raw is None and isinstance(self.cfg.get("dft"), dict):
            mode_raw = self.cfg["dft"].get("mode")
        mode = str(mode_raw or "legacy_single").strip().lower() or "legacy_single"
        if mode == "dual_family":
            raw = self.cfg.get("dft_lcao_engine")
            if raw is None and isinstance(self.cfg.get("dft"), dict):
                tracks = self.cfg["dft"].get("tracks")
                if isinstance(tracks, dict):
                    lcao = tracks.get("lcao_upscaled")
                    if isinstance(lcao, dict):
                        raw = lcao.get("engine")
        else:
            raw = topo_cfg.get("variant_dft_engine")
            if raw is None:
                raw = self.cfg.get("topology_variant_dft_engine")
            if raw is None and mode == "hybrid_qe_ref_siesta_variants":
                raw = "siesta"

        if raw is None:
            raw = self.cfg.get("dft_engine")
        if raw is None and isinstance(self.cfg.get("dft"), dict):
            raw = self.cfg["dft"].get("engine")

        engine = str(raw or "siesta").strip().lower() or "siesta"
        if engine not in {"qe", "siesta", "abacus"}:
            raise ValueError(
                f"Unsupported variant_dft_engine={engine!r}. Use 'qe', 'siesta', or 'abacus'."
            )
        return engine

    def _point_manifest_rows(
        self,
        *,
        variants: list[dict[str, Any]],
        thicknesses: list[int],
        topo_cfg: dict[str, Any],
        fermi_ev: float,
    ) -> list[dict[str, Any]]:
        scope = self._hr_scope(topo_cfg)
        rows: list[dict[str, Any]] = []
        point_idx = 0
        hr_grid = topo_cfg.get("hr_grid", {}) if isinstance(topo_cfg.get("hr_grid"), dict) else {}
        ref_thickness = hr_grid.get("reference_thickness_uc")
        if ref_thickness is None:
            ref_thickness = min(thicknesses) if thicknesses else 1
        ref_thickness = int(ref_thickness)
        mapping = str(hr_grid.get("thickness_mapping", "middle_layer_scale")).strip().lower()
        middle_role = str(hr_grid.get("middle_layer_role", "middle")).strip()
        reuse_global_hr = bool(topo_cfg.get("caveat_reuse_global_hr_dat", False))

        for v in variants:
            for d in thicknesses:
                point_name = f"point_{point_idx:03d}"
                point_dir = self.topo_dir / "hr_points" / point_name
                row = {
                    "point_index": point_idx,
                    "point_name": point_name,
                    "variant_id": v.get("variant_id"),
                    "defect_severity": float(v.get("defect_severity", 0.0)),
                    "is_pristine": bool(v.get("is_pristine", False)),
                    "thickness_uc": int(d),
                    "metadata_path": v.get("metadata_path"),
                    "variant_cif_path": v.get("cif_path"),
                    "local_point_dir": str(point_dir.resolve()),
                    "structure_path": None,
                    "siesta_slab_ldos_json": str((point_dir / "siesta_slab_ldos.json").resolve()),
                    "status": "pending",
                    "reason": None,
                    "fermi_ev": float(fermi_ev),
                    "hr_dat_path": str(v.get("hr_dat_path", self.hr_dat_path)),
                    "win_path": str(v.get("win_path")) if v.get("win_path") else None,
                    "hr_scope": scope,
                }
                if scope == "per_variant" and not reuse_global_hr:
                    row["status"] = "pending_variant"
                rows.append(row)
                point_idx += 1
        return rows

    def _run_hr_generation(
        self,
        *,
        rows: list[dict[str, Any]],
        topo_cfg: dict[str, Any],
        thicknesses: list[int],
    ) -> list[dict[str, Any]]:
        scope = self._hr_scope(topo_cfg)
        from wtec.cluster.ssh import open_ssh
        from wtec.cluster.submit import JobManager
        from wtec.structure.io import read as read_structure

        dft_engine = self._variant_dft_engine(topo_cfg)
        if dft_engine == "qe":
            from wtec.qe.parser import parse_fermi_energy
            from wtec.workflow.dft_pipeline import DFTPipeline as PipelineClass
        elif dft_engine == "siesta":
            from wtec.siesta.parser import parse_fermi_energy
            from wtec.siesta.runner import SiestaPipeline as PipelineClass
        else:
            from wtec.abacus.parser import parse_fermi_energy
            from wtec.abacus.runner import AbacusPipeline as PipelineClass

        cluster_cfg = ClusterConfig.from_env()
        run_name = str(self.cfg.get("name", "run")).strip() or "run"
        hr_grid = topo_cfg.get("hr_grid", {}) if isinstance(topo_cfg.get("hr_grid"), dict) else {}
        reuse_ok = bool(hr_grid.get("reuse_successful_points", True))
        strict_fail = str(topo_cfg.get("failure_policy", "strict")).strip().lower() == "strict"
        max_variant_jobs = max(1, int(topo_cfg.get("max_concurrent_variant_dft_jobs", 1)))
        min_th = min(thicknesses) if thicknesses else 1
        reuse_global_hr = bool(topo_cfg.get("caveat_reuse_global_hr_dat", False))

        if reuse_global_hr:
            global_hr = str(self.hr_dat_path.resolve())
            global_win_path = self.hr_dat_path.with_name(f"{self.cfg.get('material', 'material')}.win")
            global_win = str(global_win_path.resolve()) if global_win_path.exists() else None
            for row in rows:
                if str(row.get("status")) == "failed":
                    continue
                row["status"] = "ready"
                row["reason"] = "caveat_reuse_global_hr_dat"
                row["hr_dat_path"] = global_hr
                row["win_path"] = global_win
            return rows

        rows_to_run: list[dict[str, Any]] = []
        for row in rows:
            if str(row.get("status")) == "failed":
                continue
            if scope == "per_variant" and str(row.get("status")) == "pending_variant":
                if int(row.get("thickness_uc", 1)) != int(min_th):
                    continue
                cif_path = row.get("variant_cif_path") or self.cfg.get("structure_file")
                if not cif_path:
                    row["status"] = "failed"
                    row["reason"] = "missing_variant_cif"
                    continue
                point_dir = Path(str(row["local_point_dir"])).resolve()
                point_dir.mkdir(parents=True, exist_ok=True)
                structure_out = point_dir / "structure.generated.cif"
                if not structure_out.exists():
                    from wtec.structure.io import write as write_structure

                    atoms_src = read_structure(cif_path)
                    write_structure(atoms_src, structure_out, fmt="cif")
                row["structure_path"] = str(structure_out.resolve())
                row["status"] = "pending"
            if str(row.get("status")) == "pending":
                rows_to_run.append(row)

        def _run_single(row: dict[str, Any]) -> dict[str, Any]:
            structure_path = row.get("structure_path")
            if not structure_path:
                return {
                    "status": "failed",
                    "reason": "missing_point_structure",
                }
            point_dir = Path(str(row["local_point_dir"])).resolve()
            dft_dir = point_dir / "dft"
            dft_dir.mkdir(parents=True, exist_ok=True)
            hr_path = dft_dir / f"{self.cfg['material']}_hr.dat"
            win_path = dft_dir / f"{self.cfg['material']}.win"
            scf_out = dft_dir / f"{self.cfg['material']}.scf.out"
            if reuse_ok and hr_path.exists() and win_path.exists() and scf_out.exists():
                try:
                    fe = float(parse_fermi_energy(scf_out))
                except Exception:
                    fe = float(row.get("fermi_ev", 0.0))
                return {
                    "status": "ok",
                    "hr_dat_path": str(hr_path.resolve()),
                    "win_path": str(win_path.resolve()),
                    "fermi_ev": fe,
                    "hr_reused": True,
                }

            remote_base = (
                f"{self._remote_run_base(cluster_cfg)}/topology/hr_points/"
                f"{run_name}/{row['point_name']}"
            )
            atoms = read_structure(structure_path)
            common_kwargs = {
                "run_dir": dft_dir,
                "remote_base": remote_base,
                "n_nodes": int(self.cfg.get("n_nodes", 1)),
                "n_cores_per_node": cluster_cfg.mpi_cores,
                "n_cores_by_queue": cluster_cfg.mpi_cores_by_queue,
                "queue": cluster_cfg.pbs_queue,
                "queue_priority": cluster_cfg.pbs_queue_priority,
                "kpoints_scf": tuple(self.cfg.get("kpoints_scf", (8, 8, 8))),
                "kpoints_nscf": tuple(self.cfg.get("kpoints_nscf", (12, 12, 12))),
                "omp_threads": cluster_cfg.omp_threads,
                "modules": cluster_cfg.modules,
                "bin_dirs": cluster_cfg.bin_dirs,
                "live_log": self.cfg.get("_runtime_live_log", True),
                "log_poll_interval": self.cfg.get("_runtime_log_poll_interval", 5),
                "stale_log_seconds": self.cfg.get("_runtime_stale_log_seconds", 300),
            }
            with open_ssh(cluster_cfg) as ssh:
                jm = JobManager(ssh)
                if dft_engine == "qe":
                    pipeline = PipelineClass(
                        atoms,
                        self.cfg["material"],
                        jm,
                        pseudo_dir=cluster_cfg.qe_pseudo_dir,
                        qe_noncolin=self.cfg.get("qe_noncolin", True),
                        qe_lspinorb=self.cfg.get("qe_lspinorb", True),
                        qe_disable_symmetry=self.cfg.get("qe_disable_symmetry", False),
                        dispersion_cfg=self._dft_dispersion_cfg(),
                        **common_kwargs,
                    )
                elif dft_engine == "siesta":
                    siesta_cfg = self._dft_siesta_cfg()
                    pipeline = PipelineClass(
                        atoms,
                        self.cfg["material"],
                        jm,
                        pseudo_dir=str(siesta_cfg.get("pseudo_dir") or cluster_cfg.siesta_pseudo_dir),
                        basis_profile=str(siesta_cfg.get("basis_profile", "")).strip(),
                        wannier_interface=str(siesta_cfg.get("wannier_interface", "sisl")).strip().lower(),
                        spin_orbit=bool(siesta_cfg.get("spin_orbit", True)),
                        include_pao_basis=bool(siesta_cfg.get("include_pao_basis", True)),
                        mpi_np_scf=int(siesta_cfg.get("mpi_np_scf", 0)),
                        mpi_np_nscf=int(siesta_cfg.get("mpi_np_nscf", 0)),
                        mpi_np_wannier=int(siesta_cfg.get("mpi_np_wannier", 0)),
                        omp_threads_scf=int(siesta_cfg.get("omp_threads_scf", 0)),
                        omp_threads_nscf=int(siesta_cfg.get("omp_threads_nscf", 0)),
                        omp_threads_wannier=int(siesta_cfg.get("omp_threads_wannier", 0)),
                        factorization_defaults=(
                            siesta_cfg.get("factorization_defaults", {})
                            if isinstance(siesta_cfg.get("factorization_defaults"), dict)
                            else {}
                        ),
                        dm_mixing_weight=float(siesta_cfg.get("dm_mixing_weight", 0.10)),
                        dm_number_pulay=int(siesta_cfg.get("dm_number_pulay", 8)),
                        electronic_temperature_k=float(siesta_cfg.get("electronic_temperature_k", 300.0)),
                        max_scf_iterations=int(siesta_cfg.get("max_scf_iterations", 200)),
                        dispersion_cfg=self._dft_dispersion_cfg(),
                        **common_kwargs,
                    )
                else:
                    abacus_cfg = self._dft_abacus_cfg()
                    pipeline = PipelineClass(
                        atoms,
                        self.cfg["material"],
                        jm,
                        pseudo_dir=str(abacus_cfg.get("pseudo_dir") or cluster_cfg.abacus_pseudo_dir),
                        orbital_dir=str(
                            abacus_cfg.get("orbital_dir") or cluster_cfg.abacus_orbital_dir
                        ),
                        basis_type=str(abacus_cfg.get("basis_type", "lcao")).strip().lower() or "lcao",
                        ks_solver=str(abacus_cfg.get("ks_solver", "genelpa")).strip().lower() or "genelpa",
                        executable=str(abacus_cfg.get("executable", "abacus")).strip() or "abacus",
                        **common_kwargs,
                    )
                scf_meta = pipeline.run_scf()
                fe = float(parse_fermi_energy(scf_out))
                nscf_meta = pipeline.run_nscf(fe)
                wan_meta = pipeline.run_wannier(fe)
            return {
                "status": "ok",
                "hr_dat_path": str(hr_path.resolve()),
                "win_path": str(win_path.resolve()),
                "fermi_ev": fe,
                "hr_reused": False,
                "dft_jobs": {
                    "scf": scf_meta.get("job_id"),
                    "nscf": nscf_meta.get("job_id"),
                    "wannier": wan_meta.get("job_id"),
                },
            }

        failed_any = False
        if max_variant_jobs == 1:
            for row in rows_to_run:
                try:
                    result = _run_single(row)
                except Exception as exc:
                    result = {"status": "failed", "reason": f"hr_generation_failed:{type(exc).__name__}:{exc}"}
                row.update(result)
                if str(row.get("status")) == "failed":
                    failed_any = True
                    if strict_fail:
                        break
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_variant_jobs) as ex:
                fut_to_row = {ex.submit(_run_single, row): row for row in rows_to_run}
                for fut in concurrent.futures.as_completed(fut_to_row):
                    row = fut_to_row[fut]
                    try:
                        result = fut.result()
                    except Exception as exc:
                        result = {"status": "failed", "reason": f"hr_generation_failed:{type(exc).__name__}:{exc}"}
                    row.update(result)
                    if str(row.get("status")) == "failed":
                        failed_any = True

        if strict_fail and failed_any:
            for row in rows:
                if str(row.get("status")) in {"pending", "pending_variant"}:
                    row["status"] = "failed"
                    row["reason"] = "aborted_due_to_previous_variant_dft_failure"
        return rows

    @staticmethod
    def _propagate_variant_hr_rows(
        rows: list[dict[str, Any]],
        *,
        thicknesses: list[int],
    ) -> list[dict[str, Any]]:
        """For per-variant scope, copy reference-thickness HR outputs across thicknesses."""
        if not rows:
            return rows
        ref_th = min(int(t) for t in thicknesses) if thicknesses else None
        by_variant: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            vid = str(row.get("variant_id", "unknown"))
            by_variant.setdefault(vid, []).append(row)

        for _, group in by_variant.items():
            src: dict[str, Any] | None = None
            for row in sorted(group, key=lambda r: int(r.get("thickness_uc", 10**9))):
                if row.get("status") != "ok":
                    continue
                if ref_th is None or int(row.get("thickness_uc", 0)) == int(ref_th):
                    src = row
                    break
            if src is None:
                for row in group:
                    if row.get("status") == "ok":
                        src = row
                        break

            for row in group:
                if row.get("status") not in {"pending_variant", "pending"}:
                    continue
                if src is None:
                    row["status"] = "failed"
                    row["reason"] = "variant_reference_hr_missing"
                    continue
                row["status"] = "ok"
                row["hr_dat_path"] = src.get("hr_dat_path")
                row["win_path"] = src.get("win_path")
                row["fermi_ev"] = src.get("fermi_ev")
                row["hr_reused"] = True
                row["hr_source_point"] = src.get("point_name")
        return rows

    def _apply_delta_h_rows(
        self,
        *,
        rows: list[dict[str, Any]],
        topo_cfg: dict[str, Any],
    ) -> list[dict[str, Any]]:
        art_raw = topo_cfg.get("delta_h_artifact_path")
        if not art_raw:
            return rows
        artifact_path = Path(str(art_raw)).expanduser().resolve()
        if not artifact_path.exists():
            raise FileNotFoundError(f"delta_h_artifact_path not found: {artifact_path}")
        artifact = load_delta_h_artifact(artifact_path)
        strict_fail = str(topo_cfg.get("failure_policy", "strict")).strip().lower() == "strict"
        material = str(self.cfg.get("material", "material")).strip() or "material"
        mapped: dict[tuple[str, str], str] = {}
        failed_any = False

        for row in rows:
            if str(row.get("status", "")).lower() == "failed":
                continue
            hr_raw = row.get("hr_dat_path")
            if not hr_raw:
                row["status"] = "failed"
                row["reason"] = "missing_hr_dat_for_delta_h"
                failed_any = True
                continue
            hr_path = Path(str(hr_raw)).expanduser().resolve()
            if not hr_path.exists():
                row["status"] = "failed"
                row["reason"] = f"hr_dat_not_found_for_delta_h:{hr_path}"
                failed_any = True
                continue
            wp_raw = row.get("win_path")
            win_path = None
            if wp_raw:
                wp = Path(str(wp_raw)).expanduser().resolve()
                if wp.exists():
                    win_path = wp
            key = (str(hr_path), str(win_path) if win_path is not None else "")
            if key in mapped:
                row["hr_dat_path"] = mapped[key]
                row["hr_delta_h_applied"] = True
                row["delta_h_artifact_path"] = str(artifact_path)
                continue
            point_dir = Path(str(row.get("local_point_dir", self.topo_dir))).expanduser().resolve()
            out_dir = point_dir / "dft"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_hr = out_dir / f"{material}_hr.delta_h.dat"
            try:
                apply_delta_h_to_hr_file(
                    hr_dat_path=hr_path,
                    output_hr_dat_path=out_hr,
                    artifact=artifact,
                    win_path=win_path,
                )
            except Exception as exc:
                row["status"] = "failed"
                row["reason"] = f"delta_h_apply_failed:{type(exc).__name__}:{exc}"
                failed_any = True
                continue
            mapped[key] = str(out_hr.resolve())
            row["hr_dat_path"] = mapped[key]
            row["hr_delta_h_applied"] = True
            row["delta_h_artifact_path"] = str(artifact_path)

        if strict_fail and failed_any:
            for row in rows:
                if str(row.get("status", "")).lower() in {"pending", "pending_variant"}:
                    row["status"] = "failed"
                    row["reason"] = "aborted_due_to_delta_h_failure"
        return rows

    def _materialize_siesta_slab_ldos_rows(
        self,
        *,
        rows: list[dict[str, Any]],
        topo_cfg: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Ensure per-point slab-LDOS payloads exist for siesta_slab_ldos arc mode."""
        arc_engine = str(topo_cfg.get("arc_engine", "siesta_slab_ldos")).strip().lower()
        if arc_engine != "siesta_slab_ldos":
            return rows

        autogen_mode = str(
            topo_cfg.get("siesta_slab_ldos_autogen", "kwant_proxy")
        ).strip().lower()
        disabled_modes = {"none", "off", "disabled", "false", "no"}
        energy_shift = float(topo_cfg.get("energy_shift_ev", 0.0))
        model_cache: dict[str, WannierTBModel] = {}

        for row in rows:
            if str(row.get("status", "")).lower() == "failed":
                continue
            ldos_raw = row.get("siesta_slab_ldos_json")
            if not ldos_raw:
                row["status"] = "failed"
                row["reason"] = "missing_siesta_slab_ldos_json_path"
                continue
            ldos_path = Path(str(ldos_raw)).expanduser().resolve()

            if ldos_path.exists():
                try:
                    payload = json.loads(ldos_path.read_text())
                except Exception:
                    payload = {}
                if isinstance(payload, dict) and (
                    "metric" in payload or "surface_fraction" in payload
                ):
                    row["siesta_slab_ldos_source_engine"] = str(
                        payload.get("source_engine", "siesta_slab_ldos")
                    )
                    row["siesta_slab_ldos_source_kind"] = str(
                        payload.get("source_kind", "provided")
                    )
                    continue

            if autogen_mode in disabled_modes:
                row["status"] = "failed"
                row["reason"] = f"missing_siesta_slab_ldos_json:{ldos_path}"
                continue

            hr_raw = row.get("hr_dat_path")
            if not hr_raw:
                row["status"] = "failed"
                row["reason"] = "missing_hr_dat_for_arc_ldos_autogen"
                continue
            hr_path = Path(str(hr_raw)).expanduser().resolve()
            if not hr_path.exists():
                row["status"] = "failed"
                row["reason"] = f"hr_dat_not_found_for_arc_ldos_autogen:{hr_path}"
                continue

            win_path = None
            wp_raw = row.get("win_path")
            if wp_raw:
                wp = Path(str(wp_raw)).expanduser().resolve()
                if wp.exists():
                    win_path = wp

            cache_key = f"{hr_path}::{win_path if win_path else ''}"
            model = model_cache.get(cache_key)
            if model is None:
                model = WannierTBModel.from_hr_dat(
                    str(hr_path),
                    win_path=str(win_path) if win_path else None,
                )
                model_cache[cache_key] = model

            try:
                arc = compute_arc_connectivity(
                    model,
                    thickness_uc=int(row.get("thickness_uc", 1)),
                    energy_ev=float(row.get("fermi_ev", 0.0)) + energy_shift,
                    n_layers_x=int(topo_cfg.get("n_layers_x", 4)),
                    n_layers_y=int(topo_cfg.get("n_layers_y", 4)),
                    lead_axis=str(topo_cfg.get("transport_axis_primary", "x")),
                    prefer_engine="kwant",
                    hr_dat_path=str(hr_path),
                    allow_proxy_fallback=True,
                )
            except Exception as exc:
                row["status"] = "failed"
                row["reason"] = f"siesta_slab_ldos_autogen_failed:{type(exc).__name__}:{exc}"
                continue

            if str(arc.get("status", "")).lower() != "ok":
                row["status"] = "failed"
                row["reason"] = (
                    "siesta_slab_ldos_autogen_metric_failed:"
                    + str(arc.get("reason", "unknown"))
                )
                continue

            metric = float(arc.get("metric", arc.get("surface_fraction", 0.0)))
            payload = {
                "metric": float(np.clip(metric, 0.0, 1.0)),
                "surface_fraction": float(np.clip(metric, 0.0, 1.0)),
                "source_engine": str(arc.get("engine", "kwant_ldos_surface_proxy")),
                "source_kind": "autogenerated_kwant_proxy",
                "generated_by": "wtec.topology_pipeline",
                "point_name": str(row.get("point_name", "")),
                "thickness_uc": int(row.get("thickness_uc", 1)),
                "fermi_ev": float(row.get("fermi_ev", 0.0)),
            }
            ldos_path.parent.mkdir(parents=True, exist_ok=True)
            ldos_path.write_text(json.dumps(payload, indent=2))
            row["siesta_slab_ldos_source_engine"] = str(payload.get("source_engine"))
            row["siesta_slab_ldos_source_kind"] = str(payload.get("source_kind"))
        return rows

    def _tasks_from_point_rows(
        self,
        *,
        point_rows: list[dict[str, Any]],
        topo_cfg: dict[str, Any],
        fermi_ev_default: float,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Convert point-manifest rows into topology tasks and pre-failed raw rows."""
        tasks: list[dict[str, Any]] = []
        prefail_raw: list[dict[str, Any]] = []

        for row in point_rows:
            status = str(row.get("status", "pending"))
            try:
                point_index = int(row.get("point_index", -1))
            except Exception:
                point_index = -1
            point_name = str(row.get("point_name", f"point_{point_index:03d}"))
            vid = str(row.get("variant_id", "unknown"))
            sev = float(row.get("defect_severity", 0.0))
            try:
                thickness_uc = int(row.get("thickness_uc", -1))
            except Exception:
                thickness_uc = -1
            is_pristine = bool(row.get("is_pristine", False))

            if status in {"failed", "skipped"}:
                prefail_raw.append(
                    {
                        "status": "failed",
                        "reason": str(row.get("reason") or "point_manifest_failed"),
                        "point_index": point_index,
                        "point_name": point_name,
                        "variant_id": vid,
                        "thickness_uc": thickness_uc,
                        "defect_severity": sev,
                        "is_pristine": is_pristine,
                    }
                )
                continue

            hr_raw = row.get("hr_dat_path")
            if not hr_raw:
                prefail_raw.append(
                    {
                        "status": "failed",
                        "reason": "missing_hr_dat_path",
                        "point_index": point_index,
                        "point_name": point_name,
                        "variant_id": vid,
                        "thickness_uc": thickness_uc,
                        "defect_severity": sev,
                        "is_pristine": is_pristine,
                    }
                )
                continue
            hr_path = Path(str(hr_raw)).expanduser().resolve()
            if not hr_path.exists():
                prefail_raw.append(
                    {
                        "status": "failed",
                        "reason": f"hr_dat_not_found:{hr_path}",
                        "point_index": point_index,
                        "point_name": point_name,
                        "variant_id": vid,
                        "thickness_uc": thickness_uc,
                        "defect_severity": sev,
                        "is_pristine": is_pristine,
                    }
                )
                continue

            win_path_out: str | None = None
            wp_raw = row.get("win_path")
            if wp_raw:
                wp = Path(str(wp_raw)).expanduser().resolve()
                if wp.exists():
                    win_path_out = str(wp)

            tasks.append(
                {
                    "point_index": point_index,
                    "point_name": point_name,
                    "variant_id": vid,
                    "defect_severity": sev,
                    "is_pristine": is_pristine,
                    "thickness_uc": thickness_uc,
                    "hr_dat_path": str(hr_path),
                    "win_path": win_path_out,
                    "fermi_ev": float(row.get("fermi_ev", fermi_ev_default)),
                    "n_layers_x": int(topo_cfg.get("n_layers_x", 4)),
                    "n_layers_y": int(topo_cfg.get("n_layers_y", 4)),
                    "lead_axis": str(topo_cfg.get("transport_axis_primary", "x")),
                    "coarse_kmesh": topo_cfg.get("coarse_kmesh", [20, 20, 20]),
                    "refine_kmesh": topo_cfg.get("refine_kmesh", [5, 5, 5]),
                    "newton_max_iter": int(topo_cfg.get("newton_max_iter", 50)),
                    "gap_threshold_ev": float(topo_cfg.get("gap_threshold_ev", 0.05)),
                    "max_candidates": int(topo_cfg.get("max_candidates", 64)),
                    "dedup_tol": float(topo_cfg.get("dedup_tol", 0.04)),
                    "arc_engine": str(topo_cfg.get("arc_engine", "siesta_slab_ldos")),
                    "node_method": str(topo_cfg.get("node_method", "wannierberri_flux")),
                    "reference_bands_path": topo_cfg.get("reference_bands_path"),
                    "reference_tol_ev": float(topo_cfg.get("reference_tol_ev", 0.20)),
                    "metadata_path": row.get("metadata_path"),
                    "variant_cif_path": row.get("variant_cif_path"),
                    "local_point_dir": row.get("local_point_dir"),
                    "siesta_slab_ldos_json": row.get("siesta_slab_ldos_json"),
                    "siesta_slab_ldos_source_engine": row.get("siesta_slab_ldos_source_engine"),
                    "siesta_slab_ldos_source_kind": row.get("siesta_slab_ldos_source_kind"),
                    "arc_allow_proxy_fallback": bool(topo_cfg.get("arc_allow_proxy_fallback", False)),
                    "transport_probe_enabled": bool(
                        (topo_cfg.get("transport_probe", {}) or {}).get("enabled", True)
                    ),
                    "transport_probe_n_ensemble": int(
                        (topo_cfg.get("transport_probe", {}) or {}).get("n_ensemble", 1)
                    ),
                    "transport_probe_disorder_strength": float(
                        (topo_cfg.get("transport_probe", {}) or {}).get("disorder_strength", 0.0)
                    ),
                    "transport_probe_energy_shift_ev": float(
                        (topo_cfg.get("transport_probe", {}) or {}).get("energy_shift_ev", 0.0)
                    ),
                    "transport_thickness_axis": str(
                        (topo_cfg.get("transport_probe", {}) or {}).get(
                            "thickness_axis",
                            topo_cfg.get("transport_axis_aux", "z"),
                        )
                    ),
                }
            )
        return tasks, prefail_raw

    def _run_local(self, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cache = {}
        out: list[dict[str, Any]] = []
        for t in tasks:
            try:
                out.append(evaluate_topology_point(t, cache=cache))
            except Exception as exc:
                out.append(
                    {
                        "status": "failed",
                        "point_index": t.get("point_index"),
                        "point_name": t.get("point_name"),
                        "variant_id": t.get("variant_id"),
                        "thickness_uc": t.get("thickness_uc"),
                        "defect_severity": t.get("defect_severity", 0.0),
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
        return out

    def _execute_tasks(
        self,
        *,
        tasks: list[dict[str, Any]],
        topo_cfg: dict[str, Any],
        prefail_raw: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], str]:
        backend = str(topo_cfg.get("backend", "qsub")).lower().strip()
        exec_mode = str(topo_cfg.get("execution_mode", "single_batch")).lower().strip()
        strict_qsub = bool(topo_cfg.get("strict_qsub", False))
        raw: list[dict[str, Any]] = list(prefail_raw)
        backend_used = backend
        if not tasks:
            return raw, "none_no_valid_points"
        if backend == "qsub":
            try:
                eval_raw = self._run_qsub(tasks, topo_cfg)
                raw.extend(eval_raw)
                backend_used = (
                    "qsub_per_point"
                    if exec_mode in {"per_point", "per_point_qsub", "point"}
                    else "qsub_batch"
                )
            except Exception as exc:
                if strict_qsub:
                    raise
                eval_raw = self._run_local(tasks)
                raw.extend(eval_raw)
                backend_used = f"local_fallback:{type(exc).__name__}"
            return raw, backend_used
        raw.extend(self._run_local(tasks))
        return raw, "local"

    @staticmethod
    def _tiering_cfg(topo_cfg: dict[str, Any]) -> dict[str, Any]:
        raw = topo_cfg.get("tiering", {})
        if not isinstance(raw, dict):
            raw = {}
        out = {
            "mode": str(raw.get("mode", "single")).strip().lower() or "single",
            "refine_top_k_per_thickness": int(raw.get("refine_top_k_per_thickness", 2)),
            "always_include_pristine": bool(raw.get("always_include_pristine", True)),
            "selection_metric": str(raw.get("selection_metric", "S_total")).strip() or "S_total",
            "screen": {},
        }
        screen = raw.get("screen", {})
        if not isinstance(screen, dict):
            screen = {}
        out["screen"] = {
            "arc_engine": str(screen.get("arc_engine", "siesta_slab_ldos")).strip().lower() or "siesta_slab_ldos",
            "node_method": str(screen.get("node_method", "wannierberri_flux")).strip().lower() or "wannierberri_flux",
            "coarse_kmesh": screen.get("coarse_kmesh", [10, 10, 10]),
            "refine_kmesh": screen.get("refine_kmesh", [3, 3, 3]),
            "newton_max_iter": int(screen.get("newton_max_iter", 20)),
            "max_candidates": int(screen.get("max_candidates", 24)),
        }
        return out

    @staticmethod
    def _apply_screen_overrides(
        tasks: list[dict[str, Any]],
        *,
        screen_cfg: dict[str, Any],
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for task in tasks:
            t = dict(task)
            if "arc_engine" in screen_cfg:
                t["arc_engine"] = screen_cfg["arc_engine"]
            if "node_method" in screen_cfg:
                t["node_method"] = screen_cfg["node_method"]
            if "coarse_kmesh" in screen_cfg:
                t["coarse_kmesh"] = screen_cfg["coarse_kmesh"]
            if "refine_kmesh" in screen_cfg:
                t["refine_kmesh"] = screen_cfg["refine_kmesh"]
            if "newton_max_iter" in screen_cfg:
                t["newton_max_iter"] = int(screen_cfg["newton_max_iter"])
            if "max_candidates" in screen_cfg:
                t["max_candidates"] = int(screen_cfg["max_candidates"])
            out.append(t)
        return out

    def _select_refine_point_indices(
        self,
        *,
        scored_rows: list[dict[str, Any]],
        tiering_cfg: dict[str, Any],
    ) -> set[int]:
        selection_metric = str(tiering_cfg.get("selection_metric", "S_total")).strip() or "S_total"
        top_k = max(1, int(tiering_cfg.get("refine_top_k_per_thickness", 2)))
        always_include_pristine = bool(tiering_cfg.get("always_include_pristine", True))

        selected: set[int] = set()
        by_thickness: dict[int, list[dict[str, Any]]] = {}
        for row in scored_rows:
            try:
                d = int(row.get("thickness_uc", -1))
            except Exception:
                continue
            by_thickness.setdefault(d, []).append(row)

        for rows in by_thickness.values():
            candidates = [
                r
                for r in rows
                if not bool(r.get("is_pristine", False))
                and r.get(selection_metric) is not None
                and str(r.get("status", "")).lower() in {"ok", "partial"}
            ]
            candidates.sort(key=lambda r: float(r.get(selection_metric, -1.0)), reverse=True)
            for row in candidates[:top_k]:
                try:
                    selected.add(int(row.get("point_index")))
                except Exception:
                    continue
            if always_include_pristine:
                for row in rows:
                    if not bool(row.get("is_pristine", False)):
                        continue
                    try:
                        selected.add(int(row.get("point_index")))
                    except Exception:
                        continue
        return selected

    @staticmethod
    def _thread_exports(command: str, *, threads: int) -> str:
        t = max(1, int(threads))
        exports = (
            f"export OMP_NUM_THREADS={t}; "
            f"export MKL_NUM_THREADS={t}; "
            f"export OPENBLAS_NUM_THREADS={t}; "
            f"export NUMEXPR_NUM_THREADS={t}; "
        )
        return exports + command

    @staticmethod
    def _point_resource_profile(
        topo_cfg: dict[str, Any],
        *,
        total_cores: int,
    ) -> dict[str, int]:
        prof = topo_cfg.get("point_resource_profile", {})
        if not isinstance(prof, dict):
            prof = {}
        node_np = prof.get("node_phase_mpi_np", total_cores)
        if node_np is None:
            node_np = total_cores
        arc_threads = prof.get("arc_phase_threads", total_cores)
        if arc_threads is None:
            arc_threads = total_cores
        out = {
            "node_phase_mpi_np": max(1, int(node_np)),
            "node_phase_threads": max(1, int(prof.get("node_phase_threads", 1))),
            "arc_phase_mpi_np": max(1, int(prof.get("arc_phase_mpi_np", 1))),
            "arc_phase_threads": max(1, int(arc_threads)),
        }
        return out

    def _run_qsub(self, tasks: list[dict[str, Any]], topo_cfg: dict[str, Any]) -> list[dict[str, Any]]:
        mode = str(topo_cfg.get("execution_mode", "single_batch")).strip().lower()
        if mode in {"single_batch", "batch", "qsub_batch"}:
            return self._run_qsub_batch(tasks, topo_cfg)
        if mode in {"per_point", "per_point_qsub", "point"}:
            return self._run_qsub_per_point(tasks, topo_cfg)
        raise ValueError(f"Unknown topology.execution_mode: {mode!r}")

    def _run_qsub_batch(self, tasks: list[dict[str, Any]], topo_cfg: dict[str, Any]) -> list[dict[str, Any]]:
        from wtec.cluster.ssh import open_ssh
        from wtec.cluster.submit import JobManager

        cluster_cfg = ClusterConfig.from_env()
        payload_local = self.topo_dir / "topology_payload.json"
        stage_dir = self.topo_dir / "batch_stage_inputs"
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        stage_dir.mkdir(parents=True, exist_ok=True)
        worker_zip = self._worker_source_zip()
        result_name = "topology_worker_result.json"
        payload_remote_name = payload_local.name

        local_files: list[Path] = [payload_local, worker_zip]
        staged_name_by_src: dict[str, str] = {}

        def _stage_copy(src: str | Path, *, prefix: str) -> str:
            p = Path(src).expanduser().resolve()
            key = str(p)
            if key in staged_name_by_src:
                return staged_name_by_src[key]
            digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
            staged_name = f"{prefix}_{len(staged_name_by_src):03d}_{digest}_{p.name}"
            dst = stage_dir / staged_name
            if not dst.exists() or dst.stat().st_mtime < p.stat().st_mtime:
                shutil.copy2(p, dst)
            local_files.append(dst)
            staged_name_by_src[key] = staged_name
            return staged_name

        remote_tasks = []
        for t in tasks:
            c = dict(t)
            c["hr_dat_path"] = _stage_copy(t["hr_dat_path"], prefix="hr")
            if t.get("win_path"):
                c["win_path"] = _stage_copy(t["win_path"], prefix="win")
            remote_tasks.append(c)
        payload_local.write_text(json.dumps({"tasks": remote_tasks}, indent=2))

        run_name = str(self.cfg.get("name", "run")).strip() or "run"
        remote_dir = f"{self._remote_run_base(cluster_cfg)}/topology"

        with open_ssh(cluster_cfg) as ssh:
            jm = JobManager(ssh)
            queue_used = jm.resolve_queue(
                cluster_cfg.pbs_queue,
                fallback_order=cluster_cfg.pbs_queue_priority,
            )
            cores_per_node = cluster_cfg.cores_for_queue(queue_used)
            n_nodes = int(self.cfg.get("n_nodes", 1))
            total_cores = max(1, n_nodes * cores_per_node)

            python_exe = str(topo_cfg.get("cluster_python_exe", "python3"))
            worker_python = (
                f"env PYTHONPATH=$PWD/{worker_zip.name}:$PYTHONPATH {python_exe}"
            )
            mpi = MPIConfig(n_cores=total_cores, n_pool=1)
            cmd = build_command(
                worker_python,
                mpi=mpi,
                extra_args=f"-m wtec.topology.worker {payload_remote_name} {result_name}",
            )
            script = generate_script(
                PBSJobConfig(
                    job_name=f"topo_{run_name}"[:15],
                    n_nodes=n_nodes,
                    n_cores_per_node=cores_per_node,
                    walltime=str(topo_cfg.get("walltime", "01:00:00")),
                    queue=queue_used,
                    work_dir=remote_dir,
                    modules=cluster_cfg.modules,
                    env_vars={},
                ),
                [cmd],
            )
            jm.submit_and_wait(
                script,
                remote_dir=remote_dir,
                local_dir=self.topo_dir,
                retrieve_patterns=[result_name, "*.log", "*.out"],
                script_name="topology_worker.pbs",
                stage_files=local_files,
                expected_local_outputs=[result_name],
                queue_used=queue_used,
                poll_interval=int(self.cfg.get("_runtime_log_poll_interval", 5)),
                live_log=bool(self.cfg.get("_runtime_live_log", True)),
                live_files=["topology_worker.log", result_name],
                stale_log_seconds=int(self.cfg.get("_runtime_stale_log_seconds", 300)),
            )

        payload = json.loads((self.topo_dir / result_name).read_text())
        results = payload.get("results", [])
        if not isinstance(results, list):
            raise RuntimeError("Invalid topology worker result payload.")
        return results

    def _merge_point_phase_results(
        self,
        *,
        task: dict[str, Any],
        node_result: dict[str, Any],
        arc_result: dict[str, Any],
    ) -> dict[str, Any]:
        base = dict(node_result)
        base.setdefault("point_index", task.get("point_index"))
        base.setdefault("point_name", task.get("point_name"))
        base.setdefault("variant_id", task.get("variant_id"))
        base.setdefault("thickness_uc", task.get("thickness_uc"))
        base.setdefault("defect_severity", task.get("defect_severity", 0.0))
        if base.get("status") != "ok":
            return base

        if arc_result.get("status") != "ok":
            failed = dict(base)
            failed["status"] = "failed"
            failed["reason"] = f"arc_phase_failed:{arc_result.get('reason', 'unknown')}"
            failed["arc_scan"] = arc_result.get("arc_scan", {"status": "failed"})
            return failed

        merged = dict(base)
        merged["arc_scan"] = arc_result.get("arc_scan", {"status": "failed", "reason": "missing_arc_scan"})
        return merged

    def _run_qsub_per_point(self, tasks: list[dict[str, Any]], topo_cfg: dict[str, Any]) -> list[dict[str, Any]]:
        from wtec.cluster.ssh import open_ssh
        from wtec.cluster.submit import JobManager

        cluster_cfg = ClusterConfig.from_env()
        run_name = str(self.cfg.get("name", "run")).strip() or "run"
        remote_topo_root = f"{self._remote_run_base(cluster_cfg)}/topology"
        local_poll = int(self.cfg.get("_runtime_log_poll_interval", 5))
        local_live = bool(self.cfg.get("_runtime_live_log", True))
        stale_secs = int(self.cfg.get("_runtime_stale_log_seconds", 300))
        max_conc = max(1, int(topo_cfg.get("max_concurrent_point_jobs", 1)))
        raw_rows: list[tuple[int, dict[str, Any]]] = []
        job_rows_raw: list[tuple[int, dict[str, Any]]] = []
        python_exe = str(topo_cfg.get("cluster_python_exe", "python3"))
        worker_zip = self._worker_source_zip()
        point_walltime = str(topo_cfg.get("walltime_per_point", topo_cfg.get("walltime", "00:30:00")))
        remote_shared_dir = f"{remote_topo_root}/_shared"

        def _sha1_file(path: Path) -> str:
            h = hashlib.sha1()
            with path.open("rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    h.update(chunk)
            return h.hexdigest()

        stage_dir = self.topo_dir / "point_shared_stage_inputs"
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        stage_dir.mkdir(parents=True, exist_ok=True)

        digest_to_name: dict[str, str] = {}
        src_to_name: dict[str, str] = {}
        staged_files: list[Path] = []
        shared_manifest_rows: list[dict[str, Any]] = []

        def _register_shared(src: str | Path, *, prefix: str) -> str:
            p = Path(src).expanduser().resolve()
            skey = str(p)
            if skey in src_to_name:
                return src_to_name[skey]
            digest = _sha1_file(p)
            name = digest_to_name.get(digest)
            if name is None:
                name = f"{prefix}_{len(digest_to_name):03d}_{digest[:10]}_{p.name}"
                dst = stage_dir / name
                shutil.copy2(p, dst)
                staged_files.append(dst)
                digest_to_name[digest] = name
                shared_manifest_rows.append(
                    {
                        "digest": digest,
                        "shared_name": name,
                        "source_path": skey,
                        "size_bytes": int(p.stat().st_size),
                    }
                )
            src_to_name[skey] = name
            return name

        worker_zip_shared = _register_shared(worker_zip, prefix="src")
        point_shared: dict[int, dict[str, Any]] = {}
        for idx, task in enumerate(tasks):
            try:
                pidx = int(task.get("point_index", idx))
            except Exception:
                pidx = idx
            hr_name = _register_shared(task["hr_dat_path"], prefix="hr")
            win_name = None
            if task.get("win_path"):
                win_name = _register_shared(task["win_path"], prefix="win")
            ldos_name = None
            ldos_raw = task.get("siesta_slab_ldos_json")
            if ldos_raw:
                ldos_path = Path(str(ldos_raw)).expanduser().resolve()
                if ldos_path.exists():
                    ldos_name = _register_shared(ldos_path, prefix="ldos")
            point_shared[pidx] = {"hr": hr_name, "win": win_name, "siesta_slab_ldos_json": ldos_name}

        with open_ssh(cluster_cfg) as ssh:
            jm = JobManager(ssh)
            queue_used = jm.resolve_queue(
                cluster_cfg.pbs_queue,
                fallback_order=cluster_cfg.pbs_queue_priority,
            )
            cores_per_node = cluster_cfg.cores_for_queue(queue_used)
            n_nodes = int(self.cfg.get("n_nodes", 1))
            total_cores = max(1, n_nodes * cores_per_node)
            prof = self._point_resource_profile(topo_cfg, total_cores=total_cores)
            jm.stage_files(staged_files, remote_shared_dir)
            (self.topo_dir / "topology_shared_artifacts.json").write_text(
                json.dumps(
                    {
                        "remote_shared_dir": remote_shared_dir,
                        "worker_zip_shared": worker_zip_shared,
                        "artifacts": shared_manifest_rows,
                    },
                    indent=2,
                )
            )

        def _run_one(task_index: int, task: dict[str, Any]) -> tuple[int, dict[str, Any], dict[str, Any]]:
            point_name = str(task.get("point_name", f"point_{task_index:03d}"))
            try:
                point_index = int(task.get("point_index", task_index))
            except Exception:
                point_index = task_index
            local_point = self.topo_dir / point_name
            local_point.mkdir(parents=True, exist_ok=True)
            remote_dir = f"{remote_topo_root}/{point_name}"
            shared = point_shared.get(point_index, {})

            task_payload = dict(task)
            task_payload["hr_dat_path"] = str(shared.get("hr") or Path(str(task["hr_dat_path"])).name)
            if task.get("win_path"):
                task_payload["win_path"] = str(shared.get("win") or Path(str(task["win_path"])).name)
            if task.get("siesta_slab_ldos_json") and shared.get("siesta_slab_ldos_json"):
                task_payload["siesta_slab_ldos_json"] = str(shared.get("siesta_slab_ldos_json"))

            task_file = local_point / "task.json"
            task_file.write_text(json.dumps({"tasks": [task_payload]}, indent=2))
            node_result_name = "node_result.json"
            arc_result_name = "arc_result.json"

            worker_python = (
                f"env PYTHONPATH=$PWD/{worker_zip_shared}:$PYTHONPATH {python_exe}"
            )
            node_cmd = build_command(
                worker_python,
                mpi=MPIConfig(n_cores=prof["node_phase_mpi_np"], n_pool=1),
                extra_args=f"-m wtec.topology.worker {task_file.name} {node_result_name} node",
            )
            arc_cmd = build_command(
                worker_python,
                mpi=MPIConfig(n_cores=prof["arc_phase_mpi_np"], n_pool=1),
                extra_args=f"-m wtec.topology.worker {task_file.name} {arc_result_name} arc",
            )
            node_cmd = self._thread_exports(node_cmd, threads=prof["node_phase_threads"])
            arc_cmd = self._thread_exports(arc_cmd, threads=prof["arc_phase_threads"])

            prep_cmds = [
                f"cp -f {shlex.quote(remote_shared_dir.rstrip('/') + '/' + worker_zip_shared)} ./",
                f"cp -f {shlex.quote(remote_shared_dir.rstrip('/') + '/' + str(task_payload['hr_dat_path']))} ./",
            ]
            if task_payload.get("win_path"):
                prep_cmds.append(
                    f"cp -f {shlex.quote(remote_shared_dir.rstrip('/') + '/' + str(task_payload['win_path']))} ./"
                )
            if task_payload.get("siesta_slab_ldos_json"):
                prep_cmds.append(
                    f"cp -f {shlex.quote(remote_shared_dir.rstrip('/') + '/' + str(task_payload['siesta_slab_ldos_json']))} ./"
                )
            prep_cmd = "set -e; " + "; ".join(prep_cmds)

            script = generate_script(
                PBSJobConfig(
                    job_name=f"tp{task_index:03d}_{run_name}"[:15],
                    n_nodes=n_nodes,
                    n_cores_per_node=cores_per_node,
                    walltime=point_walltime,
                    queue=queue_used,
                    work_dir=remote_dir,
                    modules=cluster_cfg.modules,
                    env_vars={},
                ),
                [prep_cmd, node_cmd, arc_cmd],
            )
            try:
                with open_ssh(cluster_cfg) as ssh:
                    jm = JobManager(ssh)
                    meta = jm.submit_and_wait(
                        script,
                        remote_dir=remote_dir,
                        local_dir=local_point,
                        retrieve_patterns=[node_result_name, arc_result_name, "*.log", "*.out"],
                        script_name=f"topology_{point_name}.pbs",
                        stage_files=[task_file],
                        expected_local_outputs=[node_result_name, arc_result_name],
                        queue_used=queue_used,
                        poll_interval=local_poll,
                        live_log=local_live,
                        live_files=[node_result_name, arc_result_name],
                        stale_log_seconds=stale_secs,
                    )
                node_payload = json.loads((local_point / node_result_name).read_text())
                arc_payload = json.loads((local_point / arc_result_name).read_text())
                node_results = node_payload.get("results", [])
                arc_results = arc_payload.get("results", [])
                node_row = node_results[0] if isinstance(node_results, list) and node_results else {
                    "status": "failed",
                    "reason": "missing_node_result",
                }
                arc_row = arc_results[0] if isinstance(arc_results, list) and arc_results else {
                    "status": "failed",
                    "reason": "missing_arc_result",
                }
                merged = self._merge_point_phase_results(
                    task=task,
                    node_result=node_row,
                    arc_result=arc_row,
                )
            except Exception as exc:
                meta = {
                    "job_id": None,
                    "status": "FAILED",
                    "remote_dir": remote_dir,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
                merged = {
                    "status": "failed",
                    "reason": f"point_qsub_failed:{type(exc).__name__}:{exc}",
                    "point_index": point_index,
                    "point_name": point_name,
                    "variant_id": task.get("variant_id"),
                    "thickness_uc": task.get("thickness_uc"),
                    "defect_severity": task.get("defect_severity", 0.0),
                }

            job_row = {
                "point_index": point_index,
                "point_name": point_name,
                "variant_id": task.get("variant_id"),
                "thickness_uc": task.get("thickness_uc"),
                "queue": queue_used,
                "n_nodes": n_nodes,
                "cores_per_node": cores_per_node,
                "node_phase_mpi_np": prof["node_phase_mpi_np"],
                "node_phase_threads": prof["node_phase_threads"],
                "arc_phase_mpi_np": prof["arc_phase_mpi_np"],
                "arc_phase_threads": prof["arc_phase_threads"],
                "job_id": meta.get("job_id"),
                "status": meta.get("status"),
                "remote_dir": meta.get("remote_dir"),
                "shared_hr": task_payload.get("hr_dat_path"),
                "shared_win": task_payload.get("win_path"),
                "shared_siesta_slab_ldos_json": task_payload.get("siesta_slab_ldos_json"),
            }
            return point_index, merged, job_row

        if max_conc == 1:
            for idx, task in enumerate(tasks):
                pidx, merged, job_row = _run_one(idx, task)
                raw_rows.append((pidx, merged))
                job_rows_raw.append((pidx, job_row))
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_conc) as ex:
                futs = [ex.submit(_run_one, idx, task) for idx, task in enumerate(tasks)]
                for fut in concurrent.futures.as_completed(futs):
                    pidx, merged, job_row = fut.result()
                    raw_rows.append((pidx, merged))
                    job_rows_raw.append((pidx, job_row))

        raw_rows.sort(key=lambda x: int(x[0]))
        job_rows_raw.sort(key=lambda x: int(x[0]))
        raw = [r for _, r in raw_rows]
        job_rows = [r for _, r in job_rows_raw]
        (self.topo_dir / "topology_point_jobs.json").write_text(json.dumps({"jobs": job_rows}, indent=2))
        self._write_csv(self.topo_dir / "topology_point_jobs.csv", job_rows)
        return raw

    @staticmethod
    def _delta_node(sig: dict[str, Any] | None, base: dict[str, Any] | None) -> float | None:
        if not sig or not base:
            return None
        nc = sig.get("n_nodes")
        nb = base.get("n_nodes")
        if nc is None or nb is None:
            return None
        count_term = abs(float(nc) - float(nb)) / max(1.0, float(nb))
        ec = sig.get("mean_abs_energy_ev")
        eb = base.get("mean_abs_energy_ev")
        if ec is None or eb is None:
            energy_term = 0.0
        else:
            energy_term = abs(float(ec) - float(eb)) / max(0.05, abs(float(eb)) + 0.05)
        gc = sig.get("mean_gap_ev")
        gb = base.get("mean_gap_ev")
        if gc is None or gb is None:
            gap_term = 0.0
        else:
            gap_term = abs(float(gc) - float(gb)) / max(0.01, abs(float(gb)) + 0.01)
        return float(np.clip(0.6 * count_term + 0.3 * energy_term + 0.1 * gap_term, 0.0, 1.0))

    @staticmethod
    def _arc_deviation(arc_metric: float | None, arc_base: float | None) -> float | None:
        if arc_metric is None or arc_base is None:
            return None
        return float(np.clip(abs(float(arc_metric) - float(arc_base)), 0.0, 1.0))

    def _baseline_maps(self, raw: list[dict[str, Any]]) -> tuple[dict[int, dict[str, Any]], dict[int, float | None]]:
        # Baseline = pristine variant at each thickness; if multiple, pick first successful.
        node_base: dict[int, dict[str, Any]] = {}
        arc_base: dict[int, float | None] = {}
        for r in raw:
            if not r.get("status") == "ok":
                continue
            if not bool(r.get("is_pristine", False)):
                continue
            d = int(r["thickness_uc"])
            if d in node_base:
                continue
            node_base[d] = r.get("node_signature", {})
            arc = r.get("arc_scan", {})
            arc_base[d] = arc.get("metric") if isinstance(arc, dict) else None
        return node_base, arc_base

    def _transport_lookup(self) -> dict[int, float]:
        out: dict[int, float] = {}
        scan = self.transport_results.get("thickness_scan", {})
        if not isinstance(scan, dict) or not scan:
            return out
        # Prefer W=0.0 if available.
        key = None
        for k in scan.keys():
            if float(k) == 0.0:
                key = k
                break
        if key is None:
            key = sorted(scan.keys())[0]
        res = scan[key]
        t = res.get("thickness_uc")
        rho = res.get("rho_mean")
        if t is None or rho is None:
            return out
        for tu, rv in zip(t, rho):
            out[int(tu)] = float(rv)
        return out

    @staticmethod
    def _transport_probe_maps(raw: list[dict[str, Any]]) -> tuple[dict[int, float], dict[tuple[str, int], float]]:
        pristine: dict[int, float] = {}
        all_map: dict[tuple[str, int], float] = {}
        for r in raw:
            tp = r.get("transport_probe")
            if not isinstance(tp, dict):
                continue
            if str(tp.get("status", "")).lower() != "ok":
                continue
            rho = tp.get("rho_ohm_m")
            if rho is None:
                continue
            try:
                d = int(r.get("thickness_uc", -1))
                vid = str(r.get("variant_id", "unknown"))
                rho_v = float(rho)
            except Exception:
                continue
            all_map[(vid, d)] = rho_v
            if bool(r.get("is_pristine", False)) and d not in pristine:
                pristine[d] = rho_v
        return pristine, all_map

    def _score_results(
        self,
        raw: list[dict[str, Any]],
        *,
        w_topo: float,
        failure_policy: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        node_base, arc_base = self._baseline_maps(raw)
        rho_map = self._transport_lookup()
        probe_pristine_map, _ = self._transport_probe_maps(raw)
        soft_missing = str(failure_policy).strip().lower() == "rescale"
        all_rows: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        node_rows: list[dict[str, Any]] = []
        arc_rows: list[dict[str, Any]] = []

        def _transport_score_for_row(row: dict[str, Any]) -> float | None:
            tp = row.get("transport_probe")
            is_pristine = bool(row.get("is_pristine", False))
            try:
                d_loc = int(row.get("thickness_uc", -1))
            except Exception:
                d_loc = -1
            if isinstance(tp, dict) and str(tp.get("status", "")).lower() == "ok":
                rho_val = tp.get("rho_ohm_m")
                if rho_val is not None:
                    try:
                        rho_float = float(rho_val)
                    except Exception:
                        return 0.0 if is_pristine else None
                    base = probe_pristine_map.get(d_loc, rho_map.get(d_loc))
                    if base is None:
                        return 0.0 if is_pristine else None
                    if is_pristine:
                        return 0.0
                    denom = max(1e-30, abs(float(base)))
                    return float(np.clip(abs(rho_float - float(base)) / denom, 0.0, 1.0))
            # Fallback only for pristine baselines from transport curve.
            if is_pristine and d_loc in rho_map:
                return 0.0
            return None

        for r in raw:
            d = int(r.get("thickness_uc", -1))
            vid = str(r.get("variant_id", "unknown"))
            sev = float(r.get("defect_severity", 0.0))
            point_index = r.get("point_index")
            point_name = r.get("point_name")
            if r.get("status") != "ok":
                s_transport = _transport_score_for_row(r)
                row = build_result(
                    thickness_uc=d,
                    variant_id=vid,
                    defect_severity=sev,
                    s_arc=None,
                    delta_node=None,
                    s_transport=s_transport,
                    w_topo=w_topo,
                    allow_missing_topology=soft_missing,
                    extras={
                        "reason": r.get("reason"),
                        "is_pristine": bool(r.get("is_pristine", False)),
                        "point_index": point_index,
                        "point_name": point_name,
                        "raw_status": r.get("status"),
                        "raw_node_status": (
                            r.get("node_scan", {}).get("status")
                            if isinstance(r.get("node_scan"), dict)
                            else None
                        ),
                        "raw_arc_status": (
                            r.get("arc_scan", {}).get("status")
                            if isinstance(r.get("arc_scan"), dict)
                            else None
                        ),
                        "raw_node_reason": (
                            r.get("node_scan", {}).get("reason")
                            if isinstance(r.get("node_scan"), dict)
                            else None
                        ),
                        "raw_arc_reason": (
                            r.get("arc_scan", {}).get("reason")
                            if isinstance(r.get("arc_scan"), dict)
                            else None
                        ),
                        "transport_probe_status": (
                            r.get("transport_probe", {}).get("status")
                            if isinstance(r.get("transport_probe"), dict)
                            else None
                        ),
                        "transport_probe_G_e2_over_h": (
                            r.get("transport_probe", {}).get("G_e2_over_h")
                            if isinstance(r.get("transport_probe"), dict)
                            else None
                        ),
                        "transport_probe_rho_ohm_m": (
                            r.get("transport_probe", {}).get("rho_ohm_m")
                            if isinstance(r.get("transport_probe"), dict)
                            else None
                        ),
                        "tier_used": r.get("tier_used", "single"),
                        "refined": bool(r.get("refined", False)),
                    },
                )
                all_rows.append(row)
                if row["status"] == "failed":
                    failed.append(row)
                continue

            sig = r.get("node_signature", {})
            base_sig = node_base.get(d)
            delta_node = self._delta_node(sig, base_sig)

            arc = r.get("arc_scan", {})
            arc_metric = arc.get("metric") if isinstance(arc, dict) else None
            s_arc = self._arc_deviation(arc_metric, arc_base.get(d))

            s_transport = _transport_score_for_row(r)

            row = build_result(
                thickness_uc=d,
                variant_id=vid,
                defect_severity=sev,
                s_arc=s_arc,
                delta_node=delta_node,
                s_transport=s_transport,
                w_topo=w_topo,
                allow_missing_topology=soft_missing,
                extras={
                    "is_pristine": bool(r.get("is_pristine", False)),
                    "arc_metric": arc_metric,
                    "n_nodes": sig.get("n_nodes"),
                    "mean_abs_node_energy_ev": sig.get("mean_abs_energy_ev"),
                    "validation_status": r.get("validation", {}).get("status"),
                    "node_method": r.get("node_method_requested"),
                    "node_method_requested": r.get("node_method_requested"),
                    "node_method_effective": (
                        r.get("node_scan", {}).get("node_method_effective")
                        if isinstance(r.get("node_scan"), dict)
                        else None
                    ),
                    "node_method_warning": (
                        r.get("node_scan", {}).get("node_method_warning")
                        if isinstance(r.get("node_scan"), dict)
                        else None
                    ),
                    "arc_engine_requested": r.get("arc_engine_requested"),
                    "arc_engine": arc.get("engine") if isinstance(arc, dict) else None,
                    "arc_source_engine": arc.get("source_engine") if isinstance(arc, dict) else None,
                    "arc_source_kind": arc.get("source_kind") if isinstance(arc, dict) else None,
                    "transport_probe_status": (
                        r.get("transport_probe", {}).get("status")
                        if isinstance(r.get("transport_probe"), dict)
                        else None
                    ),
                    "transport_probe_G_e2_over_h": (
                        r.get("transport_probe", {}).get("G_e2_over_h")
                        if isinstance(r.get("transport_probe"), dict)
                        else None
                    ),
                    "transport_probe_rho_ohm_m": (
                        r.get("transport_probe", {}).get("rho_ohm_m")
                        if isinstance(r.get("transport_probe"), dict)
                        else None
                    ),
                    "raw_node_status": (
                        r.get("node_scan", {}).get("status")
                        if isinstance(r.get("node_scan"), dict)
                        else None
                    ),
                    "raw_arc_status": arc.get("status") if isinstance(arc, dict) else None,
                    "raw_node_reason": (
                        r.get("node_scan", {}).get("reason")
                        if isinstance(r.get("node_scan"), dict)
                        else None
                    ),
                    "raw_arc_reason": arc.get("reason") if isinstance(arc, dict) else None,
                    "point_index": point_index,
                    "point_name": point_name,
                    "tier_used": r.get("tier_used", "single"),
                    "refined": bool(r.get("refined", False)),
                },
            )
            all_rows.append(row)
            if row["status"] == "failed":
                failed.append(row)

            # Expanded node/arc tables
            for n in r.get("node_scan", {}).get("nodes", []) if isinstance(r.get("node_scan"), dict) else []:
                node_rows.append(
                    {
                        "point_index": point_index,
                        "point_name": point_name,
                        "variant_id": vid,
                        "thickness_uc": d,
                        "defect_severity": sev,
                        "k_frac_x": n.get("k_frac", [None, None, None])[0],
                        "k_frac_y": n.get("k_frac", [None, None, None])[1],
                        "k_frac_z": n.get("k_frac", [None, None, None])[2],
                        "gap_ev": n.get("gap_ev"),
                        "energy_rel_fermi_ev": n.get("energy_rel_fermi_ev"),
                        "chirality": n.get("chirality"),
                    }
                )
            arc_rows.append(
                {
                    "point_index": point_index,
                    "point_name": point_name,
                    "variant_id": vid,
                    "thickness_uc": d,
                    "defect_severity": sev,
                    "arc_metric": arc_metric,
                    "node_method": r.get("node_method_requested", r.get("node_method")),
                    "arc_engine": arc.get("engine") if isinstance(arc, dict) else None,
                    "arc_source_engine": arc.get("source_engine") if isinstance(arc, dict) else None,
                    "arc_status": arc.get("status") if isinstance(arc, dict) else None,
                }
            )

        return all_rows, failed, node_rows, arc_rows

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            path.write_text("")
            return
        keys = sorted(set().union(*(r.keys() for r in rows)))
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in rows:
                v = dict(row)
                if isinstance(v.get("missing"), list):
                    v["missing"] = ",".join(v["missing"])
                w.writerow(v)

    def _write_plots(self, rows: list[dict[str, Any]]) -> list[str]:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return []
        if not rows:
            return []

        thicknesses = sorted({int(r["thickness_uc"]) for r in rows})
        variants = sorted({str(r["variant_id"]) for r in rows})
        v_index = {v: i for i, v in enumerate(variants)}
        t_index = {t: i for i, t in enumerate(thicknesses)}

        def make_matrix(key: str) -> np.ndarray:
            m = np.full((len(variants), len(thicknesses)), np.nan, dtype=float)
            for r in rows:
                vv = r.get(key)
                if vv is None:
                    continue
                m[v_index[str(r["variant_id"])], t_index[int(r["thickness_uc"])]] = float(vv)
            return m

        files: list[str] = []
        for key, fname, title in [
            ("S_total", "deviation_heatmap.pdf", "Total Topology Deviation"),
            ("S_arc", "component_arc.pdf", "Arc Component"),
            ("delta_node", "component_node.pdf", "Node Component"),
            ("S_transport", "component_transport.pdf", "Transport Component"),
        ]:
            mat = make_matrix(key)
            fig, ax = plt.subplots(figsize=(max(6, len(thicknesses) * 0.5), max(4, len(variants) * 0.35)))
            im = ax.imshow(mat, aspect="auto", origin="lower", vmin=0.0, vmax=1.0, cmap="viridis")
            ax.set_xticks(range(len(thicknesses)))
            ax.set_xticklabels(thicknesses, rotation=0)
            ax.set_yticks(range(len(variants)))
            ax.set_yticklabels(variants)
            ax.set_xlabel("Thickness (uc)")
            ax.set_ylabel("Variant")
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            out = self.topo_dir / fname
            fig.tight_layout()
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            files.append(str(out))
        return files

    def run(self) -> dict[str, Any]:
        topo_cfg = self._topology_cfg()
        if not bool(topo_cfg.get("enabled", True)):
            return {"enabled": False, "status": "skipped"}
        scope = self._hr_scope(topo_cfg)

        fermi_ev = self._resolve_fermi_ev(topo_cfg)
        if fermi_ev is None:
            return {"enabled": True, "status": "failed", "reason": "missing_fermi_ev"}

        thicknesses = self._thicknesses()
        variants = self._variants(topo_cfg)
        point_rows = self._point_manifest_rows(
            variants=variants,
            thicknesses=thicknesses,
            topo_cfg=topo_cfg,
            fermi_ev=float(fermi_ev),
        )
        try:
            point_rows = self._run_hr_generation(
                rows=point_rows,
                topo_cfg=topo_cfg,
                thicknesses=thicknesses,
            )
        except Exception as exc:
            for row in point_rows:
                if str(row.get("status", "")) == "failed":
                    continue
                row["status"] = "failed"
                row["reason"] = f"hr_generation_exception:{type(exc).__name__}:{exc}"
        if scope == "per_variant":
            point_rows = self._propagate_variant_hr_rows(point_rows, thicknesses=thicknesses)
        try:
            point_rows = self._apply_delta_h_rows(rows=point_rows, topo_cfg=topo_cfg)
        except Exception as exc:
            for row in point_rows:
                if str(row.get("status", "")).lower() == "failed":
                    continue
                row["status"] = "failed"
                row["reason"] = f"delta_h_exception:{type(exc).__name__}:{exc}"

        for row in point_rows:
            status = str(row.get("status", "pending"))
            if status == "failed":
                continue
            if row.get("fermi_ev") is None:
                row["fermi_ev"] = float(fermi_ev)
            if status in {"pending", "pending_variant", "ok"}:
                hp_raw = row.get("hr_dat_path")
                if not hp_raw:
                    row["status"] = "failed"
                    row["reason"] = "missing_hr_dat_path"
                    continue
                hp = Path(str(hp_raw)).expanduser().resolve()
                if not hp.exists():
                    row["status"] = "failed"
                    row["reason"] = f"hr_dat_not_found:{hp}"
                    continue
                if status in {"pending", "pending_variant"}:
                    row["status"] = "ok"

        point_rows = self._materialize_siesta_slab_ldos_rows(
            rows=point_rows,
            topo_cfg=topo_cfg,
        )

        (self.topo_dir / "topology_point_manifest.json").write_text(
            json.dumps({"points": point_rows}, indent=2)
        )
        self._write_csv(self.topo_dir / "topology_point_manifest.csv", point_rows)

        tasks, prefail_raw = self._tasks_from_point_rows(
            point_rows=point_rows,
            topo_cfg=topo_cfg,
            fermi_ev_default=float(fermi_ev),
        )
        point_meta_by_index: dict[int, dict[str, Any]] = {}
        for row in point_rows:
            try:
                pidx = int(row.get("point_index", -1))
            except Exception:
                pidx = -1
            point_meta_by_index[pidx] = row

        backend = str(topo_cfg.get("backend", "qsub")).lower().strip()
        exec_mode = str(topo_cfg.get("execution_mode", "single_batch")).lower().strip()
        strict_qsub = bool(topo_cfg.get("strict_qsub", False))
        tier_cfg = self._tiering_cfg(topo_cfg)
        tier_mode = str(tier_cfg.get("mode", "single")).strip().lower()
        raw: list[dict[str, Any]]
        backend_used: str
        tiering_summary: dict[str, Any]
        if tier_mode == "two_tier" and tasks:
            screen_tasks = self._apply_screen_overrides(
                tasks,
                screen_cfg=tier_cfg.get("screen", {}),
            )
            screen_raw, backend_screen = self._execute_tasks(
                tasks=screen_tasks,
                topo_cfg=topo_cfg,
                prefail_raw=prefail_raw,
            )
            for row in screen_raw:
                row.setdefault("tier_used", "screen")
                row.setdefault("refined", False)
            score_cfg = topo_cfg.get("score", {})
            if isinstance(score_cfg, dict):
                if "w_topo" in score_cfg:
                    w_topo_screen = float(score_cfg.get("w_topo", 0.70))
                else:
                    w_topo_screen = 1.0 - float(score_cfg.get("w_transport", 0.30))
            else:
                w_topo_screen = 0.70
            w_topo_screen = float(np.clip(w_topo_screen, 0.0, 1.0))
            failure_policy = str(topo_cfg.get("failure_policy", "strict"))
            scored_screen, _, _, _ = self._score_results(
                screen_raw,
                w_topo=w_topo_screen,
                failure_policy=failure_policy,
            )
            refine_idx = self._select_refine_point_indices(
                scored_rows=scored_screen,
                tiering_cfg=tier_cfg,
            )
            refine_tasks = []
            for task in tasks:
                try:
                    pidx = int(task.get("point_index", -1))
                except Exception:
                    continue
                if pidx in refine_idx:
                    refine_tasks.append(task)

            refine_raw: list[dict[str, Any]] = []
            backend_refine = "none"
            if refine_tasks:
                refine_raw, backend_refine = self._execute_tasks(
                    tasks=refine_tasks,
                    topo_cfg=topo_cfg,
                    prefail_raw=[],
                )
                for row in refine_raw:
                    row["tier_used"] = "refine"
                    row["refined"] = True

            refine_by_idx: dict[int, dict[str, Any]] = {}
            for row in refine_raw:
                try:
                    pidx = int(row.get("point_index", -1))
                except Exception:
                    continue
                refine_by_idx[pidx] = row

            merged_raw: list[dict[str, Any]] = []
            for row in screen_raw:
                try:
                    pidx = int(row.get("point_index", -1))
                except Exception:
                    pidx = -1
                if pidx in refine_by_idx:
                    merged_raw.append(refine_by_idx[pidx])
                else:
                    row.setdefault("tier_used", "screen")
                    row.setdefault("refined", False)
                    merged_raw.append(row)
            raw = merged_raw
            backend_used = f"{backend_screen}+{backend_refine}" if refine_tasks else backend_screen
            tiering_summary = {
                "mode": "two_tier",
                "screen_points": len(screen_raw),
                "refine_selected_points": len(refine_tasks),
                "refine_top_k_per_thickness": int(tier_cfg.get("refine_top_k_per_thickness", 2)),
                "always_include_pristine": bool(tier_cfg.get("always_include_pristine", True)),
            }
        else:
            raw, backend_used = self._execute_tasks(
                tasks=tasks,
                topo_cfg=topo_cfg,
                prefail_raw=prefail_raw,
            )
            for row in raw:
                row.setdefault("tier_used", "single")
                row.setdefault("refined", False)
            tiering_summary = {
                "mode": "single",
                "screen_points": len(raw),
                "refine_selected_points": 0,
            }

        # Restore pristine tags and canonical point metadata.
        for r in raw:
            try:
                pidx = int(r.get("point_index", -1))
            except Exception:
                pidx = -1
            meta = point_meta_by_index.get(pidx)
            if meta:
                r.setdefault("point_name", meta.get("point_name"))
                r.setdefault("variant_id", meta.get("variant_id"))
                r.setdefault("thickness_uc", meta.get("thickness_uc"))
                r.setdefault("defect_severity", meta.get("defect_severity", 0.0))
                r["is_pristine"] = bool(meta.get("is_pristine", False))
            else:
                r["is_pristine"] = bool(r.get("is_pristine", False))

        score_cfg = topo_cfg.get("score", {})
        if isinstance(score_cfg, dict):
            if "w_topo" in score_cfg:
                w_topo = float(score_cfg.get("w_topo", 0.70))
            else:
                w_topo = 1.0 - float(score_cfg.get("w_transport", 0.30))
        else:
            w_topo = 0.70
        w_topo = float(np.clip(w_topo, 0.0, 1.0))
        failure_policy = str(topo_cfg.get("failure_policy", "strict"))
        scored, failed, node_rows, arc_rows = self._score_results(
            raw,
            w_topo=w_topo,
            failure_policy=failure_policy,
        )
        n_partial_points = sum(1 for r in scored if str(r.get("status")) == "partial")

        summary = {
            "status": "ok" if (not failed and n_partial_points == 0) else "partial",
            "backend_requested": backend,
            "backend_used": backend_used,
            "execution_mode": exec_mode,
            "strict_qsub": strict_qsub,
            "hr_scope": scope,
            "variant_dft_engine": self._variant_dft_engine(topo_cfg),
            "failure_policy": failure_policy,
            "score_w_topo": float(w_topo),
            "n_manifest_points": len(point_rows),
            "n_valid_tasks": len(tasks),
            "n_prefailed_points": len(prefail_raw),
            "n_points": len(scored),
            "n_failed_points": len(failed),
            "n_partial_points": int(n_partial_points),
            "fermi_ev": float(fermi_ev),
            "thicknesses_uc": thicknesses,
            "variants": [v["variant_id"] for v in variants],
            "delta_h_artifact_path": (
                str(Path(str(topo_cfg.get("delta_h_artifact_path"))).expanduser().resolve())
                if topo_cfg.get("delta_h_artifact_path")
                else None
            ),
            "tiering": tiering_summary,
        }
        variant_hr_map: dict[str, str] = {}
        variant_fermi_ev_map: dict[str, float] = {}
        for row in point_rows:
            if str(row.get("status", "")).lower() not in {"ok", "ready"}:
                continue
            vid = str(row.get("variant_id", "")).strip()
            if not vid:
                continue
            hrp = row.get("hr_dat_path")
            if isinstance(hrp, str) and hrp and vid not in variant_hr_map:
                variant_hr_map[vid] = hrp
            fev = row.get("fermi_ev")
            if fev is not None and vid not in variant_fermi_ev_map:
                try:
                    variant_fermi_ev_map[vid] = float(fev)
                except Exception:
                    pass
        summary["variant_hr_map"] = variant_hr_map
        summary["variant_fermi_ev_map"] = variant_fermi_ev_map
        if str(failure_policy).strip().lower() == "strict" and (failed or n_partial_points > 0):
            summary["status"] = "failed"

        out_json = {
            "summary": summary,
            "config": topo_cfg,
            "point_manifest": point_rows,
            "results": scored,
            "raw": raw,
        }
        (self.topo_dir / "topology_deviation.json").write_text(json.dumps(out_json, indent=2))
        self._write_csv(self.topo_dir / "topology_deviation.csv", scored)
        self._write_csv(self.topo_dir / "failed_points.csv", failed)
        self._write_csv(self.topo_dir / "node_table.csv", node_rows)
        self._write_csv(self.topo_dir / "arc_connectivity.csv", arc_rows)
        plot_files = self._write_plots(scored)
        summary["plots"] = plot_files
        return {"summary": summary, "results": scored, "failed_points": failed}
