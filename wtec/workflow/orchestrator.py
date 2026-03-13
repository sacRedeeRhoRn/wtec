"""High-level workflow orchestrator with checkpoint/resume state machine.

States: INIT → STRUCTURE → DFT_SCF → DFT_NSCF → WANNIER90 → TRANSPORT → ANALYSIS → DONE
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
import shlex
import zipfile
from typing import Any

import numpy as np

from wtec.rgf import (
    canonicalize_rgf_inputs,
    normalize_axis,
    normalize_rgf_blas_backend,
    normalize_rgf_mode,
    normalize_rgf_parallel_policy,
    normalize_rgf_validate_against,
    normalize_transport_engine,
    plan_execution as rgf_plan_execution,
    phase1_alignment_issues as rgf_phase1_alignment_issues,
    preflight_summary as rgf_preflight_summary,
    resolve_transport_engine,
)


STAGES = [
    "INIT",
    "STRUCTURE",
    "DFT_SCF",
    "DFT_NSCF",
    "WANNIER90",
    "TRANSPORT",
    "ANALYSIS",
    "DONE",
]

DEFAULT_TRANSPORT_RGF_ETA = 1.0e-6
DEFAULT_TRANSPORT_RGF_EXACT_SIGMA_ETA = 1.0e-8


def _wtec_state_dir() -> Path:
    env_dir = os.environ.get("WTEC_STATE_DIR")
    if isinstance(env_dir, str) and env_dir.strip():
        return Path(env_dir).expanduser().resolve()
    local_dir = (Path.cwd() / ".wtec").expanduser().resolve()
    if local_dir.exists():
        return local_dir
    return (Path.home() / ".wtec").expanduser().resolve()


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _load_init_state() -> dict[str, Any]:
    path = _wtec_state_dir() / "init_state.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def resolve_transport_rgf_eta(
    config: dict[str, Any],
    *,
    internal_sigma_mode: str | None = None,
) -> float:
    explicit_eta = config.get("transport_rgf_eta")
    if explicit_eta is not None:
        if not isinstance(explicit_eta, str) or explicit_eta.strip():
            return float(explicit_eta)
    rgf_mode = str(config.get("transport_rgf_mode", "")).strip().lower()
    sigma_backend = (
        str(config.get("transport_rgf_full_finite_sigma_backend", "native")).strip().lower()
        or "native"
    )
    sigma_source = (
        str(
            internal_sigma_mode
            if internal_sigma_mode is not None
            else config.get("_transport_rgf_internal_sigma_mode", sigma_backend)
        )
        .strip()
        .lower()
        or sigma_backend
    )
    if rgf_mode == "full_finite" and sigma_source == "kwant_exact":
        return float(DEFAULT_TRANSPORT_RGF_EXACT_SIGMA_ETA)
    return float(DEFAULT_TRANSPORT_RGF_ETA)


class TopoSlabWorkflow:
    """End-to-end workflow with checkpoint/resume support."""

    def __init__(
        self,
        config: dict,
        job_manager=None,
        *,
        checkpoint_file: Path | None = None,
    ) -> None:
        self.cfg = config
        self.jm = job_manager
        self._state: dict[str, Any] = {"stage": "INIT", "outputs": {}}
        self._checkpoint_file = checkpoint_file or (
            _wtec_state_dir() / "checkpoints" / f"{config.get('name', 'run')}.json"
        )

    # ── constructors ─────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: dict) -> "TopoSlabWorkflow":
        """Build from a config dict (loaded from JSON)."""
        # We can't keep the SSH connection open indefinitely; open per-job.
        # Pass None as job_manager; each stage will open its own connection.
        return cls(config, job_manager=None)

    # ── state machine ─────────────────────────────────────────────────────────

    def run_full(self) -> dict:
        """Run all stages in sequence."""
        self._advance_to("INIT")
        atoms = self._stage_structure()
        hr_direct = self._resolve_hr_dat_from_cfg_or_state()
        if hr_direct is not None:
            print(
                "[Orchestrator] Using configured hr_dat_path; skipping "
                "DFT_SCF/DFT_NSCF/WANNIER90."
            )
            self._state["outputs"]["dft_jobs"] = {
                "scf": {
                    "job_id": None,
                    "status": "SKIPPED",
                    "reason": "configured_hr_dat_path",
                },
                "nscf": {
                    "job_id": None,
                    "status": "SKIPPED",
                    "reason": "configured_hr_dat_path",
                },
            }
            self._state["outputs"]["wannier_job"] = {
                "job_id": None,
                "status": "SKIPPED",
                "reason": "configured_hr_dat_path",
            }
            self._state["outputs"]["hr_dat"] = str(Path(hr_direct).resolve())
            self._state["outputs"]["dft_reuse_mode"] = self._dft_reuse_mode()
            self._save_checkpoint()
            hr_dat = Path(hr_direct)
        else:
            fermi_ev = self._stage_dft(atoms)
            hr_dat = self._stage_wannier90(atoms, fermi_ev)
        transport_results = self._stage_transport(hr_dat)
        analysis_results = self._stage_analysis(transport_results, hr_dat=hr_dat)
        self._set_stage("DONE")
        self._save_checkpoint()
        return {"transport": transport_results, "analysis": analysis_results}

    def run_stage(self, stage: str) -> dict:
        """Run only the specified stage, using checkpoint outputs for prior stages."""
        self._load_checkpoint()
        stage_upper = stage.upper().replace("-", "_")
        aliases = {
            "DFT": "DFT_NSCF",
        }
        stage_upper = aliases.get(stage_upper, stage_upper)
        valid = set(STAGES) | {"DFT"}
        if stage_upper not in valid:
            raise ValueError(f"Unknown stage: {stage!r}. Valid: {sorted(valid)}")

        if stage_upper == "INIT":
            self._advance_to("INIT")
            return {"stage": "INIT"}

        if stage_upper == "STRUCTURE":
            atoms = self._stage_structure()
            return {"structure_atoms": len(atoms)}

        # Transport/analysis can run directly from explicit hr_dat_path (config)
        # or from checkpointed hr_dat, without forcing DFT/Wannier stages.
        if stage_upper in {"TRANSPORT", "ANALYSIS"}:
            hr_direct = self._resolve_hr_dat_from_cfg_or_state()
            if hr_direct is not None:
                self._state["outputs"]["hr_dat"] = str(hr_direct)
                self._save_checkpoint()
                if stage_upper == "TRANSPORT":
                    results = self._stage_transport(hr_direct)
                    return {"transport": results}
                transport_results = self._stage_transport(hr_direct)
                analysis = self._stage_analysis(transport_results, hr_dat=hr_direct)
                return {"analysis": analysis}

        atoms = self._build_atoms_from_config()
        if stage_upper in {"DFT_SCF", "DFT_NSCF"}:
            fermi_ev = self._stage_dft(atoms)
            return {"fermi_ev": fermi_ev}

        fermi_ev = self._state["outputs"].get("fermi_ev")
        if fermi_ev is None:
            fermi_ev = self._stage_dft(atoms)

        if stage_upper == "WANNIER90":
            hr_dat = self._stage_wannier90(atoms, float(fermi_ev))
            return {"hr_dat": str(hr_dat)}

        hr_dat_str = self._state["outputs"].get("hr_dat")
        hr_dat = Path(hr_dat_str) if hr_dat_str else None
        if hr_dat is None or not hr_dat.exists():
            hr_dat = self._stage_wannier90(atoms, float(fermi_ev))

        if stage_upper == "TRANSPORT":
            results = self._stage_transport(hr_dat)
            return {"transport": results}

        if stage_upper == "ANALYSIS":
            transport_results = self._stage_transport(hr_dat)
            analysis = self._stage_analysis(transport_results, hr_dat=hr_dat)
            return {"analysis": analysis}

        if stage_upper == "DONE":
            out = self.run_full()
            self._set_stage("DONE")
            self._save_checkpoint()
            return out

        raise ValueError(f"Unhandled stage: {stage!r}")

    def resume(self) -> dict:
        """Resume from last saved checkpoint."""
        self._load_checkpoint()
        current = self._state.get("stage", "INIT")
        if current not in STAGES:
            current = "INIT"
        idx = STAGES.index(current)
        print(f"[Orchestrator] Resuming from stage {current} ({idx}/{len(STAGES)-1})")
        if current == "DONE":
            return {"status": "DONE", "outputs": self._state.get("outputs", {})}

        fermi_ev = self._state.get("outputs", {}).get("fermi_ev")
        hr_dat = self._resolve_hr_dat_from_cfg_or_state()

        if current in {"INIT", "STRUCTURE", "DFT_SCF", "DFT_NSCF"}:
            atoms = self._build_atoms_from_config()
            fermi_ev = self._stage_dft(atoms)
            hr_dat = self._stage_wannier90(atoms, float(fermi_ev))
        elif current == "WANNIER90":
            atoms = self._build_atoms_from_config()
            if fermi_ev is None:
                fermi_ev = self._stage_dft(atoms)
            hr_dat = self._stage_wannier90(atoms, float(fermi_ev))
        elif current in {"TRANSPORT", "ANALYSIS"}:
            if hr_dat is None:
                atoms = self._build_atoms_from_config()
                if fermi_ev is None:
                    fermi_ev = self._stage_dft(atoms)
                hr_dat = self._stage_wannier90(atoms, float(fermi_ev))
        else:
            return self.run_full()

        if hr_dat is None:
            raise FileNotFoundError(
                "Could not resolve hr_dat for resume. Provide cfg['hr_dat_path'] "
                "or ensure checkpoint output exists."
            )
        transport_results = self._stage_transport(hr_dat)
        analysis_results = self._stage_analysis(transport_results, hr_dat=hr_dat)
        self._set_stage("DONE")
        self._save_checkpoint()
        return {"transport": transport_results, "analysis": analysis_results}

    def status(self) -> dict:
        self._load_checkpoint()
        return {
            "stage": self._state["stage"],
            "outputs": list(self._state["outputs"].keys()),
        }

    # ── stages ────────────────────────────────────────────────────────────────

    def _stage_structure(self):
        self._set_stage("STRUCTURE")
        struct_file = self.cfg["structure_file"]
        atoms = self._build_atoms_from_config()

        self._state["outputs"]["structure"] = str(struct_file)
        self._save_checkpoint()
        return atoms

    def _dft_engine(self) -> str:
        mode_raw = self.cfg.get("dft_mode")
        if mode_raw is None and isinstance(self.cfg.get("dft"), dict):
            mode_raw = self.cfg["dft"].get("mode")
        mode = str(mode_raw or "legacy_single").strip().lower() or "legacy_single"

        # Hybrid mode: this orchestrator DFT stage is the reference track.
        if mode == "hybrid_qe_ref_siesta_variants":
            raw = self.cfg.get("dft_reference_engine")
            if raw is None and isinstance(self.cfg.get("dft"), dict):
                raw = (self.cfg["dft"].get("reference") or {}).get("engine")
            if raw is None:
                raw = "qe"
        elif mode == "dual_family":
            raw = self.cfg.get("dft_pes_engine")
            if raw is None and isinstance(self.cfg.get("dft"), dict):
                tracks = self.cfg["dft"].get("tracks")
                if isinstance(tracks, dict):
                    pes = tracks.get("pes_reference")
                    if isinstance(pes, dict):
                        raw = pes.get("engine")
            if raw is None:
                raw = self.cfg.get("dft_reference_engine")
            if raw is None:
                raw = "qe"
        else:
            raw = self.cfg.get("dft_engine")
            if raw is None and isinstance(self.cfg.get("dft"), dict):
                raw = self.cfg["dft"].get("engine")

        engine = str(raw or "qe").strip().lower() or "qe"
        if engine not in {"qe", "siesta", "vasp"}:
            raise ValueError(f"Unsupported dft_engine={engine!r}. Use 'qe', 'siesta', or 'vasp'.")
        return engine

    def _transport_engine(self) -> str:
        raw = self.cfg.get("transport_engine")
        if raw is None and isinstance(self.cfg.get("transport"), dict):
            raw = self.cfg["transport"].get("engine")
        backend = str(self.cfg.get("transport_backend", "qsub")).strip().lower() or "qsub"
        return resolve_transport_engine(
            raw or "auto",
            cfg=self.cfg,
            init_state=_load_init_state(),
            backend=backend,
        )

    def _dft_reuse_mode(self) -> str:
        raw = self.cfg.get("dft_reuse_mode")
        if raw is None and isinstance(self.cfg.get("dft"), dict):
            raw = self.cfg["dft"].get("reuse_mode")
        mode = str(raw or "all").strip().lower() or "all"
        allowed = {"none", "pristine-only", "all"}
        if mode not in allowed:
            raise ValueError(f"Unsupported dft.reuse_mode={mode!r}. Use one of {sorted(allowed)}.")
        return mode

    def _dft_mode(self) -> str:
        raw = self.cfg.get("dft_mode")
        if raw is None and isinstance(self.cfg.get("dft"), dict):
            raw = self.cfg["dft"].get("mode")
        mode = str(raw or "legacy_single").strip().lower() or "legacy_single"
        allowed = {"legacy_single", "hybrid_qe_ref_siesta_variants", "dual_family"}
        if mode not in allowed:
            raise ValueError(f"Unsupported dft_mode={mode!r}. Use one of {sorted(allowed)}.")
        return mode

    def _variant_dft_engine(self) -> str:
        if self._dft_mode() == "dual_family":
            raw = self.cfg.get("dft_lcao_engine")
            if raw is None and isinstance(self.cfg.get("dft"), dict):
                tracks = self.cfg["dft"].get("tracks")
                if isinstance(tracks, dict):
                    lcao = tracks.get("lcao_upscaled")
                    if isinstance(lcao, dict):
                        raw = lcao.get("engine")
        else:
            raw = self.cfg.get("topology_variant_dft_engine")
        if raw is None and isinstance(self.cfg.get("topology"), dict):
            raw = self.cfg["topology"].get("variant_dft_engine")
        if raw is None and self._dft_mode() == "hybrid_qe_ref_siesta_variants":
            raw = "siesta"
        if raw is None:
            raw = self._dft_engine()
        engine = str(raw or "siesta").strip().lower() or "siesta"
        if engine not in {"qe", "siesta", "abacus"}:
            raise ValueError(
                f"Unsupported topology variant dft_engine={engine!r}. Use 'qe', 'siesta', or 'abacus'."
            )
        return engine

    def _pes_reference_structure_path(self) -> Path | None:
        raw = self.cfg.get("dft_pes_reference_structure_file")
        if raw is None and isinstance(self.cfg.get("dft"), dict):
            tracks = self.cfg["dft"].get("tracks")
            if isinstance(tracks, dict):
                pes = tracks.get("pes_reference")
                if isinstance(pes, dict):
                    raw = pes.get("structure_file")
        if not isinstance(raw, str) or not raw.strip():
            return None
        return Path(raw).expanduser().resolve()

    def _reference_atoms(self, fallback_atoms):
        ref_path = self._pes_reference_structure_path()
        if ref_path is None:
            if self._dft_mode() in {"dual_family", "hybrid_qe_ref_siesta_variants"}:
                raise ValueError(
                    f"{self._dft_mode()} mode requires "
                    "dft_pes_reference_structure_file for PES reference track."
                )
            return fallback_atoms
        if not ref_path.exists():
            raise FileNotFoundError(f"dft_pes_reference_structure_file not found: {ref_path}")
        if ref_path.stat().st_size == 0:
            raise ValueError(f"dft_pes_reference_structure_file is empty: {ref_path}")
        import ase.io

        return ase.io.read(str(ref_path))

    def _dft_siesta_cfg(self) -> dict[str, Any]:
        flat = self.cfg.get("dft_siesta", {})
        nested = self.cfg.get("dft", {}).get("siesta") if isinstance(self.cfg.get("dft"), dict) else {}
        out: dict[str, Any] = {}
        if isinstance(nested, dict):
            out.update(nested)
        if isinstance(flat, dict):
            out.update(flat)
        if out:
            return out
        return {}

    def _dft_vasp_cfg(self) -> dict[str, Any]:
        flat = self.cfg.get("dft_vasp", {})
        nested = self.cfg.get("dft", {}).get("vasp") if isinstance(self.cfg.get("dft"), dict) else {}
        out: dict[str, Any] = {}
        if isinstance(nested, dict):
            out.update(nested)
        if isinstance(flat, dict):
            out.update(flat)
        if out:
            return out
        return {}

    def _dft_abacus_cfg(self) -> dict[str, Any]:
        flat = self.cfg.get("dft_abacus", {})
        nested = self.cfg.get("dft", {}).get("abacus") if isinstance(self.cfg.get("dft"), dict) else {}
        out: dict[str, Any] = {}
        if isinstance(nested, dict):
            out.update(nested)
        if isinstance(flat, dict):
            out.update(flat)
        if out:
            return out
        return {}

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

    def _reference_reuse_policy(self) -> str:
        raw = self.cfg.get("dft_reference_reuse_policy")
        if raw is None and isinstance(self.cfg.get("dft"), dict):
            ref = self.cfg["dft"].get("reference")
            if isinstance(ref, dict):
                raw = ref.get("reuse_policy")
        policy = str(raw or "strict_hash").strip().lower() or "strict_hash"
        if policy not in {"strict_hash", "timestamp_only"}:
            raise ValueError(
                "dft.reference.reuse_policy must be 'strict_hash' or 'timestamp_only'."
            )
        return policy

    @staticmethod
    def _sha1_file(path: Path) -> str:
        h = hashlib.sha1()
        with path.open("rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def _qe_reference_manifest_path(self, run_dir: Path) -> Path:
        return run_dir / "qe_reference_manifest.json"

    def _qe_reference_fingerprint(self, run_dir: Path) -> dict[str, Any]:
        structure_path = self._pes_reference_structure_path()
        if structure_path is None:
            structure_path = Path(str(self.cfg.get("structure_file", ""))).expanduser()
        structure_hash = None
        if structure_path.exists() and structure_path.is_file():
            structure_hash = self._sha1_file(structure_path)
        return {
            "material": str(self.cfg.get("material", "")),
            "structure_file": str(structure_path.resolve()) if structure_path.exists() else str(structure_path),
            "structure_sha1": structure_hash,
            "kpoints_scf": list(self.cfg.get("kpoints_scf", (8, 8, 8))),
            "kpoints_nscf": list(self.cfg.get("kpoints_nscf", (12, 12, 12))),
            "qe_noncolin": bool(self.cfg.get("qe_noncolin", True)),
            "qe_lspinorb": bool(self.cfg.get("qe_lspinorb", True)),
            "qe_disable_symmetry": bool(self.cfg.get("qe_disable_symmetry", False)),
            "dft_dispersion": self._dft_dispersion_cfg(),
            "pseudo_dir": str(self.cfg.get("qe_pseudo_dir", "")),
            "n_nodes": int(self.cfg.get("n_nodes", 1)),
        }

    def _load_qe_reference_manifest(self, run_dir: Path) -> dict[str, Any] | None:
        p = self._qe_reference_manifest_path(run_dir)
        if not p.exists() or p.stat().st_size == 0:
            return None
        try:
            data = json.loads(p.read_text())
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        return data

    def _qe_reference_can_reuse(
        self,
        *,
        run_dir: Path,
        required_outputs: list[str],
    ) -> tuple[bool, str]:
        policy = self._reference_reuse_policy()
        if policy == "timestamp_only":
            for name in required_outputs:
                p = run_dir / name
                if not p.exists() or p.stat().st_size == 0:
                    return False, f"missing_output:{name}"
            return True, "timestamp_only_outputs_present"

        manifest = self._load_qe_reference_manifest(run_dir)
        if manifest is None:
            return False, "manifest_missing"
        current_fp = self._qe_reference_fingerprint(run_dir)
        saved_fp = manifest.get("fingerprint")
        if not isinstance(saved_fp, dict):
            return False, "manifest_fingerprint_missing"
        if saved_fp != current_fp:
            return False, "fingerprint_mismatch"

        outputs = manifest.get("outputs")
        if not isinstance(outputs, dict):
            return False, "manifest_outputs_missing"
        for name in required_outputs:
            p = run_dir / name
            if not p.exists() or p.stat().st_size == 0:
                return False, f"missing_output:{name}"
            meta = outputs.get(name)
            if not isinstance(meta, dict):
                return False, f"manifest_output_missing:{name}"
            expected_sha = str(meta.get("sha1", "")).strip()
            if not expected_sha:
                return False, f"manifest_sha_missing:{name}"
            actual_sha = self._sha1_file(p)
            if actual_sha != expected_sha:
                return False, f"sha_mismatch:{name}"
        return True, "strict_hash_match"

    def _write_qe_reference_manifest(
        self,
        *,
        run_dir: Path,
        fermi_ev: float,
    ) -> None:
        outputs: dict[str, Any] = {}
        for name in [
            f"{self.cfg['material']}.scf.out",
            f"{self.cfg['material']}.nscf.out",
            f"{self.cfg['material']}.win",
            f"{self.cfg['material']}.wout",
            f"{self.cfg['material']}_hr.dat",
        ]:
            p = run_dir / name
            if not p.exists() or p.stat().st_size == 0:
                continue
            outputs[name] = {
                "path": str(p.resolve()),
                "size_bytes": int(p.stat().st_size),
                "sha1": self._sha1_file(p),
            }
        manifest = {
            "fingerprint": self._qe_reference_fingerprint(run_dir),
            "fermi_ev": float(fermi_ev),
            "outputs": outputs,
            "updated_at": int(time.time()),
        }
        self._qe_reference_manifest_path(run_dir).write_text(json.dumps(manifest, indent=2))

    def _stage_dft(self, atoms) -> float:
        self._set_stage("DFT_SCF")
        from wtec.config.cluster import ClusterConfig
        from wtec.cluster.ssh import open_ssh
        from wtec.cluster.submit import JobManager

        cfg = ClusterConfig.from_env()
        run_dir = Path(self.cfg.get("run_dir", ".")) / "dft"
        run_dir.mkdir(parents=True, exist_ok=True)
        atoms_ref = self._reference_atoms(atoms)
        material = self.cfg["material"]
        scf_out = run_dir / f"{material}.scf.out"
        nscf_out = run_dir / f"{material}.nscf.out"
        resume_from_existing_scf = bool(
            self.cfg.get("resume_from_existing_scf", True)
        )
        engine = self._dft_engine()

        if engine == "qe":
            from wtec.workflow.dft_pipeline import DFTPipeline as PipelineClass
            from wtec.qe.parser import parse_fermi_energy, parse_convergence
        elif engine == "siesta":
            from wtec.siesta.runner import SiestaPipeline as PipelineClass
            from wtec.siesta.parser import parse_fermi_energy, parse_convergence
        else:  # vasp
            from wtec.vasp.runner import VaspPipeline as PipelineClass
            from wtec.vasp.parser import parse_fermi_energy, parse_convergence

        if engine == "qe" and self._dft_reuse_mode() in {"pristine-only", "all"}:
            can_reuse, reuse_reason = self._qe_reference_can_reuse(
                run_dir=run_dir,
                required_outputs=[f"{material}.scf.out", f"{material}.nscf.out"],
            )
            if can_reuse and scf_out.exists():
                fermi_ev = float(parse_fermi_energy(scf_out))
                print(f"[Orchestrator] Reusing QE reference DFT ({reuse_reason}).")
                self._state["outputs"]["fermi_ev"] = fermi_ev
                self._state["outputs"]["dft_engine"] = engine
                self._state["outputs"]["dft_mode"] = self._dft_mode()
                self._state["outputs"]["dft_jobs"] = {
                    "scf": {
                        "job_id": None,
                        "status": "SKIPPED",
                        "reason": f"qe_reference_reuse:{reuse_reason}",
                        "local_scf_out": str(scf_out),
                    },
                    "nscf": {
                        "job_id": None,
                        "status": "SKIPPED",
                        "reason": f"qe_reference_reuse:{reuse_reason}",
                        "local_nscf_out": str(nscf_out),
                    },
                }
                self._save_checkpoint()
                return fermi_ev

        scf_meta: dict[str, Any]
        fermi_ev: float

        # Fast path: local SCF output already complete and parseable.
        if (
            resume_from_existing_scf
            and scf_out.exists()
            and scf_out.stat().st_size > 0
            and parse_convergence(scf_out)
        ):
            fermi_ev = parse_fermi_energy(scf_out)
            scf_meta = {
                "job_id": None,
                "status": "SKIPPED",
                "reason": "existing_local_scf",
                "local_scf_out": str(scf_out),
            }
            print(
                f"[Orchestrator] Reusing existing SCF output, skipping SCF: {scf_out}"
            )
        else:
            fermi_ev = float("nan")
            scf_meta = {}

        with open_ssh(cfg) as ssh:
            jm = JobManager(ssh)
            common_kwargs = {
                "run_dir": run_dir,
                "remote_base": self.cfg.get("remote_workdir", cfg.remote_workdir),
                "n_nodes": self.cfg.get("n_nodes", 1),
                "n_cores_per_node": cfg.mpi_cores,
                "n_cores_by_queue": cfg.mpi_cores_by_queue,
                "queue": cfg.pbs_queue,
                "queue_priority": cfg.pbs_queue_priority,
                "kpoints_scf": tuple(self.cfg.get("kpoints_scf", (8, 8, 8))),
                "kpoints_nscf": tuple(self.cfg.get("kpoints_nscf", (12, 12, 12))),
                "omp_threads": cfg.omp_threads,
                "modules": cfg.modules,
                "bin_dirs": cfg.bin_dirs,
                "live_log": self.cfg.get("_runtime_live_log", True),
                "log_poll_interval": self.cfg.get("_runtime_log_poll_interval", 5),
                "stale_log_seconds": self.cfg.get("_runtime_stale_log_seconds", 300),
            }
            if engine == "qe":
                custom_projections = self.cfg.get("wannier_custom_projections")
                if custom_projections is not None and not isinstance(custom_projections, list):
                    raise ValueError("wannier_custom_projections must be a list[str] when provided.")
                qe_pseudo_dir = str(self.cfg.get("qe_pseudo_dir", "")).strip() or cfg.qe_pseudo_dir
                pipeline = PipelineClass(
                    atoms_ref,
                    material,
                    jm,
                    pseudo_dir=qe_pseudo_dir,
                    qe_noncolin=self.cfg.get("qe_noncolin", True),
                    qe_lspinorb=self.cfg.get("qe_lspinorb", True),
                    qe_disable_symmetry=self.cfg.get("qe_disable_symmetry", False),
                    custom_projections=custom_projections,
                    dispersion_cfg=self._dft_dispersion_cfg(),
                    **common_kwargs,
                )
            elif engine == "siesta":
                siesta_cfg = self._dft_siesta_cfg()
                pipeline = PipelineClass(
                    atoms_ref,
                    material,
                    jm,
                    pseudo_dir=cfg.resolved_siesta_pseudo_dir(
                        spin_orbit=bool(siesta_cfg.get("spin_orbit", True)),
                        explicit=str(siesta_cfg.get("pseudo_dir", "")).strip(),
                    ),
                    basis_profile=str(siesta_cfg.get("basis_profile", "")).strip(),
                    wannier_interface=str(siesta_cfg.get("wannier_interface", "sisl")).strip().lower(),
                    spin_orbit=bool(siesta_cfg.get("spin_orbit", True)),
                    include_pao_basis=bool(siesta_cfg.get("include_pao_basis", False)),
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
                    dm_mixing_weight=float(siesta_cfg.get("dm_mixing_weight", 0.18)),
                    dm_number_pulay=int(siesta_cfg.get("dm_number_pulay", 6)),
                    electronic_temperature_k=float(siesta_cfg.get("electronic_temperature_k", 300.0)),
                    max_scf_iterations=int(siesta_cfg.get("max_scf_iterations", 120)),
                    dispersion_cfg=self._dft_dispersion_cfg(),
                    **common_kwargs,
                )
            else:
                vasp_cfg = self._dft_vasp_cfg()
                pipeline = PipelineClass(
                    atoms_ref,
                    material,
                    jm,
                    pseudo_dir=str(vasp_cfg.get("pseudo_dir") or cfg.vasp_pseudo_dir),
                    executable=str(vasp_cfg.get("executable", "vasp_std")).strip() or "vasp_std",
                    encut_ev=float(vasp_cfg.get("encut_ev", 520.0)),
                    ediff=float(vasp_cfg.get("ediff", 1e-6)),
                    ismear=int(vasp_cfg.get("ismear", 0)),
                    sigma=float(vasp_cfg.get("sigma", 0.05)),
                    disable_symmetry=bool(
                        vasp_cfg.get(
                            "disable_symmetry",
                            self.cfg.get("qe_disable_symmetry", True),
                        )
                    ),
                    **common_kwargs,
                )
            if not scf_meta:
                remote_dir = f"{pipeline.remote_base}/{material}"
                remote_scf_out = f"{remote_dir.rstrip('/')}/{material}.scf.out"
                # Resume path: detect finished SCF on cluster even if local output was cleared.
                check_remote = f"test -s {shlex.quote(remote_scf_out)}"
                rc, _, _ = ssh.run(check_remote, check=False)
                if resume_from_existing_scf and rc == 0:
                    jm.retrieve(
                        remote_dir,
                        run_dir,
                        [f"{material}.scf.out", f"scf_{material}.log"],
                    )
                    if not scf_out.exists():
                        raise FileNotFoundError(
                            f"SCF output was detected remotely but not retrieved to {scf_out}"
                        )
                    if parse_convergence(scf_out):
                        fermi_ev = parse_fermi_energy(scf_out)
                        scf_meta = {
                            "job_id": None,
                            "status": "SKIPPED",
                            "reason": "existing_remote_scf",
                            "remote_scf_out": remote_scf_out,
                            "local_scf_out": str(scf_out),
                        }
                        print(
                            "[Orchestrator] Reusing completed remote SCF output, skipping SCF."
                        )
                    else:
                        scf_meta = pipeline.run_scf()
                        fermi_ev = parse_fermi_energy(scf_out)
                else:
                    scf_meta = pipeline.run_scf()
                    fermi_ev = parse_fermi_energy(scf_out)

            self._set_stage("DFT_NSCF")
            nscf_meta = pipeline.run_nscf(fermi_ev)

        self._state["outputs"]["fermi_ev"] = fermi_ev
        if self._dft_mode() == "hybrid_qe_ref_siesta_variants" and engine == "qe":
            self._state["outputs"]["fermi_ev_pristine_qe"] = float(fermi_ev)
        self._state["outputs"]["dft_engine"] = engine
        self._state["outputs"]["dft_mode"] = self._dft_mode()
        self._state["outputs"]["dft_reuse_mode"] = self._dft_reuse_mode()
        self._state["outputs"]["dft_jobs"] = {
            "scf": scf_meta,
            "nscf": nscf_meta,
        }
        self._state["last_job_id"] = nscf_meta.get("job_id")
        self._save_checkpoint()
        return fermi_ev

    def _stage_wannier90(self, atoms, fermi_ev: float) -> Path:
        self._set_stage("WANNIER90")
        from wtec.config.cluster import ClusterConfig
        from wtec.cluster.ssh import open_ssh
        from wtec.cluster.submit import JobManager

        cfg = ClusterConfig.from_env()
        run_dir = Path(self.cfg.get("run_dir", ".")) / "dft"
        atoms_ref = self._reference_atoms(atoms)
        engine = self._dft_engine()
        material = self.cfg["material"]
        hr_dat = run_dir / f"{material}_hr.dat"
        win_file = run_dir / f"{material}.win"
        wout_file = run_dir / f"{material}.wout"

        if engine == "qe" and self._dft_reuse_mode() in {"pristine-only", "all"}:
            can_reuse, reuse_reason = self._qe_reference_can_reuse(
                run_dir=run_dir,
                required_outputs=[
                    f"{material}.scf.out",
                    f"{material}.nscf.out",
                    f"{material}.win",
                    f"{material}.wout",
                    f"{material}_hr.dat",
                ],
            )
            if can_reuse and hr_dat.exists() and win_file.exists() and wout_file.exists():
                print(f"[Orchestrator] Reusing QE reference Wannier outputs ({reuse_reason}).")
                wan_meta = {
                    "job_id": None,
                    "status": "SKIPPED",
                    "reason": f"qe_reference_reuse:{reuse_reason}",
                    "local_hr_dat": str(hr_dat),
                }
                self._state["outputs"]["hr_dat"] = str(hr_dat)
                self._state["outputs"]["wannier_job"] = wan_meta
                self._state["outputs"]["dft_engine"] = engine
                self._state["outputs"]["dft_mode"] = self._dft_mode()
                self._save_checkpoint()
                return hr_dat

        with open_ssh(cfg) as ssh:
            jm = JobManager(ssh)
            common_kwargs = {
                "run_dir": run_dir,
                "remote_base": self.cfg.get("remote_workdir", cfg.remote_workdir),
                "n_nodes": self.cfg.get("n_nodes", 1),
                "n_cores_per_node": cfg.mpi_cores,
                "n_cores_by_queue": cfg.mpi_cores_by_queue,
                "queue": cfg.pbs_queue,
                "queue_priority": cfg.pbs_queue_priority,
                "kpoints_scf": tuple(self.cfg.get("kpoints_scf", (8, 8, 8))),
                "kpoints_nscf": tuple(self.cfg.get("kpoints_nscf", (12, 12, 12))),
                "omp_threads": cfg.omp_threads,
                "modules": cfg.modules,
                "bin_dirs": cfg.bin_dirs,
                "live_log": self.cfg.get("_runtime_live_log", True),
                "log_poll_interval": self.cfg.get("_runtime_log_poll_interval", 5),
                "stale_log_seconds": self.cfg.get("_runtime_stale_log_seconds", 300),
            }
            if engine == "qe":
                from wtec.workflow.dft_pipeline import DFTPipeline as PipelineClass

                custom_projections = self.cfg.get("wannier_custom_projections")
                if custom_projections is not None and not isinstance(custom_projections, list):
                    raise ValueError("wannier_custom_projections must be a list[str] when provided.")
                qe_pseudo_dir = str(self.cfg.get("qe_pseudo_dir", "")).strip() or cfg.qe_pseudo_dir
                pipeline = PipelineClass(
                    atoms_ref,
                    material,
                    jm,
                    pseudo_dir=qe_pseudo_dir,
                    qe_noncolin=self.cfg.get("qe_noncolin", True),
                    qe_lspinorb=self.cfg.get("qe_lspinorb", True),
                    qe_disable_symmetry=self.cfg.get("qe_disable_symmetry", False),
                    custom_projections=custom_projections,
                    dispersion_cfg=self._dft_dispersion_cfg(),
                    **common_kwargs,
                )
            elif engine == "siesta":
                from wtec.siesta.runner import SiestaPipeline as PipelineClass

                siesta_cfg = self._dft_siesta_cfg()
                pipeline = PipelineClass(
                    atoms_ref,
                    material,
                    jm,
                    pseudo_dir=cfg.resolved_siesta_pseudo_dir(
                        spin_orbit=bool(siesta_cfg.get("spin_orbit", True)),
                        explicit=str(siesta_cfg.get("pseudo_dir", "")).strip(),
                    ),
                    basis_profile=str(siesta_cfg.get("basis_profile", "")).strip(),
                    wannier_interface=str(siesta_cfg.get("wannier_interface", "sisl")).strip().lower(),
                    spin_orbit=bool(siesta_cfg.get("spin_orbit", True)),
                    include_pao_basis=bool(siesta_cfg.get("include_pao_basis", False)),
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
                    dm_mixing_weight=float(siesta_cfg.get("dm_mixing_weight", 0.18)),
                    dm_number_pulay=int(siesta_cfg.get("dm_number_pulay", 6)),
                    electronic_temperature_k=float(siesta_cfg.get("electronic_temperature_k", 300.0)),
                    max_scf_iterations=int(siesta_cfg.get("max_scf_iterations", 120)),
                    dispersion_cfg=self._dft_dispersion_cfg(),
                    **common_kwargs,
                )
            else:
                from wtec.vasp.runner import VaspPipeline as PipelineClass

                vasp_cfg = self._dft_vasp_cfg()
                pipeline = PipelineClass(
                    atoms_ref,
                    material,
                    jm,
                    pseudo_dir=str(vasp_cfg.get("pseudo_dir") or cfg.vasp_pseudo_dir),
                    executable=str(vasp_cfg.get("executable", "vasp_std")).strip() or "vasp_std",
                    encut_ev=float(vasp_cfg.get("encut_ev", 520.0)),
                    ediff=float(vasp_cfg.get("ediff", 1e-6)),
                    ismear=int(vasp_cfg.get("ismear", 0)),
                    sigma=float(vasp_cfg.get("sigma", 0.05)),
                    disable_symmetry=bool(
                        vasp_cfg.get(
                            "disable_symmetry",
                            self.cfg.get("qe_disable_symmetry", True),
                        )
                    ),
                    **common_kwargs,
                )
            wan_meta = pipeline.run_wannier(fermi_ev)

        self._state["outputs"]["hr_dat"] = str(hr_dat)
        self._state["outputs"]["wannier_job"] = wan_meta
        self._state["outputs"]["dft_engine"] = engine
        self._state["outputs"]["dft_mode"] = self._dft_mode()
        self._state["outputs"]["dft_reuse_mode"] = self._dft_reuse_mode()
        self._state["last_job_id"] = wan_meta.get("job_id")
        if engine == "qe":
            self._write_qe_reference_manifest(run_dir=run_dir, fermi_ev=float(fermi_ev))
        self._save_checkpoint()
        return hr_dat

    def _stage_transport(self, hr_dat: Path) -> dict:
        self._set_stage("TRANSPORT")
        requested_engine_raw = self.cfg.get("transport_engine")
        if requested_engine_raw is None and isinstance(self.cfg.get("transport"), dict):
            requested_engine_raw = self.cfg["transport"].get("engine")
        requested_engine = normalize_transport_engine(requested_engine_raw or "auto")
        cached = self._load_cached_transport_results()
        if cached is not None:
            cached_meta = cached.setdefault("meta", {}) if isinstance(cached, dict) else {}
            if isinstance(cached_meta, dict):
                cached_meta.setdefault("transport_engine_requested", requested_engine)
                cached_meta.setdefault("transport_engine_resolved", self._transport_engine())
            print(
                "[Orchestrator] Reusing cached transport result: "
                f"{self._transport_result_file()}"
            )
            self._state["outputs"]["transport_results"] = "computed"
            self._state["outputs"]["transport_job"] = {
                "status": "REUSED",
                "backend": "cached_result",
                "result_file": str(self._transport_result_file()),
            }
            self._save_checkpoint()
            return cached

        backend = str(self.cfg.get("transport_backend", "qsub")).strip().lower()
        engine = self._transport_engine()
        strict_qsub = bool(self.cfg.get("transport_strict_qsub", True))
        job_meta: dict[str, Any] | None = None

        if engine == "rgf":
            if backend != "qsub":
                raise RuntimeError(
                    "transport_engine='rgf' is currently executable only with "
                    "transport.backend='qsub'."
                )
            results, job_meta = self._stage_transport_rgf_qsub(hr_dat, label="primary")
            self._state["outputs"]["transport_results"] = "computed"
            if job_meta is not None:
                self._state["outputs"]["transport_job"] = job_meta
                if job_meta.get("job_id"):
                    self._state["last_job_id"] = job_meta.get("job_id")
            meta_payload = results.setdefault("meta", {}) if isinstance(results, dict) else {}
            if isinstance(meta_payload, dict):
                meta_payload["transport_engine_requested"] = requested_engine
                meta_payload["transport_engine_resolved"] = engine
            self._save_checkpoint()
            return results

        if backend == "qsub":
            try:
                results, job_meta = self._stage_transport_qsub(hr_dat, label="primary")
            except Exception as exc:
                if strict_qsub:
                    raise
                print(
                    "[Orchestrator] transport qsub failed; "
                    f"falling back to local: {type(exc).__name__}: {exc}"
                )
                results = self._stage_transport_local(hr_dat, label="primary")
                job_meta = {
                    "status": "LOCAL_FALLBACK",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "backend": "local",
                }
        elif backend == "local":
            results = self._stage_transport_local(hr_dat, label="primary")
            job_meta = {"status": "LOCAL", "backend": "local"}
        else:
            raise ValueError(
                f"Unknown transport_backend={backend!r}. Use 'qsub' or 'local'."
            )

        self._state["outputs"]["transport_results"] = "computed"
        if job_meta is not None:
            self._state["outputs"]["transport_job"] = job_meta
            if job_meta.get("job_id"):
                self._state["last_job_id"] = job_meta.get("job_id")
        meta_payload = results.setdefault("meta", {}) if isinstance(results, dict) else {}
        if isinstance(meta_payload, dict):
            meta_payload["transport_engine_requested"] = requested_engine
            meta_payload["transport_engine_resolved"] = engine
        self._save_checkpoint()
        return results

    def _stage_transport_local(self, hr_dat: Path, *, label: str = "primary") -> dict:
        from wtec.workflow.transport_pipeline import TransportPipeline

        engine = self._transport_engine()
        if engine != "kwant":
            raise ValueError(
                f"Unsupported local transport engine={engine!r}. Use 'kwant' or 'auto'."
            )
        tp = TransportPipeline(
            hr_dat,
            thicknesses=self.cfg.get("thicknesses"),
            disorder_strengths=self.cfg.get("disorder_strengths"),
            n_ensemble=self.cfg.get("n_ensemble", 50),
            energy=self.cfg.get("fermi_shift_eV", 0.0),
            n_jobs=self.cfg.get("n_jobs", 4),
            mfp_n_layers_z=self.cfg.get("mfp_n_layers_z", 10),
            mfp_lengths=self.cfg.get("mfp_lengths"),
            lead_onsite_eV=self.cfg.get("lead_onsite_eV", 0.0),
            base_seed=self.cfg.get("base_seed", 0),
            lead_axis=self.cfg.get("transport_axis", "x"),
            thickness_axis=self.cfg.get("thickness_axis", "z"),
            n_layers_x=self.cfg.get("transport_n_layers_x", 4),
            n_layers_y=self.cfg.get("transport_n_layers_y", 4),
            carrier_density_m3=self.cfg.get("carrier_density_m3"),
            fermi_velocity_m_per_s=self.cfg.get("fermi_velocity_m_per_s"),
            transport_engine=engine,
        )
        results = tp.run_full()
        run_dir = Path(self.cfg.get("run_dir", ".")).resolve()
        out_dir = run_dir / "transport" / str(label)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "transport_result.json"
        out_path.write_text(
            json.dumps({"transport_results": _jsonable(results)}, indent=2)
        )
        return results

    def _stage_transport_qsub(self, hr_dat: Path, *, label: str = "primary") -> tuple[dict, dict]:
        from wtec.cluster.mpi import MPIConfig, build_command
        from wtec.cluster.pbs import PBSJobConfig, generate_script
        from wtec.cluster.ssh import open_ssh
        from wtec.cluster.submit import JobManager
        from wtec.config.cluster import ClusterConfig

        cfg = ClusterConfig.from_env()
        run_dir = Path(self.cfg.get("run_dir", ".")).resolve()
        transport_dir = run_dir / "transport" / str(label)
        transport_dir.mkdir(parents=True, exist_ok=True)

        payload_name = "transport_payload.json"
        result_name = "transport_result.json"
        progress_name = "transport_progress.jsonl"
        cert_name = "transport_runtime_cert.json"
        payload_path = transport_dir / payload_name
        win_path = hr_dat.with_name(f"{self.cfg.get('material', 'material')}.win")
        logging_cfg = self.cfg.get("logging", {})
        if not isinstance(logging_cfg, dict):
            logging_cfg = {}
        runtime_log_detail = str(
            self.cfg.get(
                "runtime_logging_detail",
                logging_cfg.get("detail", "per_ensemble"),
            )
        ).strip().lower() or "per_ensemble"
        runtime_heartbeat = int(
            self.cfg.get(
                "runtime_logging_heartbeat_seconds",
                logging_cfg.get("heartbeat_seconds", 20),
            )
        )
        stream_from_start = bool(
            self.cfg.get(
                "runtime_stream_from_start",
                logging_cfg.get("stream_from_start", True),
            )
        )
        retrieve_on_failure = bool(
            self.cfg.get(
                "runtime_retrieve_on_failure",
                logging_cfg.get("retrieve_on_failure", True),
            )
        )
        transport_engine = self._transport_engine()
        if transport_engine != "kwant":
            raise RuntimeError(
                f"Unsupported qsub transport engine={transport_engine!r}. "
                "Only kwant-backed transport is executable right now."
            )
        require_mumps = bool(self.cfg.get("transport_require_mumps", True))
        kwant_mode = str(self.cfg.get("transport_kwant_mode", "auto")).strip().lower() or "auto"
        kwant_task_workers = int(self.cfg.get("transport_kwant_task_workers", 0))
        mumps_nrhs = self.cfg.get("transport_mumps_nrhs")
        mumps_ordering = self.cfg.get("transport_mumps_ordering")
        mumps_sparse_rhs = self.cfg.get("transport_mumps_sparse_rhs")
        payload = {
            "hr_dat_path": hr_dat.name,
            "win_path": win_path.name if win_path.exists() else None,
            "thicknesses": self.cfg.get("thicknesses"),
            "disorder_strengths": self.cfg.get("disorder_strengths"),
            "n_ensemble": int(self.cfg.get("n_ensemble", 50)),
            "energy": float(self.cfg.get("fermi_shift_eV", 0.0)),
            "n_jobs": int(self.cfg.get("n_jobs", 1)),
            "mfp_n_layers_z": int(self.cfg.get("mfp_n_layers_z", 10)),
            "mfp_lengths": self.cfg.get("mfp_lengths"),
            "lead_onsite_eV": float(self.cfg.get("lead_onsite_eV", 0.0)),
            "base_seed": int(self.cfg.get("base_seed", 0)),
            "lead_axis": str(self.cfg.get("transport_axis", "x")),
            "thickness_axis": str(self.cfg.get("thickness_axis", "z")),
            "n_layers_x": int(self.cfg.get("transport_n_layers_x", 4)),
            "n_layers_y": int(self.cfg.get("transport_n_layers_y", 4)),
            "carrier_density_m3": self.cfg.get("carrier_density_m3"),
            "fermi_velocity_m_per_s": self.cfg.get("fermi_velocity_m_per_s"),
            "progress_file": progress_name,
            "runtime_cert_file": cert_name,
            "logging_detail": runtime_log_detail,
            "heartbeat_seconds": max(5, runtime_heartbeat),
            "transport_engine": transport_engine,
            "require_mumps": require_mumps,
            "kwant_mode": kwant_mode,
            "kwant_task_workers": max(0, kwant_task_workers),
            "mumps_nrhs": (int(mumps_nrhs) if mumps_nrhs is not None else None),
            "mumps_ordering": (
                str(mumps_ordering).strip() if mumps_ordering is not None else None
            ),
            "mumps_sparse_rhs": (
                bool(mumps_sparse_rhs) if mumps_sparse_rhs is not None else None
            ),
        }

        worker_zip = self._worker_source_zip(transport_dir)
        stage_files: list[Path] = [worker_zip, hr_dat]
        if win_path.exists():
            stage_files.append(win_path)

        run_name = str(self.cfg.get("name", "run")).strip() or "run"
        remote_dir = f"{self._remote_run_base(cfg)}/transport/{str(label)}"
        walltime = str(self.cfg.get("transport_walltime", "01:00:00"))
        python_exe = str(
            self.cfg.get(
                "transport_cluster_python_exe",
                self.cfg.get("topology", {}).get("cluster_python_exe", "python3"),
            )
        ).strip() or "python3"

        with open_ssh(cfg) as ssh:
            jm = JobManager(ssh)
            queue_used = jm.resolve_queue(
                cfg.pbs_queue,
                fallback_order=cfg.pbs_queue_priority,
            )
            cores_per_node = cfg.cores_for_queue(queue_used)
            n_nodes = int(self.cfg.get("n_nodes", 1))
            total_cores = max(1, n_nodes * cores_per_node)
            enforce_1x64 = bool(self.cfg.get("transport_kwant_enforce_1x64", True))
            if enforce_1x64:
                if n_nodes != 1:
                    raise RuntimeError(
                        "transport_kwant_enforce_1x64 requires n_nodes=1. "
                        f"Current n_nodes={n_nodes}."
                    )
                if total_cores < 64:
                    raise RuntimeError(
                        "transport_kwant_enforce_1x64 requires at least 64 allocated cores "
                        f"(queue={queue_used}, nodes={n_nodes}, ppn={cores_per_node}, total={total_cores})."
                    )
                mpi_np = 1
                threads = 64
            else:
                raw_mpi_np = int(self.cfg.get("transport_mpi_np", 0))
                raw_threads = int(self.cfg.get("transport_threads", 0))
                # Avoid CPU oversubscription by default.
                # Auto mode (0/0): use one MPI rank and all available threads.
                if raw_mpi_np <= 0 and raw_threads <= 0:
                    mpi_np = 1
                    threads = total_cores
                elif raw_mpi_np <= 0:
                    threads = max(1, min(raw_threads, total_cores))
                    mpi_np = max(1, total_cores // threads)
                elif raw_threads <= 0:
                    mpi_np = max(1, min(raw_mpi_np, total_cores))
                    threads = max(1, total_cores // mpi_np)
                else:
                    mpi_np = max(1, min(raw_mpi_np, total_cores))
                    threads = max(1, min(raw_threads, total_cores))

                if mpi_np * threads > total_cores:
                    threads = max(1, total_cores // mpi_np)

            payload["expected_mpi_np"] = int(mpi_np)
            payload["expected_threads"] = int(threads)
            payload_path.write_text(json.dumps(payload, indent=2))
            stage_files.insert(0, payload_path)

            print(
                "[Orchestrator] Transport qsub resources: "
                f"queue={queue_used}, nodes={n_nodes}, ppn={cores_per_node}, "
                f"np={mpi_np}, threads={threads}"
            )
            cert_payload = {
                "queue": queue_used,
                "n_nodes": int(n_nodes),
                "ppn": int(cores_per_node),
                "total_cores": int(total_cores),
                "expected_mpi_np": int(mpi_np),
                "expected_threads": int(threads),
                "transport_engine": transport_engine,
                "kwant_enforce_1x64": bool(enforce_1x64),
                "require_mumps": bool(require_mumps),
                "kwant_mode": kwant_mode,
                "kwant_task_workers": int(max(0, kwant_task_workers)),
                "runtime_logging_detail": runtime_log_detail,
                "runtime_heartbeat_seconds": int(max(5, runtime_heartbeat)),
            }
            (transport_dir / cert_name).write_text(json.dumps(cert_payload, indent=2))

            worker_python = (
                f"env PYTHONPATH=$PWD/{worker_zip.name}:$PYTHONPATH {python_exe}"
            )
            threaded_single_rank = int(mpi_np) == 1 and int(threads) > 1
            cmd = build_command(
                worker_python,
                mpi=MPIConfig(
                    n_cores=mpi_np,
                    n_pool=1,
                    bind_to="none" if threaded_single_rank else "core",
                ),
                extra_args=f"-m wtec.workflow.transport_worker {payload_name} {result_name}",
            )
            cmd = self._thread_exports(
                cmd,
                threads=threads,
                full_node_threading=threaded_single_rank,
            )
            job_name = f"tr_{str(label)[:5]}_{run_name}"[:15]
            script = generate_script(
                PBSJobConfig(
                    job_name=job_name,
                    n_nodes=n_nodes,
                    n_cores_per_node=cores_per_node,
                    mpi_procs_per_node=max(1, int(mpi_np // max(1, n_nodes))),
                    omp_threads=max(1, int(threads)),
                    walltime=walltime,
                    queue=queue_used,
                    work_dir=remote_dir,
                    modules=cfg.modules,
                    env_vars={},
                ),
                [cmd],
            )
            local_script_path = transport_dir / f"transport_worker_{str(label)}.pbs"
            local_script_path.write_text(script)

            meta = jm.submit_and_wait(
                script,
                remote_dir=remote_dir,
                local_dir=transport_dir,
                retrieve_patterns=[
                    result_name,
                    progress_name,
                    cert_name,
                    "wtec_job.log",
                    f"{job_name}.log",
                    "*.out",
                ],
                script_name=f"transport_worker_{str(label)}.pbs",
                stage_files=stage_files,
                expected_local_outputs=[result_name],
                queue_used=queue_used,
                poll_interval=int(self.cfg.get("_runtime_log_poll_interval", 5)),
                live_log=bool(self.cfg.get("_runtime_live_log", True)),
                live_files=["wtec_job.log", f"{job_name}.log", progress_name, result_name],
                live_retrieve_patterns=[progress_name, "wtec_job.log", f"{job_name}.log", result_name],
                live_retrieve_interval_seconds=int(self.cfg.get("_runtime_log_poll_interval", 5)),
                stale_log_seconds=int(self.cfg.get("_runtime_stale_log_seconds", 300)),
                retrieve_on_failure=retrieve_on_failure,
                stream_from_start=stream_from_start,
            )

        out_payload = json.loads((transport_dir / result_name).read_text())
        results = out_payload.get("transport_results", {})
        if not isinstance(results, dict):
            raise RuntimeError("Invalid transport worker result payload.")
        return self._normalize_transport_results(results), meta

    def _stage_transport_rgf_qsub(self, hr_dat: Path, *, label: str = "primary") -> tuple[dict, dict]:
        from wtec.cluster.mpi import MPIConfig, build_command
        from wtec.cluster.pbs import PBSJobConfig, generate_script
        from wtec.cluster.ssh import open_ssh
        from wtec.cluster.submit import JobManager
        from wtec.config.cluster import ClusterConfig
        from wtec.transport.rgf_postprocess import (
            convert_rgf_raw_to_transport_results,
            load_rgf_raw_result,
        )

        cached = self._load_cached_transport_results(label=label)
        if cached is not None:
            result_file = self._transport_result_file(label=label)
            print(f"[Orchestrator] Reusing cached transport result: {result_file}")
            return cached, {
                "status": "REUSED",
                "backend": "cached_result",
                "result_file": str(result_file),
                "label": str(label),
            }

        cfg = ClusterConfig.from_env()
        init_state = _load_init_state()
        rgf_cluster = (
            init_state.get("rgf", {}).get("cluster", {})
            if isinstance(init_state.get("rgf"), dict)
            else {}
        )
        if not isinstance(rgf_cluster, dict) or not bool(rgf_cluster.get("ready")):
            raise RuntimeError(
                "RGF cluster router is not ready. Re-run `wtec init` first."
            )
        rgf_binary = str(rgf_cluster.get("binary_path") or "").strip()
        if not rgf_binary:
            raise RuntimeError(
                "RGF cluster router state is missing binary_path. Re-run `wtec init`."
            )

        run_dir = Path(self.cfg.get("run_dir", ".")).resolve()
        transport_dir = run_dir / "transport" / str(label)
        transport_dir.mkdir(parents=True, exist_ok=True)
        logging_cfg = self.cfg.get("logging", {})
        if not isinstance(logging_cfg, dict):
            logging_cfg = {}
        runtime_log_detail = str(
            self.cfg.get(
                "runtime_logging_detail",
                logging_cfg.get("detail", "per_ensemble"),
            )
        ).strip().lower() or "per_ensemble"
        runtime_heartbeat = int(
            self.cfg.get(
                "runtime_logging_heartbeat_seconds",
                logging_cfg.get("heartbeat_seconds", 20),
            )
        )
        stream_from_start = bool(
            self.cfg.get(
                "runtime_stream_from_start",
                logging_cfg.get("stream_from_start", True),
            )
        )
        retrieve_on_failure = bool(
            self.cfg.get(
                "runtime_retrieve_on_failure",
                logging_cfg.get("retrieve_on_failure", True),
            )
        )

        run_name = str(self.cfg.get("name", "run")).strip() or "run"
        attempt_tag = self._transport_attempt_tag(str(label))
        payload_name = self._attempt_artifact_name("transport_payload.json", attempt_tag)
        raw_result_name = self._attempt_artifact_name("transport_rgf_raw.json", attempt_tag)
        result_name = "transport_result.json"
        cert_name = "transport_runtime_cert.json"
        progress_name = self._attempt_artifact_name("transport_progress.jsonl", attempt_tag)
        runtime_log_name = self._attempt_artifact_name("wtec_job.log", attempt_tag)
        payload_path = transport_dir / payload_name
        raw_result_path = transport_dir / raw_result_name
        final_result_path = transport_dir / result_name
        cert_path = transport_dir / cert_name
        win_path = hr_dat.with_name(f"{self.cfg.get('material', 'material')}.win")

        thicknesses_raw = self.cfg.get("thicknesses")
        if thicknesses_raw is None:
            thicknesses_raw = list(range(2, 32, 2))
        disorder_raw = self.cfg.get("disorder_strengths")
        if disorder_raw is None:
            disorder_raw = [0.0]
        mfp_lengths_raw = self.cfg.get("mfp_lengths")
        if mfp_lengths_raw is None:
            mfp_lengths_raw = list(range(5, 105, 5))
        thicknesses = [int(v) for v in thicknesses_raw]
        disorder_strengths = [float(v) for v in disorder_raw]
        mfp_lengths = [int(v) for v in mfp_lengths_raw]
        lead_axis = normalize_axis(self.cfg.get("transport_axis", "x"), field_name="transport_axis")
        thickness_axis = normalize_axis(
            self.cfg.get("thickness_axis", "z"),
            field_name="thickness_axis",
        )
        rgf_mode = normalize_rgf_mode(self.cfg.get("transport_rgf_mode", "periodic_transverse"))
        rgf_periodic_axis = normalize_axis(
            self.cfg.get("transport_rgf_periodic_axis", "y"),
            field_name="transport_rgf_periodic_axis",
        )
        build_probe = rgf_cluster.get("probe", {}) if isinstance(rgf_cluster, dict) else {}
        build_blas_backend = str(
            build_probe.get("blas_backend")
            or build_probe.get("build_blas_backend")
            or ""
        ).strip().lower()
        threaded_single_point_backend = build_blas_backend not in {"none", "serial"}
        mfp_n_layers_z = int(self.cfg.get("mfp_n_layers_z", 10))
        n_layers_x = int(self.cfg.get("transport_n_layers_x", 4))
        n_layers_y = int(self.cfg.get("transport_n_layers_y", 4))
        sigma_backend = str(
            self.cfg.get("transport_rgf_full_finite_sigma_backend", "native")
        ).strip().lower() or "native"
        internal_sigma_mode = str(
            self.cfg.get("_transport_rgf_internal_sigma_mode", sigma_backend)
        ).strip().lower() or sigma_backend
        rgf_parallel_policy = normalize_rgf_parallel_policy(
            self.cfg.get("transport_rgf_parallel_policy", "auto")
        )
        rgf_threads_per_rank_raw = self.cfg.get(
            "transport_rgf_threads_per_rank",
            self.cfg.get("transport_threads", 0),
        )
        if isinstance(rgf_threads_per_rank_raw, str) and rgf_threads_per_rank_raw.strip().lower() == "auto":
            rgf_threads_per_rank: int | str = "auto"
        else:
            rgf_threads_per_rank = int(rgf_threads_per_rank_raw or 0)
        rgf_blas_backend = normalize_rgf_blas_backend(
            self.cfg.get("transport_rgf_blas_backend", "auto")
        )
        rgf_validate_against_raw = self.cfg.get("transport_rgf_validate_against")
        if rgf_validate_against_raw is None:
            rgf_validate_against_raw = "kwant" if sigma_backend == "kwant_exact" else "none"
        rgf_validate_against = normalize_rgf_validate_against(rgf_validate_against_raw)
        canonical_input = canonicalize_rgf_inputs(
            hr_dat_path=hr_dat,
            win_path=(win_path if win_path.exists() else None),
            lead_axis=lead_axis,
            thickness_axis=thickness_axis,
            mode=rgf_mode,
            periodic_axis=rgf_periodic_axis,
            out_dir=transport_dir / "canonical_rgf",
            seedname=f"{str(label)}_rgf",
        )
        canonical_hr_path = Path(canonical_input.hr_dat_path)
        canonical_win_path = (
            Path(canonical_input.win_path)
            if canonical_input.win_path is not None
            else None
        )

        payload = {
            "hr_dat_path": canonical_hr_path.name,
            "win_path": canonical_win_path.name if canonical_win_path is not None else None,
            "queue": "",
            "thicknesses": thicknesses,
            "disorder_strengths": disorder_strengths,
            "n_ensemble": int(self.cfg.get("n_ensemble", 1)),
            "base_seed": int(self.cfg.get("base_seed", 0)),
            "energy": float(self.cfg.get("fermi_shift_eV", 0.0)),
            "eta": resolve_transport_rgf_eta(
                self.cfg,
                internal_sigma_mode=internal_sigma_mode,
            ),
            "mfp_n_layers_z": mfp_n_layers_z,
            "mfp_lengths": mfp_lengths,
            "lead_axis": "x",
            "thickness_axis": "z",
            "n_layers_x": n_layers_x,
            "n_layers_y": n_layers_y,
            "transport_engine": "rgf",
            "transport_rgf_mode": rgf_mode,
            "transport_rgf_periodic_axis": "y",
            "parallel_policy": rgf_parallel_policy,
            "blas_backend_requested": rgf_blas_backend,
            "validate_against": rgf_validate_against,
            "canonicalized_input": bool(canonical_input.was_canonicalized),
            "original_lead_axis": lead_axis,
            "original_thickness_axis": thickness_axis,
            "original_periodic_axis": rgf_periodic_axis,
            "axis_permutation": list(canonical_input.permutation),
            "progress_file": progress_name,
            "logging_detail": runtime_log_detail,
            "heartbeat_seconds": max(5, runtime_heartbeat),
        }

        stage_files: list[Path] = [payload_path, canonical_hr_path]
        if canonical_win_path is not None and canonical_win_path.exists():
            stage_files.append(canonical_win_path)

        remote_dir = f"{self._remote_run_base(cfg)}/transport/{str(label)}"
        walltime = str(self.cfg.get("transport_walltime", "01:00:00"))
        python_exe = str(
            self.cfg.get(
                "transport_cluster_python_exe",
                self.cfg.get("topology", {}).get("cluster_python_exe", "python3"),
            )
        ).strip() or "python3"
        sigma_manifest_name: str | None = None
        sigma_left_name: str | None = None
        sigma_right_name: str | None = None
        worker_zip: Path | None = None
        if internal_sigma_mode == "kwant_exact":
            if rgf_mode != "full_finite":
                raise RuntimeError(
                    "_transport_rgf_internal_sigma_mode='kwant_exact' currently requires "
                    "transport_rgf_mode='full_finite'."
                )
            if len(thicknesses) != 1 or mfp_lengths:
                raise RuntimeError(
                    "_transport_rgf_internal_sigma_mode='kwant_exact' is currently supported only "
                    "for single-thickness transport jobs without MFP sweeps."
                )
            worker_zip = self._worker_source_zip(transport_dir)
            sigma_manifest_name = "sigma_manifest.json"
            sigma_left_name = "sigma_left.bin"
            sigma_right_name = "sigma_right.bin"
            payload["sigma_left_path"] = sigma_left_name
            payload["sigma_right_path"] = sigma_right_name
            stage_files.append(worker_zip)

        with open_ssh(cfg) as ssh:
            jm = JobManager(ssh)
            queue_used = jm.resolve_queue(
                cfg.pbs_queue,
                fallback_order=cfg.pbs_queue_priority,
            )
            payload["queue"] = queue_used
            n_nodes = int(self.cfg.get("n_nodes", 1))
            cores_per_node = cfg.cores_for_queue(queue_used)
            total_cores = max(1, n_nodes * cores_per_node)
            probe_nz = max([mfp_n_layers_z, *thicknesses])
            rgf_summary = rgf_preflight_summary(
                hr_dat_path=canonical_hr_path,
                lead_axis="x",
                n_layers_x=n_layers_x,
                n_layers_y=n_layers_y,
                n_layers_z=probe_nz,
                mode=rgf_mode,
                periodic_axis="y",
                thicknesses=thicknesses,
                mfp_lengths=mfp_lengths,
                disorder_strengths=disorder_strengths,
                n_ensemble=int(self.cfg.get("n_ensemble", 1)),
                queue_cores=total_cores,
            )
            alignment_issues = (
                rgf_phase1_alignment_issues(
                    hr_dat_path=canonical_hr_path,
                    lead_axis="x",
                    n_layers_x=n_layers_x,
                    n_layers_y=n_layers_y,
                    thicknesses=thicknesses,
                    mfp_n_layers_z=mfp_n_layers_z,
                    mfp_lengths=mfp_lengths,
                    mode=rgf_mode,
                    periodic_axis="y",
                )
                if rgf_mode == "periodic_transverse"
                else []
            )
            if alignment_issues:
                raise RuntimeError(
                    "Current native RGF phase requires full principal layers for every transport task.\n"
                    + "\n".join(alignment_issues)
                )
            execution_plan = rgf_plan_execution(
                mode=rgf_mode,
                queue_cores=total_cores,
                safe_rank_cap=int(rgf_summary.safe_rank_cap),
                n_work_units=int(rgf_summary.transport_task_count),
                requested_mpi_np=int(self.cfg.get("transport_mpi_np", 0)),
                requested_threads_per_rank=rgf_threads_per_rank,
                parallel_policy=rgf_parallel_policy,
                threaded_single_point_backend=threaded_single_point_backend,
            )
            mpi_np = int(execution_plan.mpi_np)
            mpi_ppn = max(1, min(cores_per_node, (mpi_np + n_nodes - 1) // max(1, n_nodes)))
            max_threads_per_rank = max(1, cores_per_node // max(1, mpi_ppn))
            omp_threads = min(max_threads_per_rank, int(execution_plan.omp_threads))
            payload["expected_mpi_np"] = mpi_np
            payload["expected_omp_threads"] = int(omp_threads)
            payload["parallel_policy_resolved"] = str(execution_plan.parallel_policy)
            payload["threaded_single_point_backend"] = bool(threaded_single_point_backend)
            payload["task_shape"] = _jsonable(rgf_summary.task_shape)
            payload_path.write_text(json.dumps(payload, indent=2))

            rc, _, _ = ssh.run(f"test -x {shlex.quote(rgf_binary)}", check=False)
            if rc != 0:
                raise RuntimeError(
                    f"Prepared RGF binary is not executable on cluster: {rgf_binary}"
                )
            jm.ensure_remote_commands(["mpirun"], modules=cfg.modules, bin_dirs=cfg.bin_dirs)

            cmd = build_command(
                shlex.quote(rgf_binary),
                mpi=MPIConfig(
                    n_cores=mpi_np,
                    bind_to=("none" if mpi_np == 1 else "core"),
                ),
                extra_args=f"{shlex.quote(payload_name)} {shlex.quote(raw_result_name)}",
            )
            cmd = self._thread_exports(
                cmd,
                threads=int(omp_threads),
                full_node_threading=bool(mpi_np == 1 and int(omp_threads) > 1),
            )
            commands: list[str] = []
            if internal_sigma_mode == "kwant_exact":
                assert worker_zip is not None
                assert sigma_manifest_name is not None
                assert sigma_left_name is not None
                assert sigma_right_name is not None
                # The exact-sigma precompute is a single-rank dense linear-algebra
                # phase. Let it see the full node cpuset instead of leaving most
                # of the PBS allocation idle on one core.
                sigma_threads = max(1, int(cores_per_node))
                sigma_cmd = build_command(
                    f"env PYTHONPATH=$PWD/{worker_zip.name}:$PYTHONPATH {python_exe}",
                    mpi=MPIConfig(n_cores=1, bind_to="none"),
                    extra_args=(
                        "-m wtec.transport.kwant_sigma_extract "
                        f"--hr-path {shlex.quote(canonical_hr_path.name)} "
                        f"--length-uc {int(n_layers_x)} "
                        f"--width-uc {int(n_layers_y)} "
                        f"--thickness-uc {int(thicknesses[0])} "
                        f"--energy-ev {float(payload['energy']):.16g} "
                        f"--eta-ev {float(payload['eta']):.16g} "
                        "--layout full_finite_principal "
                        f"--out-dir {shlex.quote('.')}"
                    ),
                )
                sigma_cmd = self._thread_exports(
                    sigma_cmd,
                    threads=sigma_threads,
                    full_node_threading=bool(sigma_threads > 1),
                )
                commands.append('echo "[wtec][runtime] sigma_extract_start $(date -Is)"')
                commands.append(sigma_cmd)
                commands.append('echo "[wtec][runtime] sigma_extract_done $(date -Is)"')
            commands.append('echo "[wtec][runtime] rgf_runner_start $(date -Is)"')
            commands.append(cmd)
            commands.append('echo "[wtec][runtime] rgf_runner_done $(date -Is)"')
            job_name = f"rgf_{str(label)[:5]}_{run_name}"[:15]
            stdout_log_name = self._attempt_artifact_name(f"{job_name}.log", attempt_tag)
            script = generate_script(
                PBSJobConfig(
                    job_name=job_name,
                    n_nodes=n_nodes,
                    n_cores_per_node=cores_per_node,
                    mpi_procs_per_node=mpi_ppn,
                    omp_threads=int(omp_threads),
                    walltime=walltime,
                    queue=queue_used,
                    work_dir=remote_dir,
                    modules=cfg.modules,
                    env_vars={},
                    stdout_path=f"{remote_dir.rstrip('/')}/{stdout_log_name}",
                    runtime_log_path=f"{remote_dir.rstrip('/')}/{runtime_log_name}",
                ),
                commands,
            )
            local_script_name = self._attempt_artifact_name(
                f"transport_rgf_{str(label)}.pbs",
                attempt_tag,
            )
            local_script_path = transport_dir / local_script_name
            local_script_path.write_text(script)

            meta = jm.submit_and_wait(
                script,
                remote_dir=remote_dir,
                local_dir=transport_dir,
                retrieve_patterns=[
                    raw_result_name,
                    progress_name,
                    runtime_log_name,
                    stdout_log_name,
                    *( [sigma_manifest_name, sigma_left_name, sigma_right_name] if sigma_manifest_name else [] ),
                    "*.out",
                ],
                script_name=local_script_name,
                stage_files=stage_files,
                expected_local_outputs=[raw_result_name],
                queue_used=queue_used,
                poll_interval=int(self.cfg.get("_runtime_log_poll_interval", 5)),
                live_log=bool(self.cfg.get("_runtime_live_log", True)),
                live_files=[runtime_log_name, stdout_log_name, progress_name, raw_result_name],
                live_retrieve_patterns=[
                    progress_name,
                    runtime_log_name,
                    stdout_log_name,
                    raw_result_name,
                    *( [sigma_manifest_name, sigma_left_name, sigma_right_name] if sigma_manifest_name else [] ),
                ],
                live_retrieve_interval_seconds=int(self.cfg.get("_runtime_log_poll_interval", 5)),
                stale_log_seconds=int(self.cfg.get("_runtime_stale_log_seconds", 300)),
                retrieve_on_failure=retrieve_on_failure,
                stream_from_start=stream_from_start,
            )
        attempt_dir = self._archive_transport_attempt(
            transport_dir,
            attempt_key=str((meta or {}).get("job_id") or int(time.time())),
            files=[
                payload_path,
                local_script_path,
                raw_result_path,
                *( [transport_dir / sigma_manifest_name, transport_dir / sigma_left_name, transport_dir / sigma_right_name] if sigma_manifest_name else [] ),
                transport_dir / progress_name,
                transport_dir / runtime_log_name,
                transport_dir / stdout_log_name,
            ],
        )

        raw_payload, runtime_cert = load_rgf_raw_result(raw_result_path)
        runtime_cert = dict(runtime_cert)
        runtime_cert["omp_threads"] = int(omp_threads)
        runtime_cert["parallel_policy"] = str(execution_plan.parallel_policy)
        runtime_cert["threaded_single_point_backend"] = bool(threaded_single_point_backend)
        runtime_cert["blas_backend"] = str(
            build_probe.get("blas_backend")
            or build_probe.get("build_blas_backend")
            or rgf_blas_backend
        )
        runtime_cert["workspace_bytes"] = int(rgf_summary.per_rank_bytes) * int(mpi_np)
        runtime_cert["max_dense_dim"] = int(
            runtime_cert.get("max_superslice_dim", rgf_summary.superslice_dim)
        )
        runtime_cert["task_shape"] = _jsonable(rgf_summary.task_shape)
        runtime_cert["build_env"] = _jsonable(
            build_probe.get("build_env", {}) if isinstance(build_probe, dict) else {}
        )
        runtime_cert["full_finite_sigma_source"] = str(internal_sigma_mode)
        runtime_cert["validate_against"] = str(rgf_validate_against)
        runtime_cert["canonicalized_input"] = bool(canonical_input.was_canonicalized)
        runtime_cert["axis_mapping"] = {
            "lead_axis_requested": lead_axis,
            "thickness_axis_requested": thickness_axis,
            "periodic_axis_requested": rgf_periodic_axis,
            "lead_axis_native": "x",
            "thickness_axis_native": "z",
            "periodic_axis_native": "y",
            "permutation": list(canonical_input.permutation),
        }
        results = convert_rgf_raw_to_transport_results(
            raw_payload,
            win_path=canonical_win_path,
            disorder_key=float(disorder_strengths[0] if disorder_strengths else 0.0),
            carrier_density_m3=self.cfg.get("carrier_density_m3"),
            fermi_velocity_m_per_s=self.cfg.get("fermi_velocity_m_per_s"),
        )
        meta_payload = results.setdefault("meta", {})
        if isinstance(meta_payload, dict):
            meta_payload["rgf_full_finite_sigma_backend"] = str(
                self.cfg.get("transport_rgf_full_finite_sigma_backend", "native")
            ).strip().lower() or "native"
            meta_payload["rgf_full_finite_sigma_source"] = str(internal_sigma_mode)
            kwant_script_cfg = str(
                self.cfg.get("transport_rgf_full_finite_kwant_script", "")
            ).strip()
            if kwant_script_cfg:
                meta_payload["rgf_full_finite_kwant_script"] = kwant_script_cfg
            meta_payload["rgf_parallel_policy"] = str(execution_plan.parallel_policy)
            meta_payload["rgf_blas_backend"] = str(runtime_cert.get("blas_backend", rgf_blas_backend))
            meta_payload["rgf_validate_against"] = str(rgf_validate_against)
            meta_payload["rgf_axis_mapping"] = _jsonable(runtime_cert.get("axis_mapping", {}))
            meta_payload["rgf_runtime_cert"] = _jsonable(runtime_cert)
            meta_payload["rgf_preflight"] = {
                "n_orb": int(rgf_summary.n_orb),
                "principal_layer_width": int(rgf_summary.principal_layer_width),
                "superslice_dim": int(rgf_summary.superslice_dim),
                "per_rank_bytes": int(rgf_summary.per_rank_bytes),
                "transport_task_count": int(rgf_summary.transport_task_count),
                "task_shape": _jsonable(rgf_summary.task_shape),
                "queue_cores": int(rgf_summary.queue_cores),
                "safe_rank_cap": int(rgf_summary.safe_rank_cap),
                "mpi_np": int(mpi_np),
                "omp_threads": int(omp_threads),
                "mode": str(rgf_summary.mode),
                "periodic_axis": rgf_summary.periodic_axis,
            }

        cert_path.write_text(json.dumps(_jsonable(runtime_cert), indent=2))
        final_result_path.write_text(
            json.dumps(
                {
                    "transport_results": _jsonable(results),
                    "runtime_cert": _jsonable(runtime_cert),
                    "transport_results_raw": _jsonable(raw_payload),
                },
                indent=2,
            )
        )
        shutil.copy2(cert_path, attempt_dir / cert_path.name)
        shutil.copy2(final_result_path, attempt_dir / final_result_path.name)
        if isinstance(meta, dict):
            meta["attempt_dir"] = str(attempt_dir)
            meta["attempt_tag"] = str(attempt_tag)
            meta["attempt_artifacts"] = {
                "payload": payload_name,
                "raw_result": raw_result_name,
                "progress": progress_name,
                "runtime_log": runtime_log_name,
                "stdout_log": stdout_log_name,
                "script": local_script_name,
            }
        if isinstance(meta_payload, dict):
            meta_payload["rgf_attempt_dir"] = str(attempt_dir)
            meta_payload["rgf_attempt_tag"] = str(attempt_tag)
            meta_payload["rgf_attempt_artifacts"] = _jsonable(
                {
                    "payload": payload_name,
                    "raw_result": raw_result_name,
                    "progress": progress_name,
                    "runtime_log": runtime_log_name,
                    "stdout_log": stdout_log_name,
                    "script": local_script_name,
                }
            )
        return self._normalize_transport_results(results), meta

    @staticmethod
    def _thread_exports(
        command: str,
        *,
        threads: int,
        full_node_threading: bool = False,
    ) -> str:
        t = max(1, int(threads))
        exports = (
            f"export OMP_NUM_THREADS={t}; "
            f"export MKL_NUM_THREADS={t}; "
            f"export OPENBLAS_NUM_THREADS={t}; "
            f"export NUMEXPR_NUM_THREADS={t}; "
        )
        if full_node_threading:
            # Single-rank threaded jobs should see the full PBS cpuset instead of
            # inheriting MPI rank pinning to a single socket/core subset.
            exports += (
                "export OMP_PROC_BIND=spread; "
                "export OMP_PLACES=threads; "
                "export I_MPI_PIN=0; "
                "export OMPI_MCA_hwloc_base_binding_policy=none; "
                "export PRTE_MCA_hwloc_base_binding_policy=none; "
            )
        else:
            exports += "export OMP_PROC_BIND=spread; export OMP_PLACES=cores; "
        return exports + command

    @staticmethod
    def _transport_attempt_tag(label: str) -> str:
        label_token = "".join(
            ch if (ch.isalnum() or ch in {"-", "_"}) else "_"
            for ch in (label or "attempt")
        )[:16] or "attempt"
        stamp = time.strftime("%Y%m%dT%H%M%S", time.localtime())
        digest = hashlib.sha1(f"{label_token}-{time.time_ns()}".encode("utf-8")).hexdigest()[:8]
        return f"{label_token}_{stamp}_{digest}"

    @staticmethod
    def _attempt_artifact_name(name: str, attempt_tag: str) -> str:
        path = Path(str(name))
        suffix = "".join(path.suffixes)
        stem = path.name[: -len(suffix)] if suffix else path.name
        return f"{stem}_{attempt_tag}{suffix}"

    @staticmethod
    def _archive_transport_attempt(
        transport_dir: Path,
        *,
        attempt_key: str,
        files: list[Path],
    ) -> Path:
        safe_key = "".join(
            ch if (ch.isalnum() or ch in {"-", "_"}) else "_"
            for ch in (attempt_key or "attempt")
        ) or "attempt"
        attempts_dir = transport_dir / "attempts"
        attempts_dir.mkdir(parents=True, exist_ok=True)
        attempt_dir = attempts_dir / f"job_{safe_key}"
        if attempt_dir.exists():
            attempt_dir = attempts_dir / f"job_{safe_key}_{int(time.time())}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        for src in files:
            if src.exists() and src.is_file():
                shutil.copy2(src, attempt_dir / src.name)
        return attempt_dir

    @staticmethod
    def _normalize_transport_results(results: dict) -> dict:
        out = dict(results)
        scan = out.get("thickness_scan")
        if isinstance(scan, dict):
            fixed = {}
            for k, v in scan.items():
                key = k
                try:
                    key = float(k)
                except Exception:
                    pass
                fixed[key] = v
            out["thickness_scan"] = fixed
        return out

    @staticmethod
    def _worker_source_zip(out_dir: Path) -> Path:
        bundle = out_dir / "wtec_src.zip"
        pkg_dir = Path(__file__).resolve().parents[1]  # .../wtec/wtec
        with zipfile.ZipFile(bundle, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in pkg_dir.rglob("*.py"):
                rel = p.relative_to(pkg_dir)
                zf.write(p, arcname=str(Path("wtec") / rel))
        return bundle

    def _remote_run_base(self, cluster_cfg) -> str:
        cfg_remote = str(self.cfg.get("remote_workdir", "")).strip()
        if cfg_remote:
            return cfg_remote.rstrip("/")
        run_name = str(self.cfg.get("name", "run")).strip() or "run"
        return f"{cluster_cfg.remote_workdir.rstrip('/')}/{run_name}"

    def _transport_result_file(self, *, label: str = "primary") -> Path:
        run_dir = Path(self.cfg.get("run_dir", ".")).resolve()
        labeled = run_dir / "transport" / str(label) / "transport_result.json"
        if labeled.exists():
            return labeled
        if str(label) == "primary":
            legacy = run_dir / "transport" / "transport_result.json"
            if legacy.exists():
                return legacy
        return labeled

    def _required_cached_rgf_sigma_source(self) -> str | None:
        if str(self.cfg.get("transport_engine", self._transport_engine())).strip().lower() != "rgf":
            return None
        if str(self.cfg.get("transport_rgf_mode", "")).strip().lower() != "full_finite":
            return None
        sigma_backend = (
            str(self.cfg.get("transport_rgf_full_finite_sigma_backend", "native")).strip().lower()
            or "native"
        )
        sigma_source = (
            str(self.cfg.get("_transport_rgf_internal_sigma_mode", sigma_backend)).strip().lower()
            or sigma_backend
        )
        return sigma_source if sigma_source == "kwant_exact" else None

    def _required_cached_rgf_eta(self) -> float | None:
        sigma_source = self._required_cached_rgf_sigma_source()
        if sigma_source != "kwant_exact":
            return None
        return resolve_transport_rgf_eta(
            self.cfg,
            internal_sigma_mode=sigma_source,
        )

    @staticmethod
    def _transport_result_sigma_source(payload: dict[str, Any]) -> str | None:
        runtime_cert = payload.get("runtime_cert", {})
        if isinstance(runtime_cert, dict):
            source = str(runtime_cert.get("full_finite_sigma_source", "")).strip().lower()
            if source:
                return source
        results = payload.get("transport_results", {})
        meta = results.get("meta", {}) if isinstance(results, dict) else {}
        if isinstance(meta, dict):
            source = str(meta.get("rgf_full_finite_sigma_source", "")).strip().lower()
            if source:
                return source
        return None

    @staticmethod
    def _transport_result_eta(payload: dict[str, Any]) -> float | None:
        raw = payload.get("transport_results_raw", {})
        if isinstance(raw, dict):
            eta = raw.get("eta")
            if eta is not None:
                try:
                    return float(eta)
                except Exception:
                    return None
        return None

    @staticmethod
    def _eta_matches(actual: float | None, expected: float) -> bool:
        if actual is None:
            return False
        tol = max(1.0e-12, abs(float(expected)) * 1.0e-6)
        return abs(float(actual) - float(expected)) <= tol

    def _load_cached_transport_results(self, *, label: str = "primary") -> dict | None:
        run_profile = str(self.cfg.get("run_profile", "strict")).strip().lower() or "strict"
        default_reuse = run_profile == "smoke"
        if not bool(self.cfg.get("reuse_transport_results", default_reuse)):
            return None
        p = self._transport_result_file(label=label)
        if not p.exists() or p.stat().st_size == 0:
            return None
        try:
            payload = json.loads(p.read_text())
            res = payload.get("transport_results", {})
            if not isinstance(res, dict):
                return None
            required_sigma_source = self._required_cached_rgf_sigma_source()
            required_eta = self._required_cached_rgf_eta()
            if required_sigma_source is not None:
                cached_sigma_source = self._transport_result_sigma_source(payload)
                if cached_sigma_source != required_sigma_source:
                    return None
                if required_eta is not None:
                    cached_eta = self._transport_result_eta(payload)
                    if not self._eta_matches(cached_eta, required_eta):
                        return None
            return self._normalize_transport_results(res)
        except Exception:
            return None

    @staticmethod
    def _extract_zero_disorder_curve(transport_results: dict) -> tuple[list[int], list[float]] | None:
        scan = transport_results.get("thickness_scan")
        if not isinstance(scan, dict) or not scan:
            return None
        key = None
        for k in scan.keys():
            try:
                if abs(float(k)) < 1e-12:
                    key = k
                    break
            except Exception:
                continue
        if key is None:
            key = sorted(scan.keys(), key=lambda x: float(x))[0]
        payload = scan.get(key, {})
        if not isinstance(payload, dict):
            return None
        t = payload.get("thickness_uc")
        rho = payload.get("rho_mean")
        if t is None or rho is None:
            return None
        try:
            t_vals = [int(v) for v in t]
            rho_vals = [float(v) for v in rho]
        except Exception:
            return None
        if len(t_vals) != len(rho_vals) or not t_vals:
            return None
        return t_vals, rho_vals

    @classmethod
    def _transport_compare_summary(cls, qe_track: dict, si_track: dict) -> dict[str, Any]:
        import numpy as np

        q = cls._extract_zero_disorder_curve(qe_track)
        s = cls._extract_zero_disorder_curve(si_track)
        if q is None or s is None:
            return {
                "status": "failed",
                "reason": "missing_zero_disorder_curve",
            }

        tq, rq = q
        ts, rs = s
        map_q = {int(t): float(r) for t, r in zip(tq, rq)}
        map_s = {int(t): float(r) for t, r in zip(ts, rs)}
        common = sorted(set(map_q).intersection(map_s))
        if len(common) < 2:
            return {
                "status": "failed",
                "reason": "insufficient_common_thickness_points",
                "common_thicknesses": common,
            }

        q_arr = np.array([map_q[t] for t in common], dtype=float)
        s_arr = np.array([map_s[t] for t in common], dtype=float)
        eps = 1e-30
        qn = q_arr / max(float(np.nanmean(np.abs(q_arr))), eps)
        sn = s_arr / max(float(np.nanmean(np.abs(s_arr))), eps)
        corr = float(np.corrcoef(qn, sn)[0, 1]) if len(common) >= 2 else float("nan")

        tq_min = int(common[int(np.argmin(q_arr))])
        ts_min = int(common[int(np.argmin(s_arr))])
        rq_min = float(np.min(q_arr))
        rs_min = float(np.min(s_arr))
        min_delta_frac = float(abs(rs_min - rq_min) / max(abs(rq_min), eps))

        return {
            "status": "ok",
            "common_thicknesses_uc": common,
            "pearson_corr_normalized_rho": corr,
            "qe_minimum": {"thickness_uc": tq_min, "rho": rq_min},
            "siesta_minimum": {"thickness_uc": ts_min, "rho": rs_min},
            "minimum_relative_deviation": min_delta_frac,
        }

    def _find_pristine_variant_hr_from_manifest(self, run_dir: Path) -> Path | None:
        manifest = run_dir / "topology" / "topology_point_manifest.json"
        if not manifest.exists() or manifest.stat().st_size == 0:
            return None
        try:
            payload = json.loads(manifest.read_text())
        except Exception:
            return None
        rows = payload.get("points", [])
        if not isinstance(rows, list):
            return None
        candidates: list[tuple[int, Path]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if not bool(row.get("is_pristine", False)):
                continue
            if str(row.get("status", "")).lower() != "ok":
                continue
            hr_raw = row.get("hr_dat_path")
            if not hr_raw:
                continue
            p = Path(str(hr_raw)).expanduser().resolve()
            if not p.exists():
                continue
            try:
                th = int(row.get("thickness_uc", 10**9))
            except Exception:
                th = 10**9
            candidates.append((th, p))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def _run_transport_track(self, hr_dat: Path, *, label: str) -> tuple[dict, dict]:
        backend = str(self.cfg.get("transport_backend", "qsub")).strip().lower()
        strict_qsub = bool(self.cfg.get("transport_strict_qsub", True))
        if backend == "qsub":
            try:
                return self._stage_transport_qsub(hr_dat, label=label)
            except Exception as exc:
                if strict_qsub:
                    raise
                res = self._stage_transport_local(hr_dat, label=label)
                return res, {
                    "status": "LOCAL_FALLBACK",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "backend": "local",
                }
        if backend == "local":
            return self._stage_transport_local(hr_dat, label=label), {"status": "LOCAL", "backend": "local"}
        raise ValueError(f"Unknown transport_backend={backend!r}. Use 'qsub' or 'local'.")

    def _anchor_transfer_cfg(self) -> dict[str, Any]:
        dft_tbl = self.cfg.get("dft", {}) if isinstance(self.cfg.get("dft"), dict) else {}
        nested = dft_tbl.get("anchor_transfer", {}) if isinstance(dft_tbl.get("anchor_transfer"), dict) else {}
        flat = self.cfg.get("dft_anchor_transfer", {}) if isinstance(self.cfg.get("dft_anchor_transfer"), dict) else {}

        out: dict[str, Any] = {
            "enabled": self._dft_mode() == "dual_family",
            "mode": "delta_h",
            "basis_policy": "strict_same_basis",
            "scope": "onsite_plus_first_shell",
            "fit_window_ev": 1.5,
            "fit_kmesh": [8, 8, 8],
            "alpha_grid_min": -0.5,
            "alpha_grid_max": 1.5,
            "alpha_grid_points": 81,
            "max_retries": 5,
            "retry_kmesh_step": 2,
            "retry_window_step_ev": 0.5,
            "reuse_existing": True,
        }
        out.update(nested)
        out.update(flat)
        out["enabled"] = bool(out.get("enabled", True))
        out["mode"] = str(out.get("mode", "delta_h")).strip().lower() or "delta_h"
        out["basis_policy"] = str(out.get("basis_policy", "strict_same_basis")).strip() or "strict_same_basis"
        out["scope"] = str(out.get("scope", "onsite_plus_first_shell")).strip() or "onsite_plus_first_shell"
        out["fit_window_ev"] = float(out.get("fit_window_ev", 1.5))
        out["fit_kmesh"] = [int(v) for v in out.get("fit_kmesh", [8, 8, 8])]
        out["alpha_grid_min"] = float(out.get("alpha_grid_min", -0.5))
        out["alpha_grid_max"] = float(out.get("alpha_grid_max", 1.5))
        out["alpha_grid_points"] = int(out.get("alpha_grid_points", 81))
        out["max_retries"] = max(1, int(out.get("max_retries", 5)))
        out["retry_kmesh_step"] = max(0, int(out.get("retry_kmesh_step", 2)))
        out["retry_window_step_ev"] = max(0.0, float(out.get("retry_window_step_ev", 0.5)))
        out["reuse_existing"] = bool(out.get("reuse_existing", True))
        return out

    def _prepare_delta_h_artifact(self, pes_hr_path: Path) -> Path | None:
        cfg_anchor = self._anchor_transfer_cfg()
        if not bool(cfg_anchor.get("enabled", False)):
            return None
        if self._dft_mode() != "dual_family":
            return None
        if str(cfg_anchor.get("mode", "delta_h")).strip().lower() != "delta_h":
            raise ValueError("Only dft.anchor_transfer.mode='delta_h' is currently supported.")

        run_dir = Path(self.cfg.get("run_dir", ".")).resolve()
        artifact_path = run_dir / "topology" / "delta_h_artifact.json"
        if (
            bool(cfg_anchor.get("reuse_existing", True))
            and artifact_path.exists()
            and artifact_path.stat().st_size > 0
        ):
            self._state["outputs"]["delta_h_artifact"] = str(artifact_path.resolve())
            self._state["outputs"]["dft_anchor_transfer"] = {
                "status": "reused",
                "artifact_path": str(artifact_path.resolve()),
            }
            self._save_checkpoint()
            return artifact_path

        from wtec.cluster.ssh import open_ssh
        from wtec.cluster.submit import JobManager
        from wtec.config.cluster import ClusterConfig
        from wtec.config.materials import get_material
        from wtec.wannier.delta_h import build_delta_h_artifact, write_delta_h_artifact

        material = str(self.cfg.get("material", "")).strip()
        if not material:
            raise ValueError("material is required for dft.anchor_transfer")
        preset = get_material(material)
        lcao_engine = self._variant_dft_engine()
        cluster_cfg = ClusterConfig.from_env()
        siesta_cfg = self._dft_siesta_cfg()
        if lcao_engine == "siesta" and bool(siesta_cfg.get("spin_orbit", True)):
            raise RuntimeError(
                "dft.anchor_transfer with LCAO engine 'siesta' and spin_orbit=true is "
                "unsupported by the current SIESTA↔Wannier bridge. "
                "In this source tree, Src/siesta2wannier90.F90:getFileNameRoot dies for "
                "non-collinear/SOC spin cases. Disable dft.anchor_transfer for this run, "
                "or use an LCAO engine whose Wannier bridge supports SOC."
            )
        anchor_queue = cluster_cfg.pbs_queue

        anchor_dir = run_dir / "dft_anchor_lcao"
        anchor_dir.mkdir(parents=True, exist_ok=True)
        atoms_ref = self._reference_atoms(self._build_atoms_from_config())

        k_scf_base = tuple(int(v) for v in self.cfg.get("kpoints_scf", (8, 8, 8)))
        k_nscf_base = tuple(int(v) for v in self.cfg.get("kpoints_nscf", (12, 12, 12)))
        dis_win_base = tuple(float(v) for v in getattr(preset, "dis_win", (-4.0, 16.0)))
        dis_froz_base = tuple(float(v) for v in getattr(preset, "dis_froz_win", (-1.0, 1.0)))
        k_step = int(cfg_anchor.get("retry_kmesh_step", 2))
        w_step = float(cfg_anchor.get("retry_window_step_ev", 0.5))
        max_retries = int(cfg_anchor.get("max_retries", 5))

        success: dict[str, Any] | None = None
        attempt_history: list[dict[str, Any]] = []
        last_error: str | None = None

        for attempt in range(max_retries):
            k_scf = tuple(max(1, int(v) + (attempt * k_step)) for v in k_scf_base)
            k_nscf = tuple(max(1, int(v) + (attempt * k_step)) for v in k_nscf_base)
            dis_win = (
                float(dis_win_base[0]) - (attempt * w_step),
                float(dis_win_base[1]) + (attempt * w_step),
            )
            dis_froz = (
                float(dis_froz_base[0]) - (attempt * 0.5 * w_step),
                float(dis_froz_base[1]) + (attempt * 0.5 * w_step),
            )
            attempt_meta: dict[str, Any] = {
                "attempt": attempt + 1,
                "kpoints_scf": list(k_scf),
                "kpoints_nscf": list(k_nscf),
                "dis_win": [float(dis_win[0]), float(dis_win[1])],
                "dis_froz_win": [float(dis_froz[0]), float(dis_froz[1])],
                "engine": lcao_engine,
            }
            try:
                with open_ssh(cluster_cfg) as ssh:
                    jm = JobManager(ssh)
                    common_kwargs = {
                        "run_dir": anchor_dir,
                        "remote_base": self.cfg.get("remote_workdir", cluster_cfg.remote_workdir),
                        "n_nodes": self.cfg.get("n_nodes", 1),
                        "n_cores_per_node": cluster_cfg.mpi_cores,
                        "n_cores_by_queue": cluster_cfg.mpi_cores_by_queue,
                        "queue": anchor_queue,
                        "queue_priority": cluster_cfg.pbs_queue_priority,
                        "kpoints_scf": k_scf,
                        "kpoints_nscf": k_nscf,
                        "omp_threads": cluster_cfg.omp_threads,
                        "modules": cluster_cfg.modules,
                        "bin_dirs": cluster_cfg.bin_dirs,
                        "live_log": self.cfg.get("_runtime_live_log", True),
                        "log_poll_interval": self.cfg.get("_runtime_log_poll_interval", 5),
                        "stale_log_seconds": self.cfg.get("_runtime_stale_log_seconds", 300),
                    }
                    if lcao_engine == "siesta":
                        from wtec.siesta.parser import parse_fermi_energy
                        from wtec.siesta.runner import SiestaPipeline as PipelineClass
                        pipeline = PipelineClass(
                            atoms_ref,
                            material,
                            jm,
                            pseudo_dir=cluster_cfg.resolved_siesta_pseudo_dir(
                                spin_orbit=bool(siesta_cfg.get("spin_orbit", True)),
                                explicit=str(siesta_cfg.get("pseudo_dir", "")).strip(),
                            ),
                            basis_profile=str(siesta_cfg.get("basis_profile", "")).strip(),
                            wannier_interface=str(
                                siesta_cfg.get("wannier_interface", "sisl")
                            ).strip().lower(),
                            spin_orbit=bool(siesta_cfg.get("spin_orbit", True)),
                            include_pao_basis=bool(siesta_cfg.get("include_pao_basis", False)),
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
                            dm_mixing_weight=float(siesta_cfg.get("dm_mixing_weight", 0.18)),
                            dm_number_pulay=int(siesta_cfg.get("dm_number_pulay", 6)),
                            electronic_temperature_k=float(
                                siesta_cfg.get("electronic_temperature_k", 300.0)
                            ),
                            max_scf_iterations=int(siesta_cfg.get("max_scf_iterations", 120)),
                            dispersion_cfg=self._dft_dispersion_cfg(),
                            **common_kwargs,
                        )
                    elif lcao_engine == "abacus":
                        from wtec.abacus.parser import parse_fermi_energy
                        from wtec.abacus.runner import AbacusPipeline as PipelineClass

                        abacus_cfg = self._dft_abacus_cfg()
                        pipeline = PipelineClass(
                            atoms_ref,
                            material,
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
                    else:
                        raise ValueError(
                            f"dft.anchor_transfer requires LCAO engine in ['siesta','abacus'], got {lcao_engine!r}."
                        )

                    scf_meta = pipeline.run_scf()
                    scf_out = anchor_dir / f"{material}.scf.out"
                    fermi_lcao = float(parse_fermi_energy(scf_out))
                    nscf_meta = pipeline.run_nscf(fermi_lcao)
                    wan_meta = pipeline.run_wannier(
                        fermi_lcao,
                        dis_win_override=dis_win,
                        dis_froz_win_override=dis_froz,
                    )

                anchor_hr = anchor_dir / f"{material}_hr.dat"
                anchor_win = anchor_dir / f"{material}.win"
                if not anchor_hr.exists():
                    raise FileNotFoundError(f"Anchor LCAO _hr.dat not found: {anchor_hr}")
                if not anchor_win.exists():
                    raise FileNotFoundError(f"Anchor LCAO .win not found: {anchor_win}")
                attempt_meta["status"] = "ok"
                attempt_meta["fermi_ev"] = float(fermi_lcao)
                attempt_meta["jobs"] = {
                    "scf": scf_meta.get("job_id"),
                    "nscf": nscf_meta.get("job_id"),
                    "wannier": wan_meta.get("job_id"),
                }
                success = {
                    "hr_dat_path": str(anchor_hr.resolve()),
                    "win_path": str(anchor_win.resolve()),
                    "fermi_ev": float(fermi_lcao),
                    "attempt": attempt + 1,
                    "jobs": attempt_meta["jobs"],
                }
                attempt_history.append(attempt_meta)
                break
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                attempt_meta["status"] = "failed"
                attempt_meta["reason"] = reason
                attempt_history.append(attempt_meta)
                last_error = reason

        if success is None:
            raise RuntimeError(
                "dft.anchor_transfer failed after "
                f"{max_retries} attempts. Last error: {last_error or 'unknown'}"
            )

        fermi_pes = self._state.get("outputs", {}).get("fermi_ev")
        if fermi_pes is None:
            fermi_pes = self.cfg.get("fermi_shift_eV", 0.0)
        try:
            fermi_pes = float(fermi_pes)
        except Exception:
            fermi_pes = 0.0

        alpha_points = max(2, int(cfg_anchor.get("alpha_grid_points", 81)))
        alpha_grid = np.linspace(
            float(cfg_anchor.get("alpha_grid_min", -0.5)),
            float(cfg_anchor.get("alpha_grid_max", 1.5)),
            alpha_points,
        )
        pes_win_path = pes_hr_path.with_name(f"{material}.win")
        artifact = build_delta_h_artifact(
            pes_hr_dat_path=pes_hr_path,
            pes_win_path=pes_win_path if pes_win_path.exists() else None,
            lcao_hr_dat_path=success["hr_dat_path"],
            lcao_win_path=success["win_path"],
            material=material,
            fermi_pes_ev=float(fermi_pes),
            fermi_lcao_ev=float(success["fermi_ev"]),
            basis_policy=str(cfg_anchor.get("basis_policy", "strict_same_basis")),
            scope=str(cfg_anchor.get("scope", "onsite_plus_first_shell")),
            fit_window_ev=float(cfg_anchor.get("fit_window_ev", 1.5)),
            fit_kmesh=tuple(int(v) for v in cfg_anchor.get("fit_kmesh", [8, 8, 8])),
            alpha_grid=alpha_grid,
            anchor_species_counts=dict(Counter(atoms_ref.get_chemical_symbols())),
        )
        artifact["attempt_history"] = attempt_history
        artifact["anchor"]["lcao_engine"] = lcao_engine
        artifact["anchor"]["lcao_hr_attempt"] = int(success["attempt"])
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        write_delta_h_artifact(artifact_path, artifact)

        self._state["outputs"]["delta_h_artifact"] = str(artifact_path.resolve())
        self._state["outputs"]["dft_anchor_transfer"] = {
            "status": "ok",
            "mode": "delta_h",
            "lcao_engine": lcao_engine,
            "artifact_path": str(artifact_path.resolve()),
            "attempts": attempt_history,
        }
        self._state["outputs"]["fermi_ev_lcao_anchor"] = float(success["fermi_ev"])
        self._save_checkpoint()
        return artifact_path

    def _stage_analysis(self, transport_results: dict, *, hr_dat: Path | None = None) -> dict:
        self._set_stage("ANALYSIS")
        from wtec.analysis.thickness_scan import detect_rho_minimum, plot_rho_vs_thickness
        from wtec.analysis.mfp_compare import summarize_mfp
        from wtec.workflow.topology_pipeline import TopologyPipeline

        run_dir = Path(self.cfg.get("run_dir", "."))
        plot_rho_vs_thickness(
            transport_results["thickness_scan"],
            outfile=run_dir / "rho_vs_thickness.pdf",
        )
        mfp_summary = summarize_mfp(transport_results.get("mfp", {}))
        rho_signature: dict[str, Any] = {
            "has_curve": False,
            "has_rho_minimum": False,
            "reason": "missing_zero_disorder_curve",
        }
        curve = self._extract_zero_disorder_curve(transport_results)
        if curve is not None:
            t_uc, rho_vals = curve
            scan = transport_results.get("thickness_scan", {})
            zero_payload = None
            if isinstance(scan, dict):
                for k, v in scan.items():
                    try:
                        if abs(float(k)) < 1e-12 and isinstance(v, dict):
                            zero_payload = v
                            break
                    except Exception:
                        continue
                if zero_payload is None and scan:
                    try:
                        first_key = sorted(scan.keys(), key=lambda x: float(x))[0]
                    except Exception:
                        first_key = list(scan.keys())[0]
                    cand = scan.get(first_key)
                    if isinstance(cand, dict):
                        zero_payload = cand
            thickness_m = (
                zero_payload.get("thickness_m")
                if isinstance(zero_payload, dict)
                else None
            )
            if isinstance(thickness_m, list) and len(thickness_m) == len(rho_vals):
                import numpy as np

                sig = detect_rho_minimum(
                    np.asarray(thickness_m, dtype=float),
                    np.asarray(rho_vals, dtype=float),
                )
            else:
                idx_min = min(range(len(rho_vals)), key=lambda i: rho_vals[i])
                sig = {
                    "has_minimum": bool(0 < idx_min < (len(rho_vals) - 1)),
                    "d_min_nm": None,
                    "rho_min": float(rho_vals[idx_min]),
                    "idx_min": int(idx_min),
                }
            rho_signature = {
                "has_curve": True,
                "thickness_uc": [int(v) for v in t_uc],
                "rho": [float(v) for v in rho_vals],
                "has_rho_minimum": bool(sig.get("has_minimum", False)),
                "rho_minimum": sig,
                "thinning_reduces_rho": bool(rho_vals[0] < rho_vals[-1]),
            }
        topo_summary: dict[str, Any] | None = None
        transport_compare: dict[str, Any] | None = None
        topology_cfg = self.cfg.get("topology", {})
        if not isinstance(topology_cfg, dict):
            topology_cfg = {}
        if self._dft_mode() in {"hybrid_qe_ref_siesta_variants", "dual_family"}:
            topology_cfg = dict(topology_cfg)
            topology_cfg.setdefault("variant_dft_engine", self._variant_dft_engine())

        hr_path = hr_dat or self._resolve_hr_dat_from_cfg_or_state()
        delta_h_artifact_path: Path | None = None
        if hr_path is not None and Path(hr_path).exists():
            try:
                delta_h_artifact_path = self._prepare_delta_h_artifact(Path(hr_path))
            except Exception as exc:
                topo_summary = {"status": "failed", "reason": f"anchor_transfer_failed:{type(exc).__name__}: {exc}"}
                self._state["outputs"]["topology_results"] = "failed"
                self._state["outputs"]["topology_summary"] = topo_summary
                self._save_checkpoint()
                return {
                    "mfp_summary": mfp_summary,
                    "rho_signature": rho_signature,
                    "topology": topo_summary,
                    "transport_compare": transport_compare,
                }
        if delta_h_artifact_path is not None:
            topology_cfg = dict(topology_cfg)
            topology_cfg["delta_h_artifact_path"] = str(delta_h_artifact_path.resolve())

        cfg_for_topology = dict(self.cfg)
        cfg_for_topology["topology"] = topology_cfg

        if hr_path is not None and Path(hr_path).exists():
            try:
                topo = TopologyPipeline(
                    hr_path,
                    run_dir=run_dir,
                    cfg=cfg_for_topology,
                    transport_results=transport_results,
                    fermi_ev_checkpoint=self._state.get("outputs", {}).get("fermi_ev"),
                )
                topo_result = topo.run()
                topo_summary = topo_result.get("summary", topo_result)
                topo_status = "computed"
                if isinstance(topo_summary, dict):
                    summary_status = str(topo_summary.get("status", "")).strip().lower()
                    if summary_status in {"failed", "skipped"}:
                        topo_status = summary_status
                self._state["outputs"]["topology_results"] = topo_status
                self._state["outputs"]["topology_summary"] = topo_summary
                if isinstance(topo_summary, dict):
                    if isinstance(topo_summary.get("variant_hr_map"), dict):
                        self._state["outputs"]["variant_hr_map"] = topo_summary.get("variant_hr_map")
                    if isinstance(topo_summary.get("variant_fermi_ev_map"), dict):
                        self._state["outputs"]["variant_fermi_ev_map"] = topo_summary.get(
                            "variant_fermi_ev_map"
                        )
            except Exception as exc:
                topo_summary = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
                self._state["outputs"]["topology_results"] = "failed"
                self._state["outputs"]["topology_summary"] = topo_summary
        else:
            topo_summary = {"status": "skipped", "reason": "missing_hr_dat"}
            self._state["outputs"]["topology_results"] = "skipped"
            self._state["outputs"]["topology_summary"] = topo_summary

        transport_policy = str(self.cfg.get("transport_policy", "")).strip().lower()
        if transport_policy == "dual_track_compare" and self._dft_mode() == "hybrid_qe_ref_siesta_variants":
            try:
                qe_hr = hr_path if hr_path is not None else self._resolve_hr_dat_from_cfg_or_state()
                si_hr = self._find_pristine_variant_hr_from_manifest(run_dir)
                if qe_hr is None or not Path(qe_hr).exists():
                    raise FileNotFoundError("missing_qe_reference_hr")
                if si_hr is None or not si_hr.exists():
                    raise FileNotFoundError("missing_siesta_pristine_variant_hr")
                qe_track = self._normalize_transport_results(transport_results)
                si_track, si_meta = self._run_transport_track(Path(si_hr), label="siesta_pristine")
                transport_compare = self._transport_compare_summary(qe_track, si_track)
                transport_compare["policy"] = "dual_track_compare"
                transport_compare["tracks"] = {
                    "qe_reference": {
                        "hr_dat_path": str(Path(qe_hr).resolve()),
                        "job": self._state.get("outputs", {}).get("transport_job"),
                    },
                    "siesta_pristine": {
                        "hr_dat_path": str(Path(si_hr).resolve()),
                        "job": si_meta,
                    },
                }
                report_dir = run_dir / "report"
                report_dir.mkdir(parents=True, exist_ok=True)
                (report_dir / "transport_compare.json").write_text(
                    json.dumps(transport_compare, indent=2)
                )
                self._state["outputs"]["transport_compare"] = "computed"
            except Exception as exc:
                transport_compare = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
                self._state["outputs"]["transport_compare"] = "failed"

        self._save_checkpoint()
        return {
            "mfp_summary": mfp_summary,
            "rho_signature": rho_signature,
            "topology": topo_summary,
            "transport_compare": transport_compare,
        }

    # ── checkpoint helpers ────────────────────────────────────────────────────

    def _set_stage(self, stage: str) -> None:
        self._state["stage"] = stage
        self._state["timestamp"] = time.time()
        print(f"[Orchestrator] Stage: {stage}")

    def _advance_to(self, stage: str) -> None:
        self._set_stage(stage)
        self._save_checkpoint()

    def _save_checkpoint(self) -> None:
        self._checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        self._checkpoint_file.write_text(json.dumps(_jsonable(self._state), indent=2))

    def _load_checkpoint(self) -> None:
        if self._checkpoint_file.exists():
            self._state = json.loads(self._checkpoint_file.read_text())

    def _resolve_hr_dat_from_cfg_or_state(self) -> Path | None:
        """Resolve hr_dat from explicit config first, then checkpoint state."""
        hr_cfg = self.cfg.get("hr_dat_path")
        if hr_cfg:
            if self._dft_reuse_mode() == "none":
                raise ValueError(
                    "Explicit hr_dat_path is not allowed when dft.reuse_mode='none'."
                )
            p = Path(hr_cfg)
            if not p.exists():
                raise FileNotFoundError(f"Configured hr_dat_path not found: {p}")
            return p

        hr_state = self._state.get("outputs", {}).get("hr_dat")
        if not hr_state:
            return None
        p = Path(hr_state)
        return p if p.exists() else None

    def _build_atoms_from_config(self):
        import ase.io
        from wtec.structure.defect import DefectBuilder

        struct_file = self.cfg["structure_file"]
        atoms = ase.io.read(struct_file)

        defect_cfg = self.cfg.get("defect")
        if not defect_cfg:
            return atoms

        db = DefectBuilder(atoms)
        dtype = defect_cfg["type"]
        sc = tuple(defect_cfg.get("supercell", [2, 2, 2]))
        if dtype == "vacancy":
            return db.vacancy(defect_cfg["site"], supercell=sc)
        if dtype == "substitution":
            return db.substitute(defect_cfg["site"], defect_cfg["element"], supercell=sc)
        if dtype == "antisite":
            return db.antisite(defect_cfg["site"], defect_cfg["site_b"], supercell=sc)
        raise ValueError(f"Unknown defect type: {dtype!r}")
