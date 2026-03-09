"""Transport pipeline: Wannier TB model → Kwant trilayer → ρ(d)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from wtec.wannier.model import WannierTBModel
from wtec.transport.conductance import (
    compute_conductance_vs_thickness,
    compute_conductance_vs_length,
)
from wtec.transport.mfp import extract_mfp_from_scaling, mfp_from_sigma


class TransportPipeline:
    """Compute thickness-dependent resistivity from a Wannier TB model."""

    def __init__(
        self,
        hr_dat_path: str | Path,
        *,
        win_path: str | Path | None = None,
        thicknesses: list[int] | None = None,
        disorder_strengths: list[float] | None = None,
        n_ensemble: int = 50,
        energy: float = 0.0,
        n_jobs: int = 4,
        mfp_n_layers_z: int = 10,
        mfp_lengths: list[int] | None = None,
        lead_onsite_eV: float = 0.0,
        base_seed: int = 0,
        lead_axis: str = "x",
        thickness_axis: str = "z",
        n_layers_x: int = 4,
        n_layers_y: int = 4,
        carrier_density_m3: float | None = None,
        fermi_velocity_m_per_s: float | None = None,
        progress_callback: Callable[..., None] | None = None,
        log_detail: str = "minimal",
        heartbeat_seconds: int = 20,
        kwant_mode: str = "auto",
        kwant_task_workers: int = 0,
    ) -> None:
        """
        Parameters
        ----------
        hr_dat_path : str or Path
            Path to *_hr.dat file.
        thicknesses : list[int] | None
            Film thicknesses in unit cells. Default: 2..30 step 2.
        disorder_strengths : list[float] | None
            Anderson disorder W values. Default: [0, 0.1, 0.3, 0.5, 1.0].
        """
        self.tb_model = WannierTBModel.from_hr_dat(hr_dat_path, win_path)
        self.thicknesses = np.array(thicknesses or list(range(2, 32, 2)))
        self.disorder_strengths = disorder_strengths or [0.0, 0.1, 0.3, 0.5, 1.0]
        self.n_ensemble = n_ensemble
        self.energy = energy
        self.n_jobs = n_jobs
        self.mfp_n_layers_z = int(mfp_n_layers_z)
        self.mfp_lengths = mfp_lengths
        self.lead_onsite_eV = float(lead_onsite_eV)
        self.base_seed = int(base_seed)
        self.lead_axis = str(lead_axis).lower()
        self.thickness_axis = str(thickness_axis).lower()
        self.n_layers_x = int(n_layers_x)
        self.n_layers_y = int(n_layers_y)
        self.carrier_density_m3 = (
            float(carrier_density_m3) if carrier_density_m3 is not None else None
        )
        self.fermi_velocity_m_per_s = (
            float(fermi_velocity_m_per_s)
            if fermi_velocity_m_per_s is not None
            else None
        )
        self._results: dict = {}
        self._progress_callback = progress_callback
        self.log_detail = str(log_detail).strip().lower() or "minimal"
        self.heartbeat_seconds = max(5, int(heartbeat_seconds))
        self.kwant_mode = str(kwant_mode).strip().lower() or "auto"
        self.kwant_task_workers = max(0, int(kwant_task_workers))

    def _emit(self, event: str, **payload: Any) -> None:
        cb = self._progress_callback
        if cb is None:
            return
        try:
            cb(event=event, **payload)
        except Exception:
            # Transport progress logging must never break the physics run.
            pass

    def run_thickness_scan(self) -> dict:
        """Compute ρ(d) for all disorder strengths.

        Returns
        -------
        dict mapping disorder_strength → conductance_result_dict
        """
        results = {}
        for idx, W in enumerate(self.disorder_strengths):
            self._emit(
                "thickness_scan_start",
                disorder_strength=float(W),
                index=int(idx),
                total=int(len(self.disorder_strengths)),
            )
            print(f"  Thickness scan W={W:.2f} eV...")
            res = compute_conductance_vs_thickness(
                self.tb_model,
                self.thicknesses,
                disorder_strength=W,
                n_ensemble=self.n_ensemble if W > 0 else 1,
                energy=self.energy,
                n_jobs=self.n_jobs,
                base_seed=self.base_seed,
                lead_onsite_eV=self.lead_onsite_eV,
                lead_axis=self.lead_axis,
                thickness_axis=self.thickness_axis,
                n_layers_x=self.n_layers_x,
                n_layers_y=self.n_layers_y,
                progress_cb=self._progress_callback,
                log_detail=self.log_detail,
                heartbeat_seconds=self.heartbeat_seconds,
                kwant_mode=self.kwant_mode,
                task_workers=self.kwant_task_workers,
            )
            results[W] = res
            self._emit(
                "thickness_scan_done",
                disorder_strength=float(W),
                index=int(idx),
                total=int(len(self.disorder_strengths)),
                n_points=int(len(res.get("thickness_uc", []))),
            )
        self._results["thickness_scan"] = results
        return results

    def run_mfp_extraction(
        self,
        disorder_strength: float,
        n_layers_z: int = 10,
        lengths: list[int] | None = None,
    ) -> dict:
        """Compute G(L) and extract MFP for a given disorder strength.

        Parameters
        ----------
        lengths : list[int] | None
            System lengths in unit cells. Default: 5..100 step 5.
        """
        lengths = np.array(lengths or list(range(5, 105, 5)))
        self._emit(
            "mfp_scan_start",
            disorder_strength=float(disorder_strength),
            n_lengths=int(len(lengths)),
        )
        print(f"  MFP extraction W={disorder_strength:.2f} eV...")

        gL_data = compute_conductance_vs_length(
            self.tb_model,
            lengths,
            disorder_strength,
            n_layers_z_fixed=n_layers_z,
            n_layers_x_fixed=self.n_layers_x,
            n_layers_y=self.n_layers_y,
            lead_axis=self.lead_axis,
            thickness_axis=self.thickness_axis,
            n_ensemble=self.n_ensemble,
            energy=self.energy,
            n_jobs=self.n_jobs,
            base_seed=self.base_seed,
            lead_onsite_eV=self.lead_onsite_eV,
            progress_cb=self._progress_callback,
            log_detail=self.log_detail,
            heartbeat_seconds=self.heartbeat_seconds,
            kwant_mode=self.kwant_mode,
            task_workers=self.kwant_task_workers,
        )
        cross_m2 = float(np.mean(gL_data["cross_section_m2"]))

        mfp_result = extract_mfp_from_scaling(
            gL_data["length_m"],
            gL_data["G_mean"],
            cross_section_m2=cross_m2,
        )
        sigma = mfp_result.get("sigma_S_per_m")
        if (
            sigma is not None
            and self.carrier_density_m3 is not None
            and self.carrier_density_m3 > 0
            and self.fermi_velocity_m_per_s is not None
            and self.fermi_velocity_m_per_s > 0
        ):
            drude = mfp_from_sigma(
                float(sigma),
                carrier_density_m3=float(self.carrier_density_m3),
                fermi_velocity_m_per_s=float(self.fermi_velocity_m_per_s),
            )
            mfp_result["mfp_drude_m"] = drude.get("mfp_m")
            mfp_result["mfp_drude_nm"] = drude.get("mfp_nm")
            mfp_result["tau_s"] = drude.get("tau_s")
            if mfp_result.get("mfp_m") is None:
                mfp_result["mfp_m"] = drude.get("mfp_m")
                mfp_result["mfp_nm"] = drude.get("mfp_nm")
            mfp_result["mfp_reference_inputs"] = {
                "carrier_density_m3": float(self.carrier_density_m3),
                "fermi_velocity_m_per_s": float(self.fermi_velocity_m_per_s),
            }
        else:
            mfp_result["mfp_reference_inputs"] = {
                "carrier_density_m3": self.carrier_density_m3,
                "fermi_velocity_m_per_s": self.fermi_velocity_m_per_s,
            }
            mfp_result.setdefault(
                "mfp_note",
                "Drude refinement unavailable: missing carrier_density_m3 or fermi_velocity_m_per_s.",
            )
        mfp_result["G_vs_L"] = gL_data
        self._results["mfp"] = mfp_result
        self._emit(
            "mfp_scan_done",
            disorder_strength=float(disorder_strength),
            n_lengths=int(len(lengths)),
            sigma_S_per_m=mfp_result.get("sigma_S_per_m"),
            mfp_nm=mfp_result.get("mfp_nm"),
        )
        return mfp_result

    def run_full(self) -> dict:
        """Run thickness scan + MFP extraction."""
        self._emit(
            "transport_run_start",
            n_disorder=int(len(self.disorder_strengths)),
            n_thickness=int(len(self.thicknesses)),
            n_ensemble=int(self.n_ensemble),
            kwant_mode=self.kwant_mode,
            kwant_task_workers=int(self.kwant_task_workers),
        )
        thickness_results = self.run_thickness_scan()
        # Use median disorder for MFP
        mid_W = self.disorder_strengths[len(self.disorder_strengths) // 2]
        mfp_result = self.run_mfp_extraction(
            disorder_strength=mid_W,
            n_layers_z=self.mfp_n_layers_z,
            lengths=self.mfp_lengths,
        )
        meta = {
            "n_ensemble": self.n_ensemble,
            "energy_eV": self.energy,
            "n_jobs_arg": self.n_jobs,
            "lead_onsite_eV": self.lead_onsite_eV,
            "base_seed": self.base_seed,
            "mfp_n_layers_z": self.mfp_n_layers_z,
            "mfp_lengths": list(self.mfp_lengths) if self.mfp_lengths else None,
            "lead_axis": self.lead_axis,
            "thickness_axis": self.thickness_axis,
            "n_layers_x": self.n_layers_x,
            "n_layers_y": self.n_layers_y,
            "carrier_density_m3": self.carrier_density_m3,
            "fermi_velocity_m_per_s": self.fermi_velocity_m_per_s,
            "log_detail": self.log_detail,
            "heartbeat_seconds": self.heartbeat_seconds,
            "kwant_mode": self.kwant_mode,
            "kwant_task_workers": self.kwant_task_workers,
        }
        self._emit(
            "transport_run_done",
            n_disorder=int(len(self.disorder_strengths)),
            n_thickness=int(len(self.thicknesses)),
            n_ensemble=int(self.n_ensemble),
        )
        return {"thickness_scan": thickness_results, "mfp": mfp_result, "meta": meta}

    @property
    def results(self) -> dict:
        return self._results
