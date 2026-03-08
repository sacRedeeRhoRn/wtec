"""MFP comparison utilities."""

from __future__ import annotations

import numpy as np


# Reference MFPs and bulk resistivities at room temperature
REFERENCE_METALS = {
    "Cu":  {"mfp_nm": 39.0,   "rho_bulk_Ohmm": 1.68e-8},
    "Au":  {"mfp_nm": 38.0,   "rho_bulk_Ohmm": 2.44e-8},
    "Pt":  {"mfp_nm": 9.0,    "rho_bulk_Ohmm": 10.6e-8},
    "W":   {"mfp_nm": 14.0,   "rho_bulk_Ohmm": 5.4e-8},
    # Weyl semimetal estimates (300 K, experimental)
    "TaP": {"mfp_nm": 1000.0, "rho_bulk_Ohmm": 1e-9},
    "NbP": {"mfp_nm": 1500.0, "rho_bulk_Ohmm": 0.8e-9},
    "CoSi":{"mfp_nm": 200.0,  "rho_bulk_Ohmm": 5e-9},
}


def summarize_mfp(mfp_result: dict) -> dict:
    """Print and return a summary of extracted MFP."""
    sigma = mfp_result.get("sigma_S_per_m")
    r2 = mfp_result.get("fit_R2")
    regime = mfp_result.get("regime", "unknown")
    mfp_nm = mfp_result.get("mfp_nm")
    mfp_drude_nm = mfp_result.get("mfp_drude_nm")

    summary = {
        "sigma_S_per_m": sigma,
        "fit_R2": r2,
        "regime": regime,
        "mfp_nm": mfp_nm,
        "mfp_drude_nm": mfp_drude_nm,
    }

    print(f"  Conductivity:  σ = {sigma:.3e} S/m" if sigma else "  σ not extracted")
    print(f"  Fit quality:   R² = {r2:.4f}" if r2 else "  R² not available")
    if mfp_nm is not None:
        print(f"  MFP (fit):     ℓ = {float(mfp_nm):.3f} nm")
    if mfp_drude_nm is not None:
        print(f"  MFP (Drude):   ℓ = {float(mfp_drude_nm):.3f} nm")
    print(f"  Transport regime: {regime}")

    if sigma:
        print("\n  Comparison to reference metals (room temperature):")
        for name, ref in REFERENCE_METALS.items():
            rho_ref = ref["rho_bulk_Ohmm"]
            sigma_ref = 1.0 / rho_ref
            ratio = sigma / sigma_ref
            print(f"    vs {name:5s}: σ_computed/σ_{name} = {ratio:.2f}")

    return summary


def plot_mfp_comparison(
    materials_mfp: dict[str, float],
    outfile=None,
) -> None:
    """Bar chart comparing MFPs across materials.

    Parameters
    ----------
    materials_mfp : dict
        {'material_name': mfp_nm, ...}
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required")

    names = list(materials_mfp.keys())
    values = list(materials_mfp.values())

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#e74c3c" if n in ("TaP", "NbP", "CoSi") else "#3498db" for n in names]
    bars = ax.bar(names, values, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("Mean free path ℓ (nm)")
    ax.set_title("MFP comparison: topological semimetals vs normal metals")

    # Add reference lines
    for metal, ref in REFERENCE_METALS.items():
        if metal in names:
            continue
        ax.axhline(ref["mfp_nm"], ls="--", alpha=0.3, color="gray")

    plt.tight_layout()
    if outfile:
        plt.savefig(str(outfile), dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
