"""ρ(d) sweep analysis and visualization."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_rho_vs_thickness(
    scan_results: dict,
    *,
    outfile: str | Path | None = None,
    reference_metal: dict | None = None,
    title: str = "Thickness-dependent resistivity",
) -> None:
    """Plot ρ(d) for all disorder strengths.

    Parameters
    ----------
    scan_results : dict
        Output of TransportPipeline.run_thickness_scan().
        Keys are disorder strengths W, values are conductance dicts.
    reference_metal : dict | None
        Optional Fuchs-Sondheimer reference curve.
        {'rho_bulk': float, 'mfp_m': float, 'label': str}
    outfile : str | Path | None
        Save figure to this path. If None, calls plt.show().
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required: pip install matplotlib")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_rho, ax_G = axes

    colors = plt.cm.viridis(np.linspace(0, 1, len(scan_results)))

    rho_samples: list[np.ndarray] = []

    for (W, res), color in zip(sorted(scan_results.items()), colors):
        d_nm = np.asarray(res["thickness_m"], dtype=float) * 1e9
        label = f"W = {W:.2f} eV"
        rho_mean = np.asarray(res["rho_mean"], dtype=float)
        rho_std = np.asarray(res["rho_std"], dtype=float)
        rho_samples.append(rho_mean)

        ax_rho.plot(d_nm, rho_mean, color=color, label=label, lw=2)
        ax_rho.fill_between(
            d_nm,
            rho_mean - rho_std,
            rho_mean + rho_std,
            color=color,
            alpha=0.2,
        )

        ax_G.plot(d_nm, np.asarray(res["G_mean"], dtype=float), color=color, label=label, lw=2)

    if reference_metal is not None:
        from wtec.transport.observables import fuchs_sondheimer_rho
        d_arr = np.asarray(list(scan_results.values())[0]["thickness_m"], dtype=float)
        rho_fs = fuchs_sondheimer_rho(
            reference_metal["rho_bulk"],
            reference_metal["mfp_m"],
            d_arr,
        )
        ax_rho.plot(
            d_arr * 1e9,
            rho_fs,
            "k--",
            lw=1.5,
            label=reference_metal.get("label", "Fuchs-Sondheimer ref"),
        )

    ax_rho.set_xlabel("Thickness d (nm)")
    ax_rho.set_ylabel("Resistivity ρ (Ω·m)")
    ax_rho.set_title(f"{title} — ρ(d)")
    ax_rho.legend(fontsize=8)
    rho_all = np.concatenate(rho_samples) if rho_samples else np.array([], dtype=float)
    rho_pos = rho_all[np.isfinite(rho_all) & (rho_all > 0.0)]
    if rho_pos.size:
        ax_rho.set_yscale("log")
    else:
        # For degenerate/minimal test geometries conductance may be zero, making
        # rho non-finite; fall back to linear scale to avoid hard failure.
        ax_rho.set_yscale("linear")

    ax_G.set_xlabel("Thickness d (nm)")
    ax_G.set_ylabel("Conductance G (e²/h)")
    ax_G.set_title(f"{title} — G(d)")
    ax_G.legend(fontsize=8)

    plt.tight_layout()
    if outfile:
        plt.savefig(str(outfile), dpi=150, bbox_inches="tight")
        print(f"Saved: {outfile}")
    else:
        plt.show()
    plt.close(fig)


def detect_rho_minimum(
    thickness_m: np.ndarray,
    rho: np.ndarray,
) -> dict:
    """Detect whether ρ(d) has a non-monotonic minimum.

    Returns
    -------
    dict with 'has_minimum', 'd_min_nm', 'rho_min'
    """
    idx_min = np.argmin(rho)
    has_min = 0 < idx_min < len(rho) - 1
    return {
        "has_minimum": bool(has_min),
        "d_min_nm": float(thickness_m[idx_min] * 1e9),
        "rho_min": float(rho[idx_min]),
        "idx_min": int(idx_min),
    }
