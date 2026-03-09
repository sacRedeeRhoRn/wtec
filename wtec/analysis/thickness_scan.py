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


def plot_three_curve_rho(
    thickness_m: np.ndarray,
    rho_pristine: np.ndarray,
    rho_defect: np.ndarray,
    *,
    rho_fs: np.ndarray | None = None,
    rho_pristine_std: np.ndarray | None = None,
    rho_defect_std: np.ndarray | None = None,
    two_channel_fit_pristine: dict | None = None,
    two_channel_fit_defect: dict | None = None,
    outfile: str | Path | None = None,
    title: str = "",
    material: str = "Weyl semimetal",
) -> None:
    """Three-curve ρ(d) diagnostic plot for topological semimetal films.

    Physics basis
    -------------
    Topological identification requires three curves:
      1. ρ_pristine(d): decreasing with decreasing d → arc-dominated transport
      2. ρ_defect(d):   higher and potentially less decreasing → arc scattering
      3. ρ_FS(d) reference: increasing with decreasing d (trivial Fuchs-Sondheimer)

    The contrast between curves 1/2 (downward trend) and 3 (upward trend)
    is the definitive experimental signature.  Annotating the two-channel fit
    intercept σ_arc_2D extracts the arc conductance quantitatively.

    Parameters
    ----------
    thickness_m : np.ndarray
        Film thicknesses in metres.
    rho_pristine, rho_defect : np.ndarray
        Resistivity (Ω·m) for pristine and defected film.
    rho_fs : np.ndarray or None
        Fuchs-Sondheimer reference (trivial metal).
    rho_pristine_std, rho_defect_std : np.ndarray or None
        Error bars on resistivity.
    two_channel_fit_pristine, two_channel_fit_defect : dict or None
        Output of fit_two_channel_conductance() for annotation.
    outfile : str or Path or None
        Save to file; None → plt.show().
    title, material : str
        Plot title strings.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        raise ImportError("matplotlib is required: pip install matplotlib")

    d_nm = np.asarray(thickness_m, dtype=float) * 1e9
    rho_p = np.asarray(rho_pristine, dtype=float)
    rho_d = np.asarray(rho_defect, dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left panel: ρ(d) three-curve signature ─────────────────────────
    ax1.plot(d_nm, rho_p, "b-o", lw=2, ms=5, label="Pristine (topological)")
    if rho_pristine_std is not None:
        rp_std = np.asarray(rho_pristine_std, dtype=float)
        ax1.fill_between(d_nm, rho_p - rp_std, rho_p + rp_std, color="b", alpha=0.15)

    ax1.plot(d_nm, rho_d, "r-s", lw=2, ms=5, label="Defect (arc suppressed)")
    if rho_defect_std is not None:
        rd_std = np.asarray(rho_defect_std, dtype=float)
        ax1.fill_between(d_nm, rho_d - rd_std, rho_d + rd_std, color="r", alpha=0.15)

    if rho_fs is not None:
        ax1.plot(d_nm, np.asarray(rho_fs, dtype=float), "k--", lw=1.5,
                 label="Fuchs-Sondheimer (trivial)")

    rho_all = np.concatenate([rho_p, rho_d])
    rho_pos = rho_all[np.isfinite(rho_all) & (rho_all > 0.0)]
    ax1.set_yscale("log" if rho_pos.size else "linear")
    ax1.set_xlabel("Film thickness d (nm)", fontsize=11)
    ax1.set_ylabel("Resistivity ρ (Ω·m)", fontsize=11)
    header = f"{material} — ρ(d) thickness signature"
    if title:
        header = f"{title}\n{header}"
    ax1.set_title(header, fontsize=10)
    ax1.legend(fontsize=9)

    # Annotate topological trend
    rho_p_fin = rho_p[np.isfinite(rho_p)]
    if len(rho_p_fin) >= 2:
        trend = "↓ decreasing (topological)" if rho_p_fin[-1] < rho_p_fin[0] else "↑ increasing"
        ax1.annotate(
            f"Pristine: {trend}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            fontsize=8,
            color="blue",
            va="top",
        )

    # ── Right panel: σ_sq(d) linear fit ────────────────────────────────
    e2h = 7.748091729e-5
    ax2.set_title("Sheet conductance σ_sq = G·(L/W)\nTwo-channel fit: σ_arc + σ_bulk·d", fontsize=10)

    def _plot_fit(ax, d_nm, thickness_m, rho, fit, color, label_base):
        G = 1.0 / np.maximum(rho, 1e-30)
        ax.plot(d_nm, G, color=color, marker="o", ms=5, lw=1.5, label=f"{label_base} G(d)")
        if fit and not np.isnan(fit.get("sigma_arc_2D_e2h", float("nan"))):
            a = fit["sigma_arc_2D_e2h"]
            b = fit["sigma_bulk_3D_e2h_per_m"]
            d_fit = np.asarray(thickness_m, dtype=float)
            G_fit = a + b * d_fit
            ax.plot(d_nm, G_fit, color=color, ls="--", lw=1.2,
                    label=f"Fit: σ_arc={a:.2f} e²/h")
            err = fit.get("sigma_arc_2D_err", float("nan"))
            topo = fit.get("topological_signal", False)
            flag = "✓ topo" if topo else "✗ trivial"
            ax.annotate(
                f"{label_base}: σ_arc = {a:.2f}±{err:.2f} e²/h {flag}",
                xy=(0.03, 0.92 if "ristine" in label_base else 0.82),
                xycoords="axes fraction",
                fontsize=8,
                color=color,
            )

    _plot_fit(ax2, d_nm, thickness_m, rho_p, two_channel_fit_pristine, "blue", "Pristine")
    _plot_fit(ax2, d_nm, thickness_m, rho_d, two_channel_fit_defect, "red", "Defect")

    ax2.set_xlabel("Film thickness d (nm)", fontsize=11)
    ax2.set_ylabel("Conductance G (arb. units)", fontsize=11)
    ax2.legend(fontsize=8)

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
