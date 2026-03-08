"""General plotting utilities for wtec."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_bands(
    k_dist: np.ndarray,
    bands: np.ndarray,
    *,
    k_labels: list[tuple[float, str]] | None = None,
    fermi_energy: float = 0.0,
    title: str = "Band structure",
    outfile: str | Path | None = None,
) -> None:
    """Plot Wannier-interpolated band structure.

    Parameters
    ----------
    k_dist : np.ndarray shape (n_k,)
        Cumulative k-path distance.
    bands : np.ndarray shape (n_k, n_bands)
        Band energies in eV.
    k_labels : list of (position, label)
        High-symmetry point positions and labels.
    fermi_energy : float
        Fermi energy (dashed horizontal line).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required")

    fig, ax = plt.subplots(figsize=(6, 5))
    for b in range(bands.shape[1]):
        ax.plot(k_dist, bands[:, b] - fermi_energy, "b-", lw=0.8, alpha=0.7)
    ax.axhline(0, color="r", ls="--", lw=0.8, label="Fermi level")

    if k_labels:
        positions = [kl[0] for kl in k_labels]
        labels = [kl[1] for kl in k_labels]
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        for pos in positions:
            ax.axvline(pos, color="k", lw=0.5, alpha=0.5)

    ax.set_ylabel("E − EF (eV)")
    ax.set_title(title)
    ax.set_xlim(k_dist[0], k_dist[-1])
    plt.tight_layout()
    if outfile:
        plt.savefig(str(outfile), dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_G_vs_L(
    gL_data: dict,
    *,
    outfile: str | Path | None = None,
) -> None:
    """Plot G(L) and 1/G(L) for MFP extraction."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required")

    L_nm = gL_data["length_m"] * 1e9
    G = gL_data["G_mean"]
    G_std = gL_data["G_std"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.errorbar(L_nm, G, yerr=G_std, fmt="o-", capsize=3)
    ax1.set_xlabel("Length L (nm)")
    ax1.set_ylabel("Conductance G (e²/h)")
    ax1.set_title("G(L)")

    ax2.plot(L_nm, 1.0 / np.where(G > 0, G, np.nan), "s-")
    ax2.set_xlabel("Length L (nm)")
    ax2.set_ylabel("1/G (h/e²)")
    ax2.set_title("1/G(L) — linear = diffusive regime")

    plt.tight_layout()
    if outfile:
        plt.savefig(str(outfile), dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
