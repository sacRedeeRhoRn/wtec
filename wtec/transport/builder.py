"""Kwant trilayer builder utilities."""

from __future__ import annotations

import numpy as np


def build_trilayer(
    tb_model,
    n_layers_z: int,
    *,
    n_layers_x: int = 1,
    n_layers_y: int = 4,
    lead_axis: str = "x",
    substrate_onsite_eV: float = 50.0,
    substrate_layers: int = 0,
):
    """Convenience wrapper around WannierTBModel.to_kwant_builder.

    Returns a non-finalized kwant.Builder.
    """
    return tb_model.to_kwant_builder(
        n_layers_z=n_layers_z,
        n_layers_x=n_layers_x,
        n_layers_y=n_layers_y,
        lead_axis=lead_axis,
        substrate_onsite_eV=substrate_onsite_eV,
        substrate_layers=substrate_layers,
    )


def visualize_system(sys, *, figsize=(10, 6)):
    """Plot Kwant system structure (requires matplotlib)."""
    try:
        import kwant
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("kwant and matplotlib are required")

    fig, ax = plt.subplots(figsize=figsize)
    kwant.plot(sys, ax=ax)
    return fig, ax
