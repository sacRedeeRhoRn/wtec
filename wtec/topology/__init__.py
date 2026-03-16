"""Topology analysis helpers for node/arc/deviation workflows."""

from .deviation import compute_s_topo, compute_s_total
from .variant_discovery import discover_variants
from .validation import validate_wannier_model
from .node_scan import scan_weyl_nodes, compute_chern_profile, compute_weyl_velocity_tensor
from .arc_scan import compute_arc_connectivity
from .hybridization_gap import (
    compute_hybridization_gap,
    arc_transmission,
    crossover_thickness,
    two_channel_conductance_model,
)
from .surface_gf import (
    lopez_sancho_surface_gf,
    surface_spectral_map_lopez_sancho,
    compute_surface_spectral_metric_ls,
)
from .wilson_loop import compute_wilson_loop_chern
from .arc_metrics import (
    fermi_arc_length_angstrom,
    compute_arc_length_from_tb,
    required_n_layers_y,
)
from .berry_curvature import (
    compute_berry_curvature_map,
    compute_anomalous_hall_conductivity,
    berry_curvature_hotspots_for_arc_sampling,
)

__all__ = [
    # deviation
    "compute_s_topo",
    "compute_s_total",
    # discovery / validation
    "discover_variants",
    "validate_wannier_model",
    # node scan
    "scan_weyl_nodes",
    "compute_chern_profile",
    "compute_weyl_velocity_tensor",
    # arc scan
    "compute_arc_connectivity",
    # hybridization gap (new)
    "compute_hybridization_gap",
    "arc_transmission",
    "crossover_thickness",
    "two_channel_conductance_model",
    # surface GF (new)
    "lopez_sancho_surface_gf",
    "surface_spectral_map_lopez_sancho",
    "compute_surface_spectral_metric_ls",
    # Wilson loop (new)
    "compute_wilson_loop_chern",
    # arc metrics (new)
    "fermi_arc_length_angstrom",
    "compute_arc_length_from_tb",
    "required_n_layers_y",
    # Berry curvature (new)
    "compute_berry_curvature_map",
    "compute_anomalous_hall_conductivity",
    "berry_curvature_hotspots_for_arc_sampling",
]
