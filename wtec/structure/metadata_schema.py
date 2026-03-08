"""Schema helpers for generated slab/defect metadata."""

from __future__ import annotations

from typing import Any


REQUIRED_TOP_LEVEL_KEYS = (
    "wtec_slab_metadata_version",
    "project",
    "layers",
    "interfaces",
    "summary",
    "export",
)


def validate_slab_metadata(metadata: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate minimal schema required by variant/topology discovery."""
    errors: list[str] = []
    if not isinstance(metadata, dict):
        return False, ["metadata_root_not_object"]

    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in metadata:
            errors.append(f"missing_top_level:{key}")

    layers = metadata.get("layers")
    if not isinstance(layers, list) or not layers:
        errors.append("layers_missing_or_empty")
    else:
        for idx, layer in enumerate(layers):
            if not isinstance(layer, dict):
                errors.append(f"layer_not_object:{idx}")
                continue
            for key in ("label", "role"):
                if key not in layer or not str(layer.get(key, "")).strip():
                    errors.append(f"layer_missing:{idx}:{key}")
            if "z_min_angstrom" not in layer or "z_max_angstrom" not in layer:
                errors.append(f"layer_missing:{idx}:z_range")

    interfaces = metadata.get("interfaces")
    if interfaces is not None and not isinstance(interfaces, list):
        errors.append("interfaces_not_list")
    elif isinstance(interfaces, list):
        for idx, iface in enumerate(interfaces):
            if not isinstance(iface, dict):
                errors.append(f"interface_not_object:{idx}")
                continue
            if "atoms_in_window" not in iface:
                errors.append(f"interface_missing:{idx}:atoms_in_window")
            else:
                try:
                    aiw = int(iface.get("atoms_in_window", 0))
                except Exception:
                    errors.append(f"interface_invalid:{idx}:atoms_in_window")
                    continue
                if aiw <= 0:
                    errors.append(f"interface_invalid:{idx}:atoms_in_window_nonpositive")

    summary = metadata.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary_not_object")
    else:
        if "atoms_after_defects" not in summary and "atoms_before_defects" not in summary:
            errors.append("summary_missing_atom_counts")

    project = metadata.get("project")
    if not isinstance(project, dict):
        errors.append("project_not_object")
    else:
        if "name" not in project or not str(project.get("name", "")).strip():
            errors.append("project_missing_name")

    export = metadata.get("export")
    if not isinstance(export, dict):
        errors.append("export_not_object")

    return (len(errors) == 0), errors
