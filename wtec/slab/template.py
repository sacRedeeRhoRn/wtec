"""Template schema and loader for slab generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectSpec:
    name: str
    seed: int
    output_dir: str
    material: str | None = None
    mp_api_key: str | None = None
    mp_api_key_env: str | None = None


@dataclass(frozen=True)
class MatchingSpec:
    max_strain_percent: float = 5.0
    max_area_ratio: float = 4.0
    max_search_supercell: int = 4


@dataclass(frozen=True)
class StackSpec:
    align_axis: str = "z"
    vacuum_angstrom: float = 18.0
    interface_gap_angstrom: float = 2.2


@dataclass(frozen=True)
class ExportSpec:
    cif_path: str
    metadata_json_path: str


@dataclass(frozen=True)
class LayerSpec:
    label: str
    role: str
    source: str
    cif_path: str | None
    mp_id: str | None
    miller: tuple[int, int, int]
    thickness_angstrom: float
    termination_index: int = 0


@dataclass(frozen=True)
class SubstitutionSpec:
    from_element: str
    to_element: str
    count: int


@dataclass(frozen=True)
class InterfaceSpec:
    between: tuple[str, str]
    vacancy_mode: str = "none"
    vacancy_window_angstrom: float = 2.0
    vacancy_seed: int | None = None
    vacancy_counts_by_element: dict[str, int] = field(default_factory=dict)
    substitutions: list[SubstitutionSpec] = field(default_factory=list)


@dataclass(frozen=True)
class SlabTemplate:
    template_path: Path
    project: ProjectSpec
    matching: MatchingSpec
    stack: StackSpec
    export: ExportSpec
    layers: list[LayerSpec]
    interfaces: list[InterfaceSpec]

    @property
    def base_dir(self) -> Path:
        return self.template_path.parent


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib  # py311+
    except ModuleNotFoundError:  # pragma: no cover (py310 fallback)
        import tomli as tomllib  # type: ignore[no-redef]

    return tomllib.loads(path.read_text())


def _require_dict(data: dict[str, Any], key: str) -> dict[str, Any]:
    val = data.get(key)
    if not isinstance(val, dict):
        raise ValueError(f"Missing or invalid TOML table: [{key}]")
    return val


def _to_int(raw: Any, name: str) -> int:
    try:
        return int(raw)
    except Exception as exc:
        raise ValueError(f"Invalid integer for {name}: {raw!r}") from exc


def _to_float(raw: Any, name: str) -> float:
    try:
        return float(raw)
    except Exception as exc:
        raise ValueError(f"Invalid float for {name}: {raw!r}") from exc


def _to_str(raw: Any, name: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Invalid string for {name}: {raw!r}")
    return raw.strip()


def _to_optional_str(raw: Any, name: str) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError(f"Invalid string for {name}: {raw!r}")
    value = raw.strip()
    return value or None


def _parse_miller(raw: Any, *, layer_label: str) -> tuple[int, int, int]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"Layer {layer_label!r}: miller must be [h, k, l]")
    return (
        _to_int(raw[0], f"{layer_label}.miller[0]"),
        _to_int(raw[1], f"{layer_label}.miller[1]"),
        _to_int(raw[2], f"{layer_label}.miller[2]"),
    )


def _parse_layer(raw: dict[str, Any]) -> LayerSpec:
    label = _to_str(raw.get("label"), "layers[].label")
    role = _to_str(raw.get("role", "layer"), f"{label}.role")
    source = _to_str(raw.get("source"), f"{label}.source").lower()
    if source not in {"cif", "mp"}:
        raise ValueError(f"Layer {label!r}: source must be 'cif' or 'mp'")

    cif_path = raw.get("cif_path")
    mp_id = raw.get("mp_id")
    if source == "cif":
        cif_path = _to_str(cif_path, f"{label}.cif_path")
        mp_id = None
    else:
        mp_id = _to_str(mp_id, f"{label}.mp_id")
        cif_path = None

    thickness = _to_float(raw.get("thickness_angstrom"), f"{label}.thickness_angstrom")
    if thickness <= 0:
        raise ValueError(f"Layer {label!r}: thickness_angstrom must be > 0")

    termination_index = _to_int(raw.get("termination_index", 0), f"{label}.termination_index")
    if termination_index < 0:
        raise ValueError(f"Layer {label!r}: termination_index must be >= 0")

    return LayerSpec(
        label=label,
        role=role,
        source=source,
        cif_path=cif_path,
        mp_id=mp_id,
        miller=_parse_miller(raw.get("miller"), layer_label=label),
        thickness_angstrom=thickness,
        termination_index=termination_index,
    )


def _parse_substitution(raw: dict[str, Any], *, where: str) -> SubstitutionSpec:
    from_el = _to_str(raw.get("from"), f"{where}.from")
    to_el = _to_str(raw.get("to"), f"{where}.to")
    count = _to_int(raw.get("count", 0), f"{where}.count")
    if count < 0:
        raise ValueError(f"{where}.count must be >= 0")
    return SubstitutionSpec(from_element=from_el, to_element=to_el, count=count)


def _parse_interface(raw: dict[str, Any], *, idx: int) -> InterfaceSpec:
    between = raw.get("between")
    where = f"interfaces[{idx}]"
    if not isinstance(between, (list, tuple)) or len(between) != 2:
        raise ValueError(f"{where}.between must be [layer_a, layer_b]")
    pair = (_to_str(between[0], f"{where}.between[0]"), _to_str(between[1], f"{where}.between[1]"))

    vacancy_mode = _to_str(raw.get("vacancy_mode", "none"), f"{where}.vacancy_mode").lower()
    if vacancy_mode not in {"none", "random_interface"}:
        raise ValueError(f"{where}.vacancy_mode must be 'none' or 'random_interface'")

    vacancy_window = _to_float(raw.get("vacancy_window_angstrom", 2.0), f"{where}.vacancy_window_angstrom")
    if vacancy_window <= 0:
        raise ValueError(f"{where}.vacancy_window_angstrom must be > 0")

    vacancy_seed_raw = raw.get("vacancy_seed")
    vacancy_seed = None if vacancy_seed_raw is None else _to_int(vacancy_seed_raw, f"{where}.vacancy_seed")

    counts_raw = raw.get("vacancy_counts_by_element", {})
    if not isinstance(counts_raw, dict):
        raise ValueError(f"{where}.vacancy_counts_by_element must be a table/object")
    counts: dict[str, int] = {}
    for key, val in counts_raw.items():
        el = _to_str(key, f"{where}.vacancy_counts_by_element key")
        cnt = _to_int(val, f"{where}.vacancy_counts_by_element[{el}]")
        if cnt < 0:
            raise ValueError(f"{where}.vacancy_counts_by_element[{el}] must be >= 0")
        counts[el] = cnt

    subs_raw = raw.get("substitutions", [])
    if not isinstance(subs_raw, list):
        raise ValueError(f"{where}.substitutions must be an array of tables")
    substitutions = [
        _parse_substitution(s, where=f"{where}.substitutions[{i}]")
        for i, s in enumerate(subs_raw)
        if isinstance(s, dict)
    ]

    return InterfaceSpec(
        between=pair,
        vacancy_mode=vacancy_mode,
        vacancy_window_angstrom=vacancy_window,
        vacancy_seed=vacancy_seed,
        vacancy_counts_by_element=counts,
        substitutions=substitutions,
    )


def load_slab_template(path: str | Path) -> SlabTemplate:
    """Parse and validate slab-generation TOML template."""
    template_path = Path(path).expanduser().resolve()
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    data = _load_toml(template_path)

    project_raw = _require_dict(data, "project")
    matching_raw = _require_dict(data, "matching")
    stack_raw = _require_dict(data, "stack")
    export_raw = _require_dict(data, "export")

    mp_raw = data.get("materials_project", {})
    if not isinstance(mp_raw, dict):
        raise ValueError("[materials_project] must be a TOML table when provided")

    project = ProjectSpec(
        name=_to_str(project_raw.get("name"), "project.name"),
        seed=_to_int(project_raw.get("seed", 0), "project.seed"),
        output_dir=_to_str(project_raw.get("output_dir", "slab_outputs"), "project.output_dir"),
        material=_to_optional_str(project_raw.get("material"), "project.material"),
        mp_api_key=(
            _to_optional_str(project_raw.get("mp_api_key"), "project.mp_api_key")
            or _to_optional_str(mp_raw.get("api_key"), "materials_project.api_key")
        ),
        mp_api_key_env=(
            _to_optional_str(project_raw.get("mp_api_key_env"), "project.mp_api_key_env")
            or _to_optional_str(mp_raw.get("api_key_env"), "materials_project.api_key_env")
        ),
    )
    matching = MatchingSpec(
        max_strain_percent=_to_float(matching_raw.get("max_strain_percent", 5.0), "matching.max_strain_percent"),
        max_area_ratio=_to_float(matching_raw.get("max_area_ratio", 4.0), "matching.max_area_ratio"),
        max_search_supercell=max(1, _to_int(matching_raw.get("max_search_supercell", 4), "matching.max_search_supercell")),
    )
    stack = StackSpec(
        align_axis=_to_str(stack_raw.get("align_axis", "z"), "stack.align_axis").lower(),
        vacuum_angstrom=_to_float(stack_raw.get("vacuum_angstrom", 18.0), "stack.vacuum_angstrom"),
        interface_gap_angstrom=_to_float(stack_raw.get("interface_gap_angstrom", 2.2), "stack.interface_gap_angstrom"),
    )
    if stack.align_axis != "z":
        raise ValueError("Only stack.align_axis='z' is currently supported")
    if stack.vacuum_angstrom <= 0:
        raise ValueError("stack.vacuum_angstrom must be > 0")

    export = ExportSpec(
        cif_path=_to_str(export_raw.get("cif_path"), "export.cif_path"),
        metadata_json_path=_to_str(export_raw.get("metadata_json_path"), "export.metadata_json_path"),
    )

    layers_raw = data.get("layers")
    if not isinstance(layers_raw, list) or not layers_raw:
        raise ValueError("Template must define at least one [[layers]] section")
    layers = [_parse_layer(layer) for layer in layers_raw if isinstance(layer, dict)]
    if len(layers) < 2:
        raise ValueError("At least two layers are required for stacking")

    labels = [layer.label for layer in layers]
    if len(set(labels)) != len(labels):
        raise ValueError("Layer labels must be unique")

    interfaces_raw = data.get("interfaces", [])
    if not isinstance(interfaces_raw, list):
        raise ValueError("interfaces must be an array of tables")
    interfaces = [_parse_interface(item, idx=i) for i, item in enumerate(interfaces_raw) if isinstance(item, dict)]
    known = set(labels)
    for iface in interfaces:
        a, b = iface.between
        if a not in known or b not in known:
            raise ValueError(f"Interface references unknown layer(s): {a!r}, {b!r}")

    return SlabTemplate(
        template_path=template_path,
        project=project,
        matching=matching,
        stack=stack,
        export=export,
        layers=layers,
        interfaces=interfaces,
    )
