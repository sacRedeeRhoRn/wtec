from wtec.structure.metadata_schema import validate_slab_metadata


def _base_metadata() -> dict:
    return {
        "wtec_slab_metadata_version": 2,
        "project": {"name": "test"},
        "layers": [
            {
                "label": "a",
                "role": "substrate",
                "z_min_angstrom": 0.0,
                "z_max_angstrom": 1.0,
            },
            {
                "label": "b",
                "role": "active",
                "z_min_angstrom": 1.0,
                "z_max_angstrom": 2.0,
            },
        ],
        "interfaces": [
            {
                "between": ["a", "b"],
                "atoms_removed": 0,
                "atoms_in_window": 12,
                "vacancies": [],
                "substitutions": [],
            }
        ],
        "summary": {"atoms_before_defects": 24, "atoms_after_defects": 24},
        "export": {"cif_path": "x.cif"},
    }


def test_metadata_schema_accepts_atoms_in_window() -> None:
    ok, errors = validate_slab_metadata(_base_metadata())
    assert ok
    assert errors == []


def test_metadata_schema_requires_atoms_in_window() -> None:
    meta = _base_metadata()
    del meta["interfaces"][0]["atoms_in_window"]
    ok, errors = validate_slab_metadata(meta)
    assert not ok
    assert any("atoms_in_window" in e for e in errors)

