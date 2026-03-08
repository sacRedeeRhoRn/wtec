from wtec.topology.variant_discovery import _severity_from_metadata


def test_severity_uses_interface_window_atoms() -> None:
    meta = {
        "interfaces": [
            {
                "atoms_removed": 1,
                "atoms_in_window": 20,
                "vacancies": [{"applied": 99}],  # ignored (already represented by atoms_removed)
                "substitutions": [{"applied": 0}],
            },
            {
                "atoms_removed": 0,
                "atoms_in_window": 20,
                "substitutions": [{"applied": 0}],
            },
        ]
    }
    sev = _severity_from_metadata(meta)
    # total_events = 1, interface_atoms_total = 40 => 1 / (0.05*40) = 0.5
    assert abs(sev - 0.5) < 1e-12


def test_severity_clips_to_one() -> None:
    meta = {
        "interfaces": [
            {
                "atoms_removed": 2,
                "atoms_in_window": 10,
                "substitutions": [{"applied": 1}],
            }
        ]
    }
    sev = _severity_from_metadata(meta)
    assert sev == 1.0

