import wtec.cli as cli


def test_transport_signature_detects_minimum_and_huge_mfp() -> None:
    transport_results = {
        "thickness_scan": {
            0.0: {
                "thickness_uc": [2, 4, 6, 8, 10],
                "thickness_m": [2e-9, 4e-9, 6e-9, 8e-9, 1.0e-8],
                "rho_mean": [0.8, 0.5, 0.4, 0.6, 0.9],
            }
        },
        "mfp": {
            "mfp_nm": 250.0,
        },
    }
    sig = cli._compute_transport_signature(transport_results, min_mfp_nm=100.0)
    assert sig["has_curve"] is True
    assert sig["has_rho_minimum"] is True
    assert sig["thinning_reduces_rho"] is True
    assert sig["rho_trend_class"] == "u_shaped_nonmonotonic"
    assert sig["mfp_available"] is True
    assert sig["mfp_huge"] is True


def test_transport_signature_handles_missing_curve() -> None:
    sig = cli._compute_transport_signature({}, min_mfp_nm=100.0)
    assert sig["has_curve"] is False
    assert sig["mfp_available"] is False


def test_transport_signature_classifies_monotonic_decrease_with_thickness() -> None:
    transport_results = {
        "thickness_scan": {
            0.0: {
                "thickness_uc": [2, 4, 6, 8],
                "thickness_m": [2e-9, 4e-9, 6e-9, 8e-9],
                "rho_mean": [1.2, 1.0, 0.8, 0.6],
            }
        },
        "mfp": {"mfp_nm": 50.0},
    }
    sig = cli._compute_transport_signature(transport_results, min_mfp_nm=100.0)
    assert sig["has_curve"] is True
    assert sig["has_rho_minimum"] is False
    assert sig["thinning_reduces_rho"] is False
    assert sig["rho_trend_class"] == "monotonic_decreasing_with_thickness"
