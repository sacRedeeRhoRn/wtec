import numpy as np

from wtec.transport.mfp import extract_mfp_from_scaling


def test_extract_mfp_from_scaling_recovers_effective_length() -> None:
    # Synthetic Landauer crossover: G(L) = G0 * l / (L + l)
    l_true_m = 2.5e-9
    g0_e2h = 16.0
    lengths_m = np.array([1e-9, 2e-9, 3e-9, 4e-9, 5e-9], dtype=float)
    g_mean_e2h = g0_e2h * l_true_m / (lengths_m + l_true_m)

    out = extract_mfp_from_scaling(
        lengths_m=lengths_m,
        G_mean=g_mean_e2h,
        cross_section_m2=1.0e-18,
    )

    assert "error" not in out
    assert out.get("mfp_m") is not None
    assert out.get("mfp_nm") is not None
    assert abs(float(out["mfp_m"]) - l_true_m) / l_true_m < 1.0e-6
    assert float(out["sigma_S_per_m"]) > 0.0
    assert float(out["fit_R2"]) > 0.999999
