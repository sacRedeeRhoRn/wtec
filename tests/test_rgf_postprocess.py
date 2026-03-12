from pathlib import Path

from wtec.transport.rgf_postprocess import convert_rgf_raw_to_transport_results


def test_convert_rgf_raw_to_transport_results_shapes_output(tmp_path: Path) -> None:
    win = tmp_path / "toy.win"
    win.write_text(
        "\n".join(
            [
                "begin unit_cell_cart",
                "ang",
                "1.0 0.0 0.0",
                "0.0 1.0 0.0",
                "0.0 0.0 1.0",
                "end unit_cell_cart",
            ]
        )
        + "\n"
    )

    raw = {
        "engine": "rgf",
        "mode": "periodic_transverse",
        "periodic_axis": "y",
        "lead_axis": "x",
        "thickness_axis": "z",
        "n_layers_x": 3,
        "n_layers_y": 1,
        "mfp_n_layers_z": 1,
        "thicknesses": [1, 2],
        "thickness_G": [1.0, 2.0],
        "thickness_p_eff": [1, 1],
        "thickness_slice_count": [1, 2],
        "thickness_superslice_dim": [4, 8],
        "mfp_lengths": [2, 4, 6],
        "length_G": [1.0, 0.5, 1.0 / 3.0],
        "length_p_eff": [1, 1, 1],
        "length_slice_count": [1, 1, 1],
        "length_superslice_dim": [4, 4, 4],
    }

    out = convert_rgf_raw_to_transport_results(
        raw,
        win_path=win,
        disorder_key=0.0,
    )

    scan = out["thickness_scan"][0.0]
    assert scan["thickness_uc"] == [1, 2]
    assert scan["G_mean"] == [1.0, 2.0]
    assert len(scan["rho_mean"]) == 2
    assert out["mfp"]["G_vs_L"]["length_uc"] == [2, 4, 6]
    assert out["meta"]["transport_engine"] == "rgf"
    assert out["meta"]["rgf_mode"] == "periodic_transverse"


def test_convert_rgf_raw_to_transport_results_supports_multi_disorder_output(
    tmp_path: Path,
) -> None:
    win = tmp_path / "toy.win"
    win.write_text(
        "\n".join(
            [
                "begin unit_cell_cart",
                "ang",
                "1.0 0.0 0.0",
                "0.0 1.0 0.0",
                "0.0 0.0 1.0",
                "end unit_cell_cart",
            ]
        )
        + "\n"
    )

    raw = {
        "engine": "rgf",
        "mode": "full_finite",
        "periodic_axis": "y",
        "lead_axis": "x",
        "thickness_axis": "z",
        "n_layers_x": 4,
        "n_layers_y": 1,
        "mfp_n_layers_z": 1,
        "disorder_strengths": [0.0, 0.2],
        "n_ensemble": 4,
        "thicknesses": [1, 2],
        "thickness_G": [[1.0, 0.8], [0.9, 0.6]],
        "thickness_G_std": [[0.0, 0.0], [0.05, 0.04]],
        "thickness_p_eff": [1, 1],
        "thickness_slice_count": [1, 2],
        "thickness_superslice_dim": [4, 8],
        "mfp_lengths": [2, 4],
        "length_disorder_strength": 0.2,
        "length_G": [0.7, 0.5],
        "length_G_std": [0.03, 0.02],
        "length_p_eff": [1, 1],
        "length_slice_count": [1, 1],
        "length_superslice_dim": [4, 4],
    }

    out = convert_rgf_raw_to_transport_results(raw, win_path=win)

    assert set(out["thickness_scan"]) == {0.0, 0.2}
    assert out["thickness_scan"][0.0]["G_std"] == [0.0, 0.0]
    assert out["thickness_scan"][0.2]["G_mean"] == [0.9, 0.6]
    assert out["thickness_scan"][0.2]["G_std"] == [0.05, 0.04]
    assert out["mfp"]["G_vs_L"]["G_std"] == [0.03, 0.02]
    assert out["mfp"]["disorder_strength"] == 0.2
    assert out["meta"]["n_ensemble"] == 4
    assert out["meta"]["disorder_strengths"] == [0.0, 0.2]
