from wtec.transport.kwant_block_extract import _group_x_layers_keep_boundary_cells


def test_group_x_layers_keep_boundary_cells_preserves_both_boundaries() -> None:
    unique_x = list(range(-8, 24))
    groups = _group_x_layers_keep_boundary_cells(unique_x, 3)
    assert groups[0] == [-8, -7, -6]
    assert groups[-1] == [21, 22, 23]
    assert groups[-2] == [16, 17, 18, 19, 20]
    assert sum(len(group) for group in groups) == len(unique_x)


def test_group_x_layers_keep_boundary_cells_falls_back_to_uniform_when_exact() -> None:
    unique_x = list(range(12))
    groups = _group_x_layers_keep_boundary_cells(unique_x, 3)
    assert groups == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
