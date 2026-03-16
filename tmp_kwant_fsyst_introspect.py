from pathlib import Path
import importlib.util
import json

spec = importlib.util.spec_from_file_location('kwant_par_test_mod', '/home/msj/MODELING/rgf_phase2_tis_compare_20260310/kwant_par_test.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

hr_path = '/home/msj/Desktop/playground/electronics/parallealization_test/TiS_hr.dat'
norb, H_R = mod.read_wannier90_hr(hr_path)
fsyst, add_cells = mod.build_system_from_HR(H_R, L=16, W=8, H=5, Ef=0.0, transport_axis=0)
info = {
    'n_sites': len(fsyst.sites),
    'lead_interfaces_len': [len(x) for x in fsyst.lead_interfaces],
    'lead_paddings_len': [len(x) for x in fsyst.lead_paddings],
    'lead_interfaces_head': [list(map(int, list(x[:16]))) for x in fsyst.lead_interfaces],
    'lead_paddings_head': [list(map(int, list(x[:16]))) for x in fsyst.lead_paddings],
    'lead0_cell_shape': list(fsyst.leads[0].cell_hamiltonian().shape),
    'lead0_hop_shape': list(fsyst.leads[0].inter_cell_hopping().shape),
    'x_min': min(int(s.tag[0]) for s in fsyst.sites),
    'x_max': max(int(s.tag[0]) for s in fsyst.sites),
    'unique_x_count': len(sorted({int(s.tag[0]) for s in fsyst.sites})),
}
print(json.dumps(info, indent=2))
