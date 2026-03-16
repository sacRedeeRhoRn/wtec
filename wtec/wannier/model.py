"""WannierTBModel: tbmodels-backed TB model with Kwant integration."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass   # avoid hard kwant import at module level


class WannierTBModel:
    """Tight-binding model loaded from Wannier90 output.

    Wraps a tbmodels.Model (or a raw HoppingData if tbmodels is unavailable)
    and exposes a Kwant Builder factory for the trilayer slab geometry.
    """

    def __init__(self, model, lattice_vectors: np.ndarray) -> None:
        """
        Parameters
        ----------
        model : tbmodels.Model
            TB model loaded via tbmodels.
        lattice_vectors : np.ndarray shape (3, 3)
            Real-space lattice vectors (rows), in Angstroms.
        """
        self._model = model
        self.lattice_vectors = np.asarray(lattice_vectors, dtype=float)
        self._num_orbs = model.size

    # ── constructors ─────────────────────────────────────────────────────────

    @classmethod
    def from_hr_dat(
        cls,
        hr_dat_path: str | Path,
        win_path: str | Path | None = None,
    ) -> "WannierTBModel":
        """Load from Wannier90 _hr.dat via tbmodels.

        Parameters
        ----------
        hr_dat_path : str or Path
            Path to *_hr.dat file.
        win_path : str or Path | None
            Path to *.win file (needed for lattice vectors).
            If None, uses a default identity lattice.
        """
        try:
            import tbmodels
        except ImportError:
            raise ImportError("tbmodels is required: pip install tbmodels")

        hr_dat_path = Path(hr_dat_path)
        folder = hr_dat_path.parent
        seedname = hr_dat_path.name.replace("_hr.dat", "")

        model = tbmodels.Model.from_wannier_folder(
            folder=str(folder),
            prefix=seedname,
        )

        # Extract lattice vectors from .win file if provided
        if win_path is not None:
            lv = _parse_lattice_from_win(win_path)
        elif (folder / f"{seedname}.win").exists():
            lv = _parse_lattice_from_win(folder / f"{seedname}.win")
        else:
            lv = model.uc if model.uc is not None else np.eye(3)

        return cls(model, lattice_vectors=np.array(lv))

    @classmethod
    def from_hopping_data(cls, hd, lattice_vectors: np.ndarray) -> "WannierTBModel":
        """Build from raw HoppingData (wtec.wannier.parser.HoppingData).

        Uses a lightweight internal model without tbmodels dependency.
        """
        return cls(_HoppingDataWrapper(hd), lattice_vectors)

    # ── properties ───────────────────────────────────────────────────────────

    @property
    def num_orbitals(self) -> int:
        return self._num_orbs

    def hamiltonian_at_k(self, k_frac: np.ndarray) -> np.ndarray:
        """Compute H(k) at fractional k-point. Returns (nw × nw) complex array."""
        return np.array(self._model.hamilton(k_frac, convention=2), dtype=complex)

    def bands(self, k_path: np.ndarray) -> np.ndarray:
        """Interpolate eigenvalues along k_path (fractional coords).

        Returns
        -------
        np.ndarray shape (n_k, num_wann)
        """
        return np.array([
            np.linalg.eigvalsh(self.hamiltonian_at_k(k))
            for k in k_path
        ])

    # ── Kwant builder factory ─────────────────────────────────────────────────

    def to_kwant_builder(
        self,
        *,
        n_layers_z: int,
        n_layers_x: int = 1,
        n_layers_y: int = 4,
        lead_axis: str = "x",
        periodic_xy: bool = True,
        substrate_onsite_eV: float = 0.0,
        substrate_layers: int = 0,
    ):
        """Build a Kwant system for the trilayer slab geometry.

        Geometry
        --------
        Lead direction is configurable via ``lead_axis``.
        Typical topological-film setup uses in-plane transport (lead_axis="x")
        and hard-wall surfaces along z.

        Parameters
        ----------
        n_layers_z : int
            Number of unit cells in the z direction (film thickness).
        n_layers_x : int
            Number of unit cells in the x direction (transport length).
        n_layers_y : int
            Number of unit cells in the y direction (transverse width).
            Must be >= 1. Values >= 4 are recommended for arc-sensitive transport.
        lead_axis : str
            Lead attachment axis: "x", "y", or "z".
        periodic_xy : bool
            If True, use kwant.wraparound for k_x, k_y averaging.
        substrate_onsite_eV : float
            Onsite energy (eV) for lead sites.
            0 gives metallic contacts; large values make contacts insulating.
        substrate_layers : int
            Number of substrate unit-cell layers to include at each surface.
            0 = hard wall (vacuum boundary).

        Returns
        -------
        kwant.Builder
        """
        try:
            import kwant
        except ImportError:
            raise ImportError(
                "kwant is required. Install it from source: "
                "pip install -e /path/to/kwant"
            )

        if n_layers_x <= 0 or n_layers_y <= 0 or n_layers_z <= 0:
            raise ValueError("n_layers_x, n_layers_y and n_layers_z must be > 0")
        if n_layers_x < 2:
            raise ValueError(
                f"n_layers_x={n_layers_x} is too small for Kwant lead attachment. "
                "The scattering region must span at least 2 unit cells along the lead "
                "axis so that it interrupts both semi-infinite leads. "
                "With n_layers_x=1 Kwant raises 'does not interrupt the lead' and "
                "conductance returns 0 silently. "
                "Set transport_n_layers_x >= 4 in your config (minimum valid: 2)."
            )
        axis_map = {"x": 0, "y": 1, "z": 2}
        axis_key = lead_axis.lower().strip()
        if axis_key not in axis_map:
            raise ValueError(f"lead_axis must be one of ['x', 'y', 'z'], got {lead_axis!r}")
        lead_axis_idx = axis_map[axis_key]

        shape = (int(n_layers_x), int(n_layers_y), int(n_layers_z))
        required_axis_cells = self.required_lead_axis_cells(
            lead_axis=axis_key,
            n_layers_x=shape[0],
            n_layers_y=shape[1],
            n_layers_z=shape[2],
        )
        if shape[lead_axis_idx] < required_axis_cells:
            axis_name = ("x", "y", "z")[lead_axis_idx]
            raise ValueError(
                f"{axis_name}-axis cells={shape[lead_axis_idx]} is too small for this "
                f"Wannier Hamiltonian (required >= {required_axis_cells} for lead_axis={axis_key}). "
                "Increase transport length along the lead axis."
            )

        nw = self._num_orbs

        # Lattice for the slab: one site per unit cell, orbital dimension = nw
        lat = kwant.lattice.general(
            self.lattice_vectors,
            norbs=nw,
        )

        sys = kwant.Builder()

        # ── Film sites ─────────────────────────────────────────────────────
        h0 = np.array(self._model.hamilton([0, 0, 0], convention=2), dtype=complex)
        for ix in range(n_layers_x):
            for iy in range(n_layers_y):
                for iz in range(n_layers_z):
                    sys[lat(ix, iy, iz)] = h0

        # ── Hoppings ───────────────────────────────────────────────────────
        self._add_hoppings_region(sys, lat, shape)

        # ── Left and Right leads (semi-infinite along lead_axis) ─────────
        lead_vec = self.lattice_vectors[lead_axis_idx]
        left_lead = self._make_lead(
            lat,
            shape=shape,
            lead_axis_idx=lead_axis_idx,
            lead_vec=lead_vec,
            onsite_eV=substrate_onsite_eV,
            substrate_layers=substrate_layers,
        )
        right_lead = left_lead.reversed()

        sys.attach_lead(left_lead)
        sys.attach_lead(right_lead)

        return sys

    def to_kwant_builder_periodic_y(
        self,
        *,
        ky_frac: float,
        n_layers_z: int,
        n_layers_x: int = 1,
        lead_axis: str = "x",
        substrate_onsite_eV: float = 0.0,
    ):
        """Build a Kwant system reduced by an exact Fourier transform along y.

        This is intended for pristine clean transport with periodic transverse
        boundary conditions along y. The returned system is finite in x and z and
        semi-infinite along the selected lead axis. Current production support is
        limited to in-plane transport along x.
        """
        try:
            import kwant
        except ImportError:
            raise ImportError(
                "kwant is required. Install it from source: "
                "pip install -e /path/to/kwant"
            )

        axis_key = str(lead_axis).lower().strip()
        if axis_key != "x":
            raise NotImplementedError(
                "Periodic-y clean transport is currently implemented only for lead_axis='x'."
            )
        if int(n_layers_x) <= 0 or int(n_layers_z) <= 0:
            raise ValueError("n_layers_x and n_layers_z must be > 0")

        shape_xz = (int(n_layers_x), int(n_layers_z))
        required_axis_cells = self.required_lead_axis_cells(
            lead_axis="x",
            n_layers_x=shape_xz[0],
            n_layers_y=1,
            n_layers_z=shape_xz[1],
            periodic_axes=("y",),
        )
        if shape_xz[0] < required_axis_cells:
            raise ValueError(
                f"x-axis cells={shape_xz[0]} is too small for this periodic-y Wannier Hamiltonian "
                f"(required >= {required_axis_cells} for lead_axis='x')."
            )

        nw = self._num_orbs
        ky = float(np.mod(float(ky_frac), 1.0))
        two_pi = 2.0 * np.pi

        lat = kwant.lattice.general(
            [self.lattice_vectors[0], self.lattice_vectors[2]],
            norbs=nw,
        )

        def _phase(ry: int) -> complex:
            return np.exp(1j * two_pi * ky * float(int(ry)))

        h0 = np.zeros((nw, nw), dtype=complex)
        hoppings_2d: dict[tuple[int, int], np.ndarray] = {}
        for rx, ry, rz, mat in self._iter_hoppings():
            phased = _phase(ry) * np.asarray(mat, dtype=complex)
            if int(rx) == 0 and int(rz) == 0:
                h0 += phased
            else:
                key = (int(rx), int(rz))
                hoppings_2d[key] = hoppings_2d.get(key, np.zeros((nw, nw), dtype=complex)) + phased

        h0 = 0.5 * (h0 + h0.conj().T)
        for (rx, rz), mat in list(hoppings_2d.items()):
            inv_key = (-int(rx), -int(rz))
            inv_mat = hoppings_2d.get(inv_key)
            if inv_mat is None:
                hoppings_2d[inv_key] = np.asarray(mat, dtype=complex).conj().T
                continue
            sym_mat = 0.5 * (np.asarray(mat, dtype=complex) + np.asarray(inv_mat, dtype=complex).conj().T)
            hoppings_2d[(int(rx), int(rz))] = sym_mat
            hoppings_2d[inv_key] = sym_mat.conj().T

        sys = kwant.Builder()
        nx, nz = shape_xz
        for ix in range(nx):
            for iz in range(nz):
                sys[lat(ix, iz)] = h0

        for (rx, rz), mat in hoppings_2d.items():
            if int(rx) < 0 or (int(rx) == 0 and int(rz) < 0):
                continue
            for ix in range(nx):
                ix2 = ix + int(rx)
                if not (0 <= ix2 < nx):
                    continue
                for iz in range(nz):
                    iz2 = iz + int(rz)
                    if 0 <= iz2 < nz:
                        try:
                            sys[lat(ix, iz), lat(ix2, iz2)] = mat
                        except Exception:
                            pass

        lead = kwant.Builder(kwant.TranslationalSymmetry(self.lattice_vectors[0]))
        onsite_lead = h0 + np.eye(nw, dtype=complex) * float(substrate_onsite_eV)
        for iz in range(nz):
            lead[lat(0, iz)] = onsite_lead

        for (rx, rz), mat in hoppings_2d.items():
            if int(rx) < 0:
                continue
            if int(rx) == 0 and int(rz) < 0:
                continue
            for iz in range(nz):
                iz2 = iz + int(rz)
                if not (0 <= iz2 < nz):
                    continue
                try:
                    lead[lat(0, iz), lat(int(rx), iz2)] = mat
                except Exception:
                    pass

        sys.attach_lead(lead)
        sys.attach_lead(lead.reversed())
        return sys

    def required_lead_axis_cells(
        self,
        *,
        lead_axis: str,
        n_layers_x: int,
        n_layers_y: int,
        n_layers_z: int,
        periodic_axes: tuple[str, ...] = (),
    ) -> int:
        """Return minimal finite-cell count along lead axis for stable lead attachment.

        Requirement is inferred from real-space hopping range that remains active
        within the provided finite cross-section.
        """
        axis_map = {"x": 0, "y": 1, "z": 2}
        axis_key = str(lead_axis).lower().strip()
        if axis_key not in axis_map:
            raise ValueError(f"lead_axis must be one of ['x', 'y', 'z'], got {lead_axis!r}")

        periodic_axis_set = set()
        for ax in periodic_axes:
            ax_key = str(ax).lower().strip()
            if ax_key not in axis_map:
                raise ValueError(f"periodic axis must be one of ['x', 'y', 'z'], got {ax!r}")
            periodic_axis_set.add(ax_key)

        shape = [int(n_layers_x), int(n_layers_y), int(n_layers_z)]
        if any(v <= 0 for v in shape):
            raise ValueError("n_layers_x, n_layers_y and n_layers_z must be > 0")

        lead_axis_idx = axis_map[axis_key]
        finite_axes = [
            ax
            for ax in (0, 1, 2)
            if ax != lead_axis_idx and ("xyz"[ax] not in periodic_axis_set)
        ]

        max_abs_span = 0
        for rx, ry, rz, _ in self._iter_hoppings():
            dR = (int(rx), int(ry), int(rz))
            # Keep only hoppings that can exist within this finite cross-section.
            if any(abs(dR[ax]) > (shape[ax] - 1) for ax in finite_axes):
                continue
            max_abs_span = max(max_abs_span, abs(dR[lead_axis_idx]))

        return max(2, int(max_abs_span))

    def _iter_hoppings(self):
        """Yield (rx, ry, rz, matrix) hoppings for full 3D slab construction."""
        model = self._model
        # tbmodels stores hoppings as {R: matrix}
        if hasattr(model, 'hop'):
            hoppings = model.hop
        else:
            hoppings = _get_hoppings_from_wrapper(model)

        for (R, matrix) in hoppings.items():
            rx, ry, rz = int(R[0]), int(R[1]), int(R[2])
            yield rx, ry, rz, np.array(matrix, dtype=complex)

    def _add_hoppings_region(self, sys, lat, shape: tuple[int, int, int]) -> None:
        """Add hoppings for a finite x-y-z scattering region."""
        nx, ny, nz = shape
        for rx, ry, rz, mat in self._iter_hoppings():
            for ix in range(nx):
                ix2 = ix + rx
                if not (0 <= ix2 < nx):
                    continue
                for iy in range(ny):
                    iy2 = iy + ry
                    if not (0 <= iy2 < ny):
                        continue
                    for iz in range(nz):
                        iz2 = iz + rz
                        if 0 <= iz2 < nz:
                            try:
                                sys[lat(ix, iy, iz), lat(ix2, iy2, iz2)] = mat
                            except Exception:
                                pass   # skip duplicates/symmetry conflicts

    def _add_hoppings_lead(
        self,
        lead,
        lat,
        *,
        shape: tuple[int, int, int],
        lead_axis_idx: int,
    ) -> None:
        """Add hoppings for one lead unit cell (translation handled by symmetry)."""
        nx, ny, nz = shape
        finite_axes = [ax for ax in (0, 1, 2) if ax != lead_axis_idx]
        ranges = [range(nx), range(ny), range(nz)]

        # Unit-cell anchor: coordinate 0 along lead axis.
        for rx, ry, rz, mat in self._iter_hoppings():
            dR = (rx, ry, rz)
            # For lead construction, keep only one translation direction.
            # Including both ±R along lead axis can create over-wide coupling blocks.
            if dR[lead_axis_idx] < 0:
                continue
            for ix in ranges[0]:
                for iy in ranges[1]:
                    for iz in ranges[2]:
                        src = [ix, iy, iz]
                        if src[lead_axis_idx] != 0:
                            continue
                        dst = [src[0] + dR[0], src[1] + dR[1], src[2] + dR[2]]
                        ok = True
                        for ax in finite_axes:
                            bound = shape[ax]
                            if not (0 <= dst[ax] < bound):
                                ok = False
                                break
                        if not ok:
                            continue
                        try:
                            lead[lat(src[0], src[1], src[2]), lat(dst[0], dst[1], dst[2])] = mat
                        except Exception:
                            pass   # skip duplicates/symmetry conflicts

    def _make_lead(
        self,
        lat,
        *,
        shape: tuple[int, int, int],
        lead_axis_idx: int,
        lead_vec: np.ndarray,
        onsite_eV: float,
        substrate_layers: int,
    ):
        """Build a semi-infinite lead along the selected lead axis."""
        import kwant
        sym = kwant.TranslationalSymmetry(lead_vec)
        lead = kwant.Builder(sym)

        h0 = np.array(self._model.hamilton([0, 0, 0], convention=2), dtype=complex)
        nx, ny, nz = shape

        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    coord = (ix, iy, iz)
                    if coord[lead_axis_idx] != 0:
                        continue
                    # Substrate: large onsite shifts all energies away from Fermi level
                    lead[lat(ix, iy, iz)] = h0 + np.eye(self._num_orbs) * onsite_eV

        self._add_hoppings_lead(
            lead,
            lat,
            shape=shape,
            lead_axis_idx=lead_axis_idx,
        )
        return lead


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_lattice_from_win(win_path) -> np.ndarray:
    """Extract unit_cell_cart block from .win file."""
    text = Path(win_path).read_text()
    import re
    m = re.search(
        r"begin unit_cell_cart\s*\n\s*(?:ang|bohr)?\s*\n(.*?)end unit_cell_cart",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return np.eye(3)
    rows = []
    for line in m.group(1).strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 3:
            rows.append([float(x) for x in parts])
    return np.array(rows)


class _HoppingDataWrapper:
    """Minimal wrapper so HoppingData looks like tbmodels.Model."""

    def __init__(self, hd) -> None:
        self._hd = hd
        self.size = hd.num_wann
        self.uc = None
        # Build hop dict from HoppingData
        self.hop = {}
        for ri, R in enumerate(hd.r_vectors):
            key = tuple(R.tolist())
            self.hop[key] = hd.H_R[ri] / hd.deg[ri]

    def hamilton(self, k, convention=2) -> np.ndarray:
        nw = self.size
        H_k = np.zeros((nw, nw), dtype=complex)
        for R, mat in self.hop.items():
            phase = np.exp(2j * np.pi * np.dot(k, R))
            H_k += phase * mat
        return H_k


def _get_hoppings_from_wrapper(model) -> dict:
    if hasattr(model, 'hop'):
        return model.hop
    raise AttributeError("Cannot extract hoppings from model")
