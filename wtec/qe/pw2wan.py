"""pw2wannier90.x input generator."""

from __future__ import annotations

from pathlib import Path


def generate(
    prefix: str,
    outdir: str = "./out",
    *,
    seedname: str | None = None,
    spin_component: str = "none",  # "none" | "up" | "down"
    write_mmn: bool = True,
    write_amn: bool = True,
    write_unk: bool = False,
    wan_mode: str = "standalone",  # "standalone" | "library"
    outfile: str | Path | None = None,
) -> str:
    """Generate pw2wannier90.x input.

    Parameters
    ----------
    prefix : str
        QE calculation prefix (matches pw.x prefix).
    seedname : str | None
        Wannier90 seedname. Defaults to prefix.
    """
    seed = seedname or prefix
    unk_line = "  write_unk = .true." if write_unk else ""

    text = f""" &inputpp
  outdir = '{outdir}'
  prefix = '{prefix}'
  seedname = '{seed}'
  spin_component = '{spin_component}'
  write_mmn = {'.' + ('true' if write_mmn else 'false') + '.'}
  write_amn = {'.' + ('true' if write_amn else 'false') + '.'}
  wan_mode = '{wan_mode}'
{unk_line}
 /
"""
    if outfile:
        Path(outfile).write_text(text)
    return text
