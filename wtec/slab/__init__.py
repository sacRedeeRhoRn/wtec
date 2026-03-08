"""Template-driven slab generation and reporting utilities."""

from .template import SlabTemplate, load_slab_template
from .generator import generate_slab_from_template
from .report import render_slab_report

__all__ = [
    "SlabTemplate",
    "load_slab_template",
    "generate_slab_from_template",
    "render_slab_report",
]

