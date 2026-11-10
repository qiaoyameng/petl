"""Optional row-level lineage tracking for petl tables.

The public surface mirrors a small subset of the petl transformation API
(``cut``, ``rename``, ``select``, ``innerjoin``, ``leftjoin`` and
``aggregate``).  Data columns and values are produced by exactly the same
rules as plain petl; provenance is kept out-of-band in parallel and exposed
through :class:`~petl.lineage.lineage.TrackedTable`.
"""

from petl.lineage.lineage import (
    Line,
    LineageLostError,
    SourceCoord,
    TrackedTable,
    TrackedSource,
    tracked,
    cut,
    rename,
    select,
    innerjoin,
    leftjoin,
    aggregate,
)

__all__ = [
    'Line',
    'LineageLostError',
    'SourceCoord',
    'TrackedTable',
    'TrackedSource',
    'tracked',
    'cut',
    'rename',
    'select',
    'innerjoin',
    'leftjoin',
    'aggregate',
]
