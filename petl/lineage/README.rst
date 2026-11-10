Row-level lineage tracking
==========================

:mod:`petl.lineage` adds optional, out-of-band provenance to a petl pipeline.
The ordinary data columns and values are unchanged; provenance is read through
a separate interface.

Quick start
-----------

.. code-block:: python

    import petl as etl
    from petl import lineage

    sales = lineage.tracked(etl.fromcsv('sales.csv'), source_id='sales.csv')
    master = lineage.tracked(etl.fromcsv('master.csv'), source_id='master.csv')

    joined = lineage.leftjoin(sales, master, key='sku')
    summary = lineage.aggregate(joined, 'sku', sum, 'qty')

    # data is a normal petl table
    list(summary)

    # provenance of each output data row
    summary.coords()
    summary.explain(0)  # frozenset of SourceCoord for the first data row

``SourceCoord(source_id, row_number)`` identifies one physical input row.
``row_number`` is 1-based over data rows; ``file_line_number`` adds the single
header row, so it is the line number in a headered file.

Supported transforms
--------------------

``cut``, ``rename``, ``select``, ``innerjoin``, ``leftjoin`` and
``aggregate`` (simple callable and multi-field forms). Each mirrors the
matching petl transform's header and value rules.

* ``cut`` / ``rename`` / ``select`` keep the single upstream line per row.
* ``innerjoin`` unions the left and right line for every match; many-to-many
  results share immutable sub-trees instead of copying history per row.
* ``leftjoin`` matches union both sides; padded, unmatched rows keep the left
  line only.
* ``aggregate`` unions the lines of every input row in the group.

Repeated values in different physical rows always have distinct provenance.

Storage and iteration
---------------------

Provenance is an immutable persistent union tree (:class:`Line`). A join of
``n`` by ``m`` rows builds ``n + m`` small nodes and shares them across all
results. Tracked stages snapshot their rows once, so the data interface and
the lineage interface can be read in any order, repeatedly, without replaying
a one-shot input iterator.

Losing tracking
---------------

Lineage only survives the transforms listed above. Applying an ordinary petl
transform (or passing an untracked table) into a lineage transform raises
:class:`LineageLostError` at construction, rather than returning a partial
provenance set that could be mistaken for a complete answer.
