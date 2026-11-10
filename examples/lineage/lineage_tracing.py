"""Row-level lineage: trace a reconciliation error back to file line numbers.

Scenario
--------
Two warehouse input files contain repeated values, unmatched rows and a
many-to-many relationship.  After a cut/select/left-join/aggregate pipeline a
reconciliation summary row looks wrong.  This script shows how to take that
output row and list the exact physical lines in the source files that fed it,
without adding any provenance column to the data.

Run from the repository root::

    python examples/lineage/lineage_tracing.py
"""

from __future__ import absolute_import, division, print_function

import csv
import os
import tempfile

import petl as etl
from petl import lineage


SALES_ROWS = [
    # store, sku, qty  -- note the two identical (S1, SKU1, 5) data rows and
    # store S3 which has no matching master record.
    ['store', 'sku', 'qty'],
    ['S1', 'SKU1', '5'],
    ['S1', 'SKU1', '5'],
    ['S2', 'SKU2', '7'],
    ['S3', 'SKU9', '9'],
]

MASTER_ROWS = [
    # sku, price -- two price rows for SKU1 create a many-to-many join;
    # SKU7 has no sales and so never appears in a left join.
    ['sku', 'price'],
    ['SKU1', '100'],
    ['SKU1', '120'],
    ['SKU2', '30'],
    ['SKU7', '55'],
]


def write_csv(path, rows):
    with open(path, 'w', newline='') as handle:
        csv.writer(handle).writerows(rows)


def load_lines(path):
    with open(path, 'r') as handle:
        return handle.read().splitlines()


def main():
    workdir = tempfile.mkdtemp(prefix='petl-lineage-')
    sales_path = os.path.join(workdir, 'sales.csv')
    master_path = os.path.join(workdir, 'master.csv')
    write_csv(sales_path, SALES_ROWS)
    write_csv(master_path, MASTER_ROWS)

    # Ordinary petl reads: data columns and values stay exactly as petl
    # produces them.  tracked() only attaches a source label and ordinals.
    sales = lineage.tracked(etl.fromcsv(sales_path), source_id='sales.csv')
    master = lineage.tracked(etl.fromcsv(master_path), source_id='master.csv')

    # A pipeline built only from lineage-aware transforms.  cut/rename/select
    # pass one provenance line through; the join unions both sides; the
    # aggregate unions every member of the group.
    clean = lineage.select(sales, lambda rec: rec['qty'] != '')
    clean = lineage.rename(clean, {'store': 'store_id'})
    clean = lineage.cut(clean, 'store_id', 'sku', 'qty')
    joined = lineage.leftjoin(clean, master, key='sku')

    def total_qty(quantities):
        return sum(int(q) for q in quantities)

    summary = lineage.aggregate(joined, 'sku', total_qty, 'qty')

    print('Reconciliation summary (ordinary data columns only):')
    print(etl.look(summary))

    # The data interface is indistinguishable from a normal petl table.
    assert list(summary) == list(
        etl.aggregate(
            etl.leftjoin(
                etl.cut(
                    etl.rename(
                        etl.select(etl.fromcsv(sales_path),
                                   lambda rec: rec['qty'] != ''),
                        {'store': 'store_id'}),
                    'store_id', 'sku', 'qty'),
                etl.fromcsv(master_path), key='sku'),
            'sku', total_qty, 'qty'))

    # --- Locate the offending summary row ---------------------------------
    # SKU1 is duplicated in sales and has two master prices, so its qty is
    # counted across a many-to-many join.  Find the output row index first.
    data_rows = list(summary)[1:]
    target_index = next(i for i, row in enumerate(data_rows)
                        if row[0] == 'SKU1')
    coords = summary.explain(target_index)

    print('\nProvenance of summary row %r:' % (data_rows[target_index],))
    file_lines = {}
    for coordinate in sorted(coords, key=lambda c: (c.source_id,
                                                     c.row_number)):
        print('  %s data row %d  ->  %s line %d'
              % (coordinate.source_id, coordinate.row_number,
                 coordinate.source_id, coordinate.file_line_number))
        file_lines.setdefault(coordinate.source_id, []).append(
            coordinate.file_line_number)

    # --- Open the original files at the reported physical lines -----------
    sources = {
        'sales.csv': (sales_path, load_lines(sales_path)),
        'master.csv': (master_path, load_lines(master_path)),
    }
    print('\nOriginal file contents at those lines:')
    for source_id, line_numbers in sorted(file_lines.items()):
        path, lines = sources[source_id]
        print('  %s (%s)' % (source_id, path))
        for line_number in line_numbers:
            print('    line %d: %s' % (line_number, lines[line_number - 1]))

    # --- Assertions documenting the required provenance identities --------
    assert coords == {
        lineage.SourceCoord('sales.csv', 1),
        lineage.SourceCoord('sales.csv', 2),
        lineage.SourceCoord('master.csv', 1),
        lineage.SourceCoord('master.csv', 2),
    }

    # The unmatched sales row (S3 / SKU9) keeps a left-only provenance set.
    sku9_index = next(i for i, row in enumerate(data_rows)
                      if row[0] == 'SKU9')
    assert summary.explain(sku9_index) == {
        lineage.SourceCoord('sales.csv', 4)}

    # Re-reading data and lineage in any order never replays the one-shot
    # CSV iterator.
    assert list(summary) == list(summary)
    assert len(summary.lineage()) == len(data_rows)

    # An unsupported transform breaks tracking loudly, not silently.
    try:
        lineage.select(etl.cut(sales, 'qty'), lambda rec: True)
    except lineage.LineageLostError:
        print('\nUnsupported transform correctly raises LineageLostError.')


if __name__ == '__main__':
    main()
