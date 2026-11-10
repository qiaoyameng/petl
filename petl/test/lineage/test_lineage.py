from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import pytest

import petl as etl
from petl.lineage import (
    Line,
    LineageLostError,
    SourceCoord,
    aggregate,
    cut,
    innerjoin,
    leftjoin,
    rename,
    select,
    tracked,
)
from petl.test.helpers import ieq


LEFT = (
    ('id', 'cat', 'amt'),
    (1, 'x', 10),
    (1, 'x', 20),   # duplicate business values with the row above
    (2, 'y', 30),
    (3, 'z', 40),   # unmatched on the left
)

RIGHT = (
    ('id', 'note'),
    (1, 'a'),
    (1, 'b'),       # many-to-many against the two id=1 left rows
    (4, 'c'),       # unmatched on the right
)


def coord(source_id, row_number):
    return SourceCoord(source_id, row_number)


def pairs(table):
    stream = table.iterwithlineage()
    next(stream)  # header
    return [(row, line.as_set()) for row, line in stream]


def test_source_assigns_independent_ordinals_to_duplicates():
    table = tracked(LEFT, 'left.csv')
    ieq(LEFT, table)
    coords = table.coords()
    assert coords[0] == frozenset({coord('left.csv', 1)})
    assert coords[1] == frozenset({coord('left.csv', 2)})
    # identical values, distinct physical source rows
    assert table._rows[0] != table._rows[1] or coords[0] != coords[1]
    assert coords[0] != coords[1]
    # file line number is one header row above the data ordinal
    line = table.lineage()[0]
    assert list(line.coords())[0].file_line_number == 2


def test_cut_preserves_values_and_lineage():
    expect = etl.cut(LEFT, 'cat', 'amt')
    actual = cut(tracked(LEFT, 'left.csv'), 'cat', 'amt')
    ieq(expect, actual)
    assert actual.coords() == tracked(LEFT, 'left.csv').coords()


def test_cut_handles_short_rows_like_petl():
    table = (('foo', 'bar', 'baz'), ('A', 1, 2), ('E', 12))
    actual = cut(tracked(table, 'f'), 'foo', 'baz')
    ieq(etl.cut(table, 'foo', 'baz'), actual)


def test_rename_changes_header_only():
    table = tracked(LEFT, 'left.csv')
    actual = rename(table, {'cat': 'category'})
    ieq(etl.rename(LEFT, {'cat': 'category'}), actual)
    assert actual.coords() == table.coords()


def test_select_row_predicate_keeps_lineage():
    table = tracked(LEFT, 'left.csv')
    predicate = lambda rec: rec['amt'] >= 20
    actual = select(table, predicate)
    ieq(etl.select(LEFT, predicate), actual)
    assert [c for r, c in pairs(actual)] == [
        frozenset({coord('left.csv', 2)}),
        frozenset({coord('left.csv', 3)}),
        frozenset({coord('left.csv', 4)}),
    ]


def test_select_field_predicate_and_complement_and_expression():
    table = tracked(LEFT, 'left.csv')
    ieq(etl.select(LEFT, 'id', lambda v: v == 1),
        select(table, 'id', lambda v: v == 1))
    predicate = lambda rec: rec['amt'] >= 20
    ieq(etl.select(LEFT, predicate, complement=True),
        select(table, predicate, complement=True))
    ieq(etl.select(LEFT, "{id} == 2"), select(table, "{id} == 2"))


def test_innerjoin_many_to_many_union_and_leftjoin_padding():
    left = tracked(LEFT, 'left.csv')
    right = tracked(RIGHT, 'right.csv')

    inner = innerjoin(left, right, key='id')
    ieq(etl.join(LEFT, RIGHT, key='id'), inner)
    inner_pairs = pairs(inner)
    id1 = [(row, cs) for row, cs in inner_pairs if row[0] == 1]
    assert len(id1) == 4  # 2 left rows x 2 right rows
    expected = {
        (coord('left.csv', 1), coord('right.csv', 1)),
        (coord('left.csv', 1), coord('right.csv', 2)),
        (coord('left.csv', 2), coord('right.csv', 1)),
        (coord('left.csv', 2), coord('right.csv', 2)),
    }
    assert {tuple(sorted(cs)) for _row, cs in id1} == expected

    joined = leftjoin(left, right, key='id')
    ieq(etl.leftjoin(LEFT, RIGHT, key='id'), joined)
    by_id = {row[0]: cs for row, cs in pairs(joined)}
    # unmatched left row keeps the left provenance only
    assert by_id[3] == frozenset({coord('left.csv', 4)})
    assert by_id[2] == frozenset({coord('left.csv', 3)})
    # right-only row (id 4) never appears in a left join
    assert 4 not in by_id


def test_join_shares_immutable_subtrees_without_copying_history():
    left = tracked(LEFT, 'left.csv')
    right = tracked(RIGHT, 'right.csv')
    joined = innerjoin(left, right, key='id')
    stream = joined.iterwithlineage()
    next(stream)
    lines = [line for _row, line in stream]
    # all four id=1 outputs are union nodes
    note_a = [line for line in lines if line.right is not None
              and line.right.coord == coord('right.csv', 1)]
    note_b = [line for line in lines if line.right is not None
              and line.right.coord == coord('right.csv', 2)]
    assert len(note_a) == 2 and note_a[0].right is note_a[1].right
    assert len(note_b) == 2 and note_b[0].right is note_b[1].right


def test_aggregate_group_lineage_is_union_of_members():
    joined = leftjoin(tracked(LEFT, 'left.csv'),
                      tracked(RIGHT, 'right.csv'), key='id')
    grouped = aggregate(joined, 'id', sum, 'amt')
    ieq(etl.aggregate(etl.leftjoin(LEFT, RIGHT, key='id'), 'id', sum, 'amt'),
        grouped)
    by_id = {row[0]: cs for row, cs in pairs(grouped)}
    assert by_id[1] == {
        coord('left.csv', 1), coord('left.csv', 2),
        coord('right.csv', 1), coord('right.csv', 2),
    }
    assert by_id[2] == frozenset({coord('left.csv', 3)})
    assert by_id[3] == frozenset({coord('left.csv', 4)})


def test_aggregate_multi_field_and_no_key_forms():
    source = tracked(LEFT, 'left.csv')
    spec = OrderedDict()
    spec['cnt'] = len
    spec['total'] = ('amt', sum)
    multi = aggregate(source, 'id', spec)
    ieq(etl.aggregate(LEFT, 'id', spec), multi)
    assert multi.explain(0) == frozenset({coord('left.csv', 1),
                                          coord('left.csv', 2)})

    whole = aggregate(source, None, sum, 'amt')
    ieq(etl.aggregate(LEFT, None, sum, 'amt'), whole)
    assert len(whole.coords()[0]) == len(LEFT) - 1


def test_aggregate_empty_table_matches_petl():
    empty = (('foo', 'bar'),)
    source = tracked(empty, 'e')
    ieq(etl.aggregate(empty, None, len), aggregate(source, None, len))
    ieq(etl.aggregate(empty, 'foo', len), aggregate(source, 'foo', len))
    multi = OrderedDict()
    multi['c'] = len
    ieq(etl.aggregate(empty, None, multi), aggregate(source, None, multi))
    ieq(etl.aggregate(empty, 'foo', multi), aggregate(source, 'foo', multi))


def test_interleaved_data_and_lineage_reads_do_not_replay_input():
    def one_shot():
        yield ('a', 'b')
        yield (1, 2)
        yield (3, 4)

    source = tracked(one_shot(), 'stream')
    first = list(source)
    lines = source.lineage()
    second = list(source)
    coords = source.coords()
    assert first == second == [('a', 'b'), (1, 2), (3, 4)]
    assert len(lines) == len(coords) == 2

    # a derived table is also safely re-readable from either interface
    derived = cut(select(source, lambda rec: rec['a'] > 0), 'b')
    assert list(derived) == list(derived)
    assert len(derived.lineage()) == 2


def test_untracked_input_loses_lineage_eagerly():
    with pytest.raises(LineageLostError):
        cut(RIGHT, 'note')
    with pytest.raises(LineageLostError):
        select(RIGHT, lambda row: True)
    with pytest.raises(LineageLostError):
        innerjoin(LEFT, tracked(RIGHT, 'r'), key='id')
    with pytest.raises(LineageLostError):
        leftjoin(tracked(LEFT, 'l'), RIGHT, key='id')
    with pytest.raises(LineageLostError):
        aggregate(RIGHT, 'id', len)


def test_native_petl_transform_breaks_lineage():
    # Bypassing the lineage-aware transform returns an ordinary petl table;
    # feeding it back into a tracked transform must fail loudly rather than
    # report a partial provenance set.
    plain = etl.cut(tracked(LEFT, 'left.csv'), 'amt')
    with pytest.raises(LineageLostError):
        select(plain, lambda row: True)


def test_line_tree_is_immutable_and_deduplicates_coords():
    leaf_a = Line.leaf(coord('f', 1))
    leaf_b = Line.leaf(coord('f', 2))
    merged = leaf_a.union(leaf_b).union(leaf_a)
    assert merged.as_set() == frozenset({coord('f', 1), coord('f', 2)})
    assert leaf_a.as_set() == frozenset({coord('f', 1)})
    assert leaf_a.union(leaf_a) is leaf_a


def test_join_with_prefixes_and_missing_marker_matches_petl():
    left = tracked(LEFT, 'l')
    right = tracked(RIGHT, 'r')
    ieq(etl.join(LEFT, RIGHT, key='id', lprefix='l_', rprefix='r_'),
        innerjoin(left, right, key='id', lprefix='l_', rprefix='r_'))
    ieq(etl.leftjoin(LEFT, RIGHT, key='id', missing='NA'),
        leftjoin(left, right, key='id', missing='NA'))
