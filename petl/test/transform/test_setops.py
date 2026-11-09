from __future__ import absolute_import, print_function, division


import gc
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime

import pytest

from petl.test.helpers import ieq
from petl.errors import ArgumentError, FieldSelectionError
from petl.transform.setops import complement, intersection, diff, \
    recordcomplement, recorddiff, keydiff, hashcomplement, hashintersection


def _test_complement_1(complement_impl):

    table1 = (('foo', 'bar'),
              ('A', 1),
              ('B', 2),
              ('C', 7))

    table2 = (('foo', 'bar'),
              ('A', 9),
              ('B', 2),
              ('B', 3))

    expectation = (('foo', 'bar'),
                   ('A', 1),
                   ('C', 7))

    result = complement_impl(table1, table2)
    ieq(expectation, result)


def _test_complement_2(complement_impl):

    tablea = (('foo', 'bar', 'baz'),
              ('A', 1, True),
              ('C', 7, False),
              ('B', 2, False),
              ('C', 9, True))

    tableb = (('x', 'y', 'z'),
              ('B', 2, False),
              ('A', 9, False),
              ('B', 3, True),
              ('C', 9, True))

    aminusb = (('foo', 'bar', 'baz'),
               ('A', 1, True),
               ('C', 7, False))

    result = complement_impl(tablea, tableb)
    ieq(aminusb, result)

    bminusa = (('x', 'y', 'z'),
               ('A', 9, False),
               ('B', 3, True))

    result = complement_impl(tableb, tablea)
    ieq(bminusa, result)


def _test_complement_3(complement_impl):

    # make sure we deal with empty tables

    table1 = (('foo', 'bar'),
              ('A', 1),
              ('B', 2))

    table2 = (('foo', 'bar'),)

    expectation = (('foo', 'bar'),
                   ('A', 1),
                   ('B', 2))
    result = complement_impl(table1, table2)
    ieq(expectation, result)
    ieq(expectation, result)

    expectation = (('foo', 'bar'),)
    result = complement_impl(table2, table1)
    ieq(expectation, result)
    ieq(expectation, result)


def _test_complement_4(complement_impl):

    # test behaviour with duplicate rows

    table1 = (('foo', 'bar'),
              ('A', 1),
              ('B', 2),
              ('B', 2),
              ('C', 7))

    table2 = (('foo', 'bar'),
              ('B', 2))

    result = complement_impl(table1, table2)
    expectation = (('foo', 'bar'),
                   ('A', 1),
                   ('B', 2),
                   ('C', 7))
    ieq(expectation, result)
    ieq(expectation, result)

    # strict behaviour
    result = complement_impl(table1, table2, strict=True)
    expectation = (('foo', 'bar'),
                   ('A', 1),
                   ('C', 7))
    ieq(expectation, result)
    ieq(expectation, result)


def _test_complement_none(complement_impl):
    # test behaviour with unsortable types
    now = datetime.now()

    ta = [['a', 'b'], [None, None]]
    tb = [['a', 'b'], [None, now]]

    expectation = (('a', 'b'), (None, None))
    result = complement_impl(ta, tb)
    ieq(expectation, result)

    ta = [['a'], [now], [None]]
    tb = [['a'], [None], [None]]

    expectation = (('a',), (now,))
    result = complement_impl(ta, tb)
    ieq(expectation, result)


def _test_complement(f):
    _test_complement_1(f)
    _test_complement_2(f)
    _test_complement_3(f)
    _test_complement_4(f)
    _test_complement_none(f)


def test_complement():
    _test_complement(complement)


def test_complement_seqtypes():
    # test complement isn't confused by list vs tuple
    ta = [['a', 'b'], ['A', 1], ['B', 2]]
    tb = [('a', 'b'), ('A', 1), ('B', 2)]
    expectation = (('a', 'b'),)
    actual = complement(ta, tb, presorted=True)
    ieq(expectation, actual)


def test_hashcomplement_seqtypes():
    # test complement isn't confused by list vs tuple
    ta = [['a', 'b'], ['A', 1], ['B', 2]]
    tb = [('a', 'b'), ('A', 1), ('B', 2)]
    expectation = (('a', 'b'),)
    actual = hashcomplement(ta, tb)
    ieq(expectation, actual)


def test_diff():

    tablea = (('foo', 'bar', 'baz'),
              ('A', 1, True),
              ('C', 7, False),
              ('B', 2, False),
              ('C', 9, True))

    tableb = (('x', 'y', 'z'),
              ('B', 2, False),
              ('A', 9, False),
              ('B', 3, True),
              ('C', 9, True))

    aminusb = (('foo', 'bar', 'baz'),
               ('A', 1, True),
               ('C', 7, False))

    bminusa = (('x', 'y', 'z'),
               ('A', 9, False),
               ('B', 3, True))

    added, subtracted = diff(tablea, tableb)
    ieq(bminusa, added)
    ieq(aminusb, subtracted)


def test_recordcomplement_1():

    table1 = (('foo', 'bar'),
              ('A', 1),
              ('B', 2),
              ('C', 7))

    table2 = (('bar', 'foo'),
              (9, 'A'),
              (2, 'B'),
              (3, 'B'))

    expectation = (('foo', 'bar'),
                   ('A', 1),
                   ('C', 7))

    result = recordcomplement(table1, table2)
    ieq(expectation, result)


def test_recordcomplement_2():

    tablea = (('foo', 'bar', 'baz'),
              ('A', 1, True),
              ('C', 7, False),
              ('B', 2, False),
              ('C', 9, True))

    tableb = (('bar', 'foo', 'baz'),
              (2, 'B', False),
              (9, 'A', False),
              (3, 'B', True),
              (9, 'C', True))

    aminusb = (('foo', 'bar', 'baz'),
               ('A', 1, True),
               ('C', 7, False))

    result = recordcomplement(tablea, tableb)
    ieq(aminusb, result)

    bminusa = (('bar', 'foo', 'baz'),
               (3, 'B', True),
               (9, 'A', False))

    result = recordcomplement(tableb, tablea)
    ieq(bminusa, result)


def test_recordcomplement_3():

    # make sure we deal with empty tables

    table1 = (('foo', 'bar'),
              ('A', 1),
              ('B', 2))

    table2 = (('bar', 'foo'),)

    expectation = (('foo', 'bar'),
                   ('A', 1),
                   ('B', 2))
    result = recordcomplement(table1, table2)
    ieq(expectation, result)
    ieq(expectation, result)

    expectation = (('bar', 'foo'),)
    result = recordcomplement(table2, table1)
    ieq(expectation, result)
    ieq(expectation, result)


def test_recordcomplement_4():

    # test behaviour with duplicate rows

    table1 = (('foo', 'bar'),
              ('A', 1),
              ('B', 2),
              ('B', 2),
              ('C', 7))

    table2 = (('bar', 'foo'),
              (2, 'B'))

    result = recordcomplement(table1, table2)
    expectation = (('foo', 'bar'),
                   ('A', 1),
                   ('B', 2),
                   ('C', 7))
    ieq(expectation, result)
    ieq(expectation, result)

    # strict behaviour
    result = recordcomplement(table1, table2, strict=True)
    expectation = (('foo', 'bar'),
                   ('A', 1),
                   ('C', 7))
    ieq(expectation, result)
    ieq(expectation, result)


def test_recorddiff():

    tablea = (('foo', 'bar', 'baz'),
              ('A', 1, True),
              ('C', 7, False),
              ('B', 2, False),
              ('C', 9, True))

    tableb = (('bar', 'foo', 'baz'),
              (2, 'B', False),
              (9, 'A', False),
              (3, 'B', True),
              (9, 'C', True))

    aminusb = (('foo', 'bar', 'baz'),
               ('A', 1, True),
               ('C', 7, False))

    bminusa = (('bar', 'foo', 'baz'),
               (3, 'B', True),
               (9, 'A', False))

    added, subtracted = recorddiff(tablea, tableb)
    ieq(aminusb, subtracted)
    ieq(bminusa, added)


def _test_intersection_1(intersection_impl):

    table1 = (('foo', 'bar'),
              ('A', 1),
              ('B', 2),
              ('C', 7))

    table2 = (('foo', 'bar'),
              ('A', 9),
              ('B', 2),
              ('B', 3))

    expectation = (('foo', 'bar'),
                   ('B', 2))

    result = intersection_impl(table1, table2)
    ieq(expectation, result)


def _test_intersection_2(intersection_impl):

    table1 = (('foo', 'bar', 'baz'),
              ('A', 1, True),
              ('C', 7, False),
              ('B', 2, False),
              ('C', 9, True))

    table2 = (('x', 'y', 'z'),
              ('B', 2, False),
              ('A', 9, False),
              ('B', 3, True),
              ('C', 9, True))

    expect = (('foo', 'bar', 'baz'),
              ('B', 2, False),
              ('C', 9, True))

    table3 = intersection_impl(table1, table2)
    ieq(expect, table3)


def _test_intersection_3(intersection_impl):

    # empty table
    table1 = (('foo', 'bar'),
              ('A', 1),
              ('B', 2),
              ('C', 7))

    table2 = (('foo', 'bar'),)

    expectation = (('foo', 'bar'),)
    result = intersection_impl(table1, table2)
    ieq(expectation, result)
    ieq(expectation, result)


def _test_intersection_4(intersection_impl):

    # duplicate rows

    table1 = (('foo', 'bar'),
              ('A', 1),
              ('B', 2),
              ('B', 2),
              ('B', 2),
              ('C', 7))

    table2 = (('foo', 'bar'),
              ('A', 9),
              ('B', 2),
              ('B', 2),
              ('B', 3))

    expectation = (('foo', 'bar'),
                   ('B', 2),
                   ('B', 2))

    result = intersection_impl(table1, table2)
    ieq(expectation, result)
    ieq(expectation, result)


def _test_intersection_empty(intersection_impl):

    table1 = (('foo', 'bar'),
              ('A', 1),
              ('B', 2),
              ('C', 7))
    table2 = (('foo', 'bar'),)
    expectation = (('foo', 'bar'),)
    result = intersection_impl(table1, table2)
    ieq(expectation, result)


def _test_intersection(intersection_impl):
    _test_intersection_1(intersection_impl)
    _test_intersection_2(intersection_impl)
    _test_intersection_3(intersection_impl)
    _test_intersection_4(intersection_impl)
    _test_intersection_empty(intersection_impl)


def test_intersection():
    _test_intersection(intersection)


def test_hashcomplement():
    _test_complement(hashcomplement)


def test_hashintersection():
    _test_intersection(hashintersection)


def _keydiff_apply(a_rows, changes):
    """Apply a keydiff change stream to a list of rows (no header)."""
    rows = [tuple(r) for r in a_rows]
    for change, key, before, after in changes:
        if change == 'delete':
            rows.remove(before)
        elif change == 'insert':
            rows.append(after)
        elif change == 'update':
            rows.remove(before)
            rows.append(after)
        else:
            raise AssertionError('unexpected change type %r' % (change,))
    return rows


def _keydiff_sample():
    """A few dozen rows exercising all change types."""
    hdr = ('id', 'day', 'item', 'qty')
    a_rows = []
    b_rows = []
    # unchanged records, cancelling one-to-one
    for i in range(20):
        a_rows.append(('k%02d' % i, 'd1', 'apple', i))
        b_rows.append(('k%02d' % i, 'd1', 'apple', i))
    # same record 3x in a, 2x in b -> exactly one delete
    a_rows += [('dup', 'd1', 'pear', 1)] * 3
    b_rows += [('dup', 'd1', 'pear', 1)] * 2
    # updates: one leftover row on each side
    for i in range(5):
        a_rows.append(('u%02d' % i, 'd1', 'fig', i))
        b_rows.append(('u%02d' % i, 'd1', 'fig', i + 100))
    # deletes: keys only in a
    for i in range(5):
        a_rows.append(('del%02d' % i, 'd1', 'plum', i))
    # inserts: keys only in b
    for i in range(5):
        b_rows.append(('ins%02d' % i, 'd1', 'kiwi', i))
    # multiple leftovers on both sides -> inserts and deletes, no updates
    for i in range(3):
        a_rows += [('multi%02d' % i, 'd1', 'old', 1),
                   ('multi%02d' % i, 'd1', 'old', 2)]
        b_rows += [('multi%02d' % i, 'd1', 'new', 3)]
    # None in the key, and None distinct from empty string
    a_rows += [(None, 'd1', 'none', 1), ('nn', 'd1', None, 5)]
    b_rows += [(None, 'd1', 'none', 2), ('nn', 'd1', '', 5)]
    return hdr, a_rows, b_rows


def test_keydiff_exact():
    a = (('id', 'sub', 'v'),
         (None, 'z', 1),
         ('k1', 'a', 1),
         ('k1', 'a', 1),
         ('k1', 'b', 2),
         ('k2', 'x', None),
         ('k3', 'p', ''),
         ('k4', 'q', 5),
         ('k6', 'n', None),
         ('k7', 'm', 1),
         ('k8', 'm', 1),
         ('k8', 'm', 2))
    # same fields, different order
    b = (('v', 'sub', 'id'),
         (1, 'z', None),
         (2, 'z', None),
         (1, 'a', 'k1'),
         (9, 'b', 'k1'),
         (None, 'x', 'k2'),
         (7, 'x', 'k2'),
         ('', 'p', 'k3'),
         ('', 'p', 'k3'),
         (10, 'z', 'k5'),
         ('', 'n', 'k6'),
         (2, 'm', 'k7'),
         (3, 'm', 'k7'),
         (3, 'm', 'k8'))
    # force the disk path with a tiny buffer
    result = keydiff(a, b, key=('id', 'sub'), buffersize=2)
    expectation = (
        ('change', 'key', 'before', 'after'),
        # None key sorts first; one pair cancels, one insert remains
        ('insert', (None, 'z'), None, (None, 'z', 2)),
        # 2x vs 1x -> a single delete
        ('delete', ('k1', 'a'), ('k1', 'a', 1), None),
        # one leftover each side -> update
        ('update', ('k1', 'b'), ('k1', 'b', 2), ('k1', 'b', 9)),
        # None value pair cancels, one insert remains
        ('insert', ('k2', 'x'), None, ('k2', 'x', 7)),
        ('insert', ('k3', 'p'), None, ('k3', 'p', '')),
        ('delete', ('k4', 'q'), ('k4', 'q', 5), None),
        ('insert', ('k5', 'z'), None, ('k5', 'z', 10)),
        # None is distinct from empty string -> update
        ('update', ('k6', 'n'), ('k6', 'n', None), ('k6', 'n', '')),
        # 1 vs 2 leftovers -> delete + inserts, no guessed pairing
        ('delete', ('k7', 'm'), ('k7', 'm', 1), None),
        ('insert', ('k7', 'm'), None, ('k7', 'm', 2)),
        ('insert', ('k7', 'm'), None, ('k7', 'm', 3)),
        # 2 vs 1 leftovers -> deletes + insert, no guessed pairing
        ('delete', ('k8', 'm'), ('k8', 'm', 1), None),
        ('delete', ('k8', 'm'), ('k8', 'm', 2), None),
        ('insert', ('k8', 'm'), None, ('k8', 'm', 3)),
    )
    ieq(expectation, result)


def test_keydiff_apply_equation_disk():
    # applying the change stream to `a` must yield exactly `b`, comparing
    # by row multiplicity; this must hold under shuffled inputs, swapped
    # column order and tiny buffers forcing the external sort path
    import random
    hdr, a_rows, b_rows = _keydiff_sample()
    tempdir = tempfile.mkdtemp()
    try:
        for shuffle in (False, True):
            for swap_cols in (False, True):
                for buffersize in (2, 5, None):
                    aa = list(a_rows)
                    bb = list(b_rows)
                    if shuffle:
                        random.Random(1234).shuffle(aa)
                        random.Random(5678).shuffle(bb)
                    b_hdr = hdr
                    if swap_cols:
                        perm = (3, 2, 1, 0)
                        b_hdr = tuple(hdr[i] for i in perm)
                        bb = [tuple(r[i] for i in perm) for r in bb]
                    ta = [hdr] + aa
                    tb = [b_hdr] + bb
                    result = keydiff(ta, tb, key=('id', 'day'),
                                     buffersize=buffersize,
                                     tempdir=tempdir)
                    changes = list(result)
                    assert changes[0] == ('change', 'key', 'before', 'after')
                    assert len(changes) > 1
                    applied = _keydiff_apply(aa, changes[1:])
                    # applied rows are in the canonical (a's) field order
                    assert Counter(applied) == Counter(map(tuple, b_rows)), \
                        (shuffle, swap_cols, buffersize)
                    # this pass is complete, temp files must be released
                    gc.collect()
                    assert os.listdir(tempdir) == [], \
                        (shuffle, swap_cols, buffersize)
    finally:
        shutil.rmtree(tempdir, ignore_errors=True)


def test_keydiff_one_pass_inputs():
    # inputs may be iterators that can only be traversed once
    hdr, a_rows, b_rows = _keydiff_sample()
    result = keydiff(iter([hdr] + a_rows), iter([hdr] + b_rows),
                     key=('id', 'day'), buffersize=4)
    changes = list(result)
    applied = _keydiff_apply(a_rows, changes[1:])
    assert Counter(applied) == Counter(map(tuple, b_rows))


def test_keydiff_empty():
    a = (('id', 'v'),)
    b = (('v', 'id'),)
    result = keydiff(a, b, key='id')
    ieq((('change', 'key', 'before', 'after'),), result)


def test_keydiff_errors():
    a = (('id', 'v'), (1, 'x'))
    b = (('id', 'v'), (1, 'x'))
    # empty key
    with pytest.raises(ArgumentError):
        keydiff(a, b, key=())
    # duplicate header
    with pytest.raises(ArgumentError):
        list(keydiff((('id', 'id'), (1, 2)), b, key='id'))
    with pytest.raises(ArgumentError):
        list(keydiff(a, (('id', 'v', 'v'), (1, 'x', 'y')), key='id'))
    # missing key field
    with pytest.raises(FieldSelectionError):
        list(keydiff(a, b, key='nope'))
    with pytest.raises(FieldSelectionError):
        list(keydiff(a, b, key=('id', 'nope')))
    # non-key field sets differ
    with pytest.raises(ArgumentError):
        list(keydiff(a, (('id', 'w'), (1, 'x')), key='id'))
    # unsupported value types
    with pytest.raises(TypeError):
        list(keydiff((('id', 'v'), (1, 1.5)), b, key='id'))
    with pytest.raises(TypeError):
        list(keydiff((('id', 'v'), (1, True)), b, key='id'))
    with pytest.raises(TypeError):
        list(keydiff(a, (('id', 'v'), (1, 2.5)), key='id'))


def test_keydiff_cleanup_on_close():
    # closing the change stream early must release this pass's temp files
    hdr, a_rows, b_rows = _keydiff_sample()
    tempdir = tempfile.mkdtemp()
    try:
        result = keydiff([hdr] + a_rows, [hdr] + b_rows, key=('id', 'day'),
                         buffersize=2, tempdir=tempdir)
        it = iter(result)
        next(it)
        next(it)
        assert os.listdir(tempdir) != []  # disk path was exercised
        it.close()
        del it
        gc.collect()
        assert os.listdir(tempdir) == []
    finally:
        shutil.rmtree(tempdir, ignore_errors=True)


def test_keydiff_cleanup_on_source_error():
    # a failing source must not leave this pass's temp files behind
    hdr, a_rows, b_rows = _keydiff_sample()
    tempdir = tempfile.mkdtemp()

    def boom():
        yield hdr
        for row in a_rows[:7]:
            yield row
        raise RuntimeError('source failed')

    try:
        result = keydiff(boom(), [hdr] + b_rows, key=('id', 'day'),
                         buffersize=2, tempdir=tempdir)
        with pytest.raises(RuntimeError):
            list(result)
        gc.collect()
        assert os.listdir(tempdir) == []
    finally:
        shutil.rmtree(tempdir, ignore_errors=True)
