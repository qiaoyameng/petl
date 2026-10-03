from __future__ import absolute_import, print_function, division


import os
import gc
import logging
from datetime import datetime
import platform

import pytest

from petl.compat import next

from petl.errors import FieldSelectionError, ArgumentError, \
    SortError
from petl.test.helpers import ieq, eq_
from petl.util import nrows
from petl.transform.basics import cat
from petl.transform.sorts import sort, mergesort, issorted


logger = logging.getLogger(__name__)
debug = logger.debug


def test_sort_1():

    table = (('foo', 'bar'),
             ('C', '2'),
             ('A', '9'),
             ('A', '6'),
             ('F', '1'),
             ('D', '10'))

    result = sort(table, 'foo')
    expectation = (('foo', 'bar'),
                   ('A', '9'),
                   ('A', '6'),
                   ('C', '2'),
                   ('D', '10'),
                   ('F', '1'))
    ieq(expectation, result)


def test_sort_2():

    table = (('foo', 'bar'),
             ('C', '2'),
             ('A', '9'),
             ('A', '6'),
             ('F', '1'),
             ('D', '10'))

    result = sort(table, key=('foo', 'bar'))
    expectation = (('foo', 'bar'),
                   ('A', '6'),
                   ('A', '9'),
                   ('C', '2'),
                   ('D', '10'),
                   ('F', '1'))
    ieq(expectation, result)

    result = sort(table)  # default is lexical sort
    expectation = (('foo', 'bar'),
                   ('A', '6'),
                   ('A', '9'),
                   ('C', '2'),
                   ('D', '10'),
                   ('F', '1'))
    ieq(expectation, result)


def test_sort_3():

    table = (('foo', 'bar'),
             ('C', '2'),
             ('A', '9'),
             ('A', '6'),
             ('F', '1'),
             ('D', '10'))

    result = sort(table, 'bar')
    expectation = (('foo', 'bar'),
                   ('F', '1'),
                   ('D', '10'),
                   ('C', '2'),
                   ('A', '6'),
                   ('A', '9'))
    ieq(expectation, result)


def test_sort_4():

    table = (('foo', 'bar'),
             ('C', 2),
             ('A', 9),
             ('A', 6),
             ('F', 1),
             ('D', 10))

    result = sort(table, 'bar')
    expectation = (('foo', 'bar'),
                   ('F', 1),
                   ('C', 2),
                   ('A', 6),
                   ('A', 9),
                   ('D', 10))
    ieq(expectation, result)


def test_sort_5():

    table = (('foo', 'bar'),
             (2.3, 2),
             (1.2, 9),
             (2.3, 6),
             (3.2, 1),
             (1.2, 10))

    expectation = (('foo', 'bar'),
                   (1.2, 9),
                   (1.2, 10),
                   (2.3, 2),
                   (2.3, 6),
                   (3.2, 1))

    # can use either field names or indices (from 1) to specify sort key
    result = sort(table, key=('foo', 'bar'))
    ieq(expectation, result)
    result = sort(table, key=(0, 1))
    ieq(expectation, result)
    result = sort(table, key=('foo', 1))
    ieq(expectation, result)
    result = sort(table, key=(0, 'bar'))
    ieq(expectation, result)


def test_sort_6():

    table = (('foo', 'bar'),
             (2.3, 2),
             (1.2, 9),
             (2.3, 6),
             (3.2, 1),
             (1.2, 10))

    expectation = (('foo', 'bar'),
                   (3.2, 1),
                   (2.3, 6),
                   (2.3, 2),
                   (1.2, 10),
                   (1.2, 9))

    result = sort(table, key=('foo', 'bar'), reverse=True)
    ieq(expectation, result)


def test_sort_buffered():

    table = (('foo', 'bar'),
             ('C', 2),
             ('A', 9),
             ('A', 6),
             ('F', 1),
             ('D', 10))

    # test sort forwards
    expectation = (('foo', 'bar'),
                   ('F', 1),
                   ('C', 2),
                   ('A', 6),
                   ('A', 9),
                   ('D', 10))
    result = sort(table, 'bar')
    ieq(expectation, result)
    result = sort(table, 'bar', buffersize=2)
    ieq(expectation, result)

    # sort in reverse
    expectation = (('foo', 'bar'),
                   ('D', 10),
                   ('A', 9),
                   ('A', 6),
                   ('C', 2),
                   ('F', 1))

    result = sort(table, 'bar', reverse=True)
    ieq(expectation, result)
    result = sort(table, 'bar', reverse=True, buffersize=2)
    ieq(expectation, result)

    # no key
    expectation = (('foo', 'bar'),
                   ('F', 1),
                   ('D', 10),
                   ('C', 2),
                   ('A', 9),
                   ('A', 6))
    result = sort(table, reverse=True)
    ieq(expectation, result)
    result = sort(table, reverse=True, buffersize=2)
    ieq(expectation, result)


def test_sort_buffered_tempdir():

    table = (('foo', 'bar'),
             ('C', 2),
             ('A', 9),
             ('A', 6),
             ('F', 1),
             ('D', 10))

    # test sort forwards
    expectation = (('foo', 'bar'),
                   ('F', 1),
                   ('C', 2),
                   ('A', 6),
                   ('A', 9),
                   ('D', 10))
    result = sort(table, 'bar')
    ieq(expectation, result)
    tempdir = 'tmp'
    if not os.path.exists(tempdir):
        os.mkdir(tempdir)
    result = sort(table, 'bar', buffersize=2, tempdir=tempdir)
    ieq(expectation, result)


def test_sort_buffered_independent():

    table = (('foo', 'bar'),
             ('C', 2),
             ('A', 9),
             ('A', 6),
             ('F', 1),
             ('D', 10))
    expectation = (('foo', 'bar'),
                   ('F', 1),
                   ('C', 2),
                   ('A', 6),
                   ('A', 9),
                   ('D', 10))

    result = sort(table, 'bar', buffersize=4)
    nrows(result)  # cause data to be cached
    # check that two row iterators are independent, i.e., consuming rows
    # from one does not affect the other
    it1 = iter(result)
    it2 = iter(result)
    eq_(expectation[0], next(it1))
    eq_(expectation[1], next(it1))
    eq_(expectation[0], next(it2))
    eq_(expectation[1], next(it2))
    eq_(expectation[2], next(it2))
    eq_(expectation[2], next(it1))


def _get_names(l):
    return [x.name for x in l]


def test_sort_buffered_cleanup():

    table = (('foo', 'bar'),
             ('C', 2),
             ('A', 9),
             ('A', 6),
             ('F', 1),
             ('D', 10))
    result = sort(table, 'bar', buffersize=2)
    debug('initially filecache should be empty')
    eq_(None, result._filecache)
    debug('pull rows through, should populate file cache')
    eq_(5, nrows(result))
    eq_(3, len(result._filecache))
    debug('check all files exist')
    filenames = _get_names(result._filecache)
    for fn in filenames:
        assert os.path.exists(fn), fn
    debug('delete object and garbage collect')
    del result
    gc.collect()
    debug('check all files have been deleted')
    for fn in filenames:
        assert not os.path.exists(fn), fn


@pytest.mark.skipif(platform.python_implementation() == 'PyPy', reason='SKIP sort cleanup test (PyPy)')
def test_sort_buffered_cleanup_open_iterator():

    table = (('foo', 'bar'),
             ('C', 2),
             ('A', 9),
             ('A', 6),
             ('F', 1),
             ('D', 10))
    # check if cleanup is robust against open iterators
    result = sort(table, 'bar', buffersize=2)
    debug('pull rows through, should populate file cache')
    eq_(5, nrows(result))
    eq_(3, len(result._filecache))
    debug('check all files exist')
    filenames = _get_names(result._filecache)
    for fn in filenames:
        assert os.path.exists(fn), fn
    debug(filenames)
    debug('open an iterator')
    it = iter(result)
    next(it)
    next(it)
    debug('delete objects and garbage collect')
    del result
    del it
    gc.collect()
    for fn in filenames:
        assert not os.path.exists(fn), fn


def test_sort_empty():
    table = (('foo', 'bar'),)
    expect = (('foo', 'bar'),)
    actual = sort(table)
    ieq(expect, actual)


def test_sort_none():

    table = (('foo', 'bar'),
             ('C', 2),
             ('A', 9),
             ('A', None),
             ('F', 1),
             ('D', 10))

    result = sort(table, 'bar')
    print(list(result))
    expectation = (('foo', 'bar'),
                   ('A', None),
                   ('F', 1),
                   ('C', 2),
                   ('A', 9),
                   ('D', 10))
    ieq(expectation, result)

    dt = datetime.now().replace

    table = (('foo', 'bar'),
             ('C', dt(hour=5)),
             ('A', dt(hour=1)),
             ('A', None),
             ('F', dt(hour=9)),
             ('D', dt(hour=17)))

    result = sort(table, 'bar')
    expectation = (('foo', 'bar'),
                   ('A', None),
                   ('A', dt(hour=1)),
                   ('C', dt(hour=5)),
                   ('F', dt(hour=9)),
                   ('D', dt(hour=17)))
    ieq(expectation, result)


def test_sort_headerless_no_keys():
    """
    Sorting a headerless table without specifying cols should be a no-op.
    """
    table = []
    result = sort(table)
    expectation = []
    ieq(expectation, result)


def test_sort_headerless_explicit():
    """
    But if you specify keys, they must exist.
    """
    table = []
    with pytest.raises(FieldSelectionError):
        for i in sort(table, 'foo'):
            pass

# TODO test sort with native comparison


def test_mergesort_1():

    table1 = (('foo', 'bar'),
              ('A', 6),
              ('C', 2),
              ('D', 10),
              ('A', 9),
              ('F', 1))

    table2 = (('foo', 'bar'),
              ('B', 3),
              ('D', 10),
              ('A', 10),
              ('F', 4))

    # should be same as concatenate then sort (but more efficient, esp. when
    # presorted)
    expect = sort(cat(table1, table2))

    actual = mergesort(table1, table2)
    ieq(expect, actual)
    ieq(expect, actual)

    actual = mergesort(sort(table1), sort(table2), presorted=True)
    ieq(expect, actual)
    ieq(expect, actual)


def test_mergesort_2():

    table1 = (('foo', 'bar'),
              ('A', 9),
              ('C', 2),
              ('D', 10),
              ('A', 6),
              ('F', 1))

    table2 = (('foo', 'baz'),
              ('B', 3),
              ('D', 10),
              ('A', 10),
              ('F', 4))

    # should be same as concatenate then sort (but more efficient, esp. when
    # presorted)
    expect = sort(cat(table1, table2), key='foo')

    actual = mergesort(table1, table2, key='foo')
    ieq(expect, actual)
    ieq(expect, actual)

    actual = mergesort(sort(table1, key='foo'), sort(table2, key='foo'),
                       key='foo', presorted=True)
    ieq(expect, actual)
    ieq(expect, actual)


def test_mergesort_3():

    table1 = (('foo', 'bar'),
              ('A', 9),
              ('C', 2),
              ('D', 10),
              ('A', 6),
              ('F', 1))

    table2 = (('foo', 'baz'),
              ('B', 3),
              ('D', 10),
              ('A', 10),
              ('F', 4))

    # should be same as concatenate then sort (but more efficient, esp. when
    # presorted)
    expect = sort(cat(table1, table2), key='foo', reverse=True)

    actual = mergesort(table1, table2, key='foo', reverse=True)
    ieq(expect, actual)
    ieq(expect, actual)

    actual = mergesort(sort(table1, key='foo', reverse=True),
                       sort(table2, key='foo', reverse=True),
                       key='foo', reverse=True, presorted=True)
    ieq(expect, actual)
    ieq(expect, actual)


def test_mergesort_4():

    table1 = (('foo', 'bar', 'baz'),
              (1, 'A', True),
              (2, 'B', None),
              (4, 'C', True))
    table2 = (('bar', 'baz', 'quux'),
              ('A', True, 42.0),
              ('B', False, 79.3),
              ('C', False, 12.4))

    expect = sort(cat(table1, table2), key='bar')

    actual = mergesort(table1, table2, key='bar')
    ieq(expect, actual)
    ieq(expect, actual)


def test_mergesort_empty():

    table1 = (('foo', 'bar'),
              ('A', 9),
              ('C', 2),
              ('D', 10),
              ('F', 1))

    table2 = (('foo', 'bar'),)

    expect = table1
    actual = mergesort(table1, table2, key='foo')
    ieq(expect, actual)
    ieq(expect, actual)


def test_issorted():

    table1 = (('foo', 'bar', 'baz'),
              ('a', 1, True),
              ('b', 3, True),
              ('b', 2))
    assert issorted(table1, key='foo')
    assert not issorted(table1, key='foo', reverse=True)
    assert not issorted(table1, key='foo', strict=True)

    table2 = (('foo', 'bar', 'baz'),
              ('b', 2, True),
              ('a', 1, True),
              ('b', 3))
    assert not issorted(table2, key='foo')

    table3 = (('foo', 'bar', 'baz'),
              ('a', 1, True),
              ('b', 2, True),
              ('b', 3))
    assert issorted(table3, key=('foo', 'bar'))
    assert issorted(table3)

    table4 = (('foo', 'bar', 'baz'),
              ('a', 1, True),
              ('b', 3, True),
              ('b', 2))
    assert not issorted(table4, key=('foo', 'bar'))
    assert not issorted(table4)

    table5 = (('foo', 'bar', 'baz'),
              ('b', 3, True),
              ('b', 2),
              ('a', 1, True))
    assert not issorted(table5, key='foo')
    assert issorted(table5, key='foo', reverse=True)
    assert not issorted(table5, key='foo', reverse=True, strict=True)


def test_sort_missing_cell_numeric():
    """ Sorting table with missing values raises IndexError #385 """
    tbl = (('a', 'b'), ('4',), ('2', '1'), ('1',))
    expect = (('a', 'b'), ('1',), ('2', '1'), ('4',))

    tbl_sorted = sort(tbl)
    ieq(expect, tbl_sorted)


def test_sort_missing_cell_text():
    """ Sorting table with missing values raises IndexError #385 """
    tbl = (('a', 'b', 'c'), ('C',), ('A', '4', '5'))
    expect = (('a', 'b', 'c'), ('A', '4', '5'), ('C',))

    tbl_sorted = sort(tbl)
    ieq(expect, tbl_sorted)


# ---------------------------------------------------------------------------
# per-field sort direction, null placement and short-row handling
# ---------------------------------------------------------------------------


def _accounting_table():
    return (('account', 'updated_at', 'note'),
            ('B', None, 'b-null'),
            ('A', 3, 'a-3'),
            ('A', 9, 'a-9'),
            ('C', 2, 'c-2'),
            ('A', 9, 'a-9-dup'),
            ('A', None, 'a-null'),
            ('B', 1, 'b-1'),
            ('C', 2, 'c-2-dup'))


_ACCOUNTING_EXPECT = (('account', 'updated_at', 'note'),
                      ('A', 9, 'a-9'),
                      ('A', 9, 'a-9-dup'),
                      ('A', 3, 'a-3'),
                      ('A', None, 'a-null'),
                      ('B', 1, 'b-1'),
                      ('B', None, 'b-null'),
                      ('C', 2, 'c-2'),
                      ('C', 2, 'c-2-dup'))


def test_sort_per_field_direction_and_nulls():
    table = _accounting_table()
    sortkey = ['account', ('updated_at', 'desc', 'last')]
    ieq(_ACCOUNTING_EXPECT, sort(table, sortkey=sortkey, cache=False))


def test_sort_per_field_buffered_matches_memory():
    # the comparison rules must give identical results in memory, across
    # chunks, after merging and when re-read from the cache
    table = _accounting_table()
    sortkey = ['account', ('updated_at', 'desc', 'last')]
    memory = list(sort(table, sortkey=sortkey, buffersize=None, cache=False))
    for buffersize in (2, 3):
        view = sort(table, sortkey=sortkey, buffersize=buffersize,
                    tempdir='tmp')
        first = list(view)
        eq_(memory, first)
        # second pass merges the cached chunk files
        eq_(memory, list(view))
        # and uncached external sort agrees too
        eq_(memory,
           list(sort(table, sortkey=sortkey, buffersize=buffersize,
                     cache=False, tempdir='tmp')))


def test_sort_equal_keys_stable_across_chunks():
    # equal sort keys must retain the original input order; whole rows are
    # never compared to break ties
    table = _accounting_table()
    sortkey = ['account', ('updated_at', 'desc', 'last')]
    result = list(sort(table, sortkey=sortkey, buffersize=2, cache=False,
                       tempdir='tmp'))
    notes = [r[2] for r in result[1:]]
    assert notes.index('a-9') < notes.index('a-9-dup')
    assert notes.index('c-2') < notes.index('c-2-dup')
    # identical rows also stay in input order (no whole-row comparison)
    t = (('k', 'v'), ('a', 1), ('a', 1), ('a', 1))
    ieq(t, sort(t, sortkey=['k'], buffersize=1, cache=False, tempdir='tmp'))


def test_sort_nulls_first_and_last_explicit():
    table = (('k', 'v'),
             ('a', None), ('a', 1), ('b', None), ('b', 0))
    # ascending value with nulls first
    expect = (('k', 'v'),
              ('a', None), ('b', None), ('b', 0), ('a', 1))
    ieq(expect, sort(table, sortkey=[('v', 'asc', 'first')], cache=False))
    ieq(expect, sort(table, sortkey=[('v', 'asc', 'first')], buffersize=2,
                     cache=False, tempdir='tmp'))
    # ascending value with nulls last
    expect = (('k', 'v'),
              ('b', 0), ('a', 1), ('a', None), ('b', None))
    ieq(expect, sort(table, sortkey=[('v', 'asc', 'last')], cache=False))
    ieq(expect, sort(table, sortkey=[('v', 'asc', 'last')], buffersize=2,
                     cache=False, tempdir='tmp'))


def test_sort_nulls_default_argument():
    table = (('k', 'v'), ('a', None), ('b', 0), ('a', 1))
    expect = (('k', 'v'), ('b', 0), ('a', 1), ('a', None))
    ieq(expect, sort(table, sortkey=['v'], nulls='last', cache=False))


def test_sort_direction_as_string():
    table = (('v',), (3,), (1,), (2,))
    ieq((('v',), (3,), (2,), (1,)),
        sort(table, sortkey=[('v', 'descending')], cache=False))
    ieq((('v',), (1,), (2,), (3,)),
        sort(table, sortkey=[('v', 'asc')], cache=False))


def test_sort_sortfield_object():
    from petl.transform.sorts import SortField
    table = (('foo', 'bar'),
             ('C',), ('A', None), ('B', 1), ('A', 2))
    expect = (('foo', 'bar'),
              ('A', 2), ('B', 1), ('C',), ('A', None))
    ieq(expect, sort(table, sortkey=[SortField('bar', True, 'last')],
                     cache=False))


def test_sort_short_rows_fill_default():
    # default 'fill' keeps the historical behaviour: absent cells compare
    # like actual None values
    table = (('foo', 'bar'), ('C',), ('A', None), ('B', 1), ('A', 2))
    expect = (('foo', 'bar'),
              ('C',), ('A', None), ('B', 1), ('A', 2))
    ieq(expect, sort(table, sortkey=['bar'], cache=False))
    ieq(expect, sort(table, sortkey=['bar'], buffersize=2, cache=False,
                     tempdir='tmp'))


def test_sort_short_rows_raise():
    table = (('foo', 'bar'), ('C',), ('A', 1))
    view = sort(table, sortkey=['bar'], short_rows='raise', cache=False)
    with pytest.raises(SortError):
        list(view)


def test_sort_short_rows_distinct():
    # explicit None and absent field stay distinct; the absent field sorts
    # immediately beyond the null position
    table = (('foo', 'bar'),
             ('A', None), ('C',), ('B', 1), ('A', 2))
    # nulls first: None (A) precedes the absent field (C)
    expect = (('foo', 'bar'),
              ('A', None), ('C',), ('B', 1), ('A', 2))
    ieq(expect, sort(table, sortkey=[('bar', 'asc', 'first')],
                     short_rows='distinct', cache=False))
    # nulls last: values, then None (A), then the absent field (C)
    expect = (('foo', 'bar'),
              ('B', 1), ('A', 2), ('A', None), ('C',))
    ieq(expect, sort(table, sortkey=[('bar', 'asc', 'last')],
                     short_rows='distinct', cache=False))
    ieq(expect, sort(table, sortkey=[('bar', 'asc', 'last')],
                     short_rows='distinct', buffersize=2, cache=False,
                     tempdir='tmp'))


def test_sort_mixed_arguments_rejected():
    table = (('foo', 'bar'), ('A', 1))
    with pytest.raises(ArgumentError):
        sort(table, key='foo', sortkey=['foo'])
    with pytest.raises(ArgumentError):
        sort(table, sortkey=['foo'], reverse=True)
    with pytest.raises(ArgumentError):
        list(sort(table, sortkey=['foo'], nulls='middle', cache=False))
    with pytest.raises(ArgumentError):
        list(sort(table, sortkey=['foo'], short_rows='bogus', cache=False))
    with pytest.raises(ArgumentError):
        list(sort(table, sortkey=[('foo', 'sideways')], cache=False))
    with pytest.raises(ArgumentError):
        sort(table, key='foo', reverse='yes')


def test_sort_legacy_key_reverse_unchanged():
    table = _accounting_table()
    legacy_memory = list(sort(table, 'updated_at', reverse=True))
    legacy_buffered = list(sort(table, 'updated_at', reverse=True,
                                buffersize=2, tempdir='tmp'))
    eq_(legacy_memory, legacy_buffered)


def test_mergesort_per_field():
    t1 = (('account', 'updated_at'), ('A', 3), ('B', None))
    t2 = (('account', 'updated_at'), ('A', 9), ('C', 2), ('B', 1))
    sortkey = ['account', ('updated_at', 'desc', 'last')]
    expect = sort(cat(t1, t2), sortkey=sortkey)
    actual = mergesort(t1, t2, sortkey=sortkey, buffersize=2)
    ieq(expect, actual)
    ieq(expect, actual)
    actual = mergesort(sort(t1, sortkey=sortkey),
                       sort(t2, sortkey=sortkey),
                       sortkey=sortkey, presorted=True)
    ieq(expect, actual)


def test_issorted_per_field():
    table = list(_ACCOUNTING_EXPECT)
    assert issorted(table, sortkey=['account', ('updated_at', 'desc', 'last')])
    assert not issorted(_accounting_table(),
                        sortkey=['account', ('updated_at', 'desc', 'last')])
    # strict detects equal adjacent keys
    dup = (('k',), (1,), (1,))
    assert issorted(dup, key=0)
    assert not issorted(dup, key=0, strict=True)
    # null placement is respected
    nulls_first = (('v',), (None,), (3,), (2,))
    assert issorted(nulls_first, sortkey=[('v', 'desc', 'first')])
    assert not issorted(nulls_first, sortkey=[('v', 'desc', 'last')])
