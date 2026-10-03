from __future__ import absolute_import, print_function, division

import os
import sys

import pytest

from petl.errors import FieldSelectionError
from petl.test.failonerror import assert_failonerror
from petl.test.helpers import ieq
from petl.transform.conversions import convert, convertall, convertnumbers, \
    replace, update, format, interpolate, convertfork

from functools import partial


def test_convert():

    table1 = (('foo', 'bar', 'baz'),
              ('A', 1, 2),
              ('B', '2', '3.4'),
              (u'B', u'3', u'7.8', True),
              ('D', 'xyz', 9.0),
              ('E', None))

    # test the simplest style - single field, lambda function
    table2 = convert(table1, 'foo', lambda s: s.lower())
    expect2 = (('foo', 'bar', 'baz'),
               ('a', 1, 2),
               ('b', '2', '3.4'),
               (u'b', u'3', u'7.8', True),
               ('d', 'xyz', 9.0),
               ('e', None))
    ieq(expect2, table2)
    ieq(expect2, table2)

    # test single field with method call
    table3 = convert(table1, 'foo', 'lower')
    expect3 = expect2
    ieq(expect3, table3)

    # test single field with method call with arguments
    table4 = convert(table1, 'foo', 'replace', 'B', 'BB')
    expect4 = (('foo', 'bar', 'baz'),
               ('A', 1, 2),
               ('BB', '2', '3.4'),
               (u'BB', u'3', u'7.8', True),
               ('D', 'xyz', 9.0),
               ('E', None))
    ieq(expect4, table4)

    # test multiple fields with the same conversion
    table5 = convert(table1, ('bar', 'baz'), str)
    expect5 = (('foo', 'bar', 'baz'),
               ('A', '1', '2'),
               ('B', '2', '3.4'),
               (u'B', u'3', u'7.8', True),
               ('D', 'xyz', '9.0'),
               ('E', 'None'))
    ieq(expect5, table5)

    # test convert with dictionary
    table6 = convert(table1, 'foo', {'A': 'Z', 'B': 'Y'})
    expect6 = (('foo', 'bar', 'baz'),
               ('Z', 1, 2),
               ('Y', '2', '3.4'),
               (u'Y', u'3', u'7.8', True),
               ('D', 'xyz', 9.0),
               ('E', None))
    ieq(expect6, table6)


def test_convert_empty():
    table = (('foo', 'bar'),)
    expect = (('foo', 'bar'),)
    actual = convert(table, 'foo', int)
    ieq(expect, actual)


def test_convert_headerless():
    table = ()
    with pytest.raises(FieldSelectionError):
        for i in convert(table, 'foo', int):
            pass


def test_convert_headerless_no_conversions():
    table = expect = ()
    actual = convert(table)
    ieq(expect, actual)


def test_convert_indexes():

    table1 = (('foo', 'bar', 'baz'),
              ('A', 1, 2),
              ('B', '2', '3.4'),
              (u'B', u'3', u'7.8', True),
              ('D', 'xyz', 9.0),
              ('E', None))

    # test the simplest style - single field, lambda function
    table2 = convert(table1, 0, lambda s: s.lower())
    expect2 = (('foo', 'bar', 'baz'),
               ('a', 1, 2),
               ('b', '2', '3.4'),
               (u'b', u'3', u'7.8', True),
               ('d', 'xyz', 9.0),
               ('e', None))
    ieq(expect2, table2)
    ieq(expect2, table2)

    # test single field with method call
    table3 = convert(table1, 0, 'lower')
    expect3 = expect2
    ieq(expect3, table3)

    # test single field with method call with arguments
    table4 = convert(table1, 0, 'replace', 'B', 'BB')
    expect4 = (('foo', 'bar', 'baz'),
               ('A', 1, 2),
               ('BB', '2', '3.4'),
               (u'BB', u'3', u'7.8', True),
               ('D', 'xyz', 9.0),
               ('E', None))
    ieq(expect4, table4)

    # test multiple fields with the same conversion
    table5a = convert(table1, (1, 2), str)
    table5b = convert(table1, (1, 'baz'), str)
    table5c = convert(table1, ('bar', 2), str)
    table5d = convert(table1, list(range(1, 3)), str)
    expect5 = (('foo', 'bar', 'baz'),
               ('A', '1', '2'),
               ('B', '2', '3.4'),
               (u'B', u'3', u'7.8', True),
               ('D', 'xyz', '9.0'),
               ('E', 'None'))
    ieq(expect5, table5a)
    ieq(expect5, table5b)
    ieq(expect5, table5c)
    ieq(expect5, table5d)

    # test convert with dictionary
    table6 = convert(table1, 0, {'A': 'Z', 'B': 'Y'})
    expect6 = (('foo', 'bar', 'baz'),
               ('Z', 1, 2),
               ('Y', '2', '3.4'),
               (u'Y', u'3', u'7.8', True),
               ('D', 'xyz', 9.0),
               ('E', None))
    ieq(expect6, table6)


def test_fieldconvert():

    table1 = (('foo', 'bar', 'baz'),
              ('A', 1, 2),
              ('B', '2', '3.4'),
              (u'B', u'3', u'7.8', True),
              ('D', 'xyz', 9.0),
              ('E', None))

    # test the style where the converters functions are passed in as a
    # dictionary
    converters = {'foo': str, 'bar': int, 'baz': float}
    table5 = convert(table1, converters, errorvalue='error')
    expect5 = (('foo', 'bar', 'baz'),
               ('A', 1, 2.0),
               ('B', 2, 3.4),
               ('B', 3, 7.8, True),  # N.B., long rows are preserved
               ('D', 'error', 9.0),
               ('E', 'error'))  # N.B., short rows are preserved
    ieq(expect5, table5)

    # test the style where the converters functions are added one at a time
    table6 = convert(table1, errorvalue='err')
    table6['foo'] = str
    table6['bar'] = int
    table6['baz'] = float
    expect6 = (('foo', 'bar', 'baz'),
               ('A', 1, 2.0),
               ('B', 2, 3.4),
               ('B', 3, 7.8, True),
               ('D', 'err', 9.0),
               ('E', 'err'))
    ieq(expect6, table6)

    # test some different converters
    table7 = convert(table1)
    table7['foo'] = 'replace', 'B', 'BB'
    expect7 = (('foo', 'bar', 'baz'),
               ('A', 1, 2),
               ('BB', '2', '3.4'),
               (u'BB', u'3', u'7.8', True),
               ('D', 'xyz', 9.0),
               ('E', None))
    ieq(expect7, table7)

    # test the style where the converters functions are passed in as a list
    converters = [str, int, float]
    table8 = convert(table1, converters, errorvalue='error')
    expect8 = (('foo', 'bar', 'baz'),
               ('A', 1, 2.0),
               ('B', 2, 3.4),
               ('B', 3, 7.8, True),  # N.B., long rows are preserved
               ('D', 'error', 9.0),
               ('E', 'error'))  # N.B., short rows are preserved
    ieq(expect8, table8)

    # test the style where the converters functions are passed in as a list
    converters = [str, None, float]
    table9 = convert(table1, converters, errorvalue='error')
    expect9 = (('foo', 'bar', 'baz'),
               ('A', 1, 2.0),
               ('B', '2', 3.4),
               ('B', u'3', 7.8, True),  # N.B., long rows are preserved
               ('D', 'xyz', 9.0),
               ('E', None))  # N.B., short rows are preserved
    ieq(expect9, table9)


def test_convertall():

    table1 = (('foo', 'bar', 'baz'),
              ('1', '3', '9'),
              ('2', '1', '7'))
    table2 = convertall(table1, int)
    expect2 = (('foo', 'bar', 'baz'),
               (1, 3, 9),
               (2, 1, 7))
    ieq(expect2, table2)
    ieq(expect2, table2)

    # test with non-string field names
    table1 = (('foo', 3, 4),
              (2, 2, 2))
    table2 = convertall(table1, lambda x: x**2)
    expect = (('foo', 3, 4),
              (4, 4, 4))
    ieq(expect, table2)


def test_convertnumbers():

    table1 = (('foo', 'bar', 'baz', 'quux'),
              ('1', '3.0', '9+3j', 'aaa'),
              ('2', '1.3', '7+2j', None))
    table2 = convertnumbers(table1)
    expect2 = (('foo', 'bar', 'baz', 'quux'),
               (1, 3.0, 9+3j, 'aaa'),
               (2, 1.3, 7+2j, None))
    ieq(expect2, table2)
    ieq(expect2, table2)


def test_convert_translate():

    table = (('foo', 'bar'),
             ('M', 12),
             ('F', 34),
             ('-', 56))

    trans = {'M': 'male', 'F': 'female'}
    result = convert(table, 'foo', trans)
    expectation = (('foo', 'bar'),
                   ('male', 12),
                   ('female', 34),
                   ('-', 56))
    ieq(expectation, result)


def test_convert_with_row():

    table = (('foo', 'bar'),
             ('a', 1),
             ('b', 2))

    expect = (('foo', 'bar'),
              ('a', 'A'),
              ('b', 'B'))

    actual = convert(table, 'bar',
                     lambda v, row: row.foo.upper(),
                     pass_row=True)
    ieq(expect, actual)


def test_convert_with_row_backwards_compat():

    table = (('foo', 'bar'),
             (' a ', 1),
             (' b ', 2))

    expect = (('foo', 'bar'),
              ('a', 1),
              ('b', 2))

    actual = convert(table, 'foo', 'strip')
    ieq(expect, actual)


def test_convert_where():

    tbl1 = (('foo', 'bar'),
            ('a', 1),
            ('b', 2))

    expect = (('foo', 'bar'),
              ('a', 1),
              ('b', 4))

    actual = convert(tbl1, 'bar', lambda v: v*2, where=lambda r: r.foo == 'b')
    ieq(expect, actual)
    ieq(expect, actual)
    actual = convert(tbl1, 'bar', lambda v: v*2, where="{foo} == 'b'")
    ieq(expect, actual)
    ieq(expect, actual)


def test_convert_failonerror():
    input_  = (('foo',), ('A',), (1,))
    cvt_    = {'foo': 'lower'}
    expect_ = (('foo',), ('a',), (None,))

    assert_failonerror(
            input_fn=partial(convert, input_, cvt_),
            expected_output=expect_)


def test_replace_where():

    tbl1 = (('foo', 'bar'),
            ('a', 1),
            ('b', 2))

    expect = (('foo', 'bar'),
              ('a', 1),
              ('b', 4))

    actual = replace(tbl1, 'bar', 2, 4, where=lambda r: r.foo == 'b')
    ieq(expect, actual)
    ieq(expect, actual)
    actual = replace(tbl1, 'bar', 2, 4, where="{foo} == 'b'")
    ieq(expect, actual)
    ieq(expect, actual)


def test_update():

    table1 = (('foo', 'bar', 'baz'),
              ('A', 1, 2),
              ('B', '2', '3.4'),
              (u'B', u'3', u'7.8', True),
              ('D', 'xyz', 9.0),
              ('E', None))

    table2 = update(table1, 'foo', 'X')
    expect2 = (('foo', 'bar', 'baz'),
               ('X', 1, 2),
               ('X', '2', '3.4'),
               ('X', u'3', u'7.8', True),
               ('X', 'xyz', 9.0),
               ('X', None))
    ieq(expect2, table2)
    ieq(expect2, table2)


def test_replace_unhashable():

    table1 = (('foo', 'bar'), ('a', ['b']), ('c', None))
    expect = (('foo', 'bar'), ('a', ['b']), ('c', []))
    actual = replace(table1, 'bar', None, [])
    ieq(expect, actual)


def test_format():

    table = (('foo', 'bar'),
             ('a', 1),
             ('b', 2))

    expect = (('foo', 'bar'),
              ('a', '01'),
              ('b', '02'))

    actual = format(table, 'bar', '{0:02d}')
    ieq(expect, actual)
    ieq(expect, actual)


def test_interpolate():

    table = (('foo', 'bar'),
             ('a', 1),
             ('b', 2))

    expect = (('foo', 'bar'),
              ('a', '01'),
              ('b', '02'))

    actual = interpolate(table, 'bar', '%02d')
    ieq(expect, actual)
    ieq(expect, actual)


def _convertfork_table():
    return (('foo', 'bar'),
            ('A', '1'),
            ('B', 'x'),
            ('C', '3'),
            ('D', 'y'))


def _check_convertfork_rejected(rejected):
    assert rejected[0] == ('row', 'field', 'value', 'error')
    assert len(rejected) == 3
    assert rejected[1][:3] == (2, 'bar', 'x')
    assert rejected[2][:3] == (4, 'bar', 'y')
    for row in rejected[1:]:
        # error details are captured as plain strings, not exceptions
        assert isinstance(row[3], str)
        assert row[3].startswith('ValueError')


def test_convertfork():
    # read the accepted output first, then the rejected output
    fork = convertfork(_convertfork_table(), 'bar', int)
    try:
        expect_accepted = (('foo', 'bar'),
                           ('A', 1),
                           ('C', 3))
        ieq(expect_accepted, fork.accepted)
        _check_convertfork_rejected(list(fork.rejected))
    finally:
        fork.close()


def test_convertfork_rejected_first():
    # the rejected output can be fully consumed before the accepted output
    fork = convertfork(_convertfork_table(), 'bar', int)
    try:
        _check_convertfork_rejected(list(fork.rejected))
        expect_accepted = (('foo', 'bar'),
                           ('A', 1),
                           ('C', 3))
        ieq(expect_accepted, fork.accepted)
    finally:
        fork.close()


def test_convertfork_interleaved():
    # the two outputs can be consumed in an interleaved manner
    fork = convertfork(_convertfork_table(), 'bar', int)
    try:
        acc = iter(fork.accepted)
        rej = iter(fork.rejected)
        assert next(acc) == ('foo', 'bar')
        assert next(rej) == ('row', 'field', 'value', 'error')
        assert next(acc) == ('A', 1)
        assert next(rej)[:3] == (2, 'bar', 'x')
        assert next(acc) == ('C', 3)
        assert next(rej)[:3] == (4, 'bar', 'y')
        with pytest.raises(StopIteration):
            next(acc)
        with pytest.raises(StopIteration):
            next(rej)
    finally:
        fork.close()


def test_convertfork_single_pass():
    # the source is read once and each value is converted at most once,
    # however the outputs are consumed
    calls = list()

    def source():
        for row in _convertfork_table():
            yield row

    def conv(v):
        calls.append(v)
        return int(v)

    fork = convertfork(source(), 'bar', conv)
    try:
        rejected = list(fork.rejected)
        accepted = list(fork.accepted)
        # read both outputs a second time, must not re-read the source or
        # re-run any conversion
        assert list(fork.accepted) == accepted
        assert list(fork.rejected) == rejected
        assert calls == ['1', 'x', '3', 'y']
        assert len(rejected) == 3
        assert len(accepted) == 3
    finally:
        fork.close()


def test_convertfork_close_reclaims_temp_state(tmpdir):
    # abandon the accepted output part way through, then close - all
    # temporary spill files must be released
    tempdir = str(tmpdir)
    fork = convertfork(_convertfork_table(), 'bar', int, tempdir=tempdir)
    acc = iter(fork.accepted)
    next(acc)
    next(acc)
    del acc
    assert len(os.listdir(tempdir)) == 2
    fork.close()
    assert os.listdir(tempdir) == []
    # outputs cannot be read after close
    with pytest.raises(RuntimeError):
        list(fork.accepted)
    with pytest.raises(RuntimeError):
        list(fork.rejected)
    # close is idempotent
    fork.close()


def test_convertfork_context_manager(tmpdir):
    tempdir = str(tmpdir)
    with convertfork(_convertfork_table(), 'bar', int,
                     tempdir=tempdir) as fork:
        _check_convertfork_rejected(list(fork.rejected))
        assert len(os.listdir(tempdir)) == 2
    assert os.listdir(tempdir) == []


def test_convertfork_abandon_rejected(tmpdir):
    # abandon the rejected output without reading it at all
    tempdir = str(tmpdir)
    fork = convertfork(_convertfork_table(), 'bar', int, tempdir=tempdir)
    expect_accepted = (('foo', 'bar'),
                       ('A', 1),
                       ('C', 3))
    ieq(expect_accepted, fork.accepted)
    fork.close()
    assert os.listdir(tempdir) == []


def test_convertfork_does_not_retain_exceptions():
    fork = convertfork(_convertfork_table(), 'bar', int)
    rejected = list(fork.rejected)
    fork.close()
    for row in rejected[1:]:
        # only plain strings are kept, no exception or traceback objects
        assert isinstance(row[3], str)
    if sys.version_info[0] >= 3:
        assert sys.exc_info() == (None, None, None)


def test_convertfork_multiple_fields():
    table = (('a', 'b'),
             ('1', '2'),
             ('x', '3'),
             ('4', 'y'))
    fork = convertfork(table, {'a': int, 'b': int})
    try:
        expect_accepted = (('a', 'b'),
                           (1, 2))
        ieq(expect_accepted, fork.accepted)
        rejected = list(fork.rejected)
        assert rejected[0] == ('row', 'field', 'value', 'error')
        assert rejected[1][:3] == (2, 'a', 'x')
        assert rejected[2][:3] == (3, 'b', 'y')
    finally:
        fork.close()


def test_convertfork_where():
    # rows skipped by the where predicate pass through unchanged
    fork = convertfork(_convertfork_table(), 'bar', int,
                       where=lambda r: r.foo != 'B')
    try:
        expect_accepted = (('foo', 'bar'),
                           ('A', 1),
                           ('B', 'x'),
                           ('C', 3))
        ieq(expect_accepted, fork.accepted)
        rejected = list(fork.rejected)
        assert rejected[0] == ('row', 'field', 'value', 'error')
        assert len(rejected) == 2
        assert rejected[1][:3] == (4, 'bar', 'y')
    finally:
        fork.close()


def test_convertfork_pass_row():
    table = (('foo', 'bar'),
             ('A', '1'),
             ('B', '2'))
    fork = convertfork(table, 'bar',
                       lambda v, row: int(v) * len(row.foo), pass_row=True)
    try:
        expect_accepted = (('foo', 'bar'),
                           ('A', 1),
                           ('B', 2))
        ieq(expect_accepted, fork.accepted)
        assert list(fork.rejected) == [('row', 'field', 'value', 'error')]
    finally:
        fork.close()


def test_convertfork_empty():
    fork = convertfork((('foo', 'bar'),), 'bar', int)
    try:
        assert list(fork.accepted) == [('foo', 'bar')]
        assert list(fork.rejected) == [('row', 'field', 'value', 'error')]
    finally:
        fork.close()


def test_convertfork_all_rejected():
    table = (('foo', 'bar'),
             ('A', 'x'),
             ('B', 'y'))
    fork = convertfork(table, 'bar', int)
    try:
        assert list(fork.accepted) == [('foo', 'bar')]
        rejected = list(fork.rejected)
        assert len(rejected) == 3
        assert rejected[1][:3] == (1, 'bar', 'x')
        assert rejected[2][:3] == (2, 'bar', 'y')
    finally:
        fork.close()


def test_convertfork_bad_field():
    fork = convertfork(_convertfork_table(), 'quux', int)
    try:
        with pytest.raises(FieldSelectionError):
            list(fork.accepted)
    finally:
        fork.close()
