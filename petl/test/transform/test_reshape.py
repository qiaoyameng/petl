from __future__ import absolute_import, print_function, division


from datetime import datetime
import os
import subprocess
import sys

import pytest

from petl.errors import ArgumentError, FieldSelectionError
from petl.test.helpers import ieq
from petl.transform.reshape import melt, recast, transpose, pivot, flatten, \
    unflatten
from petl.transform.regex import split, capture


def test_melt_1():

    table = (('id', 'gender', 'age'),
             (1, 'F', 12),
             (2, 'M', 17),
             (3, 'M', 16))

    expectation = (('id', 'variable', 'value'),
                   (1, 'gender', 'F'),
                   (1, 'age', 12),
                   (2, 'gender', 'M'),
                   (2, 'age', 17),
                   (3, 'gender', 'M'),
                   (3, 'age', 16))

    result = melt(table, key='id')
    ieq(expectation, result)

    # use field index as key
    result = melt(table, key=0)
    ieq(expectation, result)

    result = melt(table, key='id', variablefield='variable', valuefield='value')
    ieq(expectation, result)


def test_melt_2():

    table = (('id', 'time', 'height', 'weight'),
             (1, 11, 66.4, 12.2),
             (2, 16, 53.2, 17.3),
             (3, 12, 34.5, 9.4))

    expectation = (('id', 'time', 'variable', 'value'),
                   (1, 11, 'height', 66.4),
                   (1, 11, 'weight', 12.2),
                   (2, 16, 'height', 53.2),
                   (2, 16, 'weight', 17.3),
                   (3, 12, 'height', 34.5),
                   (3, 12, 'weight', 9.4))
    result = melt(table, key=('id', 'time'))
    ieq(expectation, result)

    expectation = (('id', 'time', 'variable', 'value'),
                   (1, 11, 'height', 66.4),
                   (2, 16, 'height', 53.2),
                   (3, 12, 'height', 34.5))
    result = melt(table, key=('id', 'time'), variables='height')
    print(result)
    ieq(expectation, result)


def test_melt_empty():
    table = (('foo', 'bar', 'baz'),)
    expect = (('foo', 'variable', 'value'),)
    actual = melt(table, key='foo')
    ieq(expect, actual)


def test_melt_headerless():
    table = []
    expect = []
    actual = melt(table, key='foo')
    ieq(expect, actual)


def test_melt_1_shortrow():

    table = (('id', 'gender', 'age'),
             (1, 'F', 12),
             (2, 'M', 17),
             (3, 'M'),
             (4,))

    expectation = (('id', 'variable', 'value'),
                   (1, 'gender', 'F'),
                   (1, 'age', 12),
                   (2, 'gender', 'M'),
                   (2, 'age', 17),
                   (3, 'gender', 'M'))

    result = melt(table, key='id')
    ieq(expectation, result)

    result = melt(table, key='id', variablefield='variable', valuefield='value')
    ieq(expectation, result)


def test_melt_2_shortrow():

    table = (('id', 'time', 'height', 'weight'),
             (1, 11, 66.4, 12.2),
             (2, 16, 53.2, 17.3),
             (3, 12, 34.5),
             (4, 14))

    expectation = (('id', 'time', 'variable', 'value'),
                   (1, 11, 'height', 66.4),
                   (1, 11, 'weight', 12.2),
                   (2, 16, 'height', 53.2),
                   (2, 16, 'weight', 17.3),
                   (3, 12, 'height', 34.5))
    result = melt(table, key=('id', 'time'))
    ieq(expectation, result)

    expectation = (('id', 'time', 'variable', 'value'),
                   (1, 11, 'height', 66.4),
                   (2, 16, 'height', 53.2),
                   (3, 12, 'height', 34.5))
    result = melt(table, key=('id', 'time'), variables='height')
    ieq(expectation, result)


def test_recast_1():

    table = (('id', 'variable', 'value'),
             (3, 'age', 16),
             (1, 'gender', 'F'),
             (2, 'gender', 'M'),
             (2, 'age', 17),
             (1, 'age', 12),
             (3, 'gender', 'M'))

    expectation = (('id', 'age', 'gender'),
                   (1, 12, 'F'),
                   (2, 17, 'M'),
                   (3, 16, 'M'))

    # by default lift 'variable' field, hold everything else
    result = recast(table)
    ieq(expectation, result)

    result = recast(table, variablefield='variable')
    ieq(expectation, result)

    result = recast(table, key='id', variablefield='variable')
    ieq(expectation, result)

    result = recast(table, key='id', variablefield='variable',
                    valuefield='value')
    ieq(expectation, result)


def test_recast_2():

    table = (('id', 'variable', 'value'),
             (3, 'age', 16),
             (1, 'gender', 'F'),
             (2, 'gender', 'M'),
             (2, 'age', 17),
             (1, 'age', 12),
             (3, 'gender', 'M'))

    expectation = (('id', 'gender'),
                   (1, 'F'),
                   (2, 'M'),
                   (3, 'M'))

    # can manually pick which variables you want to recast as fields
    # TODO this is awkward
    result = recast(table, key='id', variablefield={'variable': ['gender']})
    ieq(expectation, result)


def test_recast_3():

    table = (('id', 'time', 'variable', 'value'),
             (1, 11, 'weight', 66.4),
             (1, 14, 'weight', 55.2),
             (2, 12, 'weight', 53.2),
             (2, 16, 'weight', 43.3),
             (3, 12, 'weight', 34.5),
             (3, 17, 'weight', 49.4))

    expectation = (('id', 'time', 'weight'),
                   (1, 11, 66.4),
                   (1, 14, 55.2),
                   (2, 12, 53.2),
                   (2, 16, 43.3),
                   (3, 12, 34.5),
                   (3, 17, 49.4))
    result = recast(table)
    ieq(expectation, result)

    # in the absence of an aggregation function, list all values
    expectation = (('id', 'weight'),
                   (1, [66.4, 55.2]),
                   (2, [53.2, 43.3]),
                   (3, [34.5, 49.4]))
    result = recast(table, key='id')
    ieq(expectation, result)

    # max aggregation
    expectation = (('id', 'weight'),
                   (1, 66.4),
                   (2, 53.2),
                   (3, 49.4))
    result = recast(table, key='id', reducers={'weight': max})
    ieq(expectation, result)

    # min aggregation
    expectation = (('id', 'weight'),
                   (1, 55.2),
                   (2, 43.3),
                   (3, 34.5))
    result = recast(table, key='id', reducers={'weight': min})
    ieq(expectation, result)

    # mean aggregation
    expectation = (('id', 'weight'),
                   (1, 60.80),
                   (2, 48.25),
                   (3, 41.95))

    def mean(values):
        return float(sum(values)) / len(values)

    def meanf(precision):
        def f(values):
            v = mean(values)
            v = round(v, precision)
            return v
        return f

    result = recast(table, key='id', reducers={'weight': meanf(precision=2)})
    ieq(expectation, result)


def test_recast4():

    # deal with missing data
    table = (('id', 'variable', 'value'),
             (1, 'gender', 'F'),
             (2, 'age', 17),
             (1, 'age', 12),
             (3, 'gender', 'M'))
    result = recast(table, key='id')
    expect = (('id', 'age', 'gender'),
              (1, 12, 'F'),
              (2, 17, None),
              (3, None, 'M'))
    ieq(expect, result)


def test_recast_empty():
    table = (('foo', 'variable', 'value'),)
    expect = (('foo',),)
    actual = recast(table)
    ieq(expect, actual)


def test_recast_headerless():
    table = []
    expect = []
    actual = recast(table)
    ieq(expect, actual)


def test_recast_date():

    dt = datetime.now().replace
    table = (('id', 'variable', 'value'),
             (dt(hour=3), 'age', 16),
             (dt(hour=1), 'gender', 'F'),
             (dt(hour=2), 'gender', 'M'),
             (dt(hour=2), 'age', 17),
             (dt(hour=1), 'age', 12),
             (dt(hour=3), 'gender', 'M'))

    expectation = (('id', 'age', 'gender'),
                   (dt(hour=1), 12, 'F'),
                   (dt(hour=2), 17, 'M'),
                   (dt(hour=3), 16, 'M'))

    # by default lift 'variable' field, hold everything else
    result = recast(table)
    ieq(expectation, result)

    result = recast(table, variablefield='variable')
    ieq(expectation, result)

    result = recast(table, key='id', variablefield='variable')
    ieq(expectation, result)

    result = recast(table, key='id', variablefield='variable',
                    valuefield='value')
    ieq(expectation, result)


def test_melt_and_capture():

    table = (('id', 'parad0', 'parad1', 'parad2'),
             ('1', '12', '34', '56'),
             ('2', '23', '45', '67'))

    expectation = (('id', 'parasitaemia', 'day'),
                   ('1', '12', '0'),
                   ('1', '34', '1'),
                   ('1', '56', '2'),
                   ('2', '23', '0'),
                   ('2', '45', '1'),
                   ('2', '67', '2'))

    step1 = melt(table, key='id', valuefield='parasitaemia')
    step2 = capture(step1, 'variable', 'parad(\\d+)', ('day',))
    ieq(expectation, step2)


def test_melt_and_split():

    table = (('id', 'parad0', 'parad1', 'parad2', 'tempd0', 'tempd1', 'tempd2'),
             ('1', '12', '34', '56', '37.2', '37.4', '37.9'),
             ('2', '23', '45', '67', '37.1', '37.8', '36.9'))

    expectation = (('id', 'value', 'variable', 'day'),
                   ('1', '12', 'para', '0'),
                   ('1', '34', 'para', '1'),
                   ('1', '56', 'para', '2'),
                   ('1', '37.2', 'temp', '0'),
                   ('1', '37.4', 'temp', '1'),
                   ('1', '37.9', 'temp', '2'),
                   ('2', '23', 'para', '0'),
                   ('2', '45', 'para', '1'),
                   ('2', '67', 'para', '2'),
                   ('2', '37.1', 'temp', '0'),
                   ('2', '37.8', 'temp', '1'),
                   ('2', '36.9', 'temp', '2'))

    step1 = melt(table, key='id')
    step2 = split(step1, 'variable', 'd', ('variable', 'day'))
    ieq(expectation, step2)


def test_transpose():
    table1 = (('id', 'colour'),
              (1, 'blue'),
              (2, 'red'),
              (3, 'purple'),
              (5, 'yellow'),
              (7, 'orange'))
    table2 = transpose(table1)
    expect2 = (('id', 1, 2, 3, 5, 7),
               ('colour', 'blue', 'red', 'purple', 'yellow', 'orange'))
    ieq(expect2, table2)
    ieq(expect2, table2)


def test_transpose_empty():
    table1 = (('id', 'colour'),)
    table2 = transpose(table1)
    expect2 = (('id',),
               ('colour',))
    ieq(expect2, table2)


def test_pivot():

    table1 = (('region', 'gender', 'style', 'units'),
              ('east', 'boy', 'tee', 12),
              ('east', 'boy', 'golf', 14),
              ('east', 'boy', 'fancy', 7),
              ('east', 'girl', 'tee', 3),
              ('east', 'girl', 'golf', 8),
              ('east', 'girl', 'fancy', 18),
              ('west', 'boy', 'tee', 12),
              ('west', 'boy', 'golf', 15),
              ('west', 'boy', 'fancy', 8),
              ('west', 'girl', 'tee', 6),
              ('west', 'girl', 'golf', 16),
              ('west', 'girl', 'fancy', 1))

    table2 = pivot(table1, 'region', 'gender', 'units', sum)
    expect2 = (('region', 'boy', 'girl'),
               ('east', 33, 29),
               ('west', 35, 23))
    ieq(expect2, table2)
    ieq(expect2, table2)


def test_pivot_empty():

    table1 = (('region', 'gender', 'style', 'units'),)
    table2 = pivot(table1, 'region', 'gender', 'units', sum)
    expect2 = (('region',),)
    ieq(expect2, table2)


def test_pivot_headerless():
    table1 = []
    with pytest.raises(FieldSelectionError):
        for i in pivot(table1, 'region', 'gender', 'units', sum):
            pass


def test_flatten():

    table1 = (('foo', 'bar', 'baz'),
              ('A', 1, True),
              ('C', 7, False),
              ('B', 2, False),
              ('C', 9, True))

    expect1 = ('A', 1, True, 'C', 7, False, 'B', 2, False, 'C', 9, True)

    actual1 = flatten(table1)
    ieq(expect1, actual1)
    ieq(expect1, actual1)


def test_flatten_empty():

    table1 = (('foo', 'bar', 'baz'),)
    expect1 = []
    actual1 = flatten(table1)
    ieq(expect1, actual1)


def test_flatten_headerless():
    table1 = []
    expect1 = []
    actual1 = flatten(table1)
    ieq(expect1, actual1)


def test_unflatten():

    table1 = (('lines',),
              ('A',),
              (1,),
              (True,),
              ('C',),
              (7,),
              (False,),
              ('B',),
              (2,),
              (False,),
              ('C',),
              (9,))

    expect1 = (('f0', 'f1', 'f2'),
               ('A', 1, True),
               ('C', 7, False),
               ('B', 2, False),
               ('C', 9, None))

    actual1 = unflatten(table1, 'lines', 3)

    ieq(expect1, actual1)
    ieq(expect1, actual1)


def test_unflatten_2():

    inpt = ('A', 1, True, 'C', 7, False, 'B', 2, False, 'C', 9)

    expect1 = (('f0', 'f1', 'f2'),
               ('A', 1, True),
               ('C', 7, False),
               ('B', 2, False),
               ('C', 9, None))

    actual1 = unflatten(inpt, 3)

    ieq(expect1, actual1)
    ieq(expect1, actual1)


def test_unflatten_empty():

    table1 = (('lines',),)
    expect1 = (('f0', 'f1', 'f2'),)
    actual1 = unflatten(table1, 'lines', 3)
    ieq(expect1, actual1)


@pytest.mark.parametrize('header,kwargs,message', [
    ((), {}, 'invalid value field: value'),
    (('id', 'variable', 'value'), {'valuefield': 'missing'},
     'invalid value field: missing'),
    (('id', 'variable', 'value'), {'key': 'missing'},
     'invalid key field: missing'),
    (('id', 'variable', 'value'), {'variablefield': 'missing'},
     'invalid variable field: missing'),
    (('id', 'variable', 'value'), {'variablefield': {'missing': ['x']}},
     'invalid variable field: missing'),
    (('id', 'variable', 'value'), {'key': ['id', 'value']},
     'value field cannot be a key field'),
    (('id', 'variable', 'value'), {'variablefield': ['variable', 'value']},
     'value field cannot be a variable field'),
])
def test_recast_invalid_fields(header, kwargs, message):
    with pytest.raises(ArgumentError) as exc:
        list(recast([header], **kwargs))
    assert str(exc.value) == 'argument error: ' + message


def test_recast_validation_with_optimization():
    code = """
from petl import empty, recast
from petl.errors import ArgumentError
try:
    list(recast(empty()))
except ArgumentError:
    pass
else:
    raise RuntimeError('recast accepted a missing value field')
"""
    subprocess.check_call([sys.executable, '-O', '-c', code])


def _one_shot(rows):
    # a source that can only be iterated once
    return (row for row in rows)


def test_recast_scanall_discovers_late_variables():
    # the variable 'late' first appears well past the default samplesize
    late = (1, 'late', 99)
    rows = (('id', 'variable', 'value'),) + \
        tuple((1, 'a', i) for i in range(1100)) + (late,) + \
        ((2, 'a', 5),)

    # sampling mode misses the late variable
    sampled = recast(list(rows), samplesize=100)
    assert list(sampled)[0] == ('id', 'a')

    expect = (('id', 'a', 'late'),
              (1, list(range(1100)), 99),
              (2, 5, None))

    # variables are emitted in order of first occurrence, even with a
    # one-shot source that cannot be re-read
    result = recast(_one_shot(rows), scanall=True)
    ieq(expect, result)

    # list() must not pre-iterate (via __len__) and drain the one-shot source
    result = recast(_one_shot(rows), scanall=True, maxrows=50)
    assert len(list(result)) == 3
    with pytest.raises(TypeError):
        len(result)

    # the default sampling mode keeps the usual row-counting behaviour
    assert len(recast(list(rows), samplesize=100)) == 3


def test_recast_scanall_first_occurrence_order():
    rows = (('id', 'variable', 'value'),
            (1, 'zeta', 1),
            (1, 'alpha', 2),
            (2, 'zeta', 3),
            (2, 'beta', 4))
    expect = (('id', 'zeta', 'alpha', 'beta'),
              (1, 1, 2, None),
              (2, 3, None, 4))
    ieq(expect, recast(_one_shot(rows), scanall=True))


def test_recast_scanall_disk_equivalence():
    # a tiny in-memory row cap forces the spill-to-disk path; results must
    # match an ordinary re-readable table and the in-memory scanall path
    rows = (('id', 'variable', 'value'),) + \
        tuple((i % 7, 'v%d' % (i % 13), i) for i in range(200))

    # sampling mode sorts discovered variables; scanall uses first-occurrence
    expect_header = ('id',) + tuple('v%d' % j for j in range(13))
    in_memory = list(recast(list(rows), scanall=True))
    on_disk = list(recast(_one_shot(rows), scanall=True, maxrows=1))

    assert in_memory[0] == expect_header
    assert on_disk == in_memory


def test_recast_scanall_reducers_on_disk():
    rows = (('id', 'variable', 'value'),) + \
        tuple((1, 'w', float(i)) for i in range(1, 31)) + \
        ((2, 'w', 5.0),)
    expect = (('id', 'w'),
              (1, 30.0),
              (2, 5.0))
    result = recast(_one_shot(rows), key='id', reducers={'w': max},
                    scanall=True, maxrows=7)
    ieq(expect, result)


def test_recast_scanall_empty_and_header_only():
    ieq([], recast(_one_shot(()), scanall=True))
    ieq((('foo',),), recast(_one_shot((('foo', 'variable', 'value'),)),
                            scanall=True))
    ieq((('foo',),), recast(_one_shot((('foo', 'variable', 'value'),)),
                            scanall=True, maxrows=1))


def test_recast_scanall_explicit_variables_still_work():
    rows = (('id', 'variable', 'value'),
            (1, 'gender', 'F'),
            (2, 'age', 17),
            (1, 'age', 12),
            (3, 'gender', 'M'))
    expect = (('id', 'gender'),
              (1, 'F'),
              (2, None),
              (3, 'M'))
    ieq(expect, recast(list(rows), key='id',
                       variablefield={'variable': ['gender']},
                       scanall=True))


def test_recast_scanall_invalid_maxrows():
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ArgumentError):
            recast([], scanall=True, maxrows=bad)


def test_recast_scanall_cleans_up_tempfiles():
    import petl.transform.reshape as reshape

    created = []
    real_named_temporary_file = reshape.NamedTemporaryFile

    def tracking(*args, **kwargs):
        tmp = real_named_temporary_file(*args, **kwargs)
        created.append(tmp.name)
        return tmp

    reshape.NamedTemporaryFile = tracking
    try:
        rows = (('id', 'variable', 'value'),) + \
            tuple((i % 5, 'v%d' % (i % 3), i) for i in range(60))

        # normal iteration spills and removes every file
        result = recast(_one_shot(rows), scanall=True, maxrows=4)
        list(result)
        assert created
        assert not [p for p in created if os.path.exists(p)]

        # the source raising mid-stream still releases the spill files
        del created[:]

        def failing_source():
            yield ('id', 'variable', 'value')
            for i in range(40):
                yield (1, 'a', i)
            raise RuntimeError('boom')

        with pytest.raises(RuntimeError):
            list(recast(failing_source(), scanall=True, maxrows=4))
        assert created
        assert not [p for p in created if os.path.exists(p)]

        # a failing reducer also releases the spill files
        del created[:]

        def broken_reducer(values):
            raise ValueError('nope')

        with pytest.raises(ValueError):
            list(recast(_one_shot(rows), key='id',
                        reducers={'v0': broken_reducer},
                        scanall=True, maxrows=4))
        assert created
        assert not [p for p in created if os.path.exists(p)]

        # a consumer that abandons iteration triggers cleanup too
        del created[:]
        iterator = iter(recast(_one_shot(rows), scanall=True, maxrows=4))
        next(iterator)
        iterator.close()
        assert created
        assert not [p for p in created if os.path.exists(p)]
    finally:
        reshape.NamedTemporaryFile = real_named_temporary_file
