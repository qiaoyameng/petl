from __future__ import absolute_import, print_function, division


import operator


from collections import OrderedDict
import pytest

from petl.errors import AggregationError
from petl.test.helpers import ieq
from petl.util import strjoin
from petl.transform.reductions import rowreduce, aggregate, \
    mergeduplicates, Conflict, fold, aggregateaccumulate
from petl.util.base import Table


def test_rowreduce():
    
    table1 = (('foo', 'bar'),
              ('a', 3),
              ('a', 7),
              ('b', 2),
              ('b', 1),
              ('b', 9),
              ('c', 4))
    
    def sumbar(key, rows):
        return [key, sum(row[1] for row in rows)]
        
    table2 = rowreduce(table1, key='foo', reducer=sumbar, 
                       header=['foo', 'barsum'])
    expect2 = (('foo', 'barsum'),
               ('a', 10),
               ('b', 12),
               ('c', 4))
    ieq(expect2, table2)
    

def test_rowreduce_fieldnameaccess():
    
    table1 = (('foo', 'bar'),
              ('a', 3),
              ('a', 7),
              ('b', 2),
              ('b', 1),
              ('b', 9),
              ('c', 4))
    
    def sumbar(key, records):
        return [key, sum([rec['bar'] for rec in records])]
        
    table2 = rowreduce(table1, key='foo', reducer=sumbar, 
                       header=['foo', 'barsum'])
    expect2 = (('foo', 'barsum'),
               ('a', 10),
               ('b', 12),
               ('c', 4))
    ieq(expect2, table2)
    

def test_rowreduce_more():
    
    table1 = (('foo', 'bar'),
              ('aa', 3),
              ('aa', 7),
              ('bb', 2),
              ('bb', 1),
              ('bb', 9),
              ('cc', 4))
    
    def sumbar(key, records):
        return [key, sum(rec['bar'] for rec in records)]
        
    table2 = rowreduce(table1, key='foo', reducer=sumbar, 
                       header=['foo', 'barsum'])
    expect2 = (('foo', 'barsum'),
               ('aa', 10),
               ('bb', 12),
               ('cc', 4))
    ieq(expect2, table2)
    

def test_rowreduce_empty():
    table = (('foo', 'bar'),)
    expect = (('foo', 'bar'),)
    reducer = lambda key, rows: (key, [r[0] for r in rows])
    actual = rowreduce(table, key='foo', reducer=reducer, 
                       header=('foo', 'bar'))
    ieq(expect, actual)


def test_aggregate_simple():
    
    table1 = (('foo', 'bar', 'baz'),
              ('a', 3, True),
              ('a', 7, False),
              ('b', 2, True),
              ('b', 2, False),
              ('b', 9, False),
              ('c', 4, True))

    # simplest signature - aggregate whole rows
    table2 = aggregate(table1, 'foo', len)
    expect2 = (('foo', 'value'),
               ('a', 2),
               ('b', 3),
               ('c', 1))
    ieq(expect2, table2)
    ieq(expect2, table2)

    # next simplest signature - aggregate single field
    table3 = aggregate(table1, 'foo', sum, 'bar')
    expect3 = (('foo', 'value'),
               ('a', 10),
               ('b', 13),
               ('c', 4))
    ieq(expect3, table3)
    ieq(expect3, table3)
    
    # alternative signature for simple aggregation
    table4 = aggregate(table1, key=('foo', 'bar'), aggregation=list, 
                       value=('bar', 'baz'))
    expect4 = (('foo', 'bar', 'value'),
               ('a', 3, [(3, True)]),
               ('a', 7, [(7, False)]),
               ('b', 2, [(2, True), (2, False)]),
               ('b', 9, [(9, False)]),
               ('c', 4, [(4, True)]))
    ieq(expect4, table4)
    ieq(expect4, table4)
    
    table5 = aggregate(table1, 'foo', len, field='nrows')
    expect5 = (('foo', 'nrows'),
               ('a', 2),
               ('b', 3),
               ('c', 1))
    ieq(expect5, table5)
    ieq(expect5, table5)


def test_aggregate_simple_key_is_None():

    table1 = (('foo', 'bar', 'baz'),
              ('a', 3, True),
              ('a', 7, False),
              ('b', 2, True),
              ('b', 2, False),
              ('b', 9, False),
              ('c', 4, True))

    # simplest signature - aggregate whole rows
    table2 = aggregate(table1, None, len)
    expect2 = (('value',), (6,))
    ieq(expect2, table2)
    ieq(expect2, table2)

    # next simplest signature - aggregate single field
    table3 = aggregate(table1, None, sum, 'bar')
    expect3 = (('value',), (27,))
    ieq(expect3, table3)
    ieq(expect3, table3)

    # alternative signature for simple aggregation
    table4 = aggregate(table1, key=None, aggregation=list,
                       value=('bar', 'baz'))
    expect4 = (('value',),
               ( [(3, True), (7, False),
                 (2, True), (2, False), (9, False),
                 (4, True)],),)

    ieq(expect4, table4)
    ieq(expect4, table4)

    table5 = aggregate(table1, None, len, field='nrows')
    expect5 = (('nrows',), (6,))
    ieq(expect5, table5)
    ieq(expect5, table5)

def test_aggregate_multifield():
    
    table1 = (('foo', 'bar'),
              ('a', 3),
              ('a', 7),
              ('b', 2),
              ('b', 1),
              ('b', 9),
              ('c', 4))
    
    # dict arg
    
    aggregators = OrderedDict()
    aggregators['count'] = len
    aggregators['minbar'] = 'bar', min
    aggregators['maxbar'] = 'bar', max
    aggregators['sumbar'] = 'bar', sum
    aggregators['listbar'] = 'bar', list
    aggregators['bars'] = 'bar', strjoin(', ')

    table2 = aggregate(table1, 'foo', aggregators)
    expect2 = (('foo', 'count', 'minbar', 'maxbar', 'sumbar', 'listbar', 
                'bars'),
               ('a', 2, 3, 7, 10, [3, 7], '3, 7'),
               ('b', 3, 1, 9, 12, [2, 1, 9], '2, 1, 9'),
               ('c', 1, 4, 4, 4, [4], '4'))
    ieq(expect2, table2)
    ieq(expect2, table2)  # check can iterate twice
    
    # use suffix notation
    
    table3 = aggregate(table1, 'foo')
    table3['count'] = len
    table3['minbar'] = 'bar', min
    table3['maxbar'] = 'bar', max
    table3['sumbar'] = 'bar', sum
    table3['listbar'] = 'bar'  # default aggregation is list
    table3['bars'] = 'bar', strjoin(', ')
    ieq(expect2, table3)
    
    # list arg

    aggregators = [('count', len),
                   ('minbar', 'bar', min),
                   ('maxbar', 'bar', max),
                   ('sumbar', 'bar', sum),
                   ('listbar', 'bar', list),
                   ('bars', 'bar', strjoin(', '))]

    table4 = aggregate(table1, 'foo', aggregators)
    ieq(expect2, table4)
    ieq(expect2, table4)  # check can iterate twice

def test_aggregate_multifield_key_is_None():

    table1 = (('foo', 'bar'),
              ('a', 3),
              ('a', 7),
              ('b', 2),
              ('b', 1),
              ('b', 9),
              ('c', 4))

    # dict arg
    aggregators = OrderedDict()
    aggregators['count'] = len
    aggregators['minbar'] = 'bar', min
    aggregators['maxbar'] = 'bar', max
    aggregators['sumbar'] = 'bar', sum
    aggregators['listbar'] = 'bar', list
    aggregators['bars'] = 'bar', strjoin(', ')

    table2 = aggregate(table1, None, aggregators)
    expect2 = (('count', 'minbar', 'maxbar', 'sumbar', 'listbar',
                'bars'),
               (6, 1, 9, 26, [3, 7, 2, 1, 9, 4], '3, 7, 2, 1, 9, 4'))
    ieq(expect2, table2)
    ieq(expect2, table2)  # check can iterate twice

    # use suffix notation
    table3 = aggregate(table1, None)
    table3['count'] = len
    table3['minbar'] = 'bar', min
    table3['maxbar'] = 'bar', max
    table3['sumbar'] = 'bar', sum
    table3['listbar'] = 'bar'  # default aggregation is list
    table3['bars'] = 'bar', strjoin(', ')
    ieq(expect2, table3)

    # list arg
    aggregators = [('count', len),
                   ('minbar', 'bar', min),
                   ('maxbar', 'bar', max),
                   ('sumbar', 'bar', sum),
                   ('listbar', 'bar', list),
                   ('bars', 'bar', strjoin(', '))]

    table4 = aggregate(table1, None, aggregators)
    ieq(expect2, table4)
    ieq(expect2, table4)  # check can iterate twice

def test_aggregate_more():

    table1 = (('foo', 'bar'),
              ('aa', 3),
              ('aa', 7),
              ('bb', 2),
              ('bb', 1),
              ('bb', 9),
              ('cc', 4),
              ('dd', 3))

    aggregators = OrderedDict()
    aggregators['minbar'] = 'bar', min
    aggregators['maxbar'] = 'bar', max
    aggregators['sumbar'] = 'bar', sum
    aggregators['listbar'] = 'bar'  # default aggregation is list
    aggregators['bars'] = 'bar', strjoin(', ')

    table2 = aggregate(table1, 'foo', aggregators)
    expect2 = (('foo', 'minbar', 'maxbar', 'sumbar', 'listbar', 'bars'),
               ('aa', 3, 7, 10, [3, 7], '3, 7'),
               ('bb', 1, 9, 12, [2, 1, 9], '2, 1, 9'),
               ('cc', 4, 4, 4, [4], '4'),
               ('dd', 3, 3, 3, [3], '3'))
    ieq(expect2, table2)
    ieq(expect2, table2)  # check can iterate twice

    table3 = aggregate(table1, 'foo')
    table3['minbar'] = 'bar', min
    table3['maxbar'] = 'bar', max
    table3['sumbar'] = 'bar', sum
    table3['listbar'] = 'bar'  # default aggregation is list
    table3['bars'] = 'bar', strjoin(', ')
    ieq(expect2, table3)

def test_aggregate_more_key_is_None():

    table1 = (('foo', 'bar'),
              ('aa', 3),
              ('aa', 7),
              ('bb', 2),
              ('bb', 1),
              ('bb', 9),
              ('cc', 4),
              ('dd', 3))

    aggregators = OrderedDict()
    aggregators['minbar'] = 'bar', min
    aggregators['maxbar'] = 'bar', max
    aggregators['sumbar'] = 'bar', sum
    aggregators['listbar'] = 'bar'  # default aggregation is list
    aggregators['bars'] = 'bar', strjoin(', ')

    table2 = aggregate(table1, None, aggregators)
    expect2 = (('minbar', 'maxbar', 'sumbar', 'listbar', 'bars'),
               (1, 9, 29,
                   [3, 7, 2, 1, 9, 4, 3],
                   '3, 7, 2, 1, 9, 4, 3'))
    ieq(expect2, table2)
    ieq(expect2, table2)  # check can iterate twice

    table3 = aggregate(table1, None)
    table3['minbar'] = 'bar', min
    table3['maxbar'] = 'bar', max
    table3['sumbar'] = 'bar', sum
    table3['listbar'] = 'bar'  # default aggregation is list
    table3['bars'] = 'bar', strjoin(', ')
    ieq(expect2, table3)

def test_aggregate_multiple_source_fields():

    table = (('foo', 'bar', 'baz'),
             ('a', 3, True),
             ('a', 7, False),
             ('b', 2, True),
             ('b', 2, False),
             ('b', 9, False),
             ('c', 4, True))

    expect = (('foo', 'bar', 'value'),
              ('a', 3, [(3, True)]),
              ('a', 7, [(7, False)]),
              ('b', 2, [(2, True), (2, False)]),
              ('b', 9, [(9, False)]),
              ('c', 4, [(4, True)]))

    actual = aggregate(table, ('foo', 'bar'), list, ('bar', 'baz'))
    ieq(expect, actual)
    ieq(expect, actual)

    actual = aggregate(table, key=('foo', 'bar'), aggregation=list, 
                       value=('bar', 'baz'))
    ieq(expect, actual)
    ieq(expect, actual)
    
    actual = aggregate(table, key=('foo', 'bar'))
    actual['value'] = ('bar', 'baz'), list
    ieq(expect, actual)
    ieq(expect, actual)    

def test_aggregate_multiple_source_fields_key_is_None():

    table = (('foo', 'bar', 'baz'),
             ('a', 3, True),
             ('a', 7, False),
             ('b', 2, True),
             ('b', 2, False),
             ('b', 9, False),
             ('c', 4, True))

    expect  = (('value',),
               ( [(3, True), (7, False),
                 (2, True), (2, False), (9, False),
                 (4, True)],),)

    actual = aggregate(table, None, list, ('bar', 'baz'))
    ieq(expect, actual)
    ieq(expect, actual)

    actual = aggregate(table, key=None, aggregation=list,
                       value=('bar', 'baz'))
    ieq(expect, actual)
    ieq(expect, actual)

    actual = aggregate(table, key=None)
    actual['value'] = ('bar', 'baz'), list
    ieq(expect, actual)
    ieq(expect, actual)


def test_aggregate_empty():

    table = (('foo', 'bar'),)

    aggregators = OrderedDict()
    aggregators['minbar'] = 'bar', min
    aggregators['maxbar'] = 'bar', max
    aggregators['sumbar'] = 'bar', sum

    actual = aggregate(table, 'foo', aggregators)
    expect = (('foo', 'minbar', 'maxbar', 'sumbar'),)
    ieq(expect, actual)

def test_aggregate_empty_key_is_None():

    table = (('foo', 'bar'),)

    aggregators = OrderedDict()
    aggregators['minbar'] = 'bar', min
    aggregators['maxbar'] = 'bar', max
    aggregators['sumbar'] = 'bar', sum

    actual = aggregate(table, None, aggregators)
    expect = (('minbar', 'maxbar', 'sumbar'),)
    ieq(expect, actual)

def test_mergeduplicates():

    table = (('foo', 'bar', 'baz'),
             ('A', 1, 2),
             ('B', '2', None),
             ('D', 'xyz', 9.4),
             ('B', None, u'7.8', True),
             ('E', None, 42.),
             ('D', 'xyz', 12.3),
             ('A', 2, None))

    # value overrides missing
    result = mergeduplicates(table, 'foo', missing=None)
    expectation = (('foo', 'bar', 'baz'),
                   ('A', Conflict([1, 2]), 2),
                   ('B', '2', u'7.8'),
                   ('D', 'xyz', Conflict([9.4, 12.3])),
                   ('E', None, 42.))
    ieq(expectation, result)
    
    
def test_mergeduplicates_empty():
    table = (('foo', 'bar'),)
    expect = (('foo', 'bar'),)
    actual = mergeduplicates(table, key='foo')
    ieq(expect, actual)


def test_mergeduplicates_shortrows():
    table = [['foo', 'bar', 'baz'], 
             ['a', 1, True], 
             ['b', 2, True], 
             ['b', 3]]
    actual = mergeduplicates(table, 'foo')
    expect = [('foo', 'bar', 'baz'), 
              ('a', 1, True), 
              ('b', Conflict([2, 3]), True)]
    ieq(expect, actual)
        
    
def test_mergeduplicates_compoundkey():
    table = [['foo', 'bar', 'baz'], 
             ['a', 1, True], 
             ['a', 1, True], 
             ['a', 2, False],
             ['a', 2, None],
             ['c', 3, True],
             ['c', 3, False],
             ]
    actual = mergeduplicates(table, key=('foo', 'bar'))
    expect = [('foo', 'bar', 'baz'), 
              ('a', 1, True), 
              ('a', 2, False), 
              ('c', 3, Conflict([True, False]))]
    ieq(expect, actual)
        

def test_fold():

    t1 = (('id', 'count'), (1, 3), (1, 5), (2, 4), (2, 8))
    t2 = fold(t1, 'id', operator.add, 'count', presorted=True)
    expect = (('key', 'value'), (1, 8), (2, 12))
    ieq(expect, t2)
    ieq(expect, t2)


def _accumulators(events=None, fail_init=False, fail_update=False,
                  fail_finalize=False):
    def init_count():
        if events is not None:
            events.append(('init', 'count'))
        return [0]

    def update_count(state, row):
        if events is not None:
            events.append(('update', 'count', row['foo']))
        if fail_update and row['foo'] == 'b' and state[0] == 1:
            raise ValueError('update failed')
        state[0] += 1

    def finalize_count(state):
        if fail_finalize and state[0] == 2:
            raise ValueError('finalize failed')
        return state[0]

    def init_mean():
        if events is not None:
            events.append(('init', 'mean'))
        return [0, 0]

    def update_mean(state, value):
        if events is not None:
            events.append(('update', 'mean', value))
        state[0] += value
        state[1] += 1

    def finalize_mean(state):
        return state[0] / state[1]

    def init_order():
        if events is not None:
            events.append(('init', 'order'))
        return []

    def update_order(state, row):
        if events is not None:
            events.append(('update', 'order', row['baz']))
        state.append(row['baz'])

    def finalize_order(state):
        return '->'.join(state)

    init_calls = [0]

    def maybe_failing_init_count():
        init_calls[0] += 1
        if fail_init and init_calls[0] == 2:
            raise ValueError('init failed')
        return init_count()

    return OrderedDict((
        ('count', (maybe_failing_init_count, update_count,
                   finalize_count)),
        ('mean', ('bar', init_mean, update_mean, finalize_mean)),
        ('order', (init_order, update_order, finalize_order)),
    ))


class TrackingTable(Table):

    def __init__(self, rows, events):
        self.rows = rows
        self.events = events

    def __iter__(self):
        hdr = self.rows[0]
        self.events.append(('header',))
        yield hdr
        for row in self.rows[1:]:
            self.events.append(('row', row[0], row[2]))
            yield row


def test_aggregateaccumulate():
    table = (('foo', 'bar', 'baz'),
             ('a', 1, 'x'),
             ('a', 2, 'y'),
             ('b', 3, 'y'),
             ('b', 4, 'x'))

    actual = aggregateaccumulate(table, 'foo', _accumulators())
    expect = (('foo', 'count', 'mean', 'order'),
              ('a', 2, 1.5, 'x->y'),
              ('b', 2, 3.5, 'y->x'))
    ieq(expect, actual)
    ieq(expect, actual)


def test_aggregateaccumulate_empty_and_header_only():
    ieq((), aggregateaccumulate((), 'foo', _accumulators()))

    table = (('foo', 'bar', 'baz'),)
    expect = (('foo', 'count', 'mean', 'order'),)
    ieq(expect, aggregateaccumulate(table, 'foo', _accumulators()))


def test_aggregateaccumulate_state_isolation():
    events = []
    table = (('foo', 'bar', 'baz'),
             ('a', 1, 'x'),
             ('b', 2, 'y'),
             ('b', 3, 'z'),
             ('c', 4, 'z'))

    actual = aggregateaccumulate(table, 'foo', _accumulators(events))
    expect = (('foo', 'count', 'mean', 'order'),
              ('a', 1, 1.0, 'x'),
              ('b', 2, 2.5, 'y->z'),
              ('c', 1, 4.0, 'z'))
    ieq(expect, actual)
    assert events.count(('init', 'count')) == 3
    assert events.count(('init', 'mean')) == 3
    assert events.count(('init', 'order')) == 3


def test_aggregateaccumulate_compound_key_and_field_tuple():
    table = (('foo', 'qux', 'bar', 'baz'),
             ('a', 'p', 1, 10),
             ('a', 'p', 2, 20),
             ('b', 'r', 3, 30))
    accumulators = OrderedDict((
        ('count', (lambda: 0,
                   lambda state, row: state + 1,
                   lambda state: state)),
        ('sumbarbaz', (('bar', 'baz'),
                       lambda: 0,
                       lambda state, value: state + value[0] + value[1],
                       lambda state: state)),
    ))

    actual = aggregateaccumulate(table, ('foo', 'qux'), accumulators)
    expect = (('foo', 'qux', 'count', 'sumbarbaz'),
              ('a', 'p', 2, 33),
              ('b', 'r', 1, 33))
    ieq(expect, actual)


def test_aggregateaccumulate_consumes_lazily():
    events = []
    table = TrackingTable((('foo', 'bar', 'baz'),
                          ('a', 1, 'x'),
                          ('a', 2, 'y'),
                          ('b', 3, 'y'),
                          ('b', 4, 'x')), events)

    result = iter(aggregateaccumulate(
        table, 'foo', _accumulators(events)
    ))
    assert next(result) == ('foo', 'count', 'mean', 'order')
    assert events == [('header',)]

    assert next(result) == ('a', 2, 1.5, 'x->y')
    assert events == [
        ('header',),
        ('row', 'a', 'x'),
        ('init', 'count'),
        ('init', 'mean'),
        ('init', 'order'),
        ('update', 'count', 'a'),
        ('update', 'mean', 1),
        ('update', 'order', 'x'),
        ('row', 'a', 'y'),
        ('update', 'count', 'a'),
        ('update', 'mean', 2),
        ('update', 'order', 'y'),
        ('row', 'b', 'y'),
    ]
    assert not any(event[:2] == ('init', 'b') for event in events)

    assert next(result) == ('b', 2, 3.5, 'y->x')


@pytest.mark.parametrize('phase, fail_init, fail_update, fail_finalize',
                         [('initialization', True, False, False),
                          ('update', False, True, False),
                          ('finalization', False, False, True)])
def test_aggregateaccumulate_failures(phase, fail_init, fail_update,
                                      fail_finalize):
    table = (('foo', 'bar', 'baz'),
             ('a', 1, 'x'),
             ('b', 2, 'y'),
             ('b', 3, 'z'))
    result = iter(aggregateaccumulate(
        table, 'foo',
        _accumulators(fail_init=fail_init, fail_update=fail_update,
                      fail_finalize=fail_finalize)
    ))
    assert next(result) == ('foo', 'count', 'mean', 'order')

    delivered = [next(result)]
    with pytest.raises(AggregationError) as excinfo:
        next(result)

    error = excinfo.value
    assert error.key == 'b'
    assert error.metric == 'count'
    assert error.phase == phase
    assert all(row[0] != 'b' for row in delivered)
