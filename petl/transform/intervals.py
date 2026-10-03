from __future__ import absolute_import, print_function, division


from operator import itemgetter, attrgetter

from petl.compat import text_type
from petl.comparison import Comparable, comparable_itemgetter
from petl.util.base import asindices, records, Table, values, rowgroupby
from petl.errors import ArgumentError, DuplicateKeyError
from petl.transform.basics import addfield, cutout, stack
from petl.transform.sorts import sort


def tupletree(table, start='start', stop='stop', value=None):
    """
    Construct an interval tree for the given table, where each node in the tree
    is a row of the table.

    """

    import intervaltree
    tree = intervaltree.IntervalTree()
    it = iter(table)
    hdr = next(it)
    flds = list(map(text_type, hdr))
    assert start in flds, 'start field not recognised'
    assert stop in flds, 'stop field not recognised'
    getstart = itemgetter(flds.index(start))
    getstop = itemgetter(flds.index(stop))
    if value is None:
        getvalue = tuple
    else:
        valueindices = asindices(hdr, value)
        assert len(valueindices) > 0, 'invalid value field specification'
        getvalue = itemgetter(*valueindices)
    for row in it:
        tree.addi(getstart(row), getstop(row), getvalue(row))
    return tree


def facettupletrees(table, key, start='start', stop='stop', value=None):
    """
    Construct faceted interval trees for the given table, where each node in
    the tree is a row of the table.

    """

    import intervaltree
    it = iter(table)
    hdr = next(it)
    flds = list(map(text_type, hdr))
    assert start in flds, 'start field not recognised'
    assert stop in flds, 'stop field not recognised'
    getstart = itemgetter(flds.index(start))
    getstop = itemgetter(flds.index(stop))
    if value is None:
        getvalue = tuple
    else:
        valueindices = asindices(hdr, value)
        assert len(valueindices) > 0, 'invalid value field specification'
        getvalue = itemgetter(*valueindices)
    keyindices = asindices(hdr, key)
    assert len(keyindices) > 0, 'invalid key'
    getkey = itemgetter(*keyindices)

    trees = dict()
    for row in it:
        k = getkey(row)
        if k not in trees:
            trees[k] = intervaltree.IntervalTree()
        trees[k].addi(getstart(row), getstop(row), getvalue(row))
    return trees


def recordtree(table, start='start', stop='stop'):
    """
    Construct an interval tree for the given table, where each node in the
    tree is a row of the table represented as a record object.

    """

    import intervaltree
    getstart = attrgetter(start)
    getstop = attrgetter(stop)
    tree = intervaltree.IntervalTree()
    for rec in records(table):
        tree.addi(getstart(rec), getstop(rec), rec)
    return tree


def facetrecordtrees(table, key, start='start', stop='stop'):
    """
    Construct faceted interval trees for the given table, where each node in 
    the tree is a record.

    """

    import intervaltree
    getstart = attrgetter(start)
    getstop = attrgetter(stop)
    getkey = attrgetter(key)
    trees = dict()
    for rec in records(table):
        k = getkey(rec)
        if k not in trees:
            trees[k] = intervaltree.IntervalTree()
        trees[k].addi(getstart(rec), getstop(rec), rec)
    return trees


def intervallookup(table, start='start', stop='stop', value=None,
                   include_stop=False):
    """
    Construct an interval lookup for the given table. E.g.::

        >>> import petl as etl
        >>> table = [['start', 'stop', 'value'],
        ...          [1, 4, 'foo'],
        ...          [3, 7, 'bar'],
        ...          [4, 9, 'baz']]
        >>> lkp = etl.intervallookup(table, 'start', 'stop')
        >>> lkp.search(0, 1)
        []
        >>> lkp.search(1, 2)
        [(1, 4, 'foo')]
        >>> lkp.search(2, 4)
        [(1, 4, 'foo'), (3, 7, 'bar')]
        >>> lkp.search(2, 5)
        [(1, 4, 'foo'), (3, 7, 'bar'), (4, 9, 'baz')]
        >>> lkp.search(9, 14)
        []
        >>> lkp.search(19, 140)
        []
        >>> lkp.search(0)
        []
        >>> lkp.search(1)
        [(1, 4, 'foo')]
        >>> lkp.search(2)
        [(1, 4, 'foo')]
        >>> lkp.search(4)
        [(3, 7, 'bar'), (4, 9, 'baz')]
        >>> lkp.search(5)
        [(3, 7, 'bar'), (4, 9, 'baz')]

    Note start coordinates are included and stop coordinates are excluded
    from the interval. Use the `include_stop` keyword argument to include the
    upper bound of the interval when finding overlaps.

    Some examples using the `include_stop` and `value` keyword arguments::
    
        >>> import petl as etl
        >>> table = [['start', 'stop', 'value'],
        ...          [1, 4, 'foo'],
        ...          [3, 7, 'bar'],
        ...          [4, 9, 'baz']]
        >>> lkp = etl.intervallookup(table, 'start', 'stop', include_stop=True,
        ...                          value='value')
        >>> lkp.search(0, 1)
        ['foo']
        >>> lkp.search(1, 2)
        ['foo']
        >>> lkp.search(2, 4)
        ['foo', 'bar', 'baz']
        >>> lkp.search(2, 5)
        ['foo', 'bar', 'baz']
        >>> lkp.search(9, 14)
        ['baz']
        >>> lkp.search(19, 140)
        []
        >>> lkp.search(0)
        []
        >>> lkp.search(1)
        ['foo']
        >>> lkp.search(2)
        ['foo']
        >>> lkp.search(4)
        ['foo', 'bar', 'baz']
        >>> lkp.search(5)
        ['bar', 'baz']

    """

    tree = tupletree(table, start=start, stop=stop, value=value)
    return IntervalTreeLookup(tree, include_stop=include_stop)


Table.intervallookup = intervallookup


def _search_tree(tree, start, stop, include_stop):
    if stop is None:
        if include_stop:
            stop = start + 1
            start -= 1
            args = (start, stop)
        else:
            args = (start,)
    else:
        if include_stop:
            stop += 1
            start -= 1
        args = (start, stop)
    if len(args) == 2:
        results = sorted(tree.overlap(*args))
    else:
        results = sorted(tree.at(*args))
    return results


class IntervalTreeLookup(object):
    
    def __init__(self, tree, include_stop=False):
        self.tree = tree
        self.include_stop = include_stop

    def search(self, start, stop=None):
        results = _search_tree(self.tree, start, stop, self.include_stop)
        return [r.data for r in results]

    find = search


def intervallookupone(table, start='start', stop='stop', value=None,
                      include_stop=False, strict=True):
    """
    Construct an interval lookup for the given table, returning at most one
    result for each query. E.g.::

        >>> import petl as etl
        >>> table = [['start', 'stop', 'value'],
        ...          [1, 4, 'foo'],
        ...          [3, 7, 'bar'],
        ...          [4, 9, 'baz']]
        >>> lkp = etl.intervallookupone(table, 'start', 'stop', strict=False)
        >>> lkp.search(0, 1)
        >>> lkp.search(1, 2)
        (1, 4, 'foo')
        >>> lkp.search(2, 4)
        (1, 4, 'foo')
        >>> lkp.search(2, 5)
        (1, 4, 'foo')
        >>> lkp.search(9, 14)
        >>> lkp.search(19, 140)
        >>> lkp.search(0)
        >>> lkp.search(1)
        (1, 4, 'foo')
        >>> lkp.search(2)
        (1, 4, 'foo')
        >>> lkp.search(4)
        (3, 7, 'bar')
        >>> lkp.search(5)
        (3, 7, 'bar')

    If ``strict=True``, queries returning more than one result will
    raise a `DuplicateKeyError`. If ``strict=False`` and there is
    more than one result, the first result is returned.

    Note start coordinates are included and stop coordinates are excluded
    from the interval. Use the `include_stop` keyword argument to include the
    upper bound of the interval when finding overlaps.

    """

    tree = tupletree(table, start=start, stop=stop, value=value)
    return IntervalTreeLookupOne(tree, strict=strict, include_stop=include_stop)


Table.intervallookupone = intervallookupone


class IntervalTreeLookupOne(object):
    
    def __init__(self, tree, strict=True, include_stop=False):
        self.tree = tree
        self.strict = strict
        self.include_stop = include_stop
        
    def search(self, start, stop=None):
        results = _search_tree(self.tree, start, stop, self.include_stop)
        if len(results) == 0:
            return None
        elif len(results) > 1 and self.strict:
            raise DuplicateKeyError((start, stop))
        else:
            return results[0].data

    find = search


def intervalrecordlookup(table, start='start', stop='stop', include_stop=False):
    """
    As :func:`petl.transform.intervals.intervallookup` but return records
    instead of tuples.

    """

    tree = recordtree(table, start=start, stop=stop)
    return IntervalTreeLookup(tree, include_stop=include_stop)


Table.intervalrecordlookup = intervalrecordlookup


def intervalrecordlookupone(table, start='start', stop='stop',
                            include_stop=False, strict=True):
    """
    As :func:`petl.transform.intervals.intervallookupone` but return records
    instead of tuples.

    """

    tree = recordtree(table, start=start, stop=stop)
    return IntervalTreeLookupOne(tree, include_stop=include_stop, strict=strict)



Table.intervalrecordlookupone = intervalrecordlookupone


_VALIDITY_ORDER_FIELD = '__petl_validityjoin_order__'
_VALIDITY_RULE_ORDER_FIELD = '__petl_validityjoin_rule_order__'
_VALIDITY_INVALID_POLICIES = ('raise', 'ignore')


def validityjoin(left, right, time='time', key=None, lkey=None, rkey=None,
                 rstart='start', rend='end', missing=None, presorted=False,
                 buffersize=None, tempdir=None, cache=True,
                 invalid_time='raise', invalid_interval='raise',
                 lprefix=None, rprefix=None):
    """
    Join point-in-time events to rules whose validity intervals contain them.
    E.g.::

        >>> import petl as etl
        >>> events = [['customer', 'event_time', 'name'],
        ...           ['b', 1, 'no-rule'],
        ...           ['a', 2, 'nested'],
        ...           ['a', 4, 'boundary'],
        ...           ['a', 5, 'open']]
        >>> rules = [['customer', 'rule', 'start', 'end'],
        ...          ['a', 'v1', 1, 4],
        ...          ['a', 'v2', 2, 3],
        ...          ['a', 'current', 4, None]]
        >>> etl.validityjoin(events, rules, key='customer',
        ...                  time='event_time').lookall()
        +----------+------------+------------+-----------+-------+------+
        | customer | event_time | name       | rule      | start | end  |
        +==========+============+============+===========+=======+======+
        | 'b'      |          1 | 'no-rule'  | None      | None  | None |
        +----------+------------+------------+-----------+-------+------+
        | 'a'      |          2 | 'nested'   | 'v1'      |     1 |    4 |
        +----------+------------+------------+-----------+-------+------+
        | 'a'      |          2 | 'nested'   | 'v2'      |     2 |    3 |
        +----------+------------+------------+-----------+-------+------+
        | 'a'      |          4 | 'boundary' | 'current' |     4 | None |
        +----------+------------+------------+-----------+-------+------+
        | 'a'      |          5 | 'open'     | 'current' |     4 | None |
        +----------+------------+------------+-----------+-------+------+
        <BLANKLINE>


    Rule intervals are half-open: ``start <= event time < end``. A rule with
    a missing `end` value remains valid indefinitely. All overlapping rules
    for the same business key are returned, while events with no matching rule
    are retained with missing values in the right-hand fields.

    The compound business key may be specified once with `key`, or separately
    for each table with `lkey` and `rkey`. By default, output rows preserve
    the original order of `left`; multiple rules for one event preserve the
    order in `right`.

    If `presorted` is True, `left` is assumed to be sorted by its business key
    followed by `time`, and `right` is assumed to be sorted by its business
    key followed by `rstart`. The join then scans both tables once rather
    than searching the rule table for each event.

    Set `invalid_interval` to ``'ignore'`` to skip rules with missing starts
    or reversed intervals. Set `invalid_time` to ``'ignore'`` to retain events
    with missing times as unmatched rows. The default ``'raise'`` rejects
    these rows with an `ArgumentError`.

    Right-hand key fields are not duplicated in the output, matching the
    behaviour of equi-joins. Use `lprefix` and `rprefix` to make other
    conflicting field names unambiguous.

    """

    if invalid_time not in _VALIDITY_INVALID_POLICIES:
        raise ArgumentError(
            "invalid_time must be one of 'raise' or 'ignore'")
    if invalid_interval not in _VALIDITY_INVALID_POLICIES:
        raise ArgumentError(
            "invalid_interval must be one of 'raise' or 'ignore'")
    if key is None:
        if lkey is None or rkey is None:
            raise ArgumentError(
                'either key or both lkey and rkey must be provided')
    else:
        if lkey is not None or rkey is not None:
            raise ArgumentError(
                'specify either key or both lkey and rkey, but not both')
        lkey = rkey = key

    return ValidityJoinView(
        left, right, time=time, lkey=lkey, rkey=rkey, rstart=rstart,
        rend=rend, missing=missing, presorted=presorted,
        buffersize=buffersize, tempdir=tempdir, cache=cache,
        invalid_time=invalid_time, invalid_interval=invalid_interval,
        lprefix=lprefix, rprefix=rprefix)


Table.validityjoin = validityjoin


class _OrderedRows(Table):

    def __init__(self, source, field):
        self.source = stack(source)
        self.field = field

    def __iter__(self):
        it = iter(self.source)
        hdr = next(it)
        yield tuple(hdr) + (self.field,)
        for order, row in enumerate(it):
            yield tuple(row) + (order,)


class ValidityJoinView(Table):

    def __init__(self, left, right, time, lkey, rkey, rstart, rend, missing,
                 presorted, buffersize, tempdir, cache, invalid_time,
                 invalid_interval, lprefix, rprefix):
        self.time = time
        self.lkey = lkey
        self.rkey = rkey
        self.rstart = rstart
        self.rend = rend
        self.missing = missing
        self.presorted = presorted
        self.buffersize = buffersize
        self.tempdir = tempdir
        self.sort_cache = cache
        self.invalid_time = invalid_time
        self.invalid_interval = invalid_interval
        self.lprefix = lprefix
        self.rprefix = rprefix

        left = stack(left, missing=missing)
        right = stack(right, missing=missing)
        if presorted:
            self.left = left
            self.right = right
        else:
            lsortkey = _validity_field_tuple(lkey) + (time,)
            rsortkey = _validity_field_tuple(rkey) + (rstart,)
            ordered_left = _OrderedRows(left, _VALIDITY_ORDER_FIELD)
            ordered_right = _OrderedRows(right, _VALIDITY_RULE_ORDER_FIELD)
            self.left = sort(ordered_left, lsortkey,
                             buffersize=buffersize, tempdir=tempdir,
                             cache=cache)
            self.right = sort(ordered_right, rsortkey,
                              buffersize=buffersize, tempdir=tempdir,
                              cache=cache)

    def __iter__(self):
        rows = itervalidityjoin(
            self.left, self.right, time=self.time, lkey=self.lkey,
            rkey=self.rkey, rstart=self.rstart, rend=self.rend,
            missing=self.missing, invalid_time=self.invalid_time,
            invalid_interval=self.invalid_interval, lprefix=self.lprefix,
            rprefix=self.rprefix,
            rorder=None if self.presorted else _VALIDITY_RULE_ORDER_FIELD)
        if self.presorted:
            return rows
        rows = sort(rows, _VALIDITY_ORDER_FIELD, buffersize=self.buffersize,
                    tempdir=self.tempdir, cache=self.sort_cache)
        return iter(cutout(rows, _VALIDITY_ORDER_FIELD))


def _validity_field_tuple(key):
    if isinstance(key, (list, tuple)):
        return tuple(key)
    return (key,)


def _validity_output_header(lhdr, rhdr, rkind, lprefix, rprefix,
                            rorder=None):
    excluded = set(rkind)
    if rorder is not None:
        excluded.update(asindices(rhdr, rorder))
    rvind = [i for i in range(len(rhdr)) if i not in excluded]
    if lprefix is None:
        outhdr = list(lhdr)
    else:
        outhdr = [f if f == _VALIDITY_ORDER_FIELD
                  else text_type(lprefix) + text_type(f) for f in lhdr]
    if rprefix is None:
        outhdr.extend(rhdr[i] for i in rvind)
    else:
        outhdr.extend(text_type(rprefix) + text_type(rhdr[i]) for i in rvind)
    return tuple(outhdr), rvind


def _validity_emit(lrow, rrow, rvind, missing):
    outrow = list(lrow)
    if rrow is None:
        outrow.extend(missing for _ in rvind)
    else:
        outrow.extend(rrow[i] if i < len(rrow) else None for i in rvind)
    return tuple(outrow)


def _ensure_valid_rule(rule_start, rule_end, row_number, policy):
    if rule_start is None:
        if policy == 'raise':
            raise ArgumentError(
                'rule row %d has missing start value' % row_number)
        return False
    if rule_end is not None and Comparable(rule_end) < Comparable(rule_start):
        if policy == 'raise':
            raise ArgumentError(
                'rule row %d has end before start: end %r must not be '
                'before start %r' % (row_number, rule_end, rule_start))
        return False
    return True


def itervalidityjoin(left, right, time, lkey, rkey, rstart, rend, missing,
                     invalid_time, invalid_interval, lprefix, rprefix,
                     rorder=None):
    lit = iter(left)
    rit = iter(right)
    lhdr = next(lit)
    rhdr = next(rit)

    lkind = tuple(asindices(lhdr, lkey))
    rkind = tuple(asindices(rhdr, rkey))
    ltimeind = asindices(lhdr, time)
    rstartind = asindices(rhdr, rstart)
    rendind = asindices(rhdr, rend)

    outhdr, rvind = _validity_output_header(
        lhdr, rhdr, rkind, lprefix, rprefix, rorder=rorder)
    yield outhdr

    get_lkey = comparable_itemgetter(*lkind)
    get_rkey = comparable_itemgetter(*rkind)
    get_ltime = comparable_itemgetter(*ltimeind)
    get_rstart = comparable_itemgetter(*rstartind)
    get_rend = comparable_itemgetter(*rendind)
    get_rorder = None if rorder is None else itemgetter(
        *asindices(rhdr, rorder))

    def emit(lrow, entry):
        rrow = None if entry is None else entry[0]
        return _validity_emit(lrow, rrow, rvind, missing)

    state = {
        'event_number': 0,
        'rule_number': 0,
        'pending_rule': None,
        'pending_key': Comparable(None),
        'active_key': Comparable(None),
        'active_entries': [],
        'rules_exhausted': False,
    }

    def advance_rule():
        while not state['rules_exhausted']:
            try:
                candidate = next(rit)
            except StopIteration:
                state['rules_exhausted'] = True
                state['pending_rule'] = None
                return
            state['rule_number'] += 1
            candidate_start = get_rstart(candidate).inner
            candidate_end = get_rend(candidate).inner
            reported_number = state['rule_number']
            if get_rorder is not None:
                reported_number = get_rorder(candidate) + 1
            if _ensure_valid_rule(candidate_start, candidate_end,
                                  reported_number, invalid_interval):
                order = None if get_rorder is None else get_rorder(candidate)
                state['pending_rule'] = (candidate, order)
                state['pending_key'] = get_rkey(candidate)
                return

    for lrow in lit:
        state['event_number'] += 1
        event_time_value = get_ltime(lrow).inner
        if event_time_value is None:
            if invalid_time == 'raise':
                raise ArgumentError(
                    'event row %d has missing time value'
                    % state['event_number'])
            yield emit(lrow, None)
            continue

        event_key = get_lkey(lrow)
        event_time = Comparable(event_time_value)

        if state['active_key'] != event_key:
            state['active_key'] = event_key
            state['active_entries'] = []

        while not state['rules_exhausted']:
            if state['pending_rule'] is None:
                advance_rule()
                continue
            if state['pending_key'] >= event_key:
                break
            advance_rule()

        if not state['rules_exhausted'] and state['pending_rule'] is not None \
                and state['pending_key'] == event_key:
            while state['pending_rule'] is not None and \
                    state['pending_key'] == event_key and \
                    get_rstart(state['pending_rule'][0]) <= event_time:
                state['active_entries'].append(state['pending_rule'])
                advance_rule()

        surviving_entries = []
        for entry in state['active_entries']:
            rrow = entry[0]
            rule_end = get_rend(rrow)
            if rule_end.inner is None or event_time < rule_end:
                surviving_entries.append(entry)
        state['active_entries'] = surviving_entries

        matched_entries = surviving_entries
        if get_rorder is not None:
            matched_entries = sorted(matched_entries,
                                     key=lambda entry: entry[1])
        if matched_entries:
            for entry in matched_entries:
                yield emit(lrow, entry)
        else:
            yield emit(lrow, None)


def facetintervallookup(table, key, start='start', stop='stop',
                        value=None, include_stop=False):
    """

    Construct a faceted interval lookup for the given table. E.g.::

        >>> import petl as etl
        >>> table = (('type', 'start', 'stop', 'value'),
        ...          ('apple', 1, 4, 'foo'),
        ...          ('apple', 3, 7, 'bar'),
        ...          ('orange', 4, 9, 'baz'))
        >>> lkp = etl.facetintervallookup(table, key='type', start='start', stop='stop')
        >>> lkp['apple'].search(1, 2)
        [('apple', 1, 4, 'foo')]
        >>> lkp['apple'].search(2, 4)
        [('apple', 1, 4, 'foo'), ('apple', 3, 7, 'bar')]
        >>> lkp['apple'].search(2, 5)
        [('apple', 1, 4, 'foo'), ('apple', 3, 7, 'bar')]
        >>> lkp['orange'].search(2, 5)
        [('orange', 4, 9, 'baz')]
        >>> lkp['orange'].search(9, 14)
        []
        >>> lkp['orange'].search(19, 140)
        []
        >>> lkp['apple'].search(1)
        [('apple', 1, 4, 'foo')]
        >>> lkp['apple'].search(2)
        [('apple', 1, 4, 'foo')]
        >>> lkp['apple'].search(4)
        [('apple', 3, 7, 'bar')]
        >>> lkp['apple'].search(5)
        [('apple', 3, 7, 'bar')]
        >>> lkp['orange'].search(5)
        [('orange', 4, 9, 'baz')]

    """

    trees = facettupletrees(table, key, start=start, stop=stop, value=value)
    out = dict()
    for k in trees:
        out[k] = IntervalTreeLookup(trees[k], include_stop=include_stop)
    return out


Table.facetintervallookup = facetintervallookup


def facetintervallookupone(table, key, start='start', stop='stop',
                           value=None, include_stop=False, strict=True):
    """
    Construct a faceted interval lookup for the given table, returning at most
    one result for each query.
    
    If ``strict=True``, queries returning more than one result will
    raise a `DuplicateKeyError`. If ``strict=False`` and there is
    more than one result, the first result is returned.

    """
    
    trees = facettupletrees(table, key, start=start, stop=stop, value=value)
    out = dict()
    for k in trees:
        out[k] = IntervalTreeLookupOne(trees[k], include_stop=include_stop,
                                       strict=strict)
    return out


Table.facetintervallookupone = facetintervallookupone


def facetintervalrecordlookup(table, key, start='start', stop='stop',
                              include_stop=False):
    """
    As :func:`petl.transform.intervals.facetintervallookup` but return records.
    
    """

    trees = facetrecordtrees(table, key, start=start, stop=stop)
    out = dict()
    for k in trees:
        out[k] = IntervalTreeLookup(trees[k], include_stop=include_stop)
    return out


Table.facetintervalrecordlookup = facetintervalrecordlookup


def facetintervalrecordlookupone(table, key, start, stop, include_stop=False,
                                 strict=True):
    """
    As :func:`petl.transform.intervals.facetintervallookupone` but return
    records.

    """
    
    trees = facetrecordtrees(table, key, start=start, stop=stop)
    out = dict()
    for k in trees:
        out[k] = IntervalTreeLookupOne(trees[k], include_stop=include_stop,
                                       strict=strict)
    return out


Table.facetintervalrecordlookupone = facetintervalrecordlookupone


def intervaljoin(left, right, lstart='start', lstop='stop', rstart='start',
                 rstop='stop', lkey=None, rkey=None, include_stop=False,
                 lprefix=None, rprefix=None):
    """
    Join two tables by overlapping intervals. E.g.::

        >>> import petl as etl
        >>> left = [['begin', 'end', 'quux'],
        ...         [1, 2, 'a'],
        ...         [2, 4, 'b'],
        ...         [2, 5, 'c'],
        ...         [9, 14, 'd'],
        ...         [1, 1, 'e'],
        ...         [10, 10, 'f']]
        >>> right = [['start', 'stop', 'value'],
        ...          [1, 4, 'foo'],
        ...          [3, 7, 'bar'],
        ...          [4, 9, 'baz']]
        >>> table1 = etl.intervaljoin(left, right,
        ...                           lstart='begin', lstop='end',
        ...                           rstart='start', rstop='stop')
        >>> table1.lookall()
        +-------+-----+------+-------+------+-------+
        | begin | end | quux | start | stop | value |
        +=======+=====+======+=======+======+=======+
        |     1 |   2 | 'a'  |     1 |    4 | 'foo' |
        +-------+-----+------+-------+------+-------+
        |     2 |   4 | 'b'  |     1 |    4 | 'foo' |
        +-------+-----+------+-------+------+-------+
        |     2 |   4 | 'b'  |     3 |    7 | 'bar' |
        +-------+-----+------+-------+------+-------+
        |     2 |   5 | 'c'  |     1 |    4 | 'foo' |
        +-------+-----+------+-------+------+-------+
        |     2 |   5 | 'c'  |     3 |    7 | 'bar' |
        +-------+-----+------+-------+------+-------+
        |     2 |   5 | 'c'  |     4 |    9 | 'baz' |
        +-------+-----+------+-------+------+-------+

        >>> # include stop coordinate in intervals
        ... table2 = etl.intervaljoin(left, right,
        ...                           lstart='begin', lstop='end',
        ...                           rstart='start', rstop='stop',
        ...                           include_stop=True)
        >>> table2.lookall()
        +-------+-----+------+-------+------+-------+
        | begin | end | quux | start | stop | value |
        +=======+=====+======+=======+======+=======+
        |     1 |   2 | 'a'  |     1 |    4 | 'foo' |
        +-------+-----+------+-------+------+-------+
        |     2 |   4 | 'b'  |     1 |    4 | 'foo' |
        +-------+-----+------+-------+------+-------+
        |     2 |   4 | 'b'  |     3 |    7 | 'bar' |
        +-------+-----+------+-------+------+-------+
        |     2 |   4 | 'b'  |     4 |    9 | 'baz' |
        +-------+-----+------+-------+------+-------+
        |     2 |   5 | 'c'  |     1 |    4 | 'foo' |
        +-------+-----+------+-------+------+-------+
        |     2 |   5 | 'c'  |     3 |    7 | 'bar' |
        +-------+-----+------+-------+------+-------+
        |     2 |   5 | 'c'  |     4 |    9 | 'baz' |
        +-------+-----+------+-------+------+-------+
        |     9 |  14 | 'd'  |     4 |    9 | 'baz' |
        +-------+-----+------+-------+------+-------+
        |     1 |   1 | 'e'  |     1 |    4 | 'foo' |
        +-------+-----+------+-------+------+-------+

    Note start coordinates are included and stop coordinates are excluded
    from the interval. Use the `include_stop` keyword argument to include the
    upper bound of the interval when finding overlaps.

    An additional key comparison can be made, e.g.::
    
        >>> import petl as etl
        >>> left = (('fruit', 'begin', 'end'),
        ...         ('apple', 1, 2),
        ...         ('apple', 2, 4),
        ...         ('apple', 2, 5),
        ...         ('orange', 2, 5),
        ...         ('orange', 9, 14),
        ...         ('orange', 19, 140),
        ...         ('apple', 1, 1))
        >>> right = (('type', 'start', 'stop', 'value'),
        ...          ('apple', 1, 4, 'foo'),
        ...          ('apple', 3, 7, 'bar'),
        ...          ('orange', 4, 9, 'baz'))
        >>> table3 = etl.intervaljoin(left, right,
        ...                           lstart='begin', lstop='end', lkey='fruit',
        ...                           rstart='start', rstop='stop', rkey='type')
        >>> table3.lookall()
        +----------+-------+-----+----------+-------+------+-------+
        | fruit    | begin | end | type     | start | stop | value |
        +==========+=======+=====+==========+=======+======+=======+
        | 'apple'  |     1 |   2 | 'apple'  |     1 |    4 | 'foo' |
        +----------+-------+-----+----------+-------+------+-------+
        | 'apple'  |     2 |   4 | 'apple'  |     1 |    4 | 'foo' |
        +----------+-------+-----+----------+-------+------+-------+
        | 'apple'  |     2 |   4 | 'apple'  |     3 |    7 | 'bar' |
        +----------+-------+-----+----------+-------+------+-------+
        | 'apple'  |     2 |   5 | 'apple'  |     1 |    4 | 'foo' |
        +----------+-------+-----+----------+-------+------+-------+
        | 'apple'  |     2 |   5 | 'apple'  |     3 |    7 | 'bar' |
        +----------+-------+-----+----------+-------+------+-------+
        | 'orange' |     2 |   5 | 'orange' |     4 |    9 | 'baz' |
        +----------+-------+-----+----------+-------+------+-------+

    """
    
    assert (lkey is None) == (rkey is None), \
        'facet key field must be provided for both or neither table'
    return IntervalJoinView(left, right, lstart=lstart, lstop=lstop,
                            rstart=rstart, rstop=rstop, lkey=lkey,
                            rkey=rkey, include_stop=include_stop,
                            lprefix=lprefix, rprefix=rprefix)


Table.intervaljoin = intervaljoin


class IntervalJoinView(Table):
    
    def __init__(self, left, right, lstart='start', lstop='stop', 
                 rstart='start', rstop='stop', lkey=None, rkey=None,
                 include_stop=False, lprefix=None, rprefix=None):
        self.left = left
        self.lstart = lstart
        self.lstop = lstop
        self.lkey = lkey
        self.right = right
        self.rstart = rstart
        self.rstop = rstop
        self.rkey = rkey
        self.include_stop = include_stop
        self.lprefix = lprefix
        self.rprefix = rprefix

    def __iter__(self):
        return iterintervaljoin(
            left=self.left,
            right=self.right,
            lstart=self.lstart,
            lstop=self.lstop,
            rstart=self.rstart,
            rstop=self.rstop,
            lkey=self.lkey,
            rkey=self.rkey,
            include_stop=self.include_stop,
            missing=None,
            lprefix=self.lprefix,
            rprefix=self.rprefix,
            leftouter=False
        )
        

def intervalleftjoin(left, right, lstart='start', lstop='stop', rstart='start',
                     rstop='stop', lkey=None, rkey=None, include_stop=False,
                     missing=None, lprefix=None, rprefix=None):
    """
    Like :func:`petl.transform.intervals.intervaljoin` but rows from the left 
    table without a match in the right table are also included. E.g.::

        >>> import petl as etl
        >>> left = [['begin', 'end', 'quux'],
        ...         [1, 2, 'a'],
        ...         [2, 4, 'b'],
        ...         [2, 5, 'c'],
        ...         [9, 14, 'd'],
        ...         [1, 1, 'e'],
        ...         [10, 10, 'f']]
        >>> right = [['start', 'stop', 'value'],
        ...          [1, 4, 'foo'],
        ...          [3, 7, 'bar'],
        ...          [4, 9, 'baz']]
        >>> table1 = etl.intervalleftjoin(left, right,
        ...                               lstart='begin', lstop='end',
        ...                               rstart='start', rstop='stop')
        >>> table1.lookall()
        +-------+-----+------+-------+------+-------+
        | begin | end | quux | start | stop | value |
        +=======+=====+======+=======+======+=======+
        |     1 |   2 | 'a'  |     1 |    4 | 'foo' |
        +-------+-----+------+-------+------+-------+
        |     2 |   4 | 'b'  |     1 |    4 | 'foo' |
        +-------+-----+------+-------+------+-------+
        |     2 |   4 | 'b'  |     3 |    7 | 'bar' |
        +-------+-----+------+-------+------+-------+
        |     2 |   5 | 'c'  |     1 |    4 | 'foo' |
        +-------+-----+------+-------+------+-------+
        |     2 |   5 | 'c'  |     3 |    7 | 'bar' |
        +-------+-----+------+-------+------+-------+
        |     2 |   5 | 'c'  |     4 |    9 | 'baz' |
        +-------+-----+------+-------+------+-------+
        |     9 |  14 | 'd'  | None  | None | None  |
        +-------+-----+------+-------+------+-------+
        |     1 |   1 | 'e'  | None  | None | None  |
        +-------+-----+------+-------+------+-------+
        |    10 |  10 | 'f'  | None  | None | None  |
        +-------+-----+------+-------+------+-------+

    Note start coordinates are included and stop coordinates are excluded
    from the interval. Use the `include_stop` keyword argument to include the
    upper bound of the interval when finding overlaps.

    """
    
    assert (lkey is None) == (rkey is None), \
        'facet key field must be provided for both or neither table'
    return IntervalLeftJoinView(left, right, lstart=lstart, lstop=lstop,
                                rstart=rstart, rstop=rstop, lkey=lkey,
                                rkey=rkey, include_stop=include_stop,
                                missing=missing, lprefix=lprefix,
                                rprefix=rprefix)


Table.intervalleftjoin = intervalleftjoin


class IntervalLeftJoinView(Table):
    
    def __init__(self, left, right, lstart='start', lstop='stop', 
                 rstart='start', rstop='stop', lkey=None, rkey=None,
                 missing=None, include_stop=False, lprefix=None, rprefix=None):
        self.left = left
        self.lstart = lstart
        self.lstop = lstop
        self.lkey = lkey
        self.right = right
        self.rstart = rstart
        self.rstop = rstop
        self.rkey = rkey
        self.missing = missing
        self.include_stop = include_stop
        self.lprefix = lprefix
        self.rprefix = rprefix

    def __iter__(self):
        return iterintervaljoin(
            left=self.left,
            right=self.right,
            lstart=self.lstart,
            lstop=self.lstop,
            rstart=self.rstart,
            rstop=self.rstop,
            lkey=self.lkey,
            rkey=self.rkey,
            include_stop=self.include_stop,
            missing=self.missing,
            lprefix=self.lprefix,
            rprefix=self.rprefix,
            leftouter=True
        )
        

def intervalantijoin(left, right, lstart='start', lstop='stop', rstart='start',
                     rstop='stop', lkey=None, rkey=None, include_stop=False,
                     missing=None):
    """
    Return rows from the `left` table with no overlapping rows from the `right`
    table.

    Note start coordinates are included and stop coordinates are excluded
    from the interval. Use the `include_stop` keyword argument to include the
    upper bound of the interval when finding overlaps.

    """

    assert (lkey is None) == (rkey is None), \
        'facet key field must be provided for both or neither table'
    return IntervalAntiJoinView(left, right, lstart=lstart, lstop=lstop,
                                rstart=rstart, rstop=rstop, lkey=lkey,
                                rkey=rkey, include_stop=include_stop,
                                missing=missing)


Table.intervalantijoin = intervalantijoin


class IntervalAntiJoinView(Table):

    def __init__(self, left, right, lstart='start', lstop='stop',
                 rstart='start', rstop='stop', lkey=None, rkey=None,
                 missing=None, include_stop=False):
        self.left = left
        self.lstart = lstart
        self.lstop = lstop
        self.lkey = lkey
        self.right = right
        self.rstart = rstart
        self.rstop = rstop
        self.rkey = rkey
        self.missing = missing
        self.include_stop = include_stop

    def __iter__(self):
        return iterintervaljoin(
            left=self.left,
            right=self.right,
            lstart=self.lstart,
            lstop=self.lstop,
            rstart=self.rstart,
            rstop=self.rstop,
            lkey=self.lkey,
            rkey=self.rkey,
            include_stop=self.include_stop,
            missing=self.missing,
            lprefix=None,
            rprefix=None,
            leftouter=True,
            anti=True
        )


def iterintervaljoin(left, right, lstart, lstop, rstart, rstop, lkey,
                     rkey, include_stop, missing, lprefix, rprefix, leftouter,
                     anti=False):

    # create iterators and obtain fields
    lit = iter(left)
    lhdr = next(lit)
    lflds = list(map(text_type, lhdr))
    rit = iter(right)
    rhdr = next(rit)
    rflds = list(map(text_type, rhdr))

    # check fields via petl.util.asindices (raises FieldSelectionError if spec
    # is not valid)
    asindices(lhdr, lstart)
    asindices(lhdr, lstop)
    if lkey is not None:
        asindices(lhdr, lkey)
    asindices(rhdr, rstart)
    asindices(rhdr, rstop)
    if rkey is not None:
        asindices(rhdr, rkey)

    # determine output fields
    if lprefix is None:
        outhdr = list(lflds)
        if not anti:
            outhdr.extend(rflds)
    else:
        outhdr = list(lprefix + f for f in lflds)
        if not anti:
            outhdr.extend(rprefix + f for f in rflds)
    yield tuple(outhdr)
    
    # create getters for start and stop positions
    getlstart = itemgetter(lflds.index(lstart))
    getlstop = itemgetter(lflds.index(lstop))

    if rkey is None:
        # build interval lookup for right table
        lookup = intervallookup(right, rstart, rstop, include_stop=include_stop)
        search = lookup.search
        # main loop
        for lrow in lit:
            start = getlstart(lrow)
            stop = getlstop(lrow)
            rrows = search(start, stop)
            if rrows:
                if not anti:
                    for rrow in rrows:
                        outrow = list(lrow)
                        outrow.extend(rrow)
                        yield tuple(outrow)
            elif leftouter:
                outrow = list(lrow)
                if not anti:
                    outrow.extend([missing] * len(rflds))
                yield tuple(outrow)

    else:
        # build interval lookup for right table
        lookup = facetintervallookup(right, key=rkey, start=rstart,
                                     stop=rstop, include_stop=include_stop)
        search = dict()
        for f in lookup:
            search[f] = lookup[f].search
        # getter for facet key values in left table
        getlkey = itemgetter(*asindices(lflds, lkey))
        # main loop
        for lrow in lit:
            lkey = getlkey(lrow)
            start = getlstart(lrow)
            stop = getlstop(lrow)
            
            try:
                rrows = search[lkey](start, stop)
            except KeyError:
                rrows = None
            except AttributeError:
                rrows = None
                
            if rrows:
                if not anti:
                    for rrow in rrows:
                        outrow = list(lrow)
                        outrow.extend(rrow)
                        yield tuple(outrow)
            elif leftouter:
                outrow = list(lrow)
                if not anti:
                    outrow.extend([missing] * len(rflds))
                yield tuple(outrow)


def intervaljoinvalues(left, right, value, lstart='start', lstop='stop',
                       rstart='start', rstop='stop', lkey=None, rkey=None,
                       include_stop=False):
    """
    Convenience function to join the left table with values from a specific 
    field in the right hand table.
    
    Note start coordinates are included and stop coordinates are excluded
    from the interval. Use the `include_stop` keyword argument to include the
    upper bound of the interval when finding overlaps.

    """
    
    assert (lkey is None) == (rkey is None), \
        'facet key field must be provided for both or neither table'
    if lkey is None:
        lkp = intervallookup(right, start=rstart, stop=rstop, value=value,
                             include_stop=include_stop)
        f = lambda row: lkp.search(row[lstart], row[lstop])
    else:
        lkp = facetintervallookup(right, rkey, start=rstart, stop=rstop,
                                  value=value, include_stop=include_stop)
        f = lambda row: lkp[row[lkey]].search(row[lstart], row[lstop])
    return addfield(left, value, f)
        
        
Table.intervaljoinvalues = intervaljoinvalues


def intervalsubtract(left, right, lstart='start', lstop='stop', rstart='start',
                     rstop='stop', lkey=None, rkey=None, include_stop=False):
    """
    Subtract intervals in the right hand table from intervals in the left hand 
    table.
    
    """

    assert (lkey is None) == (rkey is None), \
        'facet key field must be provided for both or neither table'
    return IntervalSubtractView(left, right, lstart=lstart, lstop=lstop,
                                rstart=rstart, rstop=rstop, lkey=lkey,
                                rkey=rkey, include_stop=include_stop)


Table.intervalsubtract = intervalsubtract


class IntervalSubtractView(Table):
    
    def __init__(self, left, right, lstart='start', lstop='stop', 
                 rstart='start', rstop='stop', lkey=None, rkey=None,
                 include_stop=False):
        self.left = left
        self.lstart = lstart
        self.lstop = lstop
        self.lkey = lkey
        self.right = right
        self.rstart = rstart
        self.rstop = rstop
        self.rkey = rkey
        self.include_stop = include_stop

    def __iter__(self):
        return iterintervalsubtract(self.left, self.right, self.lstart,
                                    self.lstop, self.rstart, self.rstop,
                                    self.lkey, self.rkey, self.include_stop)
        

def iterintervalsubtract(left, right, lstart, lstop, rstart, rstop, lkey, rkey,
                         include_stop):

    # create iterators and obtain fields
    lit = iter(left)
    lhdr = next(lit)
    lflds = list(map(text_type, lhdr))
    rit = iter(right)
    rhdr = next(rit)

    # check fields via petl.util.asindices (raises FieldSelectionError if spec
    # is not valid)
    asindices(lhdr, lstart)
    asindices(lhdr, lstop)
    if lkey is not None:
        asindices(lhdr, lkey)
    asindices(rhdr, rstart)
    asindices(rhdr, rstop)
    if rkey is not None:
        asindices(rhdr, rkey)

    # determine output fields
    outhdr = list(lflds)
    yield tuple(outhdr)
    
    # create getters for start and stop positions
    lstartidx, lstopidx = asindices(lhdr, (lstart, lstop))
    getlcoords = itemgetter(lstartidx, lstopidx)
    getrcoords = itemgetter(*asindices(rhdr, (rstart, rstop)))

    if rkey is None:
        # build interval lookup for right table
        lookup = intervallookup(right, rstart, rstop, include_stop=include_stop)
        search = lookup.search
        # main loop
        for lrow in lit:
            start, stop = getlcoords(lrow)
            rrows = search(start, stop)
            if not rrows:
                yield tuple(lrow)
            else:
                rivs = sorted([getrcoords(rrow) for rrow in rrows],
                              key=itemgetter(0))  # sort by start
                for x, y in _subtract(start, stop, rivs):
                    out = list(lrow)
                    out[lstartidx] = x
                    out[lstopidx] = y
                    yield tuple(out)
                
    else:
        # build interval lookup for right table
        lookup = facetintervallookup(right, key=rkey, start=rstart, stop=rstop,
                                     include_stop=include_stop)
        # getter for facet key values in left table
        getlkey = itemgetter(*asindices(lhdr, lkey))
        # main loop
        for lrow in lit:
            lkey = getlkey(lrow)
            start, stop = getlcoords(lrow)
            try:
                rrows = lookup[lkey].search(start, stop)
            except KeyError:
                rrows = None
            except AttributeError:
                rrows = None
            if not rrows:
                yield tuple(lrow)
            else:
                rivs = sorted([getrcoords(rrow) for rrow in rrows],
                              key=itemgetter(0))  # sort by start
                for x, y in _subtract(start, stop, rivs):
                    out = list(lrow)
                    out[lstartidx] = x
                    out[lstopidx] = y
                    yield tuple(out)


from collections import namedtuple
_Interval = namedtuple('Interval', 'start stop')


def collapsedintervals(table, start='start', stop='stop', key=None):
    """
    Utility function to collapse intervals in a table. 
    
    If no facet `key` is given, returns an iterator over `(start, stop)` tuples.
    
    If facet `key` is given, returns an iterator over `(key, start, stop)`
    tuples.
    
    """
    
    if key is None:
        table = sort(table, key=start)
        for iv in _collapse(values(table, (start, stop))):
            yield iv
    else:
        table = sort(table, key=(key, start))
        for k, g in rowgroupby(table, key=key, value=(start, stop)):
            for iv in _collapse(g):
                yield (k,) + iv


Table.collapsedintervals = collapsedintervals


def _collapse(intervals):
    """
    Collapse an iterable of intervals sorted by start coord.
    
    """
    span = None
    for start, stop in intervals:
        if span is None:
            span = _Interval(start, stop)
        elif start <= span.stop < stop:
            span = _Interval(span.start, stop)
        elif start > span.stop:
            yield span
            span = _Interval(start, stop)
    if span is not None:
        yield span
    
    
def _subtract(start, stop, intervals):
    """
    Subtract intervals from a spanning interval.
    
    """
    remainder_start = start
    sub_stop = None
    for sub_start, sub_stop in _collapse(intervals):
        if remainder_start < sub_start:
            yield _Interval(remainder_start, sub_start)
        remainder_start = sub_stop
    if sub_stop is not None and sub_stop < stop:
        yield _Interval(sub_stop, stop)
