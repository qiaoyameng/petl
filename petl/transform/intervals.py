from __future__ import absolute_import, print_function, division


from collections import namedtuple
from operator import itemgetter, attrgetter

from petl.comparison import Comparable
from petl.compat import text_type


from petl.util.base import asindices, records, Table, values, rowgroupby
from petl.errors import ArgumentError, DuplicateKeyError
from petl.transform.basics import addfield
from petl.transform.sorts import sort


def validityjoin(left, right, key=None, lkey=None, rkey=None, time='time',
                 start='start', end='end', presorted=False, missing=None,
                 buffersize=None, tempdir=None, cache=True,
                 missing_time='raise', invalid_interval='raise',
                 lprefix=None, rprefix=None):
    """
    Join point-in-time event rows to rules effective at the event time.

    Rule intervals are half-open: ``start <= time < end``. A missing rule
    end means the rule is still effective. Every overlapping rule is
    returned, while event rows without a match are retained with missing
    values in the rule fields.

        >>> import petl as etl
        >>> events = [['id', 'time'], ['a', 2], ['a', 4]]
        >>> rules = [['id', 'start', 'end', 'version'],
        ...          ['a', 0, 4, 'old'],
        ...          ['a', 2, 3, 'nested'],
        ...          ['a', 4, None, 'new']]
        >>> table = etl.validityjoin(events, rules, key='id')
        >>> table.lookall()
        +-----+------+-----+-------+------+----------+
        | id  | time | id  | start | end  | version  |
        +=====+======+=====+=======+======+==========+
        | 'a' |    2 | 'a' |     0 |    4 | 'old'    |
        +-----+------+-----+-------+------+----------+
        | 'a' |    2 | 'a' |     2 |    3 | 'nested' |
        +-----+------+-----+-------+------+----------+
        | 'a' |    4 | 'a' |     4 | None | 'new'    |
        +-----+------+-----+-------+------+----------+
        <BLANKLINE>

    If `presorted` is True, the event table must be ordered by business key
    and event time, and the rule table by business key and start. That path
    performs one ordered sweep and never scans all rules for one event.

    Repeated event and rule field names are retained in the output, matching
    the behaviour of other joins. Use `lprefix` and `rprefix` when duplicate
    field names need to be predictable. Set `missing_time` or
    `invalid_interval` to ``"skip"`` to omit invalid rows instead of raising.

    """

    if key is not None and (lkey is not None or rkey is not None):
        raise ArgumentError(
            'bad key arguments: either specify key, or specify both lkey '
            'and rkey'
        )
    if (lkey is None) != (rkey is None):
        raise ArgumentError('lkey and rkey must both be specified')
    if key is not None:
        lkey = rkey = key
    lkey = _normalise_key(lkey)
    rkey = _normalise_key(rkey)
    if missing_time not in ('raise', 'skip'):
        raise ArgumentError('missing_time must be either "raise" or "skip"')
    if invalid_interval not in ('raise', 'skip'):
        raise ArgumentError(
            'invalid_interval must be either "raise" or "skip"'
        )

    return ValidityJoinView(
        left, right, lkey=lkey, rkey=rkey, time=time, start=start, end=end,
        presorted=presorted, missing=missing, buffersize=buffersize,
        tempdir=tempdir, sort_cache=cache, missing_time=missing_time,
        invalid_interval=invalid_interval, lprefix=lprefix, rprefix=rprefix
    )


Table.validityjoin = validityjoin


def _normalise_key(key):
    if key is None or isinstance(key, (list, tuple)):
        return key
    return (key,)


def _unique_field(fields, candidate):
    field = candidate
    counter = 1
    while field in fields:
        counter += 1
        field = candidate + str(counter)
    return field


def _row_value(row, index):
    return row[index] if index < len(row) else None


class _OrderedRows(Table):

    def __init__(self, table, field):
        self.table = table
        self.field = field

    def __iter__(self):
        it = iter(self.table)
        hdr = next(it)
        yield tuple(list(hdr) + [self.field])
        for order, row in enumerate(it):
            yield tuple(list(row) + [order])


class _CheckedEvents(Table):

    def __init__(self, table, time, policy, missing=None):
        self.table = table
        self.time = time
        self.policy = policy
        self.missing = missing

    def __iter__(self):
        it = iter(self.table)
        hdr = next(it)
        width = len(hdr)
        fields = list(map(text_type, hdr))
        time_idx = fields.index(self.time)
        padded_rows = (
            tuple(list(row[:width]) +
                  [self.missing] * max(0, width - len(row)))
            for row in it
        )
        yield tuple(hdr)
        for row_number, row in enumerate(padded_rows, start=1):
            if _row_value(row, time_idx) is None:
                if self.policy == 'raise':
                    raise ValueError(
                        'missing event time at row %d' % row_number
                    )
                continue
            yield row


class _CheckedRules(Table):

    def __init__(self, table, start, end, policy, missing=None):
        self.table = table
        self.start = start
        self.end = end
        self.policy = policy
        self.missing = missing

    def __iter__(self):
        it = iter(self.table)
        hdr = next(it)
        width = len(hdr)
        yield tuple(hdr)
        fields = list(map(text_type, hdr))
        start_idx = fields.index(self.start)
        end_idx = fields.index(self.end)
        padded_rows = (
            tuple(list(row[:width]) +
                  [self.missing] * max(0, width - len(row)))
            for row in it
        )
        for row_number, row in enumerate(padded_rows, start=1):
            rule_start = _row_value(row, start_idx)
            rule_end = _row_value(row, end_idx)
            if rule_start is None:
                if self.policy == 'raise':
                    raise ValueError(
                        'missing rule start at row %d' % row_number
                    )
                continue
            if rule_end is not None:
                try:
                    if rule_start > rule_end:
                        if self.policy == 'raise':
                            raise ValueError(
                                'invalid rule interval at row %d: start is '
                                'after end' % row_number
                            )
                        continue
                except TypeError:
                    if self.policy == 'raise':
                        raise ValueError(
                            'incompatible rule interval coordinates at '
                            'row %d' % row_number
                        )
                    continue
            yield row


class ValidityJoinView(Table):

    def __init__(self, left, right, lkey, rkey, time, start, end,
                 presorted=False, missing=None, buffersize=None,
                 tempdir=None, sort_cache=True, missing_time='raise',
                 invalid_interval='raise', lprefix=None, rprefix=None):
        self.left = left
        self.right = right
        self.lkey = lkey
        self.rkey = rkey
        self.time = time
        self.start = start
        self.end = end
        self.presorted = presorted
        self.missing = missing
        self.buffersize = buffersize
        self.tempdir = tempdir
        self.sort_cache = sort_cache
        self.missing_time = missing_time
        self.invalid_interval = invalid_interval
        self.lprefix = lprefix
        self.rprefix = rprefix

    def __iter__(self):
        left = self.left
        right = self.right
        left_order = right_order = None

        if not self.presorted:
            lhdr = next(iter(left))
            rhdr = next(iter(right))
            left_order = _unique_field(list(map(text_type, lhdr)),
                                       '__petl_event_order__')
            right_order = _unique_field(list(map(text_type, rhdr)),
                                        '__petl_rule_order__')
            left = _OrderedRows(left, left_order)
            right = _OrderedRows(right, right_order)

        events = _CheckedEvents(left, self.time, self.missing_time,
                                missing=self.missing)
        rules = _CheckedRules(right, self.start, self.end,
                              self.invalid_interval, missing=self.missing)

        if not self.presorted:
            left = sort(events, list(self.lkey or ()) + [self.time],
                        buffersize=self.buffersize, tempdir=self.tempdir,
                        cache=self.sort_cache)
            right = sort(rules, list(self.rkey or ()) + [self.start],
                         buffersize=self.buffersize, tempdir=self.tempdir,
                         cache=self.sort_cache)

        joined = itervalidityjoin(
            events if self.presorted else left,
            rules if self.presorted else right,
            lkey=self.lkey, rkey=self.rkey, time=self.time, start=self.start,
            end=self.end, missing=self.missing, lprefix=self.lprefix,
            rprefix=self.rprefix, invalid_interval=self.invalid_interval,
            left_order=left_order,
            right_order=right_order
        )
        if self.presorted:
            return joined
        return iter(_restore_validity_order(joined, left_order))


def _restore_validity_order(table, order_field):
    ordered = sort(table, order_field)
    it = iter(ordered)
    hdr = next(it)
    fields = list(map(text_type, hdr))
    order_index = fields.index(order_field)
    indices = [i for i in range(len(fields)) if i != order_index]
    if len(indices) == 1:
        def get_row(row):
            return (row[indices[0]],)
    else:
        get_row = itemgetter(*indices)
    yield get_row(hdr)
    for row in it:
        yield get_row(row)


_ActiveRule = namedtuple('_ActiveRule', ['end', 'order', 'row'])


def itervalidityjoin(left, right, lkey, rkey, time, start, end, missing,
                     lprefix, rprefix, invalid_interval, left_order,
                     right_order):
    events = iter(left)
    rules = iter(right)
    lhdr = next(events)
    rhdr = next(rules)
    lflds = list(map(text_type, lhdr))
    rflds = list(map(text_type, rhdr))

    asindices(lhdr, time)
    asindices(rhdr, start)
    asindices(rhdr, end)
    if lkey is not None:
        asindices(lhdr, lkey)
        asindices(rhdr, rkey)

    def prefix_header(fields, prefix, order):
        values = [field for field in fields if field != order]
        if prefix is None:
            return values
        return [text_type(prefix) + text_type(field) for field in values]

    out_lhdr = prefix_header(lflds, lprefix, left_order)
    if left_order is not None:
        out_lhdr.append(left_order)
    yield tuple(out_lhdr + prefix_header(rflds, rprefix, right_order))

    time_idx = lflds.index(time)
    start_idx = rflds.index(start)
    end_idx = rflds.index(end)
    lorder_idx = lflds.index(left_order) if left_order is not None else None
    rorder_idx = rflds.index(right_order) if right_order is not None else None
    lkey_idx = asindices(lhdr, lkey) if lkey is not None else []
    rkey_idx = asindices(rhdr, rkey) if rkey is not None else []
    if lkey_idx:
        get_event_key = itemgetter(*lkey_idx)
    else:
        def get_event_key(row):
            return None
    if rkey_idx:
        get_rule_key = itemgetter(*rkey_idx)
    else:
        def get_rule_key(row):
            return None
    right_missing = [missing] * (
        len(rflds) - (1 if right_order is not None else 0)
    )

    def remove_order(row, fields, order):
        if order is None:
            return tuple(row)
        index = fields.index(order)
        return tuple(row[:index] + row[index + 1:])

    rule = None
    rule_key = None
    active = []
    active_key = object()

    def next_rule():
        row = next(rules)
        return row, get_rule_key(row)

    for event in events:
        event_time = _row_value(event, time_idx)
        if event_time is None:
            continue
        event_key = get_event_key(event)
        if event_key != active_key:
            active_key = event_key
            active = []

        if rule is None:
            try:
                rule, rule_key = next_rule()
            except StopIteration:
                rule = None
                rule_key = None

        while rule is not None and \
                Comparable(rule_key) < Comparable(event_key):
            try:
                rule, rule_key = next_rule()
            except StopIteration:
                rule = None
                rule_key = None

        while rule is not None and rule_key == event_key:
            try:
                starts_before_event = \
                    _row_value(rule, start_idx) <= event_time
            except TypeError:
                if invalid_interval == 'raise':
                    raise ValueError(
                        'incompatible event and rule start coordinates'
                    )
                try:
                    rule, rule_key = next_rule()
                except StopIteration:
                    rule = None
                    rule_key = None
                continue
            if not starts_before_event:
                break
            rule_order = _row_value(rule, rorder_idx) \
                if rorder_idx is not None else len(active)
            active.append(_ActiveRule(_row_value(rule, end_idx),
                                      rule_order, rule))
            try:
                rule, rule_key = next_rule()
            except StopIteration:
                rule = None
                rule_key = None

        still_active = []
        for candidate in active:
            if candidate.end is None:
                still_active.append(candidate)
                continue
            try:
                is_active = event_time < candidate.end
            except TypeError:
                if invalid_interval == 'raise':
                    raise ValueError(
                        'incompatible rule interval coordinate for event'
                    )
                continue
            if is_active:
                still_active.append(candidate)
        active = still_active
        event_out = tuple(event)
        if not active:
            yield tuple(list(event_out) + right_missing)
            continue
        if rorder_idx is not None:
            matching = sorted(active, key=lambda candidate: candidate.order)
        else:
            matching = active
        for candidate in matching:
            yield tuple(list(event_out) +
                        list(remove_order(candidate.row, rflds,
                                          right_order)))


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
