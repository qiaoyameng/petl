from __future__ import absolute_import, print_function, division


import itertools
import logging
from bisect import bisect_left, insort
from collections import deque
from decimal import Decimal
from fractions import Fraction


from petl.compat import next, integer_types, string_types
from petl.errors import ArgumentError
from petl.transform.sorts import sort
from petl.util.base import Table, asindices, rowgetter


logger = logging.getLogger(__name__)
warning = logger.warning
info = logger.info
debug = logger.debug


def addrollingquantile(table, key, time, value, window, q, field=None,
                       buffersize=None, tempdir=None):
    """
    Add one or more fields holding the exact quantile(s) of the `value`
    field over a rolling window of the last `window` rows within each
    group. E.g.::

        >>> import petl as etl
        >>> table1 = [['device', 'ts', 'load'],
        ...           ['a', 1, 10],
        ...           ['a', 2, 30],
        ...           ['a', 3, 20],
        ...           ['b', 1, 7],
        ...           ['b', 2, None]]
        >>> table2 = etl.addrollingquantile(table1, key='device', time='ts',
        ...                                 value='load', window=2, q=0.5)
        >>> table2
        +--------+----+------+------------------+
        | device | ts | load | rolling_quantile |
        +========+====+======+==================+
        | 'a'    |  1 |   10 |               10 |
        +--------+----+------+------------------+
        | 'a'    |  2 |   30 |               20 |
        +--------+----+------+------------------+
        | 'a'    |  3 |   20 |               25 |
        +--------+----+------+------------------+
        | 'b'    |  1 |    7 |                7 |
        +--------+----+------+------------------+
        | 'b'    |  2 | None |                7 |
        +--------+----+------+------------------+

    Rows are grouped by `key` (a single field or a list of fields) and
    ordered within each group by the `time` field; rows sharing the same
    `time` keep their original relative order (the sort is stable). The
    output rows are restored to the original input order.

    The window covers the current row and up to `window - 1` preceding
    rows of the same group; while a group has seen fewer than `window`
    rows, the rows available so far are used. `None` values occupy a
    position in the window but do not participate in the numeric
    computation; if every value in the window is `None`, the result is
    `None`.

    Quantiles are computed by linear interpolation on the sorted window
    values at position ``(m - 1) * q``, where ``m`` is the number of
    non-`None` values in the window. The `q` argument may be a single
    number or a list of numbers, in which case one output field is added
    per quantile (name the fields with the `field` argument, or accept
    the defaults). Window values must be finite `int` or
    `decimal.Decimal` instances; floats are rejected to avoid rounding
    surprises and `NaN` ordering ambiguities. Interpolated results are
    exact: `int` or `fractions.Fraction` for integer windows,
    `decimal.Decimal` for decimal windows.

    The per-row update cost is one insertion and (once the window is
    full) one deletion in a sorted list of at most `window` values, i.e.
    ``O(log window)`` comparisons and ``O(window)`` element moves per
    row. The active state is bounded by `window` and does not grow with
    the group length, and the window is never re-sorted from scratch.

    Internally the table is sorted with
    :func:`petl.transform.sorts.sort` (once by group and time, once to
    restore the original order), so the `buffersize` and `tempdir`
    arguments are passed through to control the external sort behaviour.

    """

    return AddRollingQuantileView(table, key, time, value, window, q,
                                  field=field, buffersize=buffersize,
                                  tempdir=tempdir)


Table.addrollingquantile = addrollingquantile


def _validate_window(window):
    if isinstance(window, bool) or not isinstance(window, integer_types):
        raise ArgumentError('window must be an int, got %r' % (window,))
    if window < 1:
        raise ArgumentError('window must be at least 1, got %r' % (window,))
    return window


def _normalise_quantiles(q):
    if isinstance(q, (list, tuple)):
        qs = list(q)
        if not qs:
            raise ArgumentError('at least one quantile must be requested')
    else:
        qs = [q]
    out = list()
    for x in qs:
        if isinstance(x, bool) \
                or not isinstance(x, integer_types + (float, Decimal,
                                                      Fraction)):
            raise ArgumentError(
                'quantile must be a number in [0, 1], got %r' % (x,))
        try:
            f = Fraction(x)
        except (ValueError, OverflowError):
            raise ArgumentError(
                'quantile must be a finite number, got %r' % (x,))
        if f < 0 or f > 1:
            raise ArgumentError(
                'quantile must be in [0, 1], got %r' % (x,))
        out.append(f)
    return out


def _default_field(quantile):
    pct = quantile * 100
    if pct.denominator == 1:
        return 'rolling_q%d' % pct.numerator
    return 'rolling_q%d_%d' % (pct.numerator, pct.denominator)


def _normalise_fields(field, quantiles):
    if field is None:
        if len(quantiles) == 1:
            return ['rolling_quantile']
        return [_default_field(x) for x in quantiles]
    if isinstance(field, string_types):
        if len(quantiles) != 1:
            raise ArgumentError(
                'a single field name was given but %d quantiles were '
                'requested' % len(quantiles))
        return [field]
    fields = list(field)
    if len(fields) != len(quantiles):
        raise ArgumentError(
            'number of fields (%d) does not match number of quantiles '
            '(%d)' % (len(fields), len(quantiles)))
    return fields


def _is_valid_value(value):
    if isinstance(value, bool):
        return False
    if isinstance(value, integer_types):
        return True
    if isinstance(value, Decimal):
        return value.is_finite()
    return False


def _quantile(sorted_values, q):
    # linear interpolation at position (m - 1) * q, computed exactly
    m = len(sorted_values)
    pos = (m - 1) * q
    lo_idx = pos.numerator // pos.denominator
    frac = pos - lo_idx
    lo = sorted_values[lo_idx]
    if not frac:
        return lo
    hi = sorted_values[lo_idx + 1]
    num, den = frac.numerator, frac.denominator
    if isinstance(lo, integer_types) and isinstance(hi, integer_types):
        result = Fraction(lo * (den - num) + hi * num, den)
        # normalize integral results back to plain ints
        if result.denominator == 1:
            return result.numerator
        return result
    return (lo * (den - num) + hi * num) / den


class AddRollingQuantileView(Table):

    def __init__(self, table, key, time, value, window, q, field=None,
                 buffersize=None, tempdir=None):
        self.table = table
        self.key = key
        self.time = time
        self.value = value
        # validate arguments up front, before any data is consumed
        self.window = _validate_window(window)
        self.quantiles = _normalise_quantiles(q)
        self.fields = _normalise_fields(field, self.quantiles)
        self.buffersize = buffersize
        self.tempdir = tempdir

    def __iter__(self):
        return iteraddrollingquantile(self.table, self.key, self.time,
                                      self.value, self.window,
                                      self.quantiles, self.fields,
                                      self.buffersize, self.tempdir)


def iteraddrollingquantile(table, key, time, value, window, quantiles,
                           fields, buffersize, tempdir):
    it = iter(table)
    try:
        hdr = tuple(next(it))
    except StopIteration:
        return
    key_indices = asindices(hdr, key)
    time_index = asindices(hdr, time)[0]
    value_index = asindices(hdr, value)[0]
    n = len(hdr)
    # position of the original row order index within decorated rows
    idx = n

    # decorate each row with its original position, then order by group,
    # time and original position (so equal times keep the source order)
    decorated = itertools.chain(
        [hdr + (None,)],
        (tuple(row) + (i,) for i, row in enumerate(it)),
    )
    ordered = sort(decorated, key=key_indices + [time_index, idx],
                   buffersize=buffersize, tempdir=tempdir, cache=False)
    computed = _iterrollingquantiles(ordered, key_indices, value_index,
                                     window, quantiles)
    # restore the original row order
    restored = sort(computed, key=[idx], buffersize=buffersize,
                    tempdir=tempdir, cache=False)
    rit = iter(restored)
    next(rit)  # discard the decorated header
    yield hdr + tuple(fields)
    for row in rit:
        yield row[:n] + row[n + 1:]


def _iterrollingquantiles(ordered, key_indices, value_index, window,
                          quantiles):
    it = iter(ordered)
    try:
        hdr = next(it)
    except StopIteration:
        return
    yield hdr
    getkey = rowgetter(*key_indices)
    nquant = len(quantiles)
    # per-group state, bounded by the window size, not the group length
    window_values = deque()  # raw values in arrival order, None included
    sorted_values = list()  # sorted multiset of the non-None values
    prev_key = None
    for row in it:
        row_key = getkey(row)
        if row_key != prev_key:
            prev_key = row_key
            window_values = deque()
            sorted_values = list()
        value = row[value_index]
        if value is not None and not _is_valid_value(value):
            raise ValueError(
                'rolling quantile values must be finite int or Decimal, '
                'got %r' % (value,))
        window_values.append(value)
        if value is not None:
            insort(sorted_values, value)
        if len(window_values) > window:
            expired = window_values.popleft()
            if expired is not None:
                # remove one occurrence; with duplicate values any one of
                # the equal entries is equivalent
                del sorted_values[bisect_left(sorted_values, expired)]
        if not sorted_values:
            yield tuple(row) + (None,) * nquant
        else:
            yield tuple(row) + tuple(_quantile(sorted_values, x)
                                     for x in quantiles)
