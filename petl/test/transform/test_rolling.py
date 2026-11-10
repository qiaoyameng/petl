from __future__ import absolute_import, print_function, division


import random
from decimal import Decimal
from fractions import Fraction


import pytest


import petl.transform.rolling as rolling
from petl.errors import ArgumentError
from petl.test.helpers import ieq
from petl.transform.rolling import addrollingquantile


def _ref_quantile(sorted_vals, q):
    # slow but obviously correct: interpolation at (m - 1) * q
    m = len(sorted_vals)
    pos = (m - 1) * Fraction(q)
    lo_idx = pos.numerator // pos.denominator
    frac = pos - lo_idx
    lo = sorted_vals[lo_idx]
    if not frac:
        return lo
    hi = sorted_vals[lo_idx + 1]
    num, den = frac.numerator, frac.denominator
    if isinstance(lo, int) and isinstance(hi, int):
        return Fraction(lo * (den - num) + hi * num, den)
    return (lo * (den - num) + hi * num) / den


def slow_reference(table, key, time, value, window, qs):
    """Reference implementation: re-sorts the whole window for every
    row, then computes quantiles from scratch."""
    hdr = list(table[0])
    if isinstance(key, (list, tuple)):
        key_indices = [hdr.index(k) for k in key]
    else:
        key_indices = [hdr.index(key)]
    time_index = hdr.index(time)
    value_index = hdr.index(value)
    rows = [tuple(row) for row in table[1:]]
    order = sorted(range(len(rows)),
                   key=lambda i: (tuple(rows[i][k] for k in key_indices),
                                  rows[i][time_index], i))
    result = [None] * len(rows)
    prev_key = None
    window_values = []
    for i in order:
        row_key = tuple(rows[i][k] for k in key_indices)
        if row_key != prev_key:
            prev_key = row_key
            window_values = []
        window_values.append(rows[i][value_index])
        vals = sorted(v for v in window_values[-window:]
                      if v is not None)
        if not vals:
            result[i] = (None,) * len(qs)
        else:
            result[i] = tuple(_ref_quantile(vals, q) for q in qs)
    return result


def test_addrollingquantile():
    table = (('device', 'ts', 'load'),
             ('a', 1, 10),
             ('a', 2, 30),
             ('a', 3, 20),
             ('b', 1, 7),
             ('b', 2, None),
             ('b', 3, 9))
    actual = addrollingquantile(table, key='device', time='ts',
                                value='load', window=2, q=0.5)
    expect = (('device', 'ts', 'load', 'rolling_quantile'),
              ('a', 1, 10, 10),
              ('a', 2, 30, 20),
              ('a', 3, 20, 25),
              ('b', 1, 7, 7),
              ('b', 2, None, 7),
              ('b', 3, 9, 9))
    ieq(expect, actual)
    ieq(expect, actual)


def test_addrollingquantile_restores_source_order():
    # input is neither grouped nor time ordered; output rows must come
    # back in the original order
    table = (('device', 'ts', 'v'),
             ('b', 2, 20),
             ('a', 3, 3),
             ('b', 1, 10),
             ('a', 1, 1),
             ('a', 2, 2),
             ('b', 3, 30))
    actual = addrollingquantile(table, 'device', 'ts', 'v', 2, 0.5)
    expect = (('device', 'ts', 'v', 'rolling_quantile'),
              ('b', 2, 20, 15),
              ('a', 3, 3, Fraction(5, 2)),
              ('b', 1, 10, 10),
              ('a', 1, 1, 1),
              ('a', 2, 2, Fraction(3, 2)),
              ('b', 3, 30, 25))
    ieq(expect, actual)


def test_addrollingquantile_compound_key_stable_time():
    # rows with equal (key, time) must keep their source order; with
    # window=2 the quantile of each row depends on that order
    table = (('dev', 'zone', 'ts', 'v'),
             ('a', 'x', 1, 1),
             ('b', 'x', 1, 100),
             ('a', 'x', 1, 2),
             ('a', 'x', 2, 3),
             ('b', 'y', 1, 7))
    actual = addrollingquantile(table, ['dev', 'zone'], 'ts', 'v', 2, 0.5)
    expect = (('dev', 'zone', 'ts', 'v', 'rolling_quantile'),
              ('a', 'x', 1, 1, 1),
              ('b', 'x', 1, 100, 100),
              ('a', 'x', 1, 2, Fraction(3, 2)),
              ('a', 'x', 2, 3, Fraction(5, 2)),
              ('b', 'y', 1, 7, 7))
    ieq(expect, actual)


def test_addrollingquantile_multiple_quantiles_decimal():
    table = (('g', 't', 'v'),
             ('g', 1, Decimal('1.5')),
             ('g', 2, Decimal('2.5')),
             ('g', 3, Decimal('10')),
             ('g', 4, Decimal('20')))
    actual = addrollingquantile(table, 'g', 't', 'v', 3,
                                q=[0, Decimal('0.5'), 1])
    expect = (('g', 't', 'v', 'rolling_q0', 'rolling_q50', 'rolling_q100'),
              ('g', 1, Decimal('1.5'), Decimal('1.5'), Decimal('1.5'),
               Decimal('1.5')),
              ('g', 2, Decimal('2.5'), Decimal('1.5'), Decimal('2.0'),
               Decimal('2.5')),
              ('g', 3, Decimal('10'), Decimal('1.5'), Decimal('2.5'),
               Decimal('10')),
              ('g', 4, Decimal('20'), Decimal('2.5'), Decimal('10'),
               Decimal('20')))
    ieq(expect, actual)
    # interpolated decimal results stay decimal
    row = list(actual)[2]
    assert isinstance(row[4], Decimal)


def test_addrollingquantile_custom_fields():
    table = (('g', 't', 'v'),
             ('g', 1, 4),
             ('g', 2, 8))
    actual = addrollingquantile(table, 'g', 't', 'v', 2,
                                q=[0.25, 0.75], field=['lo', 'hi'])
    expect = (('g', 't', 'v', 'lo', 'hi'),
              ('g', 1, 4, 4, 4),
              ('g', 2, 8, 5, 7))
    ieq(expect, actual)


def test_addrollingquantile_none_semantics():
    # None occupies a window slot but is not a number; a window of only
    # None values yields None
    table = (('g', 't', 'v'),
             ('g', 1, None),
             ('g', 2, None),
             ('g', 3, 5),
             ('g', 4, None),
             ('g', 5, None))
    actual = addrollingquantile(table, 'g', 't', 'v', 2, 0.5)
    expect = (('g', 't', 'v', 'rolling_quantile'),
              ('g', 1, None, None),
              ('g', 2, None, None),
              ('g', 3, 5, 5),       # window [None, 5]
              ('g', 4, None, 5),    # window [5, None]
              ('g', 5, None, None))  # window [None, None]
    ieq(expect, actual)


def test_addrollingquantile_window_one():
    table = (('g', 't', 'v'),
             ('g', 1, 3),
             ('g', 2, 1),
             ('g', 3, 2))
    actual = addrollingquantile(table, 'g', 't', 'v', 1, 0.5)
    expect = (('g', 't', 'v', 'rolling_quantile'),
              ('g', 1, 3, 3),
              ('g', 2, 1, 1),
              ('g', 3, 2, 2))
    ieq(expect, actual)


def test_addrollingquantile_empty():
    actual = addrollingquantile([], 'g', 't', 'v', 2, 0.5)
    ieq([], actual)
    actual = addrollingquantile([('g', 't', 'v')], 'g', 't', 'v', 2, 0.5)
    ieq([('g', 't', 'v', 'rolling_quantile')], actual)


class _Exploding(object):
    # a table that must never be consumed
    def __iter__(self):
        raise AssertionError('table was consumed')


def test_addrollingquantile_invalid_args_fail_before_consuming_data():
    table = _Exploding()
    bad = dict(key='g', time='t', value='v')
    for window in (0, -1, 1.5, '3', True, None):
        with pytest.raises(ArgumentError):
            addrollingquantile(table, window=window, q=0.5, **bad)
    for q in (-0.1, 1.5, 2, '0.5', None, float('nan'), float('inf'),
              [], [0.5, 2], [0.5, 'x']):
        with pytest.raises(ArgumentError):
            addrollingquantile(table, window=3, q=q, **bad)
    # field/quantile count mismatch
    with pytest.raises(ArgumentError):
        addrollingquantile(table, window=3, q=[0.25, 0.75], field='q', **bad)
    with pytest.raises(ArgumentError):
        addrollingquantile(table, window=3, q=[0.25, 0.75],
                           field=['a', 'b', 'c'], **bad)
    # valid arguments do not touch the data either (lazy evaluation)
    addrollingquantile(table, window=3, q=0.5, **bad)


def test_addrollingquantile_invalid_values():
    base = (('g', 't', 'v'),)
    for bad_value in (1.5, float('nan'), True, '5',
                      Decimal('NaN'), Decimal('Infinity')):
        table = base + (('g', 1, bad_value),)
        actual = addrollingquantile(table, 'g', 't', 'v', 2, 0.5)
        with pytest.raises(ValueError):
            list(actual)


def test_addrollingquantile_expiring_duplicate():
    # the expiring value is one of several equal values; the remaining
    # duplicates must stay in the window
    table = (('g', 't', 'v'),
             ('g', 1, 7),
             ('g', 2, 7),
             ('g', 3, 1),
             ('g', 4, 7),
             ('g', 5, 7))
    actual = addrollingquantile(table, 'g', 't', 'v', 2, 0.5)
    expect = (('g', 't', 'v', 'rolling_quantile'),
              ('g', 1, 7, 7),
              ('g', 2, 7, 7),
              ('g', 3, 1, 4),        # one 7 expired, window is [7, 1]
              ('g', 4, 7, 4),        # window is [1, 7]
              ('g', 5, 7, 7))        # window is [7, 7]
    ieq(expect, actual)


def test_addrollingquantile_many_duplicates_vs_reference():
    # short sequences over a tiny value domain, so windows are full of
    # equal values and expiry almost always removes one of several
    # duplicates; compare against the slow per-window-sort reference
    rng = random.Random(42)
    for window in (1, 2, 3, 5, 8):
        for _ in range(5):
            rows = [('g', 't', 'v')]
            n = rng.randint(1, 30)
            for t in range(n):
                group = rng.choice('ab')
                value = rng.choice([None, 1, 1, 1, 2, 2, 3])
                rows.append((group, t, value))
            # shuffle the row order to exercise grouping, stable time
            # order and final restoration of the source order
            body = rows[1:]
            rng.shuffle(body)
            table = tuple([rows[0]] + body)
            qs = [0, Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), 1]
            actual = addrollingquantile(table, 'g', 't', 'v', window, qs)
            actual_rows = list(actual)
            expect_tail = slow_reference(table, 'g', 't', 'v', window, qs)
            assert len(actual_rows) == len(table)
            for i, row in enumerate(actual_rows[1:]):
                assert tuple(row[:3]) == tuple(table[1:][i])
                assert tuple(row[3:]) == expect_tail[i]


def test_addrollingquantile_update_cost_operation_counts(monkeypatch):
    # Instrument the sorted-window maintenance to show the update cost:
    # exactly one insertion per non-None row and one deletion per
    # expired non-None value, i.e. O(1) structural updates per row, no
    # matter how long the group is. The slow reference instead re-sorts
    # the whole window for every row.
    calls = {'insort': 0, 'bisect': 0}
    real_insort = rolling.insort
    real_bisect = rolling.bisect_left

    def counting_insort(lst, x):
        calls['insort'] += 1
        return real_insort(lst, x)

    def counting_bisect(lst, x):
        calls['bisect'] += 1
        return real_bisect(lst, x)

    monkeypatch.setattr(rolling, 'insort', counting_insort)
    monkeypatch.setattr(rolling, 'bisect_left', counting_bisect)

    def run(nrows, ngroups, window):
        calls['insort'] = 0
        calls['bisect'] = 0
        rows = [('g', 't', 'v')]
        for t in range(nrows):
            rows.append(('g%d' % (t % ngroups), t, t % 7))
        # iterate explicitly: list(view) would trigger Table.__len__ and
        # hence a second full pass over the pipeline
        for _ in addrollingquantile(tuple(rows), 'g', 't', 'v', window,
                                    0.5):
            pass
        return calls['insort'], calls['bisect']

    window = 10
    # one long group: one insert per row, one delete per expired row
    ins, dele = run(1000, 1, window)
    assert (ins, dele) == (1000, 1000 - window)
    # doubling the group length doubles the work: cost grows with the
    # number of rows, not with the group length squared
    ins2, dele2 = run(2000, 1, window)
    assert (ins2, dele2) == (2000, 2000 - window)
    # splitting the same rows into many groups does not change the
    # per-row cost (only fewer expirations during warm-up)
    ins3, dele3 = run(1000, 100, window)
    assert ins3 == 1000
    assert dele3 == 1000 - 100 * window


def test_addrollingquantile_external_sort():
    # a tiny sort buffer forces petl's chunked external sort; results
    # must match the in-memory path
    rng = random.Random(7)
    body = [(rng.choice('ab'), t, rng.choice([None, 1, 2, 2, 3]))
            for t in range(50)]
    rng.shuffle(body)
    table = tuple([('g', 't', 'v')] + body)
    qs = [Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)]
    expect = [tuple(row) for row in
              addrollingquantile(table, 'g', 't', 'v', 4, qs)]
    actual = [tuple(row) for row in
              addrollingquantile(table, 'g', 't', 'v', 4, qs, buffersize=3)]
    assert expect == actual
