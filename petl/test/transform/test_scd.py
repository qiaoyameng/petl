from __future__ import absolute_import, print_function, division


import gc
import glob
import os
import tempfile
from datetime import datetime

import pytest

from petl.errors import DuplicateKeyError, SCD2Error
from petl.test.helpers import eq_, ieq
from petl.transform.scd import scd2


HIST_HEADER = ('cust', 'tier', 'city', 'valid_from', 'valid_to')
SNAP_HEADER = ('cust', 'tier', 'city')
KEY = 'cust'
TRACKED = ('tier', 'city')
VALID_FROM = 'valid_from'
VALID_TO = 'valid_to'


def _snapshot(rows):
    return [SNAP_HEADER] + list(rows)


def _history(rows):
    return [HIST_HEADER] + list(rows)


def test_scd2_three_days_change_disappear_reappear():
    # day 1: three brand new keys
    day1 = _snapshot([
        ('C1', 'gold', 'BJ'),
        ('C2', 'silver', 'SH'),
        ('C3', 'bronze', 'GZ'),
    ])
    hist = scd2(_history([]), day1, KEY, TRACKED, VALID_FROM, VALID_TO, 1)
    ieq((
        HIST_HEADER,
        ('C1', 'gold', 'BJ', 1, None),
        ('C2', 'silver', 'SH', 1, None),
        ('C3', 'bronze', 'GZ', 1, None),
    ), hist)

    # day 2: C1 unchanged, C2 changes, C3 disappears
    day2 = _snapshot([
        ('C1', 'gold', 'BJ'),
        ('C2', 'platinum', 'SH'),
    ])
    hist = scd2(hist, day2, KEY, TRACKED, VALID_FROM, VALID_TO, 2)
    ieq((
        HIST_HEADER,
        ('C1', 'gold', 'BJ', 1, None),
        ('C2', 'silver', 'SH', 1, 2),
        ('C2', 'platinum', 'SH', 2, None),
        ('C3', 'bronze', 'GZ', 1, 2),
    ), hist)

    # day 3: C1 changes, C2 unchanged, C3 reappears with the same values
    day3 = _snapshot([
        ('C1', 'gold', 'TJ'),
        ('C2', 'platinum', 'SH'),
        ('C3', 'bronze', 'GZ'),
    ])
    hist = scd2(hist, day3, KEY, TRACKED, VALID_FROM, VALID_TO, 3)
    expectation = (
        HIST_HEADER,
        ('C1', 'gold', 'BJ', 1, 3),
        ('C1', 'gold', 'TJ', 3, None),
        ('C2', 'silver', 'SH', 1, 2),
        ('C2', 'platinum', 'SH', 2, None),
        ('C3', 'bronze', 'GZ', 1, 2),
        ('C3', 'bronze', 'GZ', 3, None),
    )
    ieq(expectation, hist)

    # every key matches at most one version at every point in time
    rows = expectation[1:]
    for point in (0.5, 1, 1.5, 2, 2.5, 3, 3.5, 100):
        for key in ('C1', 'C2', 'C3'):
            hits = [r for r in rows
                    if r[0] == key
                    and r[3] <= point
                    and (r[4] is None or point < r[4])]
            assert len(hits) <= 1, (point, key, hits)


def test_scd2_reapply_same_snapshot_is_idempotent():
    day1 = _snapshot([
        ('C1', 'gold', 'BJ'),
        ('C2', 'silver', 'SH'),
        ('C3', 'bronze', 'GZ'),
    ])
    hist = scd2(_history([]), day1, KEY, TRACKED, VALID_FROM, VALID_TO, 1)
    day2 = _snapshot([
        ('C1', 'gold', 'BJ'),
        ('C2', 'platinum', 'SH'),
    ])
    hist = scd2(hist, day2, KEY, TRACKED, VALID_FROM, VALID_TO, 2)
    day3 = _snapshot([
        ('C1', 'gold', 'TJ'),
        ('C2', 'platinum', 'SH'),
        ('C3', 'bronze', 'GZ'),
    ])
    first = list(scd2(hist, day3, KEY, TRACKED, VALID_FROM, VALID_TO, 3))
    second = list(scd2(hist, day3, KEY, TRACKED, VALID_FROM, VALID_TO, 3))
    eq_(first, second)
    assert all(row[3] <= 3 for row in first[1:])


def test_scd2_none_and_empty_string_are_distinct():
    hist = _history([('C1', None, 'BJ', 1, None)])

    unchanged = scd2(hist, _snapshot([('C1', None, 'BJ')]), KEY, TRACKED,
                     VALID_FROM, VALID_TO, 2)
    eq_(2, len(list(unchanged)))

    changed = scd2(hist, _snapshot([('C1', '', 'BJ')]), KEY, TRACKED,
                   VALID_FROM, VALID_TO, 2)
    rows = list(changed)
    eq_(3, len(rows))
    eq_(None, rows[1][1])
    eq_(2, rows[1][4])
    eq_('', rows[2][1])
    eq_(None, rows[2][4])


def test_scd2_rejects_duplicate_snapshot_key():
    snapshot = _snapshot([
        ('C1', 'gold', 'BJ'),
        ('C1', 'silver', 'SH'),
    ])
    with pytest.raises(DuplicateKeyError):
        list(scd2(_history([]), snapshot, KEY, TRACKED,
                  VALID_FROM, VALID_TO, 1))


def test_scd2_rejects_overlapping_history_intervals():
    hist = _history([
        ('C1', 'gold', 'BJ', 1, 3),
        ('C1', 'silver', 'SH', 2, None),
    ])
    with pytest.raises(SCD2Error):
        list(scd2(hist, _snapshot([('C1', 'silver', 'SH')]), KEY, TRACKED,
                  VALID_FROM, VALID_TO, 4))


def test_scd2_rejects_two_open_versions():
    hist = _history([
        ('C1', 'gold', 'BJ', 1, None),
        ('C1', 'silver', 'SH', 2, None),
    ])
    with pytest.raises(SCD2Error):
        list(scd2(hist, _snapshot([('C1', 'silver', 'SH')]), KEY, TRACKED,
                  VALID_FROM, VALID_TO, 3))


def test_scd2_rejects_backdated_and_same_point_updates():
    hist = _history([('C1', 'gold', 'BJ', 5, None)])
    snapshot = _snapshot([('C1', 'silver', 'SH')])
    for effective in (5, 4):
        with pytest.raises(SCD2Error):
            list(scd2(hist, snapshot, KEY, TRACKED, VALID_FROM,
                      VALID_TO, effective))


def test_scd2_rejects_reappear_inside_closed_interval():
    hist = _history([('C1', 'gold', 'BJ', 1, 5)])
    with pytest.raises(SCD2Error):
        list(scd2(hist, _snapshot([('C1', 'gold', 'BJ')]), KEY, TRACKED,
                  VALID_FROM, VALID_TO, 3))


def test_scd2_reappears_after_gap_as_new_version():
    hist = _history([
        ('C1', 'gold', 'BJ', 1, 2),
        ('C1', 'silver', 'SH', 2, 4),
    ])
    out = list(scd2(hist, _snapshot([('C1', 'gold', 'BJ')]), KEY, TRACKED,
                    VALID_FROM, VALID_TO, 5))
    eq_((
        HIST_HEADER,
        ('C1', 'gold', 'BJ', 1, 2),
        ('C1', 'silver', 'SH', 2, 4),
        ('C1', 'gold', 'BJ', 5, None),
    ), tuple(out))


def test_scd2_empty_snapshot_closes_every_current_version():
    hist = _history([
        ('C1', 'gold', 'BJ', 1, None),
        ('C2', 'silver', 'SH', 1, None),
    ])
    out = list(scd2(hist, _snapshot([]), KEY, TRACKED,
                    VALID_FROM, VALID_TO, 2))
    eq_(3, len(out))
    assert all(row[4] == 2 for row in out[1:])
    assert all(row[3] == 1 for row in out[1:])


def test_scd2_composite_key_and_datetime_effective():
    header = ('region', 'cust', 'tier', 'score', VALID_FROM, VALID_TO)
    hist = [header,
            ('R1', 'C1', 'gold', 10, datetime(2026, 1, 1), None)]
    snapshot = [('region', 'cust', 'tier', 'score'),
                ('R1', 'C1', 'gold', 20),
                ('R2', 'C9', 'bronze', 1)]
    effective = datetime(2026, 1, 2)
    out = list(scd2(hist, snapshot, ('region', 'cust'), ('tier', 'score'),
                    VALID_FROM, VALID_TO, effective))
    eq_(4, len(out))
    closed, opened, new = out[1], out[2], out[3]
    eq_(effective, closed[5])
    eq_(20, opened[3])
    eq_(None, opened[5])
    eq_(('R2', 'C9'), (new[0], new[1]))
    eq_(effective, new[4])


def test_scd2_accepts_one_shot_iterators():

    def one_shot(header, rows):
        yield header
        for row in rows:
            yield row

    hist = one_shot(HIST_HEADER, [
        ('C1', 'gold', 'BJ', 1, None),
        ('C2', 'silver', 'SH', 1, None),
    ])
    snapshot = one_shot(SNAP_HEADER, [
        ('C1', 'gold', 'BJ'),
        ('C3', 'bronze', 'GZ'),
    ])
    out = list(scd2(hist, snapshot, KEY, TRACKED, VALID_FROM, VALID_TO, 2,
                    buffersize=1))
    eq_((
        HIST_HEADER,
        ('C1', 'gold', 'BJ', 1, None),
        ('C2', 'silver', 'SH', 1, 2),
        ('C3', 'bronze', 'GZ', 2, None),
    ), tuple(out))


def test_scd2_reiterable_view_serves_consistent_rows():
    view = scd2(_history([('C1', 'gold', 'BJ', 1, None)]),
                _snapshot([('C1', 'silver', 'SH')]), KEY, TRACKED,
                VALID_FROM, VALID_TO, 2)
    eq_(list(view), list(view))


def test_scd2_spill_files_reclaimed_on_full_iteration():
    tempdir = tempfile.mkdtemp(prefix='scd2-test-')
    try:
        hist_rows = [('C%d' % i, 'gold', 'BJ', 1, None) for i in range(12)]
        snap_rows = [('C%d' % i, 'silver', 'SH') for i in range(12)]
        view = scd2(_history(hist_rows), _snapshot(snap_rows), KEY, TRACKED,
                    VALID_FROM, VALID_TO, 2, buffersize=3, tempdir=tempdir)
        iterator = iter(view)
        next(iterator)
        next(iterator)
        # overflow is spooled while the input does not fit the buffer
        assert glob.glob(os.path.join(tempdir, '*'))
        iterator.close()
        eq_(25, len(list(view)))
        # the spill cache survives a full pass so the view stays
        # re-iterable, but is removed once the view is garbage collected
        assert glob.glob(os.path.join(tempdir, '*'))
        del view
        gc.collect()
        eq_([], glob.glob(os.path.join(tempdir, '*')))
    finally:
        if os.path.exists(tempdir):
            os.rmdir(tempdir)


def test_scd2_spill_files_reclaimed_on_early_close():
    tempdir = tempfile.mkdtemp(prefix='scd2-test-')
    try:
        hist_rows = [('C%d' % i, 'gold', 'BJ', 1, None) for i in range(12)]
        snap_rows = [('C%d' % i, 'silver', 'SH') for i in range(12)]
        view = scd2(_history(hist_rows), _snapshot(snap_rows), KEY, TRACKED,
                    VALID_FROM, VALID_TO, 2, buffersize=3, tempdir=tempdir)
        iterator = iter(view)
        next(iterator)
        next(iterator)
        assert glob.glob(os.path.join(tempdir, '*'))
        iterator.close()
        del view
        del iterator
        gc.collect()
        eq_([], glob.glob(os.path.join(tempdir, '*')))
    finally:
        if os.path.exists(tempdir):
            os.rmdir(tempdir)


def test_scd2_spill_files_reclaimed_on_error():
    tempdir = tempfile.mkdtemp(prefix='scd2-test-')
    try:
        hist_rows = [('C%d' % i, 'gold', 'BJ', 1, None) for i in range(12)]

        def broken_snapshot():
            yield SNAP_HEADER
            for i in range(12):
                yield ('C%d' % i, 'silver', 'SH')
            yield ('C3', 'duplicate', 'XX')

        view = scd2(_history(hist_rows), broken_snapshot(), KEY, TRACKED,
                    VALID_FROM, VALID_TO, 2, buffersize=3, tempdir=tempdir)
        with pytest.raises(DuplicateKeyError):
            list(view)
        del view
        gc.collect()
        eq_([], glob.glob(os.path.join(tempdir, '*')))
    finally:
        if os.path.exists(tempdir):
            os.rmdir(tempdir)


def test_scd2_requires_header_rows():
    with pytest.raises(SCD2Error):
        list(scd2([], _snapshot([('C1', 'gold', 'BJ')]), KEY, TRACKED,
                  VALID_FROM, VALID_TO, 1))
    with pytest.raises(SCD2Error):
        list(scd2(_history([('C1', 'gold', 'BJ', 1, None)]), [], KEY,
                  TRACKED, VALID_FROM, VALID_TO, 2))


def test_scd2_rejects_empty_interval_in_history():
    hist = _history([('C1', 'gold', 'BJ', 3, 3)])
    with pytest.raises(SCD2Error):
        list(scd2(hist, _snapshot([('C1', 'gold', 'BJ')]), KEY, TRACKED,
                  VALID_FROM, VALID_TO, 4))


def test_scd2_data_row_equal_to_dummy_header_is_not_skipped():
    # the internal positional dummy header looks like string data; a real
    # history row carrying those exact values must still be processed
    weird = ('scd2_f0', 'x', 'y', 1, None)
    out = list(scd2(_history([weird]),
                    _snapshot([('scd2_f0', 'x', 'y')]), KEY, TRACKED,
                    VALID_FROM, VALID_TO, 2))
    eq_(2, len(out))
    eq_(weird, out[1])


def test_scd2_identical_duplicate_snapshot_rows_are_rejected():
    snapshot = _snapshot([
        ('C1', 'gold', 'BJ'),
        ('C1', 'gold', 'BJ'),
    ])
    with pytest.raises(DuplicateKeyError):
        list(scd2(_history([]), snapshot, KEY, TRACKED,
                  VALID_FROM, VALID_TO, 1))


def test_scd2_unchanged_snapshot_row_keeps_single_version():
    hist = _history([('C1', 'gold', 'BJ', 1, None)])
    out = list(scd2(hist, _snapshot([('C1', 'gold', 'BJ')]), KEY, TRACKED,
                    VALID_FROM, VALID_TO, 2))
    eq_((
        HIST_HEADER,
        ('C1', 'gold', 'BJ', 1, None),
    ), tuple(out))
