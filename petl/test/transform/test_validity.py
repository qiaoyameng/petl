from __future__ import absolute_import, print_function, division

from datetime import date

import pytest

import petl as etl
from petl.errors import ArgumentError
from petl.test.helpers import ieq


def test_validityjoin_boundaries_nesting_and_open_end():
    events = (
        ('id', 'event_time', 'event_name'),
        ('a', date(2024, 1, 1), 'start'),
        ('a', date(2024, 1, 2), 'inside'),
        ('a', date(2024, 1, 4), 'boundary'),
        ('a', date(2024, 2, 1), 'later'),
        ('b', date(2024, 1, 2), 'other'),
        ('c', date(2024, 1, 2), 'missing'),
    )
    rules = (
        ('id', 'start', 'end', 'rule'),
        ('a', date(2024, 1, 1), date(2024, 1, 4), 'v1'),
        ('a', date(2024, 1, 2), date(2024, 1, 3), 'nested'),
        ('a', date(2024, 1, 4), None, 'open'),
    )
    expect = (
        ('id', 'event_time', 'event_name', 'id', 'start', 'end', 'rule'),
        ('a', date(2024, 1, 1), 'start', 'a', date(2024, 1, 1),
         date(2024, 1, 4), 'v1'),
        ('a', date(2024, 1, 2), 'inside', 'a', date(2024, 1, 1),
         date(2024, 1, 4), 'v1'),
        ('a', date(2024, 1, 2), 'inside', 'a', date(2024, 1, 2),
         date(2024, 1, 3), 'nested'),
        ('a', date(2024, 1, 4), 'boundary', 'a', date(2024, 1, 4), None, 'open'),
        ('a', date(2024, 2, 1), 'later', 'a', date(2024, 1, 4), None, 'open'),
        ('b', date(2024, 1, 2), 'other', None, None, None, None),
        ('c', date(2024, 1, 2), 'missing', None, None, None, None),
    )

    actual = etl.validityjoin(events, rules, key='id', time='event_time')
    ieq(expect, actual)
    ieq(expect, actual)


def test_validityjoin_presorted_matches_sorted_scan():
    events = (
        ('tenant', 'product', 'event_time', 'event_name'),
        ('a', 'x', 1, 'a1'),
        ('a', 'x', 2, 'a2'),
        ('a', 'x', 4, 'a4'),
        ('a', 'x', 4, 'a4-duplicate'),
        ('a', 'y', 2, 'ay2'),
        ('b', 'x', 2, 'b2'),
    )
    rules = (
        ('tenant', 'product', 'start', 'end', 'rule'),
        ('a', 'x', 0, 4, 'closed'),
        ('a', 'x', 2, 3, 'nested'),
        ('a', 'x', 4, None, 'open'),
        ('b', 'x', 1, None, 'b-open'),
    )

    unsorted = etl.validityjoin(
        events, rules, key=('tenant', 'product'), time='event_time'
    )
    sorted_scan = etl.validityjoin(
        events, rules, key=('tenant', 'product'), time='event_time',
        presorted=True
    )
    ieq(unsorted, sorted_scan)
    ieq(unsorted, sorted_scan)


def test_validityjoin_presorted_scans_rules_once():
    class CountingRules(object):
        def __init__(self):
            self.iterations = 0
            self.next_calls = 0

        def __iter__(self):
            self.iterations += 1
            rows = (
                ('id', 'start', 'end', 'rule'),
                ('a', 0, 5, 'active'),
                ('b', 6, None, 'later'),
            )
            for row in rows:
                self.next_calls += 1
                yield row

    events = (
        ('id', 'time'),
        ('a', 1),
        ('a', 2),
        ('a', 5),
        ('b', 6),
    )
    rules = CountingRules()

    result = etl.validityjoin(events, rules, key='id', presorted=True)
    [row for row in result]

    assert rules.iterations == 1
    assert rules.next_calls == 3


def test_validityjoin_restores_event_order_and_rule_order():
    events = (
        ('id', 'event_time', 'n'),
        ('a', 5, 1),
        ('a', 1, 2),
        ('a', 3, 3),
        ('a', 3, 4),
    )
    rules = (
        ('id', 'start', 'end', 'rule'),
        ('a', 0, 10, 'first'),
        ('a', 2, 8, 'second'),
        ('a', 3, 4, 'third'),
    )

    rows = list(etl.validityjoin(events, rules, key='id', time='event_time'))
    assert [row[2] for row in rows[1:]] == [
        1, 1, 2, 3, 3, 3, 4, 4, 4
    ]
    assert [row[-1] for row in rows if row[2] == 3 and row[1] == 3] == [
        'first', 'second', 'third'
    ]


def test_validityjoin_field_prefixes():
    events = (('id', 'time'), ('a', 1))
    rules = (('id', 'start', 'end'), ('a', 0, None))

    actual = etl.validityjoin(
        events, rules, key='id', lprefix='event_', rprefix='rule_'
    )
    expect = (
        ('event_id', 'event_time', 'rule_id', 'rule_start', 'rule_end'),
        ('a', 1, 'a', 0, None),
    )
    ieq(expect, actual)


def test_validityjoin_without_business_key():
    events = (('time', 'event'), (2, 'event'))
    rules = (
        ('start', 'end', 'rule'),
        (0, 5, 'global'),
        (2, None, 'overlapping-global'),
    )

    actual = etl.validityjoin(events, rules)
    expect = (
        ('time', 'event', 'start', 'end', 'rule'),
        (2, 'event', 0, 5, 'global'),
        (2, 'event', 2, None, 'overlapping-global'),
    )
    ieq(expect, actual)


def test_validityjoin_rejects_missing_event_time():
    events = (('id', 'time'), ('a', None))
    rules = (('id', 'start', 'end'), ('a', 0, None))

    with pytest.raises(ValueError, match='missing event time'):
        list(etl.validityjoin(events, rules, key='id'))

    actual = etl.validityjoin(
        events, rules, key='id', missing_time='skip'
    )
    ieq((('id', 'time', 'id', 'start', 'end'),), actual)


def test_validityjoin_rejects_invalid_intervals():
    events = (('id', 'time'), ('a', 1), ('b', 1))
    rules = (
        ('id', 'start', 'end'),
        ('a', 2, 1),
        ('a', None, None),
        ('b', 0, None),
    )

    with pytest.raises(ValueError, match='invalid rule interval'):
        list(etl.validityjoin(events, rules, key='id'))

    expect = (
        ('id', 'time', 'id', 'start', 'end'),
        ('a', 1, None, None, None),
        ('b', 1, 'b', 0, None),
    )
    actual = etl.validityjoin(
        events, rules, key='id', invalid_interval='skip'
    )
    ieq(expect, actual)


def test_validityjoin_skips_incompatible_coordinates():
    events = (('id', 'time'), ('a', date(2024, 1, 1)), ('b', 1))
    rules = (
        ('id', 'start', 'end', 'rule'),
        ('a', 0, None, 'wrong-type'),
        ('b', 0, None, 'compatible'),
    )

    with pytest.raises(ValueError, match='incompatible'):
        [row for row in etl.validityjoin(events, rules, key='id')]

    actual = etl.validityjoin(
        events, rules, key='id', invalid_interval='skip'
    )
    expect = (
        ('id', 'time', 'id', 'start', 'end', 'rule'),
        ('a', date(2024, 1, 1), None, None, None, None),
        ('b', 1, 'b', 0, None, 'compatible'),
    )
    ieq(expect, actual)


def test_validityjoin_validates_policy_arguments():
    events = (('id', 'time'),)
    rules = (('id', 'start', 'end'),)

    with pytest.raises(ArgumentError):
        etl.validityjoin(events, rules, key='id', missing_time='drop')
    with pytest.raises(ArgumentError):
        etl.validityjoin(
            events, rules, key='id', invalid_interval='drop'
        )
