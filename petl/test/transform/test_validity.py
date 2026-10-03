from __future__ import absolute_import, print_function, division


import pytest

import petl as etl
from petl.errors import ArgumentError
from petl.test.helpers import eq_, ieq


def test_validityjoin_compound_boundaries_ordering_and_left_preservation():

    events = (('tenant', 'product', 'event_id', 'time', 'note'),
              ('a', 'x', 'e0', 0, 'before-start'),
              ('a', 'x', 'e4', 1, 'at-start'),
              ('a', 'x', 'e1', 4, 'boundary'),
              ('a', 'x', 'e1', 4, 'duplicate-boundary'),
              ('a', 'x', 'e2', 5, 'overlap'),
              ('a', 'y', 'e3', 5, 'other-product'),
              ('b', 'x', 'e5', 5, 'other-tenant'))
    rules = (('tenant', 'product', 'rule_id', 'start', 'end', 'version'),
             ('a', 'x', 'old', 1, 4, 'v1'),
             ('a', 'x', 'nested', 4, 5, 'v2-nested'),
             ('a', 'x', 'current', 4, None, 'v2-open'),
             ('a', 'x', 'wide', 2, 8, 'wide'),
             ('a', 'y', 'only-y', 1, None, 'y'),
             ('b', 'x', 'only-b', 1, None, 'b'))

    actual = etl.validityjoin(events, rules, key=('tenant', 'product'),
                              time='time')
    expect = (
        ('tenant', 'product', 'event_id', 'time', 'note',
         'rule_id', 'start', 'end', 'version'),
        ('a', 'x', 'e0', 0, 'before-start', None, None, None, None),
        ('a', 'x', 'e4', 1, 'at-start', 'old', 1, 4, 'v1'),
        ('a', 'x', 'e1', 4, 'boundary', 'nested', 4, 5, 'v2-nested'),
        ('a', 'x', 'e1', 4, 'boundary', 'current', 4, None, 'v2-open'),
        ('a', 'x', 'e1', 4, 'boundary', 'wide', 2, 8, 'wide'),
        ('a', 'x', 'e1', 4, 'duplicate-boundary', 'nested', 4, 5,
         'v2-nested'),
        ('a', 'x', 'e1', 4, 'duplicate-boundary', 'current', 4, None,
         'v2-open'),
        ('a', 'x', 'e1', 4, 'duplicate-boundary', 'wide', 2, 8, 'wide'),
        ('a', 'x', 'e2', 5, 'overlap', 'current', 4, None, 'v2-open'),
        ('a', 'x', 'e2', 5, 'overlap', 'wide', 2, 8, 'wide'),
        ('a', 'y', 'e3', 5, 'other-product', 'only-y', 1, None, 'y'),
        ('b', 'x', 'e5', 5, 'other-tenant', 'only-b', 1, None, 'b'))
    ieq(expect, actual)
    ieq(expect, actual)


def test_validityjoin_presorted_scans_rules_once():

    class CountingRules(object):

        def __init__(self, rows):
            self.rows = rows
            self.data_rows_read = 0

        def __iter__(self):
            it = iter(self.rows)
            yield next(it)
            for row in it:
                self.data_rows_read += 1
                yield row

    events = (('tenant', 'event_id', 'time'),
              ('a', 'e1', 2),
              ('a', 'e2', 4),
              ('a', 'e3', 5),
              ('b', 'e4', 1))
    rules_rows = (('tenant', 'rule_id', 'start', 'end'),
                  ('a', 'early', 1, 4),
                  ('a', 'nested', 2, 3),
                  ('a', 'current', 4, None),
                  ('b', 'only-b', 1, None))
    rules = CountingRules(rules_rows)

    actual = etl.validityjoin(events, rules, key='tenant', time='time',
                              presorted=True)
    expect = (('tenant', 'event_id', 'time', 'rule_id', 'start', 'end'),
              ('a', 'e1', 2, 'early', 1, 4),
              ('a', 'e1', 2, 'nested', 2, 3),
              ('a', 'e2', 4, 'current', 4, None),
              ('a', 'e3', 5, 'current', 4, None),
              ('b', 'e4', 1, 'only-b', 1, None))
    ieq(expect, actual)
    eq_(4, rules.data_rows_read)


def test_validityjoin_separate_keys_prefixes_and_conflicting_fields():

    events = (('cust', 'at', 'label'),
              ('a', 2, 'event-label'))
    rules = (('customer', 'from', 'to', 'label'),
             ('a', 1, None, 'rule-label'))
    expect = (('e_cust', 'e_at', 'e_label', 'r_from', 'r_to', 'r_label'),
              ('a', 2, 'event-label', 1, None, 'rule-label'))

    actual = etl.validityjoin(events, rules, lkey='cust', rkey='customer',
                              time='at', rstart='from', rend='to',
                              lprefix='e_', rprefix='r_')
    ieq(expect, actual)


def test_validityjoin_invalid_rows_can_be_rejected_or_ignored():

    events = (('tenant', 'event_id', 'time'),
              ('a', 'valid', 2),
              ('a', 'missing-time', None))
    rules = (('tenant', 'rule_id', 'start', 'end'),
             ('a', 'valid', 1, None),
             ('a', 'missing-start', None, 5),
             ('a', 'reversed', 5, 4))

    with pytest.raises(ArgumentError):
        list(etl.validityjoin(events, rules, key='tenant'))
    with pytest.raises(ArgumentError):
        list(etl.validityjoin(events[:2], rules, key='tenant',
                              invalid_time='ignore'))
    with pytest.raises(ArgumentError):
        list(etl.validityjoin(events, rules[:2], key='tenant'))

    actual = etl.validityjoin(events, rules, key='tenant',
                              invalid_time='ignore',
                              invalid_interval='ignore')
    expect = (('tenant', 'event_id', 'time', 'rule_id', 'start', 'end'),
              ('a', 'valid', 2, 'valid', 1, None),
              ('a', 'missing-time', None, None, None, None))
    ieq(expect, actual)
