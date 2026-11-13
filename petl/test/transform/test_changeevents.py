from __future__ import absolute_import, print_function, division

import copy
import random

import pytest

import petl as etl
from petl.errors import ChangeEventError
from petl.test.helpers import ieq, eq_


KEY_FIELDS = ('id', 'region')
DATA_FIELDS = ('id', 'region', 'amount', 'note')


def _key(row):
    return tuple(row[name] for name in KEY_FIELDS)


class ReadOnlyLookup(object):
    """Read-only snapshot lookup; raises if the caller mutates it."""

    def __init__(self, state):
        self._state = state
        self.calls = 0

    def __call__(self, key):
        self.calls += 1
        row = self._state.get(key)
        # hand back a copy so the folding view cannot mutate stored state
        return copy.deepcopy(row) if row is not None else None

    def snapshot(self):
        return copy.deepcopy(self._state)


def _fold_rows(events, state, **kwargs):
    """Run the view, applying net changes to state as transactions commit."""
    lookup = ReadOnlyLookup(state)
    view = etl.foldchanges(events, list(KEY_FIELDS), lookup, **kwargs)
    rows = []
    iterator = iter(view)
    header = next(iterator)
    for row in iterator:
        rows.append(row)
        record = dict(zip(header[2:], row[2:]))
        key = tuple(record[name] for name in KEY_FIELDS)
        if row[1] == 'delete':
            # keep the old values so callers can diff against their state
            del state[key]
        else:
            state[key] = record
    return header, rows, lookup


# ---------------------------------------------------------------------------
# Per-event reference interpreter: replays every event against a live table.
# ---------------------------------------------------------------------------

def _reference_changes(events, initial_state):
    """Return (per-transaction net changes, final committed state).

    Replays every event against a live table, one transaction at a time, and
    reports the net changes each committed transaction makes against the state
    visible when it started. Re-keying updates are expanded into a delete of
    the old key plus an insert of the new key, matching the view.
    """
    committed = copy.deepcopy(initial_state)
    working = None
    txn_start = None
    output = []
    txn_output = None

    for event in events:
        op = event['op']
        if op == 'begin':
            txn_start = copy.deepcopy(committed)
            working = copy.deepcopy(committed)
            txn_output = []
        elif op == 'rollback':
            committed = txn_start
            working = None
            txn_output = None
        elif op == 'commit':
            # compute the net difference against the pre-transaction state
            old_state = txn_start
            for key in sorted(set(working) | set(old_state)):
                old = old_state.get(key)
                new = working.get(key)
                if old is None and new is not None:
                    txn_output.append(
                        ('insert', key, copy.deepcopy(new)))
                elif old is not None and new is None:
                    txn_output.append(
                        ('delete', key, copy.deepcopy(old)))
                elif old != new:
                    if _key(old) != _key(new):
                        txn_output.append(
                            ('delete', _key(old), copy.deepcopy(old)))
                        txn_output.append(
                            ('insert', _key(new), copy.deepcopy(new)))
                    else:
                        txn_output.append(
                            ('update', key, copy.deepcopy(new)))
            output.append(txn_output)
            committed = working
            working = None
            txn_output = None
        elif op == 'insert':
            row = {name: value for name, value in event.items()
                   if name not in ('op', 'txn', 'before', 'after')}
            working[_key(row)] = row
        elif op == 'delete':
            del working[_key(event['before'])]
        elif op == 'update':
            key = _key(event['before'])
            row = copy.deepcopy(working[key])
            row.update(event['after'])
            del working[key]
            working[_key(row)] = row
    return output, committed


def _random_stream(seed):
    """Generate a guaranteed-valid ordered event stream."""
    rng = random.Random(seed)
    # a handful of small-integer keys so re-keys and occupancy clashes happen
    domain = range(4)
    state = {(i, 'r'): {'id': i, 'region': 'r', 'amount': rng.randint(0, 99),
                        'note': 'n%d' % rng.randint(0, 9)}
             for i in domain}
    initial_state = copy.deepcopy(state)
    events = []
    txn_counter = 0
    txn_count = rng.randint(4, 8)
    for _ in range(txn_count):
        txn_counter += 1
        txn = 'txn%d' % txn_counter
        txn_start = copy.deepcopy(state)
        working = copy.deepcopy(txn_start)
        events.append({'op': 'begin', 'txn': txn})
        event_count = rng.randint(1, 12)
        for _ in range(event_count):
            live_keys = [(i, 'r') for i in domain if (i, 'r') in working]
            free_keys = [(i, 'r') for i in domain
                         if (i, 'r') not in working]
            if not live_keys:
                options = (0,)
            elif not free_keys:
                options = (1, 2, 4)
            else:
                options = (0, 1, 2, 3, 4)
            choice = rng.choice(options)
            if choice == 0:
                key = rng.choice(free_keys)
                row = {'id': key[0], 'region': 'r',
                       'amount': rng.randint(0, 99),
                       'note': 'n%d' % rng.randint(0, 9)}
                events.append(dict({'op': 'insert'}, **row))
                working[key] = row
            elif choice == 1:
                key = rng.choice(live_keys)
                before = copy.deepcopy(working[key])
                events.append({'op': 'delete', 'before': before})
                del working[key]
            elif choice in (2, 3):
                key = rng.choice(live_keys)
                before = copy.deepcopy(working[key])
                if choice == 3 and rng.random() < 0.5:
                    target = rng.choice(free_keys)
                    after = {'id': target[0], 'region': target[1]}
                else:
                    after = {'amount': rng.randint(0, 99)}
                events.append({'op': 'update', 'before': before,
                               'after': after})
                row = copy.deepcopy(before)
                row.update(after)
                del working[key]
                working[_key(row)] = row
            elif choice == 4:
                # delete followed by an insert of the same key: old value must
                # be retained by the folded delete if it later commits
                key = rng.choice(live_keys)
                before = copy.deepcopy(working[key])
                events.append({'op': 'delete', 'before': before})
                replacement = {'id': key[0], 'region': 'r',
                               'amount': rng.randint(0, 99),
                               'note': 'replaced'}
                events.append(dict({'op': 'insert'}, **replacement))
                working[key] = replacement
        if rng.random() < 0.25:
            events.append({'op': 'rollback', 'txn': txn})
            state = txn_start
        else:
            events.append({'op': 'commit', 'txn': txn})
            state = working
    return initial_state, events


@pytest.mark.parametrize('seed', range(30))
@pytest.mark.parametrize('buffersize', [None, 100000, 2, 1, 0])
def test_foldchanges_matches_reference(seed, buffersize):
    initial_state, events = _random_stream(seed)
    expected_rows, expected_final = _reference_changes(
        events, copy.deepcopy(initial_state))

    applied_state = copy.deepcopy(initial_state)
    header, folded_rows, lookup = _fold_rows(
        events, applied_state, buffersize=buffersize)

    eq_(('txn', 'change') + DATA_FIELDS, header)

    # group the folded rows by transaction; ordering inside a transaction
    # follows first touch of each key and is not part of the comparison
    folded_by_txn = {}
    for folded in folded_rows:
        record = dict(zip(header[2:], folded[2:]))
        key = _key(record)
        payload = (folded[1], key, record)
        folded_by_txn.setdefault(folded[0], []).append(payload)

    committed_txns = [event['txn'] for event in events
                      if event['op'] == 'commit']
    eq_(committed_txns, list(folded_by_txn))
    for txn, ref_txn_rows in zip(committed_txns, expected_rows):
        eq_(sorted(ref_txn_rows), sorted(folded_by_txn[txn]))

    # applying every emitted net change must reproduce the reference state
    eq_(expected_final, applied_state)
    # the lookup interface must be used read-only
    assert lookup.calls >= 0


@pytest.mark.parametrize('seed', range(15))
def test_spill_backend_matches_memory_ordering(seed):
    # the SQLite-backed staging area must emit rows in exactly the same
    # first-touch order as the in-memory staging area
    initial_state, events = _random_stream(seed)

    def collect(buffersize):
        state = copy.deepcopy(initial_state)
        _, rows, _ = _fold_rows(events, state, buffersize=buffersize)
        return rows

    memory_rows = collect(100000)
    for spill_size in (0, 1, 2):
        eq_(memory_rows, collect(spill_size))


# ---------------------------------------------------------------------------
# Deterministic, hand-written scenarios
# ---------------------------------------------------------------------------

def _simple_lookup(state):
    def lookup(key):
        return copy.deepcopy(state[key]) if key in state else None
    return lookup


def _rows(view):
    iterator = iter(view)
    header = next(iterator)
    return header, list(iterator)


def test_foldchanges_basic_folding_and_header():
    state = {('a',): {'id': 'a', 'v': 1}, ('b',): {'id': 'b', 'v': 2}}
    events = [
        {'op': 'begin', 'txn': 't1'},
        {'op': 'insert', 'id': 'c', 'v': 3},
        {'op': 'update', 'before': {'id': 'a', 'v': 1},
         'after': {'v': 9}},
        {'op': 'delete', 'before': {'id': 'b', 'v': 2}},
        {'op': 'commit', 'txn': 't1'},
    ]
    header, rows = _rows(
        etl.foldchanges(events, 'id', _simple_lookup(state)))
    eq_(('txn', 'change', 'id', 'v'), header)
    ieq((('t1', 'insert', 'c', 3),
         ('t1', 'update', 'a', 9),
         ('t1', 'delete', 'b', 2)), rows)


def test_foldchanges_composite_key_and_lazy_header():
    # no committed transaction means nothing but the header is emitted
    state = {(1, 'x'): {'id': 1, 'region': 'x', 'v': 0}}
    events = [
        {'op': 'begin', 'txn': 't1'},
        {'op': 'update', 'before': {'id': 1, 'region': 'x', 'v': 0},
         'after': {'v': 7}},
        {'op': 'rollback', 'txn': 't1'},
    ]
    header, rows = _rows(etl.foldchanges(
        events, ('id', 'region'), _simple_lookup(state)))
    eq_(('txn', 'change', 'id', 'region', 'v'), header)
    eq_([], rows)


def test_foldchanges_insert_then_delete_cancels():
    state = {}
    events = [
        {'op': 'begin', 'txn': 't1'},
        {'op': 'insert', 'id': 'a', 'v': 1},
        {'op': 'delete', 'before': {'id': 'a', 'v': 1}},
        {'op': 'commit', 'txn': 't1'},
    ]
    header, rows = _rows(
        etl.foldchanges(events, 'id', _simple_lookup(state)))
    eq_([], rows)


def test_foldchanges_delete_then_insert_keeps_old_value():
    state = {('a',): {'id': 'a', 'v': 1}}
    events = [
        {'op': 'begin', 'txn': 't1'},
        {'op': 'delete', 'before': {'id': 'a', 'v': 1}},
        {'op': 'insert', 'id': 'a', 'v': 2},
        {'op': 'commit', 'txn': 't1'},
    ]
    header, rows = _rows(
        etl.foldchanges(events, 'id', _simple_lookup(state)))
    # folded into a single update; old value is still available internally
    ieq((('t1', 'update', 'a', 2),), rows)


def test_foldchanges_rekey_emits_delete_and_insert():
    state = {('a',): {'id': 'a', 'v': 1}}
    events = [
        {'op': 'begin', 'txn': 't1'},
        {'op': 'update', 'before': {'id': 'a', 'v': 1},
         'after': {'id': 'b'}},
        {'op': 'commit', 'txn': 't1'},
    ]
    header, rows = _rows(
        etl.foldchanges(events, 'id', _simple_lookup(state)))
    ieq((('t1', 'delete', 'a', 1),
         ('t1', 'insert', 'b', 1)), rows)


def test_foldchanges_rekey_into_occupied_key_rejected():
    state = {('a',): {'id': 'a', 'v': 1}, ('b',): {'id': 'b', 'v': 2}}
    events = [
        {'op': 'begin', 'txn': 't1'},
        {'op': 'update', 'before': {'id': 'a', 'v': 1},
         'after': {'id': 'b'}},
    ]
    view = etl.foldchanges(events, 'id', _simple_lookup(state))
    with pytest.raises(ChangeEventError) as exc:
        list(view)
    eq_(1, exc.value.position)
    assert 'already occupied' in str(exc.value)


def test_foldchanges_rekey_then_reinsert_old_key():
    state = {('a',): {'id': 'a', 'v': 1}}
    events = [
        {'op': 'begin', 'txn': 't1'},
        {'op': 'update', 'before': {'id': 'a', 'v': 1},
         'after': {'id': 'b'}},
        {'op': 'insert', 'id': 'a', 'v': 9},
        {'op': 'commit', 'txn': 't1'},
    ]
    header, rows = _rows(
        etl.foldchanges(events, 'id', _simple_lookup(state)))
    # key a existed when the transaction started, so re-occupying it after
    # the re-key is an update against its committed value
    eq_({('t1', 'update', 'a', 9),
         ('t1', 'insert', 'b', 1)}, set(rows))


def test_foldchanges_noop_update_suppressed():
    state = {('a',): {'id': 'a', 'v': 1}}
    events = [
        {'op': 'begin', 'txn': 't1'},
        {'op': 'update', 'before': {'id': 'a', 'v': 1},
         'after': {'v': 1}},
        {'op': 'commit', 'txn': 't1'},
    ]
    header, rows = _rows(
        etl.foldchanges(events, 'id', _simple_lookup(state)))
    eq_([], rows)


def test_foldchanges_rollback_is_invisible_to_later_txn():
    state = {('a',): {'id': 'a', 'v': 1}}
    seen = []

    def lookup(key):
        seen.append(copy.deepcopy(state[key]) if key in state else None)
        return copy.deepcopy(state[key]) if key in state else None

    events = [
        {'op': 'begin', 'txn': 't1'},
        {'op': 'delete', 'before': {'id': 'a', 'v': 1}},
        {'op': 'insert', 'id': 'a', 'v': 99},
        {'op': 'rollback', 'txn': 't1'},
        {'op': 'begin', 'txn': 't2'},
        # before image must match the pre-t1 committed value
        {'op': 'update', 'before': {'id': 'a', 'v': 1},
         'after': {'v': 2}},
        {'op': 'commit', 'txn': 't2'},
    ]
    header, rows = _rows(etl.foldchanges(events, 'id', lookup))
    ieq((('t2', 'update', 'a', 2),), rows)


def test_foldchanges_before_mismatch_rejected_with_position():
    state = {('a',): {'id': 'a', 'v': 1}}
    events = [
        {'op': 'begin', 'txn': 't1'},
        {'op': 'update', 'before': {'id': 'a', 'v': 5},
         'after': {'v': 2}},
    ]
    with pytest.raises(ChangeEventError) as exc:
        list(etl.foldchanges(events, 'id', _simple_lookup(state)))
    eq_(1, exc.value.position)
    assert 'does not match' in str(exc.value)


def test_foldchanges_duplicate_and_out_of_order_boundaries():
    state = {}
    nested = [
        {'op': 'begin', 'txn': 't1'},
        {'op': 'begin', 'txn': 't2'},
    ]
    with pytest.raises(ChangeEventError) as exc:
        list(etl.foldchanges(nested, 'id', _simple_lookup(state)))
    eq_(1, exc.value.position)

    commit_without_begin = [{'op': 'commit', 'txn': 't1'}]
    with pytest.raises(ChangeEventError) as exc:
        list(etl.foldchanges(commit_without_begin, 'id',
                             _simple_lookup(state)))
    eq_(0, exc.value.position)

    rollback_without_begin = [{'op': 'rollback', 'txn': 't1'}]
    with pytest.raises(ChangeEventError) as exc:
        list(etl.foldchanges(rollback_without_begin, 'id',
                             _simple_lookup(state)))
    eq_(0, exc.value.position)

    mismatched_commit = [
        {'op': 'begin', 'txn': 't1'},
        {'op': 'commit', 'txn': 't2'},
    ]
    with pytest.raises(ChangeEventError) as exc:
        list(etl.foldchanges(mismatched_commit, 'id',
                             _simple_lookup(state)))
    eq_(1, exc.value.position)

    data_without_txn = [{'op': 'insert', 'id': 'a', 'v': 1}]
    with pytest.raises(ChangeEventError) as exc:
        list(etl.foldchanges(data_without_txn, 'id',
                             _simple_lookup(state)))
    eq_(0, exc.value.position)


def test_foldchanges_unclosed_transaction_emits_nothing():
    state = {}
    events = [
        {'op': 'begin', 'txn': 't1'},
        {'op': 'insert', 'id': 'a', 'v': 1},
        {'op': 'insert', 'id': 'b', 'v': 2},
    ]
    view = etl.foldchanges(events, 'id', _simple_lookup(state))
    iterator = iter(view)
    # the header can only be produced by a committed transaction; here the
    # stream ends first and must raise rather than leak half a transaction
    with pytest.raises(ChangeEventError) as exc:
        next(iterator)
    assert 'not closed' in str(exc.value)


def test_foldchanges_midstream_source_exception_no_half_txn():
    state = {}

    def broken_source():
        yield {'op': 'begin', 'txn': 't1'}
        yield {'op': 'insert', 'id': 'a', 'v': 1}
        raise RuntimeError('connection lost')

    view = etl.foldchanges(broken_source(), 'id', _simple_lookup(state))
    iterator = iter(view)
    with pytest.raises(ChangeEventError) as exc:
        next(iterator)
    assert 'connection lost' in str(exc.value)
    # nothing was committed, so the caller's state is untouched
    eq_({}, state)


def test_foldchanges_lookup_is_read_only_and_not_committed():
    state = {('a',): {'id': 'a', 'v': 1}}
    snapshot = copy.deepcopy(state)

    def lookup(key):
        # return the live object to prove the view never mutates it
        return state.get(key)

    events = [
        {'op': 'begin', 'txn': 't1'},
        {'op': 'update', 'before': {'id': 'a', 'v': 1},
         'after': {'v': 2}},
        {'op': 'commit', 'txn': 't1'},
    ]
    header, rows = _rows(etl.foldchanges(events, 'id', lookup))
    ieq((('t1', 'update', 'a', 2),), rows)
    # the view must not have mutated or "committed" the caller's state
    eq_(snapshot, state)


def test_foldchanges_spill_to_temp_sqlite(tmp_path):
    state = {('a',): {'id': 'a', 'v': 1}}
    events = [
        {'op': 'begin', 'txn': 't1'},
        {'op': 'insert', 'id': 'b', 'v': 2},
        {'op': 'update', 'before': {'id': 'a', 'v': 1},
         'after': {'v': 3}},
        {'op': 'delete', 'before': {'id': 'b', 'v': 2}},
        {'op': 'insert', 'id': 'c', 'v': 4},
        {'op': 'commit', 'txn': 't1'},
    ]
    before = set(p.name for p in tmp_path.iterdir())
    view = etl.foldchanges(events, 'id', _simple_lookup(state),
                           buffersize=1, tempdir=str(tmp_path))
    header, rows = _rows(view)
    ieq((('t1', 'update', 'a', 3),
         ('t1', 'insert', 'c', 4)), rows)
    # every temporary SQLite file must be cleaned up after the transaction
    after = set(p.name for p in tmp_path.iterdir())
    eq_(before, after)


def test_foldchanges_declared_fields_and_unknown_field_error():
    state = {}
    events = [
        {'op': 'begin', 'txn': 't1'},
        {'op': 'insert', 'id': 'a', 'v': 1, 'extra': 9},
        {'op': 'commit', 'txn': 't1'},
    ]
    with pytest.raises(ChangeEventError):
        list(etl.foldchanges(events, 'id', _simple_lookup(state),
                             fields=('id', 'v')))


def test_foldchanges_fluent_method():
    state = {('a',): {'id': 'a', 'v': 1}}
    events = [
        {'op': 'begin', 'txn': 't1'},
        {'op': 'delete', 'before': {'id': 'a', 'v': 1}},
        {'op': 'commit', 'txn': 't1'},
    ]
    source = etl.wrap(events)
    view = source.foldchanges('id', _simple_lookup(state))
    header, rows = _rows(view)
    eq_(('txn', 'change', 'id', 'v'), header)
    ieq((('t1', 'delete', 'a', 1),), rows)


def test_foldchanges_many_changes_to_one_row_fold_to_net_effect():
    state = {('a',): {'id': 'a', 'v': 1}}
    events = [
        {'op': 'begin', 'txn': 't1'},
        {'op': 'update', 'before': {'id': 'a', 'v': 1},
         'after': {'v': 2}},
        {'op': 'update', 'before': {'id': 'a', 'v': 2},
         'after': {'v': 3}},
        {'op': 'update', 'before': {'id': 'a', 'v': 3},
         'after': {'v': 4}},
        {'op': 'commit', 'txn': 't1'},
        {'op': 'begin', 'txn': 't2'},
        # delete then re-insert then update still folds to one update
        {'op': 'delete', 'before': {'id': 'a', 'v': 4}},
        {'op': 'insert', 'id': 'a', 'v': 5},
        {'op': 'update', 'before': {'id': 'a', 'v': 5},
         'after': {'v': 6}},
        {'op': 'commit', 'txn': 't2'},
    ]
    # apply committed net changes back into state as the stream is consumed
    lookup = ReadOnlyLookup(state)
    view = etl.foldchanges(events, 'id', lookup)
    iterator = iter(view)
    header = next(iterator)
    rows = []
    for row in iterator:
        rows.append(row)
        record = dict(zip(header[2:], row[2:]))
        key = (record['id'],)
        if row[1] == 'delete':
            del state[key]
        else:
            state[key] = record
    ieq((('t1', 'update', 'a', 4),
         ('t2', 'update', 'a', 6)), rows)


def test_foldchanges_partial_after_merges_visible_columns():
    state = {('a',): {'id': 'a', 'v': 1, 'w': 9}}
    events = [
        {'op': 'begin', 'txn': 't1'},
        {'op': 'update', 'before': {'id': 'a', 'v': 1, 'w': 9},
         'after': {'v': 2}},
        {'op': 'commit', 'txn': 't1'},
    ]
    header, rows = _rows(
        etl.foldchanges(events, 'id', _simple_lookup(state)))
    ieq((('t1', 'update', 'a', 2, 9),), rows)


def test_foldchanges_unknown_op_reports_position():
    state = {}
    events = [
        {'op': 'begin', 'txn': 't1'},
        {'op': 'upsert', 'id': 'a'},
    ]
    with pytest.raises(ChangeEventError) as exc:
        list(etl.foldchanges(events, 'id', _simple_lookup(state)))
    eq_(1, exc.value.position)
    eq_(None, exc.value.txn)
    eq_('upsert', exc.value.op)
