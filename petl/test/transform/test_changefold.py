# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import os

import pytest

import petl as etl
from petl.errors import ArgumentError, ChangeConsistencyError, \
    ChangeSourceError, ChangeStreamError, TransactionBoundaryError, \
    UnclosedTransactionError
from petl.test.helpers import eq_
from petl.transform.changefold import NetChange


# --------------------------------------------------------------------------
# A reference interpreter that replays every event against a plain dict.
# It does not fold anything: it mirrors the semantics the folding view must
# reproduce, including before-image validation and key-move occupancy checks.
# --------------------------------------------------------------------------


class ReferenceInterpreter(object):

    def __init__(self, initial):
        # key tuple -> row dict; rows are copied so references cannot leak
        self.state = {tuple(k): dict(v) for k, v in initial.items()}
        self.open = False
        self.visible = None
        self.committed = []

    def _key(self, row, fields):
        return tuple(row[f] for f in fields)

    def run(self, events, fields):
        for position, event in enumerate(events, 1):
            kind = event['event']
            if kind == 'begin':
                if self.open:
                    raise TransactionBoundaryError(position, 'nested begin')
                self.open = True
                self.visible = dict(self.state)
            elif kind == 'commit':
                if not self.open:
                    raise TransactionBoundaryError(position, 'commit no begin')
                self.state = self.visible
                self.open = False
                self.visible = None
                snapshot = {k: dict(v) for k, v in self.state.items()}
                self.committed.append(snapshot)
            elif kind == 'rollback':
                if not self.open:
                    raise TransactionBoundaryError(
                        position, 'rollback no begin')
                self.open = False
                self.visible = None
            elif kind == 'insert':
                self._require_open(kind, position)
                after = event['after']
                key = self._key(after, fields)
                if key in self.visible:
                    raise ChangeConsistencyError(
                        position, 'occupied target %r' % (key,))
                self.visible[key] = dict(after)
            elif kind == 'update':
                self._require_open(kind, position)
                before, after = event['before'], event['after']
                old_key = self._key(before, fields)
                new_key = self._key(after, fields)
                if old_key not in self.visible \
                        or self.visible[old_key] != before:
                    raise ChangeConsistencyError(
                        position, 'stale before image for %r' % (old_key,))
                if new_key != old_key and new_key in self.visible:
                    raise ChangeConsistencyError(
                        position, 'occupied target %r' % (new_key,))
                del self.visible[old_key]
                self.visible[new_key] = dict(after)
            elif kind == 'delete':
                self._require_open(kind, position)
                before = event['before']
                key = self._key(before, fields)
                if key not in self.visible or self.visible[key] != before:
                    raise ChangeConsistencyError(
                        position, 'stale before image for %r' % (key,))
                del self.visible[key]
            else:
                raise ChangeStreamError(position, 'unknown event %r' % kind)
        if self.open:
            raise UnclosedTransactionError(
                len(events), 'unclosed transaction')

    def _require_open(self, kind, position):
        if not self.open:
            raise TransactionBoundaryError(
                position, '%s outside transaction' % kind)


def apply_net_changes(state, changes):
    """Apply folded NetChanges exactly as a downstream sink would."""
    for change in changes:
        key = tuple(change.key)
        if change.operation == 'insert':
            assert change.before is None
            assert key not in state, 'insert into occupied key %r' % (key,)
            state[key] = dict(change.after)
        elif change.operation == 'update':
            assert key in state, 'update of missing key %r' % (key,)
            assert state[key] == change.before
            state[key] = dict(change.after)
        elif change.operation == 'delete':
            assert key in state, 'delete of missing key %r' % (key,)
            assert state[key] == change.before
            del state[key]
        else:  # pragma: no cover
            raise AssertionError('unknown operation %r' % change.operation)


def run_folded(events, fields, initial, buffersize=None, tempdir=None):
    """Drive the folding view and apply each committed transaction.

    Returns ``(state, folded, snapshots)`` where ``snapshots`` holds the
    applied state after every committed transaction, for row-by-row
    comparison against the reference interpreter.
    """
    state = {tuple(k): dict(v) for k, v in initial.items()}

    def lookup(key):
        row = state.get(tuple(key))
        return None if row is None else dict(row)

    view = etl.foldchanges(events, fields, lookup, buffersize=buffersize,
                           tempdir=tempdir)
    folded = []
    snapshots = []
    for ordinal, changes in view.transactions():
        apply_net_changes(state, changes)
        folded.append((ordinal, changes))
        snapshots.append({k: dict(v) for k, v in state.items()})
    return state, folded, snapshots


# --------------------------------------------------------------------------
# Randomised differential testing against the reference interpreter.
# --------------------------------------------------------------------------


import random


KEY_FIELDS = ('id',)


def _row(key, value):
    return {'id': key, 'v': value}


def _valid_event_stream(rng):
    """Return ``(events, initial)`` with the pre-stream committed rows."""
    initial = {}
    for key in rng.sample(range(200, 300), 3):
        initial[(key,)] = _row(key, rng.randrange(1000))
    events = []
    committed = {k: dict(v) for k, v in initial.items()}
    for _ in range(rng.randrange(3, 9)):
        events.append({'event': 'begin'})
        txn = {k: dict(v) for k, v in committed.items()}
        for _ in range(rng.randrange(1, 8)):
            free = [k for k in range(60) if (k,) not in txn]
            choice = rng.randrange(3)
            if choice == 0 and free:
                key = rng.choice(free)
                after = _row(key, rng.randrange(1000))
                events.append({'event': 'insert', 'after': after})
                txn[(key,)] = dict(after)
            elif choice == 1 and txn:
                key = rng.choice(sorted(txn))
                before = dict(txn[key])
                if free and rng.random() < 0.2:
                    new_key = rng.choice(free)
                else:
                    new_key = key[0]
                after = _row(new_key, rng.randrange(1000))
                events.append({'event': 'update', 'before': before,
                               'after': after})
                del txn[key]
                txn[(new_key,)] = dict(after)
            elif txn:
                key = rng.choice(sorted(txn))
                before = dict(txn[key])
                events.append({'event': 'delete', 'before': before})
                del txn[key]
        if rng.random() < 0.8:
            events.append({'event': 'commit'})
            committed = txn
        else:
            events.append({'event': 'rollback'})
    return events, initial


def test_folded_matches_reference_random():
    rng = random.Random(20261113)
    for _ in range(60):
        events, initial = _valid_event_stream(rng)
        reference = ReferenceInterpreter(initial)
        reference.run(events, KEY_FIELDS)

        folded_state, folded, snapshots = run_folded(
            events, KEY_FIELDS, initial)

        committed_count = sum(1 for e in events if e['event'] == 'commit')
        eq_(committed_count, len(folded))
        eq_(list(range(1, committed_count + 1)),
            [ordinal for ordinal, _ in folded])
        # row-by-row comparison after every single committed transaction
        eq_(reference.committed, snapshots)
        eq_(reference.state, folded_state)


def test_folded_matches_reference_random_spilled():
    rng = random.Random(998877)
    for _ in range(30):
        events, initial = _valid_event_stream(rng)
        reference = ReferenceInterpreter(initial)
        reference.run(events, KEY_FIELDS)
        folded_state, folded, snapshots = run_folded(
            events, KEY_FIELDS, initial, buffersize=1)
        eq_(reference.committed, snapshots)
        eq_(reference.state, folded_state)


# --------------------------------------------------------------------------
# Deterministic scenarios for each required piece of semantics.
# --------------------------------------------------------------------------


def _state_and_view(initial, events, **kwargs):
    state = {tuple(k): dict(v) for k, v in initial.items()}

    def lookup(key):
        row = state.get(tuple(key))
        return None if row is None else dict(row)

    view = etl.foldchanges(events, 'id', lookup, **kwargs)
    return state, view


def test_consecutive_updates_fold_to_one_update():
    initial = {(1,): _row(1, 'a')}
    events = [
        {'event': 'begin'},
        {'event': 'update', 'before': _row(1, 'a'), 'after': _row(1, 'b')},
        {'event': 'update', 'before': _row(1, 'b'), 'after': _row(1, 'c')},
        {'event': 'commit'},
    ]
    state, view = _state_and_view(initial, events)
    ordinal, changes = list(view.transactions())[0]
    eq_(1, ordinal)
    eq_((NetChange('update', (1,), _row(1, 'a'), _row(1, 'c')),), changes)
    apply_net_changes(state, changes)
    eq_({(1,): _row(1, 'c')}, state)


def test_insert_then_delete_cancels():
    state, view = _state_and_view({}, [
        {'event': 'begin'},
        {'event': 'insert', 'after': _row(7, 'a')},
        {'event': 'delete', 'before': _row(7, 'a')},
        {'event': 'commit'},
    ])
    ordinal, changes = list(view.transactions())[0]
    eq_((), changes)
    eq_({}, state)


def test_delete_then_insert_keeps_original_before():
    initial = {(5,): _row(5, 'old')}
    state, view = _state_and_view(initial, [
        {'event': 'begin'},
        {'event': 'delete', 'before': _row(5, 'old')},
        {'event': 'insert', 'after': _row(5, 'new')},
        {'event': 'commit'},
    ])
    _, changes = list(view.transactions())[0]
    eq_((NetChange('update', (5,), _row(5, 'old'), _row(5, 'new')),),
        changes)
    apply_net_changes(state, changes)
    eq_({(5,): _row(5, 'new')}, state)


def test_key_move_emits_delete_and_insert():
    initial = {(1,): _row(1, 'a')}
    state, view = _state_and_view(initial, [
        {'event': 'begin'},
        {'event': 'update', 'before': _row(1, 'a'),
         'after': _row(2, 'a')},
        {'event': 'commit'},
    ])
    _, changes = list(view.transactions())[0]
    eq_((
        NetChange('delete', (1,), _row(1, 'a'), None),
        NetChange('insert', (2,), None, _row(2, 'a')),
    ), changes)
    apply_net_changes(state, changes)
    eq_({(2,): _row(2, 'a')}, state)


def test_key_move_then_update_new_key_stays_insert():
    initial = {(1,): _row(1, 'a')}
    events = [
        {'event': 'begin'},
        {'event': 'update', 'before': _row(1, 'a'), 'after': _row(2, 'a')},
        {'event': 'update', 'before': _row(2, 'a'), 'after': _row(2, 'b')},
        {'event': 'commit'},
    ]
    state, view = _state_and_view(initial, events)
    _, changes = list(view.transactions())[0]
    eq_((
        NetChange('delete', (1,), _row(1, 'a'), None),
        NetChange('insert', (2,), None, _row(2, 'b')),
    ), changes)
    apply_net_changes(state, changes)
    eq_({(2,): _row(2, 'b')}, state)


def test_key_move_into_occupied_target_rejected():
    initial = {(1,): _row(1, 'a'), (2,): _row(2, 'b')}
    events = [
        {'event': 'begin'},
        {'event': 'update', 'before': _row(1, 'a'), 'after': _row(2, 'x')},
    ]
    _, view = _state_and_view(initial, events)
    with pytest.raises(ChangeConsistencyError) as exc:
        list(view.transactions())
    assert 'occupied' in str(exc.value)
    eq_(2, exc.value.position)


def test_swap_keys_is_ordered_and_occupancy_checked():
    initial = {(1,): _row(1, 'a'), (2,): _row(2, 'b')}
    events = [
        {'event': 'begin'},
        {'event': 'update', 'before': _row(1, 'a'), 'after': _row(3, 'a')},
        {'event': 'update', 'before': _row(2, 'b'), 'after': _row(1, 'b')},
        {'event': 'update', 'before': _row(3, 'a'), 'after': _row(2, 'a')},
        {'event': 'commit'},
    ]
    state, view = _state_and_view(initial, events)
    _, changes = list(view.transactions())[0]
    apply_net_changes(state, changes)
    eq_({(1,): _row(1, 'b'), (2,): _row(2, 'a')}, state)


def test_before_image_mismatch_rejected_with_position():
    initial = {(1,): _row(1, 'a')}
    events = [
        {'event': 'begin'},
        {'event': 'update', 'before': _row(1, 'stale'),
         'after': _row(1, 'b')},
    ]
    _, view = _state_and_view(initial, events)
    with pytest.raises(ChangeConsistencyError) as exc:
        list(view.transactions())
    eq_(2, exc.value.position)
    assert 'does not match' in str(exc.value)


def test_update_missing_row_rejected():
    events = [
        {'event': 'begin'},
        {'event': 'delete', 'before': _row(99, 'a')},
    ]
    _, view = _state_and_view({}, events)
    with pytest.raises(ChangeConsistencyError):
        list(view.transactions())


def test_insert_occupied_key_rejected():
    initial = {(1,): _row(1, 'a')}
    events = [
        {'event': 'begin'},
        {'event': 'insert', 'after': _row(1, 'b')},
    ]
    _, view = _state_and_view(initial, events)
    with pytest.raises(ChangeConsistencyError):
        list(view.transactions())


def test_nested_begin_reports_position():
    events = [
        {'event': 'begin'},
        {'event': 'begin'},
    ]
    _, view = _state_and_view({}, events)
    with pytest.raises(TransactionBoundaryError) as exc:
        list(view.transactions())
    eq_(2, exc.value.position)


def test_commit_without_begin_reports_position():
    _, view = _state_and_view({}, [{'event': 'commit'}])
    with pytest.raises(TransactionBoundaryError) as exc:
        list(view.transactions())
    eq_(1, exc.value.position)


def test_rollback_without_begin_reports_position():
    _, view = _state_and_view({}, [{'event': 'rollback'}])
    with pytest.raises(TransactionBoundaryError) as exc:
        list(view.transactions())
    eq_(1, exc.value.position)


def test_row_event_without_transaction_rejected():
    _, view = _state_and_view({}, [
        {'event': 'insert', 'after': _row(1, 'a')},
    ])
    with pytest.raises(TransactionBoundaryError) as exc:
        list(view.transactions())
    eq_(1, exc.value.position)


def test_unknown_event_type_rejected():
    _, view = _state_and_view({}, [
        {'event': 'begin'},
        {'event': 'upsert', 'after': _row(1, 'a')},
        {'event': 'rollback'},
    ])
    with pytest.raises(ChangeStreamError) as exc:
        list(view.transactions())
    eq_(2, exc.value.position)


def test_rollback_hides_transaction_and_preserves_later_state():
    initial = {(1,): _row(1, 'a')}
    events = [
        {'event': 'begin'},
        {'event': 'update', 'before': _row(1, 'a'), 'after': _row(1, 'b')},
        {'event': 'rollback'},
        {'event': 'begin'},
        {'event': 'delete', 'before': _row(1, 'a')},
        {'event': 'commit'},
    ]
    state, view = _state_and_view(initial, events)
    transactions = list(view.transactions())
    eq_(1, len(transactions))  # rolled back txn produces no output
    _, changes = transactions[0]
    eq_((NetChange('delete', (1,), _row(1, 'a'), None),), changes)
    apply_net_changes(state, changes)
    eq_({}, state)


def test_nothing_emitted_before_commit():
    initial = {}
    state = {}

    def lookup(key):
        return state.get(tuple(key))

    view = etl.foldchanges([
        {'event': 'begin'},
        {'event': 'insert', 'after': _row(1, 'a')},
    ], 'id', lookup)
    it = iter(view.transactions())
    # stream ends unclosed: consuming raises and yields nothing
    with pytest.raises(UnclosedTransactionError):
        next(it)
    with pytest.raises(StopIteration):
        next(it)


def test_compound_primary_key():
    fields = ('tenant', 'id')

    def row(tenant, key, value):
        return {'tenant': tenant, 'id': key, 'v': value}

    initial = {(1, 9): row(1, 9, 'a')}
    state = {tuple(k): dict(v) for k, v in initial.items()}

    def lookup(key):
        found = state.get(tuple(key))
        return None if found is None else dict(found)

    events = [
        {'event': 'begin'},
        {'event': 'update', 'before': row(1, 9, 'a'),
         'after': row(1, 9, 'b')},
        {'event': 'insert', 'after': row(2, 9, 'z')},
        {'event': 'commit'},
    ]
    view = etl.foldchanges(events, fields, lookup)
    _, changes = list(view.transactions())[0]
    eq_((
        NetChange('update', (1, 9), row(1, 9, 'a'), row(1, 9, 'b')),
        NetChange('insert', (2, 9), None, row(2, 9, 'z')),
    ), changes)
    apply_net_changes(state, changes)
    eq_({(1, 9): row(1, 9, 'b'), (2, 9): row(2, 9, 'z')}, state)


def test_tuple_row_events_supported():
    state, view = _state_and_view({}, [
        ('begin',),
        ('insert', _row(1, 'a')),
        ('update', _row(1, 'a'), _row(1, 'b')),
        ('delete', _row(1, 'b')),
        ('commit',),
    ])
    _, changes = list(view.transactions())[0]
    eq_((), changes)  # insert + update + delete on a new key cancels


def test_view_header_and_fluent_form():
    state, view = _state_and_view({}, [
        {'event': 'begin'},
        {'event': 'insert', 'after': _row(1, 'a')},
        {'event': 'commit'},
    ])
    rows = list(view)
    eq_(('transaction', 'changes'), rows[0])
    eq_(1, rows[1][0])
    # fluent form on a petl-wrapped event table
    wrapped = etl.wrap([
        {'event': 'begin'},
        {'event': 'insert', 'after': _row(2, 'x')},
        {'event': 'commit'},
    ])

    def lookup(key):
        return None

    eq_(1, len(list(wrapped.foldchanges('id', lookup).transactions())))


def test_lookup_is_required_and_callable():
    with pytest.raises(ArgumentError):
        etl.foldchanges([], 'id')
    with pytest.raises(ArgumentError):
        etl.foldchanges([], 'id', lookup='not-callable')


def test_empty_key_rejected():
    with pytest.raises(ArgumentError):
        etl.foldchanges([], [], lookup=lambda key: None)


def test_negative_buffersize_rejected():
    with pytest.raises(ArgumentError):
        etl.foldchanges([], 'id', lookup=lambda key: None, buffersize=-1)


def test_source_failure_mid_transaction_emits_no_partial_txn():
    emitted = []

    def source():
        yield {'event': 'begin'}
        yield {'event': 'commit'}
        yield {'event': 'begin'}
        yield {'event': 'insert', 'after': _row(1, 'a')}
        raise RuntimeError('collector crashed')

    def lookup(key):
        return None

    view = etl.foldchanges(source(), 'id', lookup)
    iterator = iter(view.transactions())
    # only the first (fully committed) transaction may come out
    ordinal, changes = next(iterator)
    eq_(1, ordinal)
    emitted.append(changes)
    with pytest.raises(ChangeSourceError) as exc:
        next(iterator)
    eq_('collector crashed', str(exc.value.__cause__))
    with pytest.raises(StopIteration):
        next(iterator)
    eq_(1, len(emitted))
    eq_(4, exc.value.position)


def test_source_failure_after_event_then_nothing_else():
    def source():
        yield {'event': 'begin'}
        yield {'event': 'insert', 'after': _row(1, 'a')}
        raise ValueError('boom at read')

    def lookup(key):
        return None

    view = etl.foldchanges(source(), 'id', lookup)
    with pytest.raises(ChangeSourceError):
        list(view.transactions())


def test_unclosed_transaction_at_end_of_stream():
    def lookup(key):
        return None

    view = etl.foldchanges([
        {'event': 'begin'},
        {'event': 'insert', 'after': _row(1, 'a')},
    ], 'id', lookup)
    with pytest.raises(UnclosedTransactionError):
        list(view.transactions())


def test_generator_close_mid_transaction_emits_nothing():
    def source():
        yield {'event': 'begin'}
        yield {'event': 'insert', 'after': _row(1, 'a')}
        yield {'event': 'commit'}
        yield {'event': 'begin'}
        yield {'event': 'insert', 'after': _row(2, 'b')}
        yield {'event': 'commit'}

    def lookup(key):
        return None

    view = etl.foldchanges(source(), 'id', lookup)
    iterator = iter(view.transactions())
    first = next(iterator)
    eq_(1, first[0])
    # abandon iteration inside the second transaction; closing the
    # generator must not flush any half transaction
    iterator.close()
    with pytest.raises(StopIteration):
        next(iterator)


def test_spill_uses_temp_sqlite_and_cleans_up(tmp_path):
    before = set(os.listdir(str(tmp_path)))

    initial = {}
    state = {}

    def lookup(key):
        return state.get(tuple(key))

    events = [{'event': 'begin'}]
    for key in range(10):
        events.append({'event': 'insert', 'after': _row(key, 'v%s' % key)})
    events.append({'event': 'commit'})
    view = etl.foldchanges(events, 'id', lookup, buffersize=2,
                           tempdir=str(tmp_path))
    _, changes = list(view.transactions())[0]
    eq_(10, len(changes))
    apply_net_changes(state, changes)
    eq_(10, len(state))
    # no temporary database survives the transaction
    after = set(os.listdir(str(tmp_path)))
    eq_(before, after)


def test_spill_preserves_first_touch_order(tmp_path):
    def lookup(key):
        return None

    events = [{'event': 'begin'}]
    for key in (5, 1, 9, 2, 7):
        events.append({'event': 'insert', 'after': _row(key, 'v')})
    events.append({'event': 'commit'})
    view = etl.foldchanges(events, 'id', lookup, buffersize=1,
                           tempdir=str(tmp_path))
    _, changes = list(view.transactions())[0]
    eq_((5, 1, 9, 2, 7), tuple(c.key[0] for c in changes))


def test_boundary_event_with_images_rejected():
    _, view = _state_and_view({}, [
        {'event': 'begin', 'after': _row(1, 'a')},
    ])
    with pytest.raises(ChangeStreamError) as exc:
        list(view.transactions())
    eq_(1, exc.value.position)


def test_insert_with_before_image_rejected():
    _, view = _state_and_view({}, [
        {'event': 'begin'},
        {'event': 'insert', 'before': _row(1, 'a'), 'after': _row(1, 'a')},
    ])
    with pytest.raises(ChangeStreamError) as exc:
        list(view.transactions())
    eq_(2, exc.value.position)


def test_delete_with_after_image_rejected():
    _, view = _state_and_view({}, [
        {'event': 'begin'},
        {'event': 'delete', 'before': _row(1, 'a'), 'after': _row(1, 'a')},
    ])
    with pytest.raises(ChangeStreamError) as exc:
        list(view.transactions())
    eq_(2, exc.value.position)


def test_missing_after_image_rejected():
    _, view = _state_and_view({}, [
        {'event': 'begin'},
        {'event': 'insert'},
    ])
    with pytest.raises(ChangeStreamError):
        list(view.transactions())


def test_missing_key_field_in_image_rejected():
    _, view = _state_and_view({}, [
        {'event': 'begin'},
        {'event': 'insert', 'after': {'v': 'a'}},
    ])
    with pytest.raises(ChangeStreamError) as exc:
        list(view.transactions())
    assert 'compound key' in str(exc.value)


def test_lookup_failure_wrapped_with_position():
    def bad_lookup(key):
        raise RuntimeError('db down')

    view = etl.foldchanges([
        {'event': 'begin'},
        {'event': 'delete', 'before': _row(1, 'a')},
        {'event': 'commit'},
    ], 'id', bad_lookup)
    with pytest.raises(ChangeSourceError) as exc:
        list(view.transactions())
    eq_(2, exc.value.position)
    eq_('db down', str(exc.value.__cause__))


def test_double_commit_reports_position():
    _, view = _state_and_view({}, [
        {'event': 'begin'},
        {'event': 'commit'},
        {'event': 'commit'},
    ])
    with pytest.raises(TransactionBoundaryError) as exc:
        list(view.transactions())
    eq_(3, exc.value.position)


def test_commit_after_rollback_starts_fresh():
    state, view = _state_and_view({}, [
        {'event': 'begin'},
        {'event': 'insert', 'after': _row(1, 'a')},
        {'event': 'rollback'},
        {'event': 'begin'},
        {'event': 'insert', 'after': _row(1, 'b')},
        {'event': 'commit'},
    ])
    transactions = list(view.transactions())
    eq_(1, len(transactions))
    ordinal, changes = transactions[0]
    eq_(1, ordinal)
    eq_((NetChange('insert', (1,), None, _row(1, 'b')),), changes)
    apply_net_changes(state, changes)
    eq_({(1,): _row(1, 'b')}, state)


def test_insert_delete_insert_same_key():
    _, view = _state_and_view({}, [
        {'event': 'begin'},
        {'event': 'insert', 'after': _row(1, 'a')},
        {'event': 'delete', 'before': _row(1, 'a')},
        {'event': 'insert', 'after': _row(1, 'b')},
        {'event': 'commit'},
    ])
    _, changes = list(view.transactions())[0]
    eq_((NetChange('insert', (1,), None, _row(1, 'b')),), changes)


def test_key_move_back_to_original_key():
    initial = {(1,): _row(1, 'a')}
    state, view = _state_and_view(initial, [
        {'event': 'begin'},
        {'event': 'update', 'before': _row(1, 'a'), 'after': _row(2, 'a')},
        {'event': 'update', 'before': _row(2, 'a'), 'after': _row(1, 'a')},
        {'event': 'commit'},
    ])
    _, changes = list(view.transactions())[0]
    eq_((), changes)  # move out and back cancels entirely
    eq_({(1,): _row(1, 'a')}, state)


def test_staging_spills_after_exceeding_budget(tmp_path):
    from petl.transform.changefold import _Staging, _VISIBLE

    staging = _Staging(buffersize=2, tempdir=str(tmp_path))
    for key in range(4):
        staging.put((key,), _VISIBLE, _row(key, key))
    assert staging.spilled
    # order is preserved across the spill
    eq_([0, 1, 2, 3],
        [key[0] for key, _, _ in staging.items_in_order()])
    staging.close()
    eq_([], os.listdir(str(tmp_path)))


def test_staging_stays_in_memory_within_budget():
    from petl.transform.changefold import _Staging, _VISIBLE

    staging = _Staging(buffersize=4)
    for key in range(3):
        staging.put((key,), _VISIBLE, _row(key, key))
    assert not staging.spilled
    eq_([0, 1, 2],
        [key[0] for key, _, _ in staging.items_in_order()])
    staging.close()


def test_staging_no_budget_never_spills():
    from petl.transform.changefold import _Staging, _VISIBLE

    staging = _Staging(buffersize=None)
    for key in range(3):
        staging.put((key,), _VISIBLE, _row(key, key))
    assert not staging.spilled
    staging.close()
