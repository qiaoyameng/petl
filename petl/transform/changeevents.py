from __future__ import absolute_import, print_function, division

import os
import sqlite3
import tempfile

import petl.config as config
from petl.compat import pickle
from petl.errors import ChangeEventError
from petl.util.base import Table


OP_BEGIN = 'begin'
OP_INSERT = 'insert'
OP_UPDATE = 'update'
OP_DELETE = 'delete'
OP_COMMIT = 'commit'
OP_ROLLBACK = 'rollback'

CHANGE_INSERT = 'insert'
CHANGE_UPDATE = 'update'
CHANGE_DELETE = 'delete'

_META_FIELDS = ('op', 'txn', 'before', 'after')
_KNOWN_OPS = (OP_BEGIN, OP_INSERT, OP_UPDATE, OP_DELETE, OP_COMMIT,
              OP_ROLLBACK)

def foldchanges(source, key, lookup, fields=None, buffersize=None,
                tempdir=None):
    """
    Fold an ordered stream of table change events into the net changes made by
    each committed transaction.

    The `source` is an iterable of scalar dictionaries (events). Each event
    carries an ``'op'`` field, one of ``'begin'``, ``'insert'``, ``'update'``,
    ``'delete'``, ``'commit'`` or ``'rollback'``. Transaction boundary events
    carry a ``'txn'`` identifier; data events may carry one as well, in which
    case it must match the active transaction. Only a single transaction may be
    active at any time.

    `key` is a field name or sequence of field names making up the (possibly
    composite) primary key of the table. `lookup` is a read-only callback with
    signature ``lookup(key_tuple)`` returning the currently committed row as a
    dictionary, or ``None`` if no row with that key exists. It is only called
    for keys that a transaction has not already staged, and is never used to
    commit anything on behalf of the caller.

    Every ``'update'`` and ``'delete'`` event carries a ``'before'`` image,
    which must match the value currently visible to the transaction (the
    committed value as modified by earlier events in the same transaction).
    ``'update'`` events also carry an ``'after'`` image. Changing a primary key
    is only allowed when the target key is not occupied.

    Nothing is emitted for a transaction until it commits, and a rolled back
    transaction leaves the state visible to later transactions untouched. An
    ``'insert'`` followed by a ``'delete'`` of the same key cancels out, and a
    ``'delete'`` followed by an ``'insert'`` keeps the original old value.

    Returns a table whose header is ``('txn', 'change', <field>...)`` where
    ``change`` is one of ``'insert'``, ``'update'`` or ``'delete'``. Delete
    rows report the pre-transaction values; insert and update rows report the
    new values. A re-keying update is emitted as a delete of the old key
    followed by an insert of the new key. Rows whose net effect is a no-op are
    not emitted.

    If `fields` is given it fixes the output data columns. Otherwise the
    columns are discovered from the events; discovering a new column after the
    output header has been emitted raises a
    :class:`petl.errors.ChangeEventError`, in which case pass `fields`
    explicitly.

    Once a transaction stages more than `buffersize` distinct keys its staging
    area is moved to a temporary SQLite database under `tempdir`. If
    `buffersize` is ``None`` the value of ``petl.config.change_buffersize`` is
    used. Malformed streams, including duplicate or out-of-order transaction
    boundaries, raise :class:`petl.errors.ChangeEventError` carrying the
    zero-based position of the offending event.

    """

    return FoldChangesView(source, key=key, lookup=lookup, fields=fields,
                           buffersize=buffersize, tempdir=tempdir)


class FoldChangesView(Table):

    def __init__(self, source, key, lookup, fields=None, buffersize=None,
                 tempdir=None):
        self.source = source
        self.key = (key,) if not isinstance(key, (list, tuple)) else tuple(key)
        self.lookup_fn = lookup
        self.fields = list(fields) if fields is not None else []
        if buffersize is None:
            self.buffersize = config.change_buffersize
        else:
            self.buffersize = buffersize
        self.tempdir = tempdir

    def __iter__(self):
        return _iterfoldchanges(self)


class _MemoryStaging(object):

    def __init__(self):
        # key tuple -> {'origin': dict or None, 'current': dict or None}
        # current is None for a key that the transaction deletes; origin is
        # None for a key that did not exist when the transaction started.
        self._entries = {}

    def __contains__(self, key):
        return key in self._entries

    def __len__(self):
        return len(self._entries)

    def get(self, key):
        entry = self._entries.get(key)
        if entry is None:
            return None
        return entry['current']

    def put(self, key, origin, current):
        self._entries[key] = {'origin': origin, 'current': current}

    def pop(self, key):
        return self._entries.pop(key)

    def items(self):
        for key, entry in self._entries.items():
            yield key, entry['origin'], entry['current']

    def close(self):
        pass


class _SqliteStaging(object):

    def __init__(self, tempdir=None):
        handle = tempfile.NamedTemporaryFile(delete=False, suffix='.sqlite',
                                             dir=tempdir)
        self.path = handle.name
        handle.close()
        self.connection = sqlite3.connect(self.path)
        self.connection.execute(
            'CREATE TABLE stage ('
            'seq INTEGER PRIMARY KEY AUTOINCREMENT, '
            'key BLOB UNIQUE, origin BLOB, current BLOB)')
        self.connection.commit()

    def __contains__(self, key):
        cur = self.connection.execute(
            'SELECT 1 FROM stage WHERE key = ?', (self._dumps(key),))
        return cur.fetchone() is not None

    def __len__(self):
        return self.connection.execute(
            'SELECT COUNT(*) FROM stage').fetchone()[0]

    @staticmethod
    def _dumps(value):
        return sqlite3.Binary(
            pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))

    @staticmethod
    def _loads(blob):
        if blob is None:
            return None
        return pickle.loads(bytes(blob))

    def get(self, key):
        cur = self.connection.execute(
            'SELECT current FROM stage WHERE key = ?',
            (self._dumps(key),))
        row = cur.fetchone()
        if row is None:
            return None
        return self._loads(row[0])

    def put(self, key, origin, current):
        key_blob = self._dumps(key)
        origin_blob = None if origin is None else self._dumps(origin)
        current_blob = None if current is None else self._dumps(current)
        cur = self.connection.execute(
            'SELECT 1 FROM stage WHERE key = ?', (key_blob,))
        if cur.fetchone() is not None:
            # update in place so the original first-touch sequence position
            # (and thus the emitted row order) is preserved
            self.connection.execute(
                'UPDATE stage SET origin = ?, current = ? WHERE key = ?',
                (origin_blob, current_blob, key_blob))
        else:
            self.connection.execute(
                'INSERT INTO stage (key, origin, current) '
                'VALUES (?, ?, ?)', (key_blob, origin_blob, current_blob))

    def pop(self, key):
        cur = self.connection.execute(
            'SELECT origin, current FROM stage WHERE key = ?',
            (self._dumps(key),))
        row = cur.fetchone()
        if row is None:
            raise KeyError(key)
        self.connection.execute('DELETE FROM stage WHERE key = ?',
                                (self._dumps(key),))
        return {'origin': self._loads(row[0]), 'current': self._loads(row[1])}

    def items(self):
        cur = self.connection.execute(
            'SELECT key, origin, current FROM stage ORDER BY seq')
        for key_blob, origin_blob, current_blob in cur:
            yield (self._loads(key_blob), self._loads(origin_blob),
                   self._loads(current_blob))

    def commit(self):
        self.connection.commit()

    def close(self):
        try:
            self.connection.close()
        finally:
            if os.path.exists(self.path):
                os.remove(self.path)


def _iterfoldchanges(view):
    source = view.source
    key_fields = view.key
    lookup = view.lookup_fn
    declared_fields = list(view.fields)
    fields = list(declared_fields)
    buffersize = view.buffersize

    active_txn = None
    staging = _MemoryStaging()
    header_emitted = False
    position = -1

    def error(message, event, txn=None):
        op = event.get('op') if isinstance(event, dict) else None
        event_txn = event.get('txn', txn) if isinstance(event, dict) else txn
        raise ChangeEventError(message, position=position,
                               txn=event_txn, op=op)

    def key_of(row, event):
        try:
            return tuple(row[name] for name in key_fields)
        except KeyError as missing:
            error('row is missing key field %r' % (missing.args[0],), event)

    def note_fields(row, event):
        for name in row:
            if name not in fields:
                if declared_fields:
                    error('field %r is not among the declared fields %r'
                          % (name, declared_fields), event)
                if header_emitted:
                    error('field %r first seen after the output header was '
                          'emitted; pass fields= to declare the output '
                          'columns' % name, event)
                fields.append(name)

    def visible(key):
        # Value of key as seen by the active transaction.
        if key in staging:
            return staging.get(key)
        return lookup(key)

    def occupied(key):
        # A key may be inserted into only when the transaction sees no row
        # there. A staged entry exists for every key the transaction touched,
        # including deletes (current is None) and re-key vacancies.
        if key not in staging:
            return lookup(key) is not None
        return staging.get(key) is not None

    def check_before(event, current):
        before = event.get('before')
        if not isinstance(before, dict):
            error('update and delete events require a dict "before" image',
                  event)
        if current is None:
            error('"before" image %r does not match any visible row'
                  % (before,), event)
        for name, old_value in before.items():
            if current.get(name) != old_value:
                error('"before" image %r does not match visible row %r'
                      % (before, current), event)

    def stage_put(key, origin, current):
        staging.put(key, origin, current)
        _maybe_spill()

    def _maybe_spill():
        nonlocal staging
        if buffersize is not None and len(staging) > buffersize \
                and isinstance(staging, _MemoryStaging):
            spilled = _SqliteStaging(view.tempdir)
            for staged_key, origin, current in staging.items():
                spilled.put(staged_key, origin, current)
            spilled.commit()
            staging.close()
            staging = spilled

    try:
        for position, event in enumerate(source):
            if not isinstance(event, dict):
                raise ChangeEventError(
                    'change events must be dictionaries, got %r'
                    % (type(event).__name__,), position=position)
            op = event.get('op')
            if op not in _KNOWN_OPS:
                error('unknown or missing event op %r' % (op,), event)

            if op == OP_BEGIN:
                if active_txn is not None:
                    error('nested begin while transaction %r is still open'
                          % active_txn, event)
                txn = event.get('txn')
                if txn is None:
                    error('begin event requires a "txn" identifier', event)
                active_txn = txn
                staging = _MemoryStaging()
                continue

            if op in (OP_COMMIT, OP_ROLLBACK):
                if active_txn is None:
                    error('%s without an open transaction' % op, event)
                txn = event.get('txn', active_txn)
                if txn != active_txn:
                    error('%s for transaction %r but active transaction is %r'
                          % (op, txn, active_txn), event)

                if op == OP_ROLLBACK:
                    staging.close()
                    staging = _MemoryStaging()
                    active_txn = None
                    continue

                # commit: emit the folded net changes for this transaction
                if not header_emitted:
                    yield tuple(['txn', 'change'] + fields)
                    header_emitted = True
                for staged_key, origin, current in staging.items():
                    if origin is None:
                        if current is not None:
                            note_fields(current, event)
                            yield _row(txn, CHANGE_INSERT, current, fields)
                    elif current is None:
                        yield _row(txn, CHANGE_DELETE, origin, fields)
                    else:
                        if origin != current:
                            note_fields(current, event)
                            yield _row(txn, CHANGE_UPDATE, current, fields)
                if isinstance(staging, _SqliteStaging):
                    staging.commit()
                staging.close()
                staging = _MemoryStaging()
                active_txn = None
                continue

            # data events require an active transaction
            if active_txn is None:
                error('%s event without an open transaction' % op, event)
            txn = event.get('txn', active_txn)
            if txn != active_txn:
                error('%s event for transaction %r but active transaction is '
                      '%r' % (op, txn, active_txn), event)

            if op == OP_INSERT:
                if 'before' in event:
                    error('insert events must not carry a "before" image',
                          event)
                row = {name: value for name, value in event.items()
                       if name not in _META_FIELDS}
                note_fields(row, event)
                new_key = key_of(row, event)
                if occupied(new_key):
                    error('insert target key %r is already occupied'
                          % (new_key,), event)
                if new_key in staging:
                    # re-using a key deleted or vacated earlier in this
                    # transaction: keep its committed origin (None for a key
                    # that did not exist when the transaction started)
                    origin = staging.pop(new_key)['origin']
                else:
                    origin = None
                stage_put(new_key, origin, dict(row))

            elif op == OP_DELETE:
                before = event.get('before')
                if not isinstance(before, dict):
                    error('delete events require a dict "before" image',
                          event)
                note_fields(before, event)
                old_key = key_of(before, event)
                current = visible(old_key)
                check_before(event, current)
                if old_key in staging:
                    entry = staging.pop(old_key)
                    origin = entry['origin']
                else:
                    origin = dict(current)
                stage_put(old_key, origin, None)

            else:  # update
                before = event.get('before')
                after = event.get('after')
                if not isinstance(before, dict):
                    error('update events require a dict "before" image',
                          event)
                if not isinstance(after, dict):
                    error('update events require a dict "after" image',
                          event)
                note_fields(before, event)
                note_fields(after, event)
                old_key = key_of(before, event)
                current = visible(old_key)
                check_before(event, current)
                new_row = dict(current)
                new_row.update(after)
                new_key = key_of(new_row, event)
                if old_key in staging:
                    entry = staging.pop(old_key)
                    origin = entry['origin']
                else:
                    origin = dict(current)
                if new_key != old_key and occupied(new_key):
                    # restore the old entry before rejecting the move
                    stage_put(old_key, origin, current)
                    error('re-key target %r is already occupied'
                          % (new_key,), event)
                if new_key != old_key:
                    # split the move into independent per-key changes: the old
                    # key loses its committed row (a delete), and the new key
                    # gains a row that did not exist there at txn start (an
                    # insert). Each entry therefore carries its own origin.
                    stage_put(old_key, origin, None)
                    origin = None
                if new_key in staging:
                    # moving into a key deleted or vacated earlier in this
                    # transaction: keep that key's own committed origin
                    origin = staging.pop(new_key)['origin']
                stage_put(new_key, origin, new_row)

        if active_txn is not None:
            raise ChangeEventError(
                'stream ended with transaction %r not closed by commit or '
                'rollback' % active_txn, position=position + 1,
                txn=active_txn)
        if not header_emitted:
            yield tuple(['txn', 'change'] + fields)
    except ChangeEventError:
        raise
    except Exception as cause:
        raise ChangeEventError(
            'source raised %s: %s' % (type(cause).__name__, cause),
            position=position + 1, txn=active_txn)
    finally:
        staging.close()


def _row(txn, change, row, fields):
    return tuple([txn, change] + [row.get(name) for name in fields])


Table.foldchanges = foldchanges
