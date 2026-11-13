# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import os
import pickle
import sqlite3
import tempfile
from collections import namedtuple, OrderedDict

from petl.compat import text_type
from petl.errors import ArgumentError, ChangeConsistencyError, \
    ChangeEventError, ChangeSourceError, ChangeStreamError, \
    TransactionBoundaryError, UnclosedTransactionError
from petl.util.base import Table


# A single folded net change for one table row.
#   operation: one of 'insert', 'update', 'delete'
#   key:       tuple of primary key values in caller order
#   before:    full row visible before the transaction (None for inserts)
#   after:     full row visible after the transaction (None for deletes)
NetChange = namedtuple('NetChange', ('operation', 'key', 'before', 'after'))


_INSERT = 'insert'
_UPDATE = 'update'
_DELETE = 'delete'
_BEGIN = 'begin'
_COMMIT = 'commit'
_ROLLBACK = 'rollback'
_BOUNDARY = frozenset((_BEGIN, _COMMIT, _ROLLBACK))
_ROW_EVENTS = frozenset((_INSERT, _UPDATE, _DELETE))
_VALID_EVENTS = _BOUNDARY | _ROW_EVENTS
_EVENT_TYPE_KEYS = ('event', 'op', 'type', 'action')

# state of one key within the open transaction
_VISIBLE = 'visible'   # the key exists, state holds the current image
_TOMBSTONE = 'tomb'    # the key does not exist within the transaction


def foldchanges(table, key, lookup=None, buffersize=None, tempdir=None):
    """
    Fold an ordered stream of row-level change events into the net change of
    each committed transaction. E.g.::

        >>> import petl as etl
        >>> events = [
        ...     {'event': 'begin'},
        ...     {'event': 'insert', 'after': {'id': 1, 'v': 'a'}},
        ...     {'event': 'update',
        ...      'before': {'id': 1, 'v': 'a'},
        ...      'after': {'id': 1, 'v': 'b'}},
        ...     {'event': 'commit'},
        ...     {'event': 'begin'},
        ...     {'event': 'delete', 'before': {'id': 1, 'v': 'b'}},
        ...     {'event': 'rollback'},
        ... ]
        >>> committed = {}
        >>> def lookup(key):
        ...     return committed.get(tuple(key))
        >>> out = []
        >>> for _, changes in etl.foldchanges(events, 'id',
        ...                                   lookup).transactions():
        ...     for change in changes:
        ...         out.append((change.operation, change.after))
        ...         if change.operation == 'delete':
        ...             committed.pop(change.key, None)
        ...         else:
        ...             committed[change.key] = change.after
        >>> out
        [('insert', {'id': 1, 'v': 'b'})]

    The input is an ordered sequence of scalar dict events. Row events carry
    full row images in their ``before`` and/or ``after`` fields:

    * ``{'event': 'begin'}``
    * ``{'event': 'insert', 'after': {...}}``
    * ``{'event': 'update', 'before': {...}, 'after': {...}}``
    * ``{'event': 'delete', 'before': {...}}``
    * ``{'event': 'commit'}``
    * ``{'event': 'rollback'}``

    Tuple rows are accepted as well, with the event type in the first column:
    ``('begin',)``, ``('insert', after)``, ``('update', before, after)`` and
    ``('delete', before)``.

    The output view yields a header row plus one row per committed
    transaction, holding the transaction ordinal and a tuple of
    :class:`NetChange` objects in event order. Use
    :meth:`FoldedChangesView.transactions` to iterate the same data without
    the header. Nothing is emitted before ``commit``; a ``rollback`` discards
    the transaction and leaves the state visible to later transactions
    untouched.

    The `key` argument names one field or a list of fields forming the
    compound primary key. The `lookup` callable receives a key tuple and
    returns the current committed row (a scalar dict) or ``None`` when the key
    is free; it is only used for read-only queries and is never asked to
    commit anything.

    If `buffersize` is an int and a transaction stages more than that many
    distinct keys, the staged images spill into a temporary SQLite database
    created under `tempdir`. The temporary database is closed and deleted
    before the transaction finishes.
    """

    return FoldedChangesView(table, key=key, lookup=lookup,
                             buffersize=buffersize, tempdir=tempdir)


Table.foldchanges = foldchanges


class FoldedChangesView(Table):
    """
    View folding ordered change events into per-transaction net changes.

    Only one transaction may be active at a time. ``update`` and ``delete``
    events carry a ``before`` image that must equal the row currently visible
    inside the transaction. A key change checks that the target key is free.
    Inserting then deleting the same key cancels out; deleting a key and then
    re-inserting it still reports the original row as the ``before`` image.
    """

    def __init__(self, table, key, lookup=None, buffersize=None,
                 tempdir=None):
        if lookup is None:
            raise ArgumentError('a read-only lookup callable is required')
        if not callable(lookup):
            raise ArgumentError('lookup must be callable')
        if isinstance(key, (list, tuple)):
            if not key:
                raise ArgumentError('key must name at least one field')
            self.key = tuple(text_type(k) for k in key)
        else:
            self.key = (text_type(key),)
        self.table = table
        self._lookup = lookup
        if buffersize is not None and buffersize < 0:
            raise ArgumentError('buffersize must be non-negative or None')
        self.buffersize = buffersize
        self.tempdir = tempdir

    def __iter__(self):
        yield ('transaction', 'changes')
        for ordinal, changes in self.transactions():
            yield (ordinal, changes)

    def transactions(self):
        """Yield ``(ordinal, (NetChange, ...))`` for each committed txn."""
        return iterfoldedchanges(self.table, self.key, self._lookup,
                                 self.buffersize, self.tempdir)


def _normalize_event(row, position):
    """Return ``(event_type, before, after)`` for one input row."""
    if isinstance(row, dict):
        event_type = None
        for name in _EVENT_TYPE_KEYS:
            if name in row:
                event_type = row[name]
                break
        if event_type is None:
            raise ChangeStreamError(position, 'event has no event type')
        return text_type(event_type), row.get('before'), row.get('after')

    if hasattr(row, '_asdict'):
        row = tuple(row)
    if isinstance(row, (text_type, bytes)):
        row = (row,)
    else:
        try:
            row = tuple(row)
        except TypeError:
            raise ChangeStreamError(position, 'event is not a scalar row')

    if not row:
        raise ChangeStreamError(position, 'empty event row')
    event_type = text_type(row[0])
    before = after = None
    if len(row) == 1:
        pass
    elif len(row) == 2:
        if event_type in (_UPDATE, _DELETE):
            before = row[1]
        else:
            after = row[1]
    elif len(row) == 3:
        before, after = row[1], row[2]
    else:
        raise ChangeStreamError(position,
                                'event row must have 1 to 3 columns')
    return event_type, before, after


def _check_event_images(event_type, before, after, position):
    if event_type in _BOUNDARY:
        if before is not None or after is not None:
            raise ChangeStreamError(
                position, '%s event must not carry row images'
                % (event_type,))
    elif event_type == _INSERT:
        if before is not None:
            raise ChangeStreamError(
                position, 'insert event must not carry a before image')
    elif event_type == _DELETE:
        if after is not None:
            raise ChangeStreamError(
                position, 'delete event must not carry an after image')


def _row_key(row, key, position):
    try:
        return tuple(row[field] for field in key)
    except (KeyError, TypeError):
        raise ChangeStreamError(
            position, 'row image is missing part of the compound key %r'
            % (key,))


def _check_image(image, position, label):
    if not isinstance(image, dict):
        raise ChangeStreamError(
            position, '%s image must be a scalar dict, got %r'
            % (label, type(image).__name__))


def _images_equal(left, right):
    if left is None or right is None:
        return left is right
    return left == right


class _MemoryStaging(object):
    """In-memory staging area for one open transaction."""

    def __init__(self):
        # key -> (state, image, order); state is _VISIBLE/_TOMBSTONE
        self._rows = OrderedDict()
        self._spill_count = 0

    def __len__(self):
        return len(self._rows)

    def get(self, key):
        entry = self._rows.get(key)
        if entry is None:
            return None
        return entry[0], entry[1]

    def put(self, key, state, image):
        existing = self._rows.get(key)
        if existing is None:
            order = self._spill_count
            self._spill_count += 1
        else:
            order = existing[2]
        self._rows[key] = (state, image, order)

    def items_in_order(self):
        return ((key, state, image)
                for key, (state, image, order) in self._rows.items())

    def close(self):
        self._rows.clear()


class _SqliteStaging(object):
    """SQLite-backed staging area used once the memory budget is exceeded."""

    def __init__(self, tempdir):
        fd, self.path = tempfile.mkstemp(prefix='petl-foldchanges-',
                                         suffix='.sqlite', dir=tempdir)
        os.close(fd)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute(
            'CREATE TABLE staged ('
            'k BLOB PRIMARY KEY, '
            'state TEXT NOT NULL, '
            'image BLOB, '
            'ord INTEGER NOT NULL)')
        self.connection.commit()
        self._ord = 0

    def __len__(self):
        cursor = self.connection.execute('SELECT COUNT(*) FROM staged')
        return cursor.fetchone()[0]

    def get(self, key):
        cursor = self.connection.execute(
            'SELECT state, image FROM staged WHERE k = ?',
            (pickle.dumps(key, -1),))
        row = cursor.fetchone()
        if row is None:
            return None
        state, image = row
        return state, pickle.loads(image) if image is not None else None

    def put(self, key, state, image):
        serialized_key = pickle.dumps(key, -1)
        serialized_image = pickle.dumps(image, -1) if image is not None \
            else None
        cursor = self.connection.execute(
            'SELECT ord FROM staged WHERE k = ?', (serialized_key,))
        row = cursor.fetchone()
        if row is None:
            order = self._ord
            self._ord += 1
            self.connection.execute(
                'INSERT INTO staged (k, state, image, ord) '
                'VALUES (?, ?, ?, ?)',
                (serialized_key, state, serialized_image, order))
        else:
            self.connection.execute(
                'UPDATE staged SET state = ?, image = ? WHERE k = ?',
                (state, serialized_image, serialized_key))

    def items_in_order(self):
        cursor = self.connection.execute(
            'SELECT k, state, image FROM staged ORDER BY ord')
        for serialized_key, state, serialized_image in cursor:
            key = pickle.loads(serialized_key)
            image = pickle.loads(serialized_image) \
                if serialized_image is not None else None
            yield key, state, image

    def close(self):
        try:
            self.connection.close()
        finally:
            try:
                os.unlink(self.path)
            except OSError:
                pass


class _Staging(object):
    """Start staging in memory, spilling to a temporary SQLite database."""

    def __init__(self, buffersize=None, tempdir=None):
        self._buffersize = buffersize
        self._tempdir = tempdir
        self._store = _MemoryStaging()
        self.spilled = False

    def _maybe_spill(self):
        if self.spilled or self._buffersize is None:
            return
        if len(self._store) <= self._buffersize:
            return
        store = _SqliteStaging(self._tempdir)
        try:
            for key, state, image in self._store.items_in_order():
                store.put(key, state, image)
        except Exception:
            store.close()
            raise
        self._store.close()
        self._store = store
        self.spilled = True

    def __len__(self):
        return len(self._store)

    def get(self, key):
        return self._store.get(key)

    def put(self, key, state, image):
        self._store.put(key, state, image)
        self._maybe_spill()

    def items_in_order(self):
        return self._store.items_in_order()

    def close(self):
        self._store.close()


class _Transaction(object):
    """State of the single open transaction.

    ``net`` stores the accumulated :class:`NetChange` per key. A key move is
    represented as a ``delete`` under the old key (before image retained) plus
    an ``insert`` under the new key; touching either side again folds into
    those two entries.
    """

    def __init__(self, key, lookup, buffersize, tempdir):
        self.key = key
        self._lookup = lookup
        self.staging = _Staging(buffersize, tempdir)
        # key -> NetChange being accumulated, in first-touch order
        self.net = OrderedDict()

    def close(self):
        self.staging.close()

    def _visible_row(self, key):
        entry = self.staging.get(key)
        if entry is not None:
            state, image = entry
            if state == _VISIBLE:
                return image
            return None
        return self._lookup(key)

    def _key_free(self, key):
        return self._visible_row(key) is None

    def _require_visible(self, key, before, position):
        visible = self._visible_row(key)
        if visible is None:
            raise ChangeConsistencyError(
                position, 'no row visible for key %r' % (key,))
        if not _images_equal(visible, before):
            raise ChangeConsistencyError(
                position,
                'before image %r does not match the visible row %r '
                'for key %r' % (before, visible, key))
        return visible

    def _record_insert(self, key, after, position):
        entry = self.net.get(key)
        if entry is None:
            self.net[key] = NetChange(_INSERT, key, None, after)
        elif entry.operation == _DELETE:
            # delete followed by an insert keeps the original before image
            self.net[key] = NetChange(_UPDATE, key, entry.before, after)
        else:
            raise ChangeConsistencyError(
                position, 'key %r is already inserted' % (key,))

    def _record_update(self, key, before, after):
        entry = self.net.get(key)
        if entry is None:
            self.net[key] = NetChange(_UPDATE, key, before, after)
        elif entry.operation == _INSERT:
            self.net[key] = NetChange(_INSERT, key, None, after)
        elif entry.operation == _UPDATE:
            self.net[key] = NetChange(_UPDATE, key, entry.before, after)
        else:  # delete entry: an update folds into an insert on this side
            self.net[key] = NetChange(_UPDATE, key, entry.before, after)

    def _record_delete(self, key, before, position):
        entry = self.net.get(key)
        if entry is None:
            self.net[key] = NetChange(_DELETE, key, before, None)
        elif entry.operation == _INSERT:
            # insert followed by delete cancels out entirely
            del self.net[key]
        elif entry.operation == _UPDATE:
            self.net[key] = NetChange(_DELETE, key, entry.before, None)
        else:
            raise ChangeConsistencyError(
                position, 'key %r is already deleted' % (key,))

    def apply_insert(self, after, position):
        _check_image(after, position, 'after')
        key = _row_key(after, self.key, position)
        if not self._key_free(key):
            raise ChangeConsistencyError(
                position, 'key %r is already occupied' % (key,))
        self.staging.put(key, _VISIBLE, after)
        self._record_insert(key, after, position)

    def apply_update(self, before, after, position):
        _check_image(before, position, 'before')
        _check_image(after, position, 'after')
        old_key = _row_key(before, self.key, position)
        new_key = _row_key(after, self.key, position)
        self._require_visible(old_key, before, position)
        if new_key != old_key and not self._key_free(new_key):
            raise ChangeConsistencyError(
                position, 'target key %r is already occupied' % (new_key,))
        if new_key == old_key:
            self.staging.put(old_key, _VISIBLE, after)
            self._record_update(old_key, before, after)
            return
        self.staging.put(old_key, _TOMBSTONE, None)
        self.staging.put(new_key, _VISIBLE, after)
        self._record_delete(old_key, before, position)
        self._record_insert(new_key, after, position)

    def apply_delete(self, before, position):
        _check_image(before, position, 'before')
        key = _row_key(before, self.key, position)
        self._require_visible(key, before, position)
        self.staging.put(key, _TOMBSTONE, None)
        self._record_delete(key, before, position)

    def net_changes(self):
        # an update whose before and after images are identical (e.g. a key
        # moved away and back) is not a net change
        return tuple(
            change for change in self.net.values()
            if not (change.operation == _UPDATE
                    and _images_equal(change.before, change.after)))


def iterfoldedchanges(table, key, lookup, buffersize=None, tempdir=None):
    """Generator yielding ``(ordinal, (NetChange, ...))`` per committed txn."""
    if isinstance(key, (list, tuple)):
        key = tuple(text_type(k) for k in key)
    else:
        key = (text_type(key),)

    position = 0
    transaction = None
    ordinal = 0

    try:
        source = iter(table)
        while True:
            try:
                row = next(source)
            except StopIteration:
                break
            position += 1
            event_type, before, after = _normalize_event(row, position)

            if event_type not in _VALID_EVENTS:
                raise ChangeStreamError(
                    position, 'unknown event type %r' % (event_type,))
            _check_event_images(event_type, before, after, position)

            if event_type == _BEGIN:
                if transaction is not None:
                    raise TransactionBoundaryError(
                        position,
                        'begin while transaction %s is still open'
                        % (ordinal + 1,))
                transaction = _Transaction(key, lookup, buffersize, tempdir)
            elif event_type in (_COMMIT, _ROLLBACK):
                if transaction is None:
                    raise TransactionBoundaryError(
                        position,
                        '%s without a matching begin' % (event_type,))
                if event_type == _COMMIT:
                    ordinal += 1
                    changes = transaction.net_changes()
                    transaction.close()
                    transaction = None
                    yield ordinal, changes
                else:
                    transaction.close()
                    transaction = None
            elif transaction is None:
                raise TransactionBoundaryError(
                    position,
                    '%s event outside of a transaction' % (event_type,))
            elif event_type == _INSERT:
                transaction.apply_insert(after, position)
            elif event_type == _UPDATE:
                transaction.apply_update(before, after, position)
            else:
                transaction.apply_delete(before, position)

        if transaction is not None:
            transaction.close()
            transaction = None
            raise UnclosedTransactionError(
                position, 'transaction %s was not committed or rolled back'
                % (ordinal + 1,))
    except ChangeEventError:
        raise
    except Exception as exc:
        raise ChangeSourceError(
            position, 'event source failed: %s' % (exc,)) from exc
    finally:
        if transaction is not None:
            # caller stopped iterating mid-transaction: emit nothing further
            transaction.close()
            transaction = None
