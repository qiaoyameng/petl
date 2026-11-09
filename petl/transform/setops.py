from __future__ import absolute_import, print_function, division

import itertools

from collections import Counter
from petl.compat import next, text_type
from petl.comparison import Comparable
from petl.errors import ArgumentError, FieldSelectionError
from petl.util.base import header, Table
from petl.transform.sorts import sort
from petl.transform.basics import cut


def complement(a, b, presorted=False, buffersize=None, tempdir=None,
               cache=True, strict=False):
    """
    Return rows in `a` that are not in `b`. E.g.::

        >>> import petl as etl
        >>> a = [['foo', 'bar', 'baz'],
        ...      ['A', 1, True],
        ...      ['C', 7, False],
        ...      ['B', 2, False],
        ...      ['C', 9, True]]
        >>> b = [['x', 'y', 'z'],
        ...      ['B', 2, False],
        ...      ['A', 9, False],
        ...      ['B', 3, True],
        ...      ['C', 9, True]]
        >>> aminusb = etl.complement(a, b)
        >>> aminusb
        +-----+-----+-------+
        | foo | bar | baz   |
        +=====+=====+=======+
        | 'A' |   1 | True  |
        +-----+-----+-------+
        | 'C' |   7 | False |
        +-----+-----+-------+

        >>> bminusa = etl.complement(b, a)
        >>> bminusa
        +-----+---+-------+
        | x   | y | z     |
        +=====+===+=======+
        | 'A' | 9 | False |
        +-----+---+-------+
        | 'B' | 3 | True  |
        +-----+---+-------+

    Note that the field names of each table are ignored - rows are simply
    compared following a lexical sort. See also the
    :func:`petl.transform.setops.recordcomplement` function.

    If `presorted` is True, it is assumed that the data are already sorted by
    the given key, and the `buffersize`, `tempdir` and `cache` arguments are
    ignored. Otherwise, the data are sorted, see also the discussion of the
    `buffersize`, `tempdir` and `cache` arguments under the
    :func:`petl.transform.sorts.sort` function.

    Note that the default behaviour is not strictly set-like, because 
    duplicate rows are counted separately, e.g.::
    
        >>> a = [['foo', 'bar'],
        ...      ['A', 1],
        ...      ['B', 2],
        ...      ['B', 2],
        ...      ['C', 7]]
        >>> b = [['foo', 'bar'],
        ...      ['B', 2]]
        >>> aminusb = etl.complement(a, b)
        >>> aminusb
        +-----+-----+
        | foo | bar |
        +=====+=====+
        | 'A' |   1 |
        +-----+-----+
        | 'B' |   2 |
        +-----+-----+
        | 'C' |   7 |
        +-----+-----+

    This behaviour can be changed with the `strict` keyword argument, e.g.::

        >>> aminusb = etl.complement(a, b, strict=True)
        >>> aminusb
        +-----+-----+
        | foo | bar |
        +=====+=====+
        | 'A' |   1 |
        +-----+-----+
        | 'C' |   7 |
        +-----+-----+

    .. versionchanged:: 1.1.0
    
    If `strict` is `True` then strict set-like behaviour is used, i.e., 
    only rows in `a` not found in `b` are returned.

    """

    return ComplementView(a, b, presorted=presorted, buffersize=buffersize,
                          tempdir=tempdir, cache=cache, strict=strict)


Table.complement = complement


class ComplementView(Table):

    def __init__(self, a, b, presorted=False, buffersize=None, tempdir=None,
                 cache=True, strict=False):
        if presorted:
            self.a = a
            self.b = b
        else:
            self.a = sort(a, buffersize=buffersize, tempdir=tempdir,
                          cache=cache)
            self.b = sort(b, buffersize=buffersize, tempdir=tempdir,
                          cache=cache)
        self.strict = strict

    def __iter__(self):
        return itercomplement(self.a, self.b, self.strict)


def itercomplement(ta, tb, strict):
    # coerce rows to tuples to ensure hashable and comparable
    ita = (tuple(row) for row in iter(ta))
    itb = (tuple(row) for row in iter(tb))
    ahdr = tuple(next(ita))
    next(itb)  # ignore b fields
    yield ahdr

    try:
        a = next(ita)
    except StopIteration:
        pass
    else:
        try:
            b = next(itb)
        except StopIteration:
            yield a
            for row in ita:
                yield row
        else:
            # we want the elements in a that are not in b
            while True:
                if b is None or Comparable(a) < Comparable(b):
                    yield a
                    try:
                        a = next(ita)
                    except StopIteration:
                        break
                elif a == b:
                    try:
                        a = next(ita)
                    except StopIteration:
                        break
                    if not strict:
                        try:
                            b = next(itb)
                        except StopIteration:
                            b = None
                else:
                    try:
                        b = next(itb)
                    except StopIteration:
                        b = None


def recordcomplement(a, b, buffersize=None, tempdir=None, cache=True,
                     strict=False):
    """
    Find records in `a` that are not in `b`. E.g.::

        >>> import petl as etl
        >>> a = [['foo', 'bar', 'baz'],
        ...      ['A', 1, True],
        ...      ['C', 7, False],
        ...      ['B', 2, False],
        ...      ['C', 9, True]]
        >>> b = [['bar', 'foo', 'baz'],
        ...      [2, 'B', False],
        ...      [9, 'A', False],
        ...      [3, 'B', True],
        ...      [9, 'C', True]]
        >>> aminusb = etl.recordcomplement(a, b)
        >>> aminusb
        +-----+-----+-------+
        | foo | bar | baz   |
        +=====+=====+=======+
        | 'A' |   1 | True  |
        +-----+-----+-------+
        | 'C' |   7 | False |
        +-----+-----+-------+

        >>> bminusa = etl.recordcomplement(b, a)
        >>> bminusa
        +-----+-----+-------+
        | bar | foo | baz   |
        +=====+=====+=======+
        |   3 | 'B' | True  |
        +-----+-----+-------+
        |   9 | 'A' | False |
        +-----+-----+-------+

    Note that both tables must have the same set of fields, but that the order
    of the fields does not matter. See also the
    :func:`petl.transform.setops.complement` function.

    See also the discussion of the `buffersize`, `tempdir` and `cache` arguments
    under the :func:`petl.transform.sorts.sort` function.

    """

    # TODO possible with only one pass?

    ha = header(a)
    hb = header(b)
    assert set(ha) == set(hb), 'both tables must have the same set of fields'
    # make sure fields are in the same order
    bv = cut(b, *ha)
    return complement(a, bv, buffersize=buffersize, tempdir=tempdir,
                      cache=cache, strict=strict)


Table.recordcomplement = recordcomplement


def diff(a, b, presorted=False, buffersize=None, tempdir=None, cache=True,
         strict=False):
    """
    Find the difference between rows in two tables. Returns a pair of tables.
    E.g.::

        >>> import petl as etl
        >>> a = [['foo', 'bar', 'baz'],
        ...      ['A', 1, True],
        ...      ['C', 7, False],
        ...      ['B', 2, False],
        ...      ['C', 9, True]]
        >>> b = [['x', 'y', 'z'],
        ...      ['B', 2, False],
        ...      ['A', 9, False],
        ...      ['B', 3, True],
        ...      ['C', 9, True]]
        >>> added, subtracted = etl.diff(a, b)
        >>> # rows in b not in a
        ... added
        +-----+---+-------+
        | x   | y | z     |
        +=====+===+=======+
        | 'A' | 9 | False |
        +-----+---+-------+
        | 'B' | 3 | True  |
        +-----+---+-------+

        >>> # rows in a not in b
        ... subtracted
        +-----+-----+-------+
        | foo | bar | baz   |
        +=====+=====+=======+
        | 'A' |   1 | True  |
        +-----+-----+-------+
        | 'C' |   7 | False |
        +-----+-----+-------+

    Convenient shorthand for ``(complement(b, a), complement(a, b))``. See also
    :func:`petl.transform.setops.complement`.

    If `presorted` is True, it is assumed that the data are already sorted by
    the given key, and the `buffersize`, `tempdir` and `cache` arguments are
    ignored. Otherwise, the data are sorted, see also the discussion of the
    `buffersize`, `tempdir` and `cache` arguments under the
    :func:`petl.transform.sorts.sort` function.

    .. versionchanged:: 1.1.0

    If `strict` is `True` then strict set-like behaviour is used.

    """

    if not presorted:
        a = sort(a)
        b = sort(b)
    added = complement(b, a, presorted=True, buffersize=buffersize,
                       tempdir=tempdir, cache=cache, strict=strict)
    subtracted = complement(a, b, presorted=True, buffersize=buffersize,
                            tempdir=tempdir, cache=cache, strict=strict)
    return added, subtracted


Table.diff = diff


def recorddiff(a, b, buffersize=None, tempdir=None, cache=True, strict=False):
    """
    Find the difference between records in two tables. E.g.::

        >>> import petl as etl
        >>> a = [['foo', 'bar', 'baz'],
        ...      ['A', 1, True],
        ...      ['C', 7, False],
        ...      ['B', 2, False],
        ...      ['C', 9, True]]
        >>> b = [['bar', 'foo', 'baz'],
        ...      [2, 'B', False],
        ...      [9, 'A', False],
        ...      [3, 'B', True],
        ...      [9, 'C', True]]
        >>> added, subtracted = etl.recorddiff(a, b)
        >>> added
        +-----+-----+-------+
        | bar | foo | baz   |
        +=====+=====+=======+
        |   3 | 'B' | True  |
        +-----+-----+-------+
        |   9 | 'A' | False |
        +-----+-----+-------+

        >>> subtracted
        +-----+-----+-------+
        | foo | bar | baz   |
        +=====+=====+=======+
        | 'A' |   1 | True  |
        +-----+-----+-------+
        | 'C' |   7 | False |
        +-----+-----+-------+

    Convenient shorthand for
    ``(recordcomplement(b, a), recordcomplement(a, b))``. See also
    :func:`petl.transform.setops.recordcomplement`.

    See also the discussion of the `buffersize`, `tempdir` and `cache`
    arguments under the :func:`petl.transform.sorts.sort` function.

    .. versionchanged:: 1.1.0

    If `strict` is `True` then strict set-like behaviour is used.

    """

    added = recordcomplement(b, a, buffersize=buffersize, tempdir=tempdir,
                             cache=cache, strict=strict)
    subtracted = recordcomplement(a, b, buffersize=buffersize, tempdir=tempdir,
                                  cache=cache, strict=strict)
    return added, subtracted


Table.recorddiff = recorddiff


def keydiff(a, b, key, buffersize=None, tempdir=None, cache=True):
    """
    Reconcile two tables by a common (compound) key and return a lazy
    stream of changes that transforms `a` into `b`. E.g.::

        >>> import petl as etl
        >>> a = [['id', 'name', 'qty'],
        ...      [1, 'apple', 3],
        ...      [2, 'pear', 1],
        ...      [3, 'plum', 7]]
        >>> b = [['qty', 'name', 'id'],
        ...      [3, 'apple', 1],
        ...      [5, 'pear', 2],
        ...      [2, 'fig', 4]]
        >>> changes = etl.keydiff(a, b, key='id')
        >>> changes
        +----------+------+----------------+----------------+
        | change   | key  | before         | after          |
        +==========+======+================+================+
        | 'update' | (2,) | (2, 'pear', 1) | (2, 'pear', 5) |
        +----------+------+----------------+----------------+
        | 'delete' | (3,) | (3, 'plum', 7) | None           |
        +----------+------+----------------+----------------+
        | 'insert' | (4,) | None           | (4, 'fig', 2)  |
        +----------+------+----------------+----------------+

    The two tables may list their fields in any order; rows are aligned by
    field name before comparison. Both tables must contain every field in
    `key`, must not have duplicate field names, and must have the same set
    of non-key fields, otherwise an error is raised. Note that validation
    is performed lazily, when the returned table is iterated, so that
    one-pass (iterator) inputs are supported.

    Within each key, rows from the two tables are first cancelled
    one-to-one by their complete row values, so if the same record appears
    three times in `a` and twice in `b`, exactly one ``delete`` change is
    emitted. If exactly one row remains on each side after cancellation,
    an ``update`` change is emitted, keeping the ``before`` and ``after``
    rows. In all other cases the remaining rows are emitted as ``delete``
    (from ``a``) or ``insert`` (from ``b``) changes - no attempt is made
    to guess pairings.

    The output table has the fields ``('change', 'key', 'before',
    'after')``. The ``key`` value is a tuple of the key field values, in
    the order the key fields were given. The ``before`` and ``after``
    values are tuples of the complete row values in a canonical field
    order (key fields first, in the order given, followed by the non-key
    fields in the order they appear in `a`), and are ``None`` for
    ``insert`` and ``delete`` changes respectively. Changes are emitted in
    a stable order, sorted by key and then by row content.

    Only `str`, `int` and `None` values are currently supported (`bool`
    values are rejected, because ``True == 1`` would corrupt the
    cancellation logic). ``None`` is distinct from the empty string, and
    key values may be ``None``.

    Internally both tables are sorted using
    :func:`petl.transform.sorts.sort`, so the `buffersize`, `tempdir` and
    `cache` arguments behave as described there, and arbitrarily large
    keys are handled via external sorting without materialising a whole
    key group in memory. If the returned iterator is closed early, or the
    underlying sources fail while being iterated, temporary files created
    for the current pass are cleaned up.

    See also :func:`petl.transform.setops.diff` and
    :func:`petl.transform.setops.recorddiff`, whose semantics are
    unchanged by this function.

    .. versionadded:: 0.1.dev2

    """

    return KeyDiffView(a, b, key, buffersize=buffersize, tempdir=tempdir,
                       cache=cache)


Table.keydiff = keydiff


_KEYDIFF_INSERT = 'insert'
_KEYDIFF_DELETE = 'delete'
_KEYDIFF_UPDATE = 'update'


class KeyDiffView(Table):

    def __init__(self, a, b, key, buffersize=None, tempdir=None,
                 cache=True):
        self.a = a
        self.b = b
        if isinstance(key, (text_type, int)):
            key = (key,)
        self.key = tuple(key)
        if not self.key:
            raise ArgumentError('at least one key field is required')
        self.buffersize = buffersize
        self.tempdir = tempdir
        self._cache = cache

    def __len__(self):
        # N.B., the base class computes len() by iterating the whole table,
        # which list() also uses as a length hint; for a lazy change stream
        # over possibly one-pass sources that would consume the inputs an
        # extra time, so the length is reported as unknown instead
        raise TypeError(
            'length of a keydiff stream is not known without consuming it')

    def __iter__(self):
        key = self.key
        ita = iter(self.a)
        itb = iter(self.b)
        try:
            hda = tuple(next(ita))
        except StopIteration:
            raise ArgumentError('table `a` has no header row')
        try:
            hdb = tuple(next(itb))
        except StopIteration:
            raise ArgumentError('table `b` has no header row')
        _keydiff_validate_header(hda, key, 'a')
        _keydiff_validate_header(hdb, key, 'b')
        keyset = set(key)
        othersa = tuple(f for f in hda if f not in keyset)
        othersb = tuple(f for f in hdb if f not in keyset)
        if set(othersa) != set(othersb):
            raise ArgumentError(
                'both tables must have the same set of non-key fields, '
                'found %r and %r' % (sorted(set(othersa)),
                                     sorted(set(othersb))))
        # canonical field order: key fields (as given), then non-key
        # fields in the order they appear in `a`
        canonical = tuple(key) + othersa
        projecta = _keydiff_projector(hda, canonical)
        projectb = _keydiff_projector(hdb, canonical)
        # project both tables into the canonical field order, then sort
        # lexicographically by the whole row, which groups rows by key and
        # orders rows within each key by content; sorting reuses the
        # external (chunked) sort, so inputs may be traversed only once
        # and arbitrarily large keys are supported
        ta = itertools.chain([canonical], (projecta(row) for row in ita))
        tb = itertools.chain([canonical], (projectb(row) for row in itb))
        sa = sort(ta, buffersize=self.buffersize, tempdir=self.tempdir,
                  cache=self._cache)
        sb = sort(tb, buffersize=self.buffersize, tempdir=self.tempdir,
                  cache=self._cache)
        return _iterkeydiff(sa, sb, len(key))


def _keydiff_validate_header(hdr, key, side):
    if len(set(hdr)) != len(hdr):
        dupes = sorted({f for f in hdr if hdr.count(f) > 1})
        raise ArgumentError(
            'table `%s` has duplicate field names: %r' % (side, dupes))
    for f in key:
        if f not in hdr:
            raise FieldSelectionError(f)


def _keydiff_projector(hdr, canonical):
    indices = tuple(hdr.index(f) for f in canonical)

    def project(row):
        row = tuple(row)
        out = tuple(row[i] for i in indices)
        for v in out:
            if v is not None and not isinstance(v, text_type) \
                    and (not isinstance(v, int) or isinstance(v, bool)):
                raise TypeError(
                    'keydiff supports str, int and None values only, '
                    'found %r' % (v,))
        return out

    return project


def _iterkeydiff(sa, sb, nkeys):
    # N.B., this is a generator; if it is closed early (e.g., the consumer
    # abandons the stream) or the sources raise while being iterated, the
    # finally block closes the underlying sort iterators, which releases
    # any temporary chunk files created for this pass
    sita = iter(sa)
    sitb = iter(sb)
    try:
        next(sita)  # discard canonical header
        next(sitb)
        yield ('change', 'key', 'before', 'after')
        a = next(sita, None)
        b = next(sitb, None)
        while a is not None or b is not None:
            if a is not None and (b is None or Comparable(a[:nkeys])
                                  < Comparable(b[:nkeys])):
                # key present in `a` only, all rows are deletes
                yield (_KEYDIFF_DELETE, a[:nkeys], a, None)
                a = next(sita, None)
            elif b is not None and (a is None or Comparable(b[:nkeys])
                                    < Comparable(a[:nkeys])):
                # key present in `b` only, all rows are inserts
                yield (_KEYDIFF_INSERT, b[:nkeys], None, b)
                b = next(sitb, None)
            else:
                # key present in both tables, reconcile the group; both
                # sides are sorted by complete row content within the key,
                # so equal rows can be cancelled one-to-one in a single
                # streaming pass, buffering at most one leftover row per
                # side (needed to detect the update case)
                curkey = a[:nkeys]
                pending = []  # (side, row) leftovers, in encounter order
                streaming = False
                while (a is not None and a[:nkeys] == curkey) \
                        or (b is not None and b[:nkeys] == curkey):
                    ina = a is not None and a[:nkeys] == curkey
                    inb = b is not None and b[:nkeys] == curkey
                    if ina and inb and a == b:
                        # rows cancel one-to-one
                        a = next(sita, None)
                        b = next(sitb, None)
                        continue
                    if ina and (not inb or Comparable(a) < Comparable(b)):
                        side, row = 'a', a
                        a = next(sita, None)
                    else:
                        side, row = 'b', b
                        b = next(sitb, None)
                    if streaming:
                        yield _keydiff_change(side, row, nkeys)
                    else:
                        pending.append((side, row))
                        counts = Counter(s for s, _ in pending)
                        if counts['a'] > 1 or counts['b'] > 1:
                            # an update is no longer possible for this
                            # key, flush the buffer and stream the rest
                            for s, r in pending:
                                yield _keydiff_change(s, r, nkeys)
                            pending = []
                            streaming = True
                if len(pending) == 2 and pending[0][0] != pending[1][0]:
                    # exactly one leftover row on each side -> update
                    if pending[0][0] == 'a':
                        before, after = pending[0][1], pending[1][1]
                    else:
                        before, after = pending[1][1], pending[0][1]
                    yield (_KEYDIFF_UPDATE, curkey, before, after)
                else:
                    for s, r in pending:
                        yield _keydiff_change(s, r, nkeys)
    finally:
        for it in (sita, sitb):
            close = getattr(it, 'close', None)
            if close is not None:
                close()


def _keydiff_change(side, row, nkeys):
    if side == 'a':
        return (_KEYDIFF_DELETE, row[:nkeys], row, None)
    else:
        return (_KEYDIFF_INSERT, row[:nkeys], None, row)


def intersection(a, b, presorted=False, buffersize=None, tempdir=None,
                 cache=True):
    """
    Return rows in `a` that are also in `b`. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar', 'baz'],
        ...           ['A', 1, True],
        ...           ['C', 7, False],
        ...           ['B', 2, False],
        ...           ['C', 9, True]]
        >>> table2 = [['x', 'y', 'z'],
        ...           ['B', 2, False],
        ...           ['A', 9, False],
        ...           ['B', 3, True],
        ...           ['C', 9, True]]
        >>> table3 = etl.intersection(table1, table2)
        >>> table3
        +-----+-----+-------+
        | foo | bar | baz   |
        +=====+=====+=======+
        | 'B' |   2 | False |
        +-----+-----+-------+
        | 'C' |   9 | True  |
        +-----+-----+-------+

    If `presorted` is True, it is assumed that the data are already sorted by
    the given key, and the `buffersize`, `tempdir` and `cache` arguments are
    ignored. Otherwise, the data are sorted, see also the discussion of the
    `buffersize`, `tempdir` and `cache` arguments under the
    :func:`petl.transform.sorts.sort` function.

    """

    return IntersectionView(a, b, presorted=presorted, buffersize=buffersize,
                            tempdir=tempdir, cache=cache)


Table.intersection = intersection


class IntersectionView(Table):

    def __init__(self, a, b, presorted=False, buffersize=None, tempdir=None,
                 cache=True):
        if presorted:
            self.a = a
            self.b = b
        else:
            self.a = sort(a, buffersize=buffersize, tempdir=tempdir,
                          cache=cache)
            self.b = sort(b, buffersize=buffersize, tempdir=tempdir,
                          cache=cache)

    def __iter__(self):
        return iterintersection(self.a, self.b)


def iterintersection(a, b):
    ita = iter(a)
    itb = iter(b)
    ahdr = next(ita)
    next(itb)  # ignore b header
    yield tuple(ahdr)
    try:
        a = tuple(next(ita))
        b = tuple(next(itb))
        while True:
            if Comparable(a) < Comparable(b):
                a = tuple(next(ita))
            elif a == b:
                yield a
                a = tuple(next(ita))
                b = tuple(next(itb))
            else:
                b = tuple(next(itb))
    except StopIteration:
        pass


def hashcomplement(a, b, strict=False):
    """
    Alternative implementation of :func:`petl.transform.setops.complement`,
    where the complement is executed by constructing an in-memory set for all
    rows found in the right hand table, then iterating over rows from the
    left hand table.

    May be faster and/or more resource efficient where the right table is small
    and the left table is large.
    
    .. versionchanged:: 1.1.0
    
    If `strict` is `True` then strict set-like behaviour is used, i.e., 
    only rows in `a` not found in `b` are returned.

    """

    return HashComplementView(a, b, strict=strict)


Table.hashcomplement = hashcomplement


class HashComplementView(Table):

    def __init__(self, a, b, strict=False):
        self.a = a
        self.b = b
        self.strict = strict

    def __iter__(self):
        return iterhashcomplement(self.a, self.b, self.strict)


def iterhashcomplement(a, b, strict):
    ita = iter(a)
    ahdr = next(ita)
    yield tuple(ahdr)
    itb = iter(b)
    next(itb)  # discard b header, assume same as a

    # N.B., need to account for possibility of duplicate rows
    bcnt = Counter(tuple(row) for row in itb)
    for ar in ita:
        t = tuple(ar)
        if bcnt[t] > 0:
            if not strict:
                bcnt[t] -= 1
        else:
            yield t


def hashintersection(a, b):
    """
    Alternative implementation of
    :func:`petl.transform.setops.intersection`, where the intersection
    is executed by constructing an in-memory set for all rows found in the
    right hand table, then iterating over rows from the left hand table.

    May be faster and/or more resource efficient where the right table is small
    and the left table is large.

    """

    return HashIntersectionView(a, b)


Table.hashintersection = hashintersection


class HashIntersectionView(Table):

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __iter__(self):
        return iterhashintersection(self.a, self.b)


def iterhashintersection(a, b):
    ita = iter(a)
    ahdr = next(ita)
    yield tuple(ahdr)
    itb = iter(b)
    next(itb)  # discard b header, assume same as a

    # N.B., need to account for possibility of duplicate rows
    bcnt = Counter(tuple(row) for row in itb)
    for ar in ita:
        t = tuple(ar)
        if bcnt[t] > 0:
            yield t
            bcnt[t] -= 1
