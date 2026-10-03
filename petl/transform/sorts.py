from __future__ import absolute_import, print_function, division


import os
import heapq
from tempfile import NamedTemporaryFile
import itertools
import logging
from collections import namedtuple
import operator
from petl.compat import pickle, next, text_type, string_types


import petl.config as config
from petl.comparison import sortable_itemgetter
from petl.errors import ArgumentError
from petl.util.base import Table, asindices


logger = logging.getLogger(__name__)
warning = logger.warning
info = logger.info
debug = logger.debug


_NULLS_FIRST = 'first'
_NULLS_LAST = 'last'
_NULLS_AUTO = 'auto'
_VALID_NULLS = (_NULLS_FIRST, _NULLS_LAST, _NULLS_AUTO)

_SHORT_FILL = 'fill'
_SHORT_RAISE = 'raise'
_SHORT_DISTINCT = 'distinct'
_VALID_SHORT_ROWS = (_SHORT_FILL, _SHORT_RAISE, _SHORT_DISTINCT)


class SortField(namedtuple('SortField', ['field', 'reverse', 'nulls'])):
    """Specification of a single sort key field.

    `field` is a field name or index, `reverse` is ``True`` for descending
    order and `nulls` is one of ``'first'``, ``'last'`` or ``'auto'``
    (``'auto'`` places null values first when ascending and last when
    descending).
    """

    __slots__ = ()

    def __new__(cls, field, reverse=False, nulls=_NULLS_AUTO):
        return super(SortField, cls).__new__(cls, field, reverse, nulls)


def _as_bool_direction(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, string_types):
        token = value.strip().lower()
        if token in ('asc', 'ascending'):
            return False
        if token in ('desc', 'descending'):
            return True
    raise ArgumentError(
        "sort direction must be True/False or 'asc'/'desc', found %r"
        % (value,))


def _check_nulls(value):
    if value not in _VALID_NULLS:
        raise ArgumentError(
            "nulls must be one of 'first', 'last' or 'auto', found %r"
            % (value,))


def _check_short_rows(value):
    if value not in _VALID_SHORT_ROWS:
        raise ArgumentError(
            "short_rows must be one of 'fill', 'raise' or 'distinct', "
            "found %r" % (value,))


def _parse_sortkey(sortkey, default_nulls):
    """Normalise a user-provided sortkey into a list of
    ``(field, descending, nulls)`` tuples."""
    if isinstance(sortkey, SortField):
        sortkey = [sortkey]
    elif not isinstance(sortkey, (list, tuple)):
        sortkey = [sortkey]
    specs = []
    for item in sortkey:
        if isinstance(item, SortField):
            field, descending, nulls = item
            descending = _as_bool_direction(descending)
        elif isinstance(item, (list, tuple)) and not isinstance(
                item, string_types):
            if len(item) == 1:
                field = item[0]
                descending = False
                nulls = default_nulls
            elif len(item) == 2:
                field, direction = item
                descending = _as_bool_direction(direction)
                nulls = default_nulls
            elif len(item) == 3:
                field, direction, nulls = item
                descending = _as_bool_direction(direction)
            else:
                raise ArgumentError(
                    'sortkey field spec must have 1, 2 or 3 elements, '
                    'found %r' % (item,))
        else:
            field = item
            descending = False
            nulls = default_nulls
        _check_nulls(nulls)
        specs.append((field, descending, nulls))
    if not specs:
        raise ArgumentError('sortkey must contain at least one field')
    return specs


def sort(table, key=None, reverse=False, sortkey=None, nulls=_NULLS_AUTO,
         short_rows=_SHORT_FILL, buffersize=None, tempdir=None, cache=True):
    """
    Sort the table. Field names or indices (from zero) can be used to specify
    the key. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar'],
        ...           ['C', 2],
        ...           ['A', 9],
        ...           ['A', 6],
        ...           ['F', 1],
        ...           ['D', 10]]
        >>> table2 = etl.sort(table1, 'foo')
        >>> table2
        +-----+-----+
        | foo | bar |
        +=====+=====+
        | 'A' |   9 |
        +-----+-----+
        | 'A' |   6 |
        +-----+-----+
        | 'C' |   2 |
        +-----+-----+
        | 'D' |  10 |
        +-----+-----+
        | 'F' |   1 |
        +-----+-----+

        >>> # sorting by compound key is supported
        ... table3 = etl.sort(table1, key=['foo', 'bar'])
        >>> table3
        +-----+-----+
        | foo | bar |
        +=====+=====+
        | 'A' |   6 |
        +-----+-----+
        | 'A' |   9 |
        +-----+-----+
        | 'C' |   2 |
        +-----+-----+
        | 'D' |  10 |
        +-----+-----+
        | 'F' |   1 |
        +-----+-----+

        >>> # if no key is specified, the default is a lexical sort
        ... table4 = etl.sort(table1)
        >>> table4
        +-----+-----+
        | foo | bar |
        +=====+=====+
        | 'A' |   6 |
        +-----+-----+
        | 'A' |   9 |
        +-----+-----+
        | 'C' |   2 |
        +-----+-----+
        | 'D' |  10 |
        +-----+-----+
        | 'F' |   1 |
        +-----+-----+

    Per-field ascending/descending order and null placement can be given via
    the `sortkey` argument, e.g. to sort an accounting export by account
    ascending then ``updated_at`` descending, with missing dates placed last
    within each account::

        >>> table5 = [['account', 'updated_at'],
        ...           ['B', None],
        ...           ['A', 3],
        ...           ['A', 9],
        ...           ['C', 2]]
        >>> table6 = etl.sort(table5,
        ...                   sortkey=['account',
        ...                            ('updated_at', 'desc', 'last')])
        >>> table6
        +---------+------------+
        | account | updated_at |
        +=========+============+
        | 'A'     |          9 |
        +---------+------------+
        | 'A'     |          3 |
        +---------+------------+
        | 'B'     | None       |
        +---------+------------+
        | 'C'     |          2 |
        +---------+------------+

    Each element of `sortkey` may be a bare field name/index, a
    ``(field, reverse)`` or ``(field, reverse, nulls)`` tuple, or a
    :class:`SortField`. Direction may be ``True``/``False`` or the strings
    ``'asc'``/``'desc'``; `nulls` may be ``'first'``, ``'last'`` or
    ``'auto'``. The `nulls` argument gives the default for every field and
    is overridden by a per-field `nulls` value.

    The `short_rows` argument controls how rows that are too short to
    contain a key field are handled: ``'fill'`` (the default, for backwards
    compatibility) treats the missing field as an actual ``None`` value;
    ``'raise'`` raises a :class:`petl.errors.SortError`; ``'distinct'``
    keeps the absent field separate from explicit ``None`` values, with
    absent fields sorting immediately beyond the null position (after
    ``None`` when nulls are first; after everything else when nulls are
    last).

    Rows with equal sort keys keep their original input order; whole rows are
    never compared to break ties.

    The `buffersize` argument should be an `int` or `None`.

    If the number of rows in the table is less than `buffersize`, the table
    will be sorted in memory. Otherwise, the table is sorted in chunks of
    no more than `buffersize` rows, each chunk is written to a temporary
    file, and then a merge sort is performed on the temporary files.

    If `buffersize` is `None`, the value of
    `petl.config.sort_buffersize` will be used. By default this is
    set to 100000 rows, but can be changed, e.g.::

        >>> import petl.config
        >>> petl.config.sort_buffersize = 500000

    If `petl.config.sort_buffersize` is set to `None`, this forces
    all sorting to be done entirely in memory.

    By default the results of the sort will be cached, and so a second pass
    over the sorted table will yield rows from the cache and will not repeat
    the sort operation. To turn off caching, set the `cache` argument to
    `False`.

    The `key`/`reverse` and `sortkey` calling styles cannot be mixed:
    passing both `key` and `sortkey`, or `reverse=True` together with
    `sortkey`, raises an :class:`petl.errors.ArgumentError`.

    """

    return SortView(table, key=key, reverse=reverse, sortkey=sortkey,
                    nulls=nulls, short_rows=short_rows,
                    buffersize=buffersize,
                    tempdir=tempdir, cache=cache)


Table.sort = sort


def _iterchunk(fn):
    # reopen so iterators from file cache are independent
    debug('iterchunk, opening %s' % fn)
    with open(fn, 'rb') as f:
        try:
            while True:
                yield pickle.load(f)
        except EOFError:
            pass
    debug('end of iterchunk, closed %s' % fn)


def _stabilized_merge(getkey, *iterables):
    """Stable k-way merge of iterables already ordered by `getkey`.

    Heap entries are ``(key, chunk_order, seq, row, iterator)``. The
    ``(chunk_order, seq)`` pair is globally unique, so ties between equal
    sort keys are resolved by original input position without ever comparing
    whole rows.
    """

    heap = []
    for order, iterable in enumerate(iterables):
        it = iter(iterable)
        try:
            first = next(it)
        except StopIteration:
            continue
        heap.append((getkey(first), order, 0, first, it))
    if not heap:
        return
    heapq.heapify(heap)
    while heap:
        key, order, seq, row, it = heapq.heappop(heap)
        yield row
        try:
            nxt = next(it)
        except StopIteration:
            continue
        heapq.heappush(heap, (getkey(nxt), order, seq + 1, nxt, it))


def _identitykey(row):
    return row


def _rowmergesorted(getkey, *iterables):
    # N.B., a single stable merge implementation is used whether or not the
    # sort direction is descending, because direction is encoded in the key
    if getkey is None:
        return _stabilized_merge(_identitykey, *iterables)
    return _stabilized_merge(getkey, *iterables)


def _build_keygetter(hdr, fieldspecs, reverse, nulls, short_rows):
    """Construct the key function and resolved field specs for a header.

    Returns ``(getkey, specs)`` where `specs` is a list of
    ``(index, descending, nulls, missing)`` tuples. Returns
    ``(None, None)`` for a legacy lexical sort (no key fields).
    """

    if fieldspecs is None:
        # legacy calling convention: key/reverse
        if hdr is None:
            indices = []
        else:
            indices = list(range(len(hdr)))
        specs = [(i, reverse, nulls, short_rows) for i in indices]
        if not specs:
            return None, None
        return sortable_itemgetter(*specs), specs

    # new calling convention
    if reverse:
        raise ArgumentError(
            "reverse must not be used together with sortkey; specify "
            "per-field direction in sortkey instead")
    parsed = _parse_sortkey(fieldspecs, nulls)
    if hdr is None:
        raise ArgumentError('a header is required to resolve sortkey fields')
    indices = asindices(hdr, [f for f, _, _ in parsed])
    specs = [(index, descending, field_nulls, short_rows)
             for index, (_, descending, field_nulls)
             in zip(indices, parsed)]
    return sortable_itemgetter(*specs), specs


class SortView(Table):
    def __init__(self, source, key=None, reverse=False, sortkey=None,
                 nulls=_NULLS_AUTO, short_rows=_SHORT_FILL, buffersize=None,
                 tempdir=None, cache=True):
        if key is not None and sortkey is not None:
            raise ArgumentError(
                "key and sortkey cannot be used together; choose one way "
                "of specifying the sort order")
        if sortkey is not None and reverse:
            raise ArgumentError(
                "reverse=True cannot be used together with sortkey; "
                "specify per-field direction in sortkey instead")
        if not isinstance(reverse, bool):
            raise ArgumentError(
                'reverse must be True or False, found %r' % (reverse,))
        _check_nulls(nulls)
        _check_short_rows(short_rows)
        self.source = source
        self.key = key
        self.reverse = reverse
        self.sortkey = sortkey
        self.nulls = nulls
        self.short_rows = short_rows
        if buffersize is None:
            self.buffersize = config.sort_buffersize
        else:
            self.buffersize = buffersize
        self.tempdir = tempdir
        self._cache = cache
        self._hdrcache = None
        self._memcache = None
        self._filecache = None
        self._getkey = None

    def clearcache(self):
        debug('clear cache')
        self._hdrcache = None
        self._memcache = None
        self._filecache = None
        self._getkey = None

    def __iter__(self):
        if self._cache and self._memcache is not None:
            return self._iterfrommemcache()
        elif self._cache and self._filecache is not None:
            return self._iterfromfilecache()
        else:
            return self._iternocache()

    def _iterfrommemcache(self):
        debug('iterate from memory cache')
        yield tuple(self._hdrcache)
        for row in self._memcache:
            yield tuple(row)

    def _iterfromfilecache(self):
        # create a reference to the filecache here, so cleanup happens in the
        # correct order
        filecache = self._filecache
        filenames = list(map(operator.attrgetter('name'), filecache))
        debug('iterate from file cache: %r', filenames)
        yield tuple(self._hdrcache)
        chunkiters = [_iterchunk(fn) for fn in filenames]
        rows = _rowmergesorted(self._getkey, *chunkiters)
        try:
            for row in rows:
                yield tuple(row)
        finally:
            debug('attempt cleanup from generator')
            # N.B., need to ensure that any open files are closed **before**
            # temporary files are deleted, as deletion will fail on Windows
            # if file is in use (i.e., still open)
            del chunkiters
            del rows
            del filecache
            debug('exiting generator')

    def _iternocache(self):
        debug('iterate without cache')
        self.clearcache()
        it = iter(self.source)

        try:
            hdr = next(it)
        except StopIteration:
            if self.key is None and self.sortkey is None:
                return  # nothing to do on a table without headers
            hdr = []
        yield tuple(hdr)

        if self.sortkey is not None:
            # new calling convention: per-field direction and null placement
            fieldspecs = self.sortkey
            getkey, _ = _build_keygetter(
                hdr, fieldspecs, False, self.nulls, self.short_rows)
        elif self.key is not None:
            # legacy calling convention: a single direction for all key
            # fields; null position follows the global reverse via 'auto'
            fieldspecs = [(idx, self.reverse, self.nulls)
                          for idx in asindices(hdr, self.key)]
            getkey, _ = _build_keygetter(
                hdr, fieldspecs, False, self.nulls, self.short_rows)
        else:
            # lexical sort over every header field
            getkey, _ = _build_keygetter(
                hdr, None, self.reverse, self.nulls, self.short_rows)

        # initialise the first chunk
        rows = list(itertools.islice(it, 0, self.buffersize))
        rows.sort(key=getkey)

        # have we exhausted the source iterator?
        if self.buffersize is None or len(rows) < self.buffersize:
            # yes, table fits within sort buffer

            if self._cache:
                debug('caching mem')
                self._hdrcache = hdr
                self._memcache = rows
                self._getkey = getkey

            for row in rows:
                yield tuple(row)

        else:
            # no, table is too big, need to sort in chunks

            chunkfiles = []

            while rows:

                # dump the chunk
                with NamedTemporaryFile(dir=self.tempdir, delete=False,
                                        mode='wb') as f:
                    # N.B., we **don't** want the file to be deleted on
                    # close, but we **do** want the file to be deleted when
                    # self is garbage collected, or when the program exits.
                    # When all references to the wrapper are gone, the file
                    # should get deleted.
                    wrapper = _NamedTempFileDeleteOnGC(f.name)
                    debug('created temporary chunk file %s' % f.name)
                    for row in rows:
                        pickle.dump(row, f, protocol=-1)
                    f.flush()
                    chunkfiles.append(wrapper)

                # grab the next chunk
                rows = list(itertools.islice(it, 0, self.buffersize))
                rows.sort(key=getkey)

            if self._cache:
                debug('caching files')
                self._hdrcache = hdr
                self._filecache = chunkfiles
                self._getkey = getkey

            chunkiters = [_iterchunk(f.name) for f in chunkfiles]
            for row in _rowmergesorted(getkey, *chunkiters):
                yield tuple(row)


class _NamedTempFileDeleteOnGC(object):

    def __init__(self, name):
        self.name = name

    def delete(self, unlink=os.unlink, log=logger.debug):
        name = self.name
        try:
            log('deleting %s' % name)
            unlink(name)
        except Exception as e:
            log('exception deleting %s: %s' % (name, e))
            raise
        else:
            log('deleted %s' % name)

    def __del__(self):
        self.delete()

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


def mergesort(*tables, **kwargs):
    """
    Combine multiple input tables into one sorted output table. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar'],
        ...           ['A', 9],
        ...           ['C', 2],
        ...           ['D', 10],
        ...           ['A', 6],
        ...           ['F', 1]]
        >>> table2 = [['foo', 'bar'],
        ...           ['B', 3],
        ...           ['D', 10],
        ...           ['A', 10],
        ...           ['F', 4]]
        >>> table3 = etl.mergesort(table1, table2, key='foo')
        >>> table3.lookall()
        +-----+-----+
        | foo | bar |
        +=====+=====+
        | 'A' |   9 |
        +-----+-----+
        | 'A' |   6 |
        +-----+-----+
        | 'A' |  10 |
        +-----+-----+
        | 'B' |   3 |
        +-----+-----+
        | 'C' |   2 |
        +-----+-----+
        | 'D' |  10 |
        +-----+-----+
        | 'D' |  10 |
        +-----+-----+
        | 'F' |   1 |
        +-----+-----+
        | 'F' |   4 |
        +-----+-----+

    If the input tables are already sorted by the given key, give
    ``presorted=True`` as a keyword argument.

    This function is equivalent to concatenating the input tables using
    :func:`cat` then sorting, however this function will typically be more
    efficient, especially if the input tables are presorted.

    Keyword arguments:

    key : string or tuple of strings, optional
        Field name or tuple of fields to sort by (defaults to `None`, lexical
        sort)
    reverse : bool, optional
        `True` for descending order (defaults to `False`)
    sortkey : sequence, optional
        Per-field sort specifications, as accepted by :func:`sort`; cannot be
        combined with `key` or `reverse=True`
    nulls : {'first', 'last', 'auto'}, optional
        Default null placement for the sort fields (defaults to ``'auto'``)
    short_rows : {'fill', 'raise', 'distinct'}, optional
        How to treat rows missing a key field (defaults to ``'fill'``)
    presorted : bool, optional
        `True` if inputs are already sorted by the given key (defaults to
        `False`)
    missing : object
        Value to fill with when input tables have different fields (defaults
        to `None`)
    header : sequence of strings, optional
        Specify a fixed header for the output table
    buffersize : int, optional
        Limit the number of rows in memory per input table when inputs are
        not presorted

    """

    return MergeSortView(tables, **kwargs)


Table.mergesort = mergesort


class MergeSortView(Table):
    def __init__(self, tables, key=None, reverse=False, sortkey=None,
                 nulls=_NULLS_AUTO, short_rows=_SHORT_FILL, presorted=False,
                 missing=None, header=None, buffersize=None, tempdir=None,
                 cache=True):
        if key is not None and sortkey is not None:
            raise ArgumentError(
                "key and sortkey cannot be used together; choose one way "
                "of specifying the sort order")
        if sortkey is not None and reverse:
            raise ArgumentError(
                "reverse=True cannot be used together with sortkey; "
                "specify per-field direction in sortkey instead")
        if not isinstance(reverse, bool):
            raise ArgumentError(
                'reverse must be True or False, found %r' % (reverse,))
        _check_nulls(nulls)
        _check_short_rows(short_rows)
        self.key = key
        self.reverse = reverse
        self.sortkey = sortkey
        self.nulls = nulls
        self.short_rows = short_rows
        if presorted:
            self.tables = tables
        else:
            self.tables = [sort(t, key=key, reverse=reverse, sortkey=sortkey,
                                nulls=nulls, short_rows=short_rows,
                                buffersize=buffersize, tempdir=tempdir,
                                cache=cache)
                           for t in tables]
        self.missing = missing
        self._header = header

    def __iter__(self):
        return itermergesort(self.tables, self.key, self.reverse,
                             self.sortkey, self.nulls, self.short_rows,
                             self._header, self.missing)


def itermergesort(sources, key, reverse, sortkey, nulls, short_rows, header,
                  missing):
    # first need to standardise headers of all input tables
    # borrow this from itercat - TODO remove code smells

    its = [iter(t) for t in sources]
    src_hdrs = []
    for it in its:
        try:
            src_hdrs.append(next(it))
        except StopIteration:
            src_hdrs.append([])

    if header is None:
        # determine output fields by gathering all the fields in the sources
        outhdr = list()
        for hdr in src_hdrs:
            for f in list(map(text_type, hdr)):
                if f not in outhdr:
                    # add any new fields as we find them
                    outhdr.append(f)
    else:
        # predetermined output fields
        outhdr = header
    yield tuple(outhdr)

    def _standardisedata(it, hdr, ofs):
        flds = list(map(text_type, hdr))
        # now construct and yield the data rows
        for _row in it:
            try:
                # should be quickest to do this way
                yield tuple(_row[flds.index(fo)] if fo in flds else missing
                            for fo in ofs)
            except IndexError:
                # handle short rows
                outrow = [missing] * len(ofs)
                for i, fi in enumerate(flds):
                    try:
                        outrow[ofs.index(fi)] = _row[i]
                    except IndexError:
                        pass  # be relaxed about short rows
                yield tuple(outrow)

    # wrap all iterators to standardise fields
    sits = [_standardisedata(it, hdr, outhdr)
            for hdr, it in zip(src_hdrs, its)]

    # now determine the key function using the same comparison rules as
    # sort(), so in-memory sorting, chunking, merging and cache re-reads all
    # agree on ordering
    if sortkey is not None:
        getkey, _ = _build_keygetter(outhdr, sortkey, False, nulls,
                                     short_rows)
    elif key is not None:
        fieldspecs = [(idx, reverse, nulls)
                      for idx in asindices(outhdr, key)]
        getkey, _ = _build_keygetter(outhdr, fieldspecs, False, nulls,
                                     short_rows)
    else:
        getkey, _ = _build_keygetter(outhdr, None, reverse, nulls,
                                     short_rows)

    # OK, do the stable merge sort
    for row in _rowmergesorted(getkey, *sits):
        yield row


def issorted(table, key=None, reverse=False, sortkey=None, nulls='auto',
             short_rows='fill', strict=False):
    """
    Return True if the table is ordered (i.e., sorted) by the given key.
    Supports the same `key`/`reverse` and `sortkey`/`nulls`/`short_rows`
    arguments as :func:`sort`. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar', 'baz'],
        ...           ['a', 1, True],
        ...           ['b', 3, True],
        ...           ['b', 2]]
        >>> etl.issorted(table1, key='foo')
        True
        >>> etl.issorted(table1, key='bar')
        False
        >>> etl.issorted(table1, key='foo', strict=True)
        False
        >>> etl.issorted(table1, key='foo', reverse=True)
        False

    """

    it = iter(table)
    try:
        hdr = [text_type(f) for f in next(it)]
    except StopIteration:
        return True

    if key is not None and sortkey is not None:
        raise ArgumentError(
            "key and sortkey cannot be used together; choose one way "
            "of specifying the sort order")
    if sortkey is not None and reverse:
        raise ArgumentError(
            "reverse=True cannot be used together with sortkey; "
            "specify per-field direction in sortkey instead")

    if sortkey is not None:
        getkey, _ = _build_keygetter(hdr, sortkey, False, nulls, short_rows)
    elif key is not None:
        fieldspecs = [(idx, reverse, nulls)
                      for idx in asindices(hdr, key)]
        getkey, _ = _build_keygetter(hdr, fieldspecs, False, nulls,
                                     short_rows)
    else:
        getkey, _ = _build_keygetter(hdr, None, reverse, nulls, short_rows)

    try:
        prevkey = getkey(next(it))
    except StopIteration:
        return True
    for curr in it:
        currkey = getkey(curr)
        if strict:
            # each subsequent key must be strictly greater
            if not prevkey < currkey:
                return False
        else:
            # keys must be non-decreasing
            if currkey < prevkey:
                return False
        prevkey = currkey
    return True


Table.issorted = issorted
