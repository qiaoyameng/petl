from __future__ import absolute_import, print_function, division


from petl.comparison import comparable_itemgetter
from petl.errors import DuplicateKeyError, SCD2Error
from petl.transform.sorts import sort
from petl.util.base import Table, asindices, rowgetter


def scd2(history, snapshot, natural_key, tracked_fields, valid_from,
         valid_to, effective, buffersize=None, tempdir=None):
    """
    Apply a full daily snapshot to a type 2 slowly changing dimension
    history table and return the new history table.

    E.g.::

        >>> import petl as etl
        >>> history = [['cust', 'tier', 'valid_from', 'valid_to']]
        >>> day1 = [['cust', 'tier'], ['C1', 'gold'], ['C2', 'silver']]
        >>> h = etl.scd2(history, day1, 'cust', 'tier',
        ...              'valid_from', 'valid_to', 1)
        >>> h.lookall()
        +------+----------+------------+----------+
        | cust | tier     | valid_from | valid_to |
        +======+==========+============+==========+
        | 'C1' | 'gold'   |          1 | None     |
        +------+----------+------------+----------+
        | 'C2' | 'silver' |          1 | None     |
        +------+----------+------------+----------+

    A later snapshot closes changed and missing keys while leaving
    unchanged rows alone::

        >>> day2 = [['cust', 'tier'], ['C1', 'platinum']]
        >>> etl.scd2(h, day2, 'cust', 'tier',
        ...          'valid_from', 'valid_to', 2).lookall()
        +------+------------+------------+----------+
        | cust | tier       | valid_from | valid_to |
        +======+============+============+==========+
        | 'C1' | 'gold'     |          1 |        2 |
        +------+------------+------------+----------+
        | 'C1' | 'platinum' |          2 | None     |
        +------+------------+------------+----------+
        | 'C2' | 'silver'   |          1 |        2 |
        +------+------------+------------+----------+

    Versions are valid on the half-open interval ``[valid_from, valid_to)``;
    a row whose `valid_to` is `None` is the current version of its business
    key. For each key found in `snapshot`, at the point in time given by
    `effective`:

    - if the key has no current version, a new version is opened, which
      also covers a key that disappeared earlier and now reappears;
    - if the current version has identical tracked values, the existing
      row is kept untouched, so reapplying the same snapshot adds no rows;
    - if the tracked values differ, the current version is closed by
      setting its `valid_to` to `effective` and a new version with
      `valid_from` equal to `effective` is inserted.

    Keys present in the history but missing from the snapshot have their
    current version closed (set `valid_to` to `effective`); already closed
    versions are left as they are.

    `None` and the empty string are distinct values and never treated as
    equal.

    The following conditions are rejected (the history is left unchanged):

    - duplicate business keys within `snapshot`;
    - overlapping version intervals for the same key in `history`;
    - an `effective` time not strictly later than the `valid_from` of the
      current version (a back-dated or same-point update), including a
      same-point update with different tracked values;
    - an `effective` time that falls inside an already closed interval.

    Both inputs may be one-off iterators. They are sorted with the existing
    external merge sort (:func:`petl.transform.sorts.sort`), so at most
    `buffersize` rows of each input are held in memory; overflow is spooled
    to temporary files. The sort views are built once and cached, so a
    one-off input is read a single time and the returned table stays
    re-iterable; the spill files are deleted when the returned table is
    garbage collected, and partial iterations release their open file
    handles when the iterator is closed or fails.

    Keyword arguments:

    history : table
        Existing type 2 history table. Must contain the natural key,
        tracked, `valid_from` and `valid_to` fields. An empty table carrying
        the correct header is accepted.
    snapshot : table
        Full snapshot of every key active at `effective`, containing the
        natural key and tracked fields.
    natural_key : string or tuple of strings
        Field name(s) identifying a business entity.
    tracked_fields : string or tuple of strings
        Field name(s) whose values are versioned.
    valid_from : string
        Name of the interval start field in `history`.
    valid_to : string
        Name of the interval end field in `history`; `None` marks the
        current version.
    effective : any ordered value
        The point in time at which `snapshot` becomes effective.
    buffersize : int, optional
        Maximum number of rows buffered in memory while sorting each input;
        defaults to `petl.config.sort_buffersize`.
    tempdir : string, optional
        Directory used for sort spill files.

    """

    return SCD2View(history, snapshot, natural_key, tracked_fields,
                    valid_from, valid_to, effective,
                    buffersize=buffersize, tempdir=tempdir)


Table.scd2 = scd2


class SCD2View(Table):

    # The two input tables are sorted once and the sort views keep their own
    # spill-file/memory cache, so re-iterating this view never re-reads a
    # one-off input iterator.

    def __init__(self, history, snapshot, natural_key, tracked_fields,
                 valid_from, valid_to, effective, buffersize=None,
                 tempdir=None):
        self.history = history
        self.snapshot = snapshot
        if isinstance(natural_key, (list, tuple)):
            self.natural_key = tuple(natural_key)
        else:
            self.natural_key = (natural_key,)
        if isinstance(tracked_fields, (list, tuple)):
            self.tracked_fields = tuple(tracked_fields)
        else:
            self.tracked_fields = (tracked_fields,)
        self.valid_from = valid_from
        self.valid_to = valid_to
        self.effective = effective
        self.buffersize = buffersize
        self.tempdir = tempdir
        self._prepared = False
        self._hist_hdr = None
        self._snap_hdr = None
        self._hist_sorted = None
        self._snap_sorted = None
        self._hist_dummy = None
        self._snap_dummy = None

    def _prepare(self):
        if self._prepared:
            return
        self._hist_hdr, self._hist_sorted, self._hist_dummy = _sorted(
            self.history, list(self.natural_key) + [self.valid_from],
            self.buffersize, self.tempdir)
        self._snap_hdr, self._snap_sorted, self._snap_dummy = _sorted(
            self.snapshot, list(self.natural_key),
            self.buffersize, self.tempdir)
        self._prepared = True

    def __iter__(self):
        self._prepare()
        for row in iterscd2(
                self._hist_hdr, self._hist_sorted, self._hist_dummy,
                self._snap_hdr, self._snap_sorted, self._snap_dummy,
                self.natural_key, self.tracked_fields, self.valid_from,
                self.valid_to, self.effective):
            yield row


class _ProbeTable(Table):
    """Wrap a table iterator whose header row has already been consumed so it
    can be handed to :func:`sort`, which expects a header row first. A
    positional dummy header of the right width is yielded and echoed back by
    the sort view; callers skip it and keep the real header separately."""

    def __init__(self, rows, header):
        self._rows = rows
        self.dummy = tuple('scd2_f%d' % i for i in range(len(header)))

    def __iter__(self):
        yield self.dummy
        for row in self._rows:
            yield tuple(row)


def _sorted(table, key_fields, buffersize, tempdir):
    it = iter(table)
    try:
        hdr = tuple(next(it))
    except StopIteration:
        return tuple(), None, None
    # only the header has been consumed; `it` still yields every data row,
    # so one-off generator inputs are read a single time
    wrapped = _ProbeTable(it, hdr)
    key_indices = asindices(hdr, key_fields)
    dummy_key = [wrapped.dummy[i] for i in key_indices]
    # cache=True lets the sort view serve further iterations from its own
    # memory/spill cache instead of re-reading the (possibly one-off) source
    sorted_view = sort(wrapped, key=dummy_key, buffersize=buffersize,
                       tempdir=tempdir, cache=True)
    return hdr, sorted_view, wrapped.dummy


class _GroupStream(object):
    """Iterate a key-sorted row iterator one business-key group at a time,
    with the ability to remember a single row read ahead."""

    def __init__(self, rows, key_getter, first=None):
        self._it = iter(rows)
        self._key_getter = key_getter
        self._ahead = first

    def next_group(self):
        first = self._ahead
        if first is not None:
            self._ahead = None
        else:
            try:
                first = next(self._it)
            except StopIteration:
                return None, []
        key = self._key_getter(first)
        group = [first]
        for row in self._it:
            if self._key_getter(row) == key:
                group.append(row)
            else:
                self._ahead = row
                break
        return key, group


def iterscd2(hist_hdr, hist_sorted, hist_dummy, snap_hdr, snap_sorted,
             snap_dummy, natural_key, tracked_fields, valid_from,
             valid_to, effective):

    if not tracked_fields:
        raise SCD2Error('at least one tracked field is required')
    if valid_from == valid_to:
        raise SCD2Error('valid_from and valid_to must name different fields')

    if not hist_hdr:
        raise SCD2Error('history table has no header row')
    if not snap_hdr:
        raise SCD2Error('snapshot table has no header row')

    hkey_idx = asindices(hist_hdr, natural_key)
    hfrom_idx = asindices(hist_hdr, (valid_from,))[0]
    hto_idx = asindices(hist_hdr, (valid_to,))[0]
    htrack_idx = asindices(hist_hdr, tracked_fields)
    skey_idx = asindices(snap_hdr, natural_key)
    strack_idx = asindices(snap_hdr, tracked_fields)

    yield tuple(hist_hdr)

    hist_getkey = comparable_itemgetter(*hkey_idx)
    snap_getkey = comparable_itemgetter(*skey_idx)
    hist_values = rowgetter(*htrack_idx)
    snap_values = rowgetter(*strack_idx)

    hist_it = iter(hist_sorted)
    snap_it = iter(snap_sorted)
    hist_first = next(hist_it)
    snap_first = next(snap_it)
    # the sort view echoes the exact dummy header object; identify it by
    # identity so a data row with the same values is never skipped by
    # mistake. A future sort that drops the header simply yields no match.
    if hist_first is hist_dummy:
        hist_first = next(hist_it, None)
    if snap_first is snap_dummy:
        snap_first = next(snap_it, None)
    hist_stream = _GroupStream(hist_it, hist_getkey, hist_first)
    snap_stream = _GroupStream(snap_it, snap_getkey, snap_first)

    hist_key, hist_group = hist_stream.next_group()
    snap_key, snap_group = snap_stream.next_group()

    while hist_key is not None or snap_key is not None:

        if hist_key is None or (snap_key is not None
                                and snap_key < hist_key):
            # brand new key, or a key that reappears after every one of its
            # history rows has already been emitted
            if len(snap_group) > 1:
                raise DuplicateKeyError(
                    tuple(snap_group[0][i] for i in skey_idx))
            yield _open_version_from_snapshot(
                hist_hdr, hkey_idx, htrack_idx, hfrom_idx, hto_idx,
                snap_group[0], skey_idx, strack_idx, effective)
            snap_key, snap_group = snap_stream.next_group()
            continue

        if snap_key is None or hist_key < snap_key:
            # key missing from the snapshot: close the current version
            for row in _process_history_group(
                    hist_group, effective, None, hfrom_idx, hto_idx,
                    htrack_idx, hist_values):
                yield row
            hist_key, hist_group = hist_stream.next_group()
            continue

        # same business key on both sides
        if len(snap_group) > 1:
            raise DuplicateKeyError(
                tuple(snap_group[0][i] for i in skey_idx))
        for row in _process_history_group(
                hist_group, effective, snap_values(snap_group[0]),
                hfrom_idx, hto_idx, htrack_idx, hist_values):
            yield row
        hist_key, hist_group = hist_stream.next_group()
        snap_key, snap_group = snap_stream.next_group()


def _process_history_group(hist_group, effective, snap_tracked, hfrom_idx,
                           hto_idx, htrack_idx, hist_values):
    """Validate the history rows of one business key and yield the rows
    representing the key after the snapshot is applied.

    `snap_tracked` is `None` when the key is absent from the snapshot,
    otherwise it is the tuple of tracked values from the snapshot row.
    """

    current = None
    previous_end = None
    for row in hist_group:
        row_from = row[hfrom_idx]
        row_to = row[hto_idx]
        if row_to is not None and not row_from < row_to:
            raise SCD2Error(
                'history interval is not an empty half-open interval '
                '[valid_from, valid_to): %r' % (row,))
        if previous_end is not None and row_from < previous_end:
            raise SCD2Error(
                'overlapping history intervals for the same business key: '
                '%r' % (row,))
        if row_to is None:
            if current is not None:
                raise SCD2Error(
                    'more than one current version (valid_to is None) for '
                    'the same business key: %r' % (row,))
            current = row
        previous_end = row_to

    if current is None:
        # all versions are already closed; only a reappearing key changes
        for row in hist_group:
            yield tuple(row)
        if snap_tracked is not None:
            if previous_end is not None and effective < previous_end:
                raise SCD2Error(
                    'effective time %r precedes an already closed version '
                    'of the same business key' % (effective,))
            yield _open_version_from_values(
                hist_group[0], htrack_idx, hfrom_idx, hto_idx,
                snap_tracked, effective)
        return

    current_from = current[hfrom_idx]
    if not current_from < effective:
        raise SCD2Error(
            'effective time %r must be strictly later than the current '
            'version start %r' % (effective, current_from))

    for row in hist_group:
        if row is not current:
            yield tuple(row)
        elif snap_tracked is None:
            # disappeared: close the current version
            new_row = list(row)
            new_row[hto_idx] = effective
            yield tuple(new_row)
        elif snap_tracked == hist_values(row):
            # unchanged; this also makes reapplying the snapshot idempotent
            yield tuple(row)
        else:
            # changed: close the old version and open a new one
            new_row = list(row)
            new_row[hto_idx] = effective
            yield tuple(new_row)
            yield _open_version_from_values(
                row, htrack_idx, hfrom_idx, hto_idx, snap_tracked,
                effective)


def _open_version_from_snapshot(hist_hdr, hkey_idx, htrack_idx, hfrom_idx,
                                hto_idx, snap_row, skey_idx, strack_idx,
                                effective):
    new_row = [None] * len(hist_hdr)
    for src_index, dest_index in zip(skey_idx, hkey_idx):
        new_row[dest_index] = snap_row[src_index]
    values = tuple(snap_row[i] for i in strack_idx)
    for dest_index, value in zip(htrack_idx, values):
        new_row[dest_index] = value
    new_row[hfrom_idx] = effective
    new_row[hto_idx] = None
    return tuple(new_row)


def _open_version_from_values(base_row, htrack_idx, hfrom_idx, hto_idx,
                              tracked_values, effective):
    # keep the natural key (and any non-versioned columns) from the previous
    # row, replace the tracked values and open a fresh current interval
    new_row = list(base_row)
    for dest_index, value in zip(htrack_idx, tracked_values):
        new_row[dest_index] = value
    new_row[hfrom_idx] = effective
    new_row[hto_idx] = None
    return tuple(new_row)
