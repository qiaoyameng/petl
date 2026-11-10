"""Row-level lineage tracking for petl.

This module is deliberately separate from the petl transformation machinery:
ordinary data columns and values are produced by the same rules as plain
petl (see the per-transform references below), while provenance is carried
out-of-band and exposed through :class:`TrackedTable`.

Design notes
------------

* Every physical input row is assigned an independent, stable ordinal by
  :class:`TrackedSource`, so two input rows holding identical values have
  distinct provenance.
* Provenance is stored as an immutable persistent union tree
  (:class:`Line`).  Joining ``n`` left rows with ``m`` right rows builds
  ``n + m`` small nodes and lets every result row share sub-trees; it never
  copies the accumulated upstream history into each result.
* A tracked table snapshots its current stage into tuples on first use, so
  the data interface (:meth:`TrackedTable.__iter__`) and the lineage
  interface (:meth:`TrackedTable.iterwithlineage`) can be read in any order
  and any number of times without replaying a one-shot input iterator.
* Passing a table whose lineage is unknown into a tracked transform raises
  :class:`LineageLostError` rather than silently returning incomplete
  provenance.
"""

from __future__ import absolute_import, division, print_function

from collections import OrderedDict
import itertools
import operator

from petl.comparison import Comparable, comparable_itemgetter
from petl.compat import callable, string_types, text_type
from petl.errors import ArgumentError, FieldSelectionError
from petl.transform.joins import keys_from_args
from petl.util.base import (
    Record,
    Table,
    asindices,
    expr,
    rowgetter,
)


class LineageLostError(Exception):
    """Raised when lineage cannot be established or is no longer available.

    This is raised eagerly (at transform construction or table read time) so
    callers never mistake a partial provenance set for a complete one.
    """


class SourceCoord(tuple):
    """Immutable coordinate of one physical input row.

    ``source_id`` labels the input table (e.g. a file name); ``row_number`` is
    the 1-based ordinal of the physical data row within that table.  For a
    table read from a file with a single header line, the physical file line
    number is ``row_number + 1``.
    """

    __slots__ = ()

    def __new__(cls, source_id, row_number):
        return tuple.__new__(cls, (source_id, row_number))

    @property
    def source_id(self):
        return self[0]

    @property
    def row_number(self):
        return self[1]

    @property
    def file_line_number(self):
        """Line number in a headered file (one header row above the data)."""
        return self[1] + 1

    def __repr__(self):
        return 'SourceCoord(%r, %d)' % (self.source_id, self.row_number)


class Line(object):
    """Immutable persistent set of :class:`SourceCoord` objects.

    The tree is a union of one or more leaves.  Instances are never mutated;
    :meth:`union` returns a new root sharing its children, so provenance
    history is shared instead of being copied into every output row.
    """

    __slots__ = ('coord', 'left', 'right')

    def __init__(self, coord=None, left=None, right=None):
        self.coord = coord
        self.left = left
        self.right = right

    @classmethod
    def leaf(cls, coord):
        return cls(coord=coord)

    def union(self, other):
        if other is None:
            return self
        if self is other:
            return self
        return Line(left=self, right=other)

    def coords(self):
        """Yield all source coordinates, ordered by their first discovery."""

        # Iterative DFS so deep pipelines do not consume Python recursion.
        stack = [self]
        while stack:
            node = stack.pop()
            if node is None:
                continue
            if node.coord is not None:
                yield node.coord
            else:
                # Push right first so the left subtree is visited first.
                stack.append(node.right)
                stack.append(node.left)

    def as_set(self):
        return frozenset(self.coords())

    def __len__(self):
        return sum(1 for _ in self.coords())

    def __eq__(self, other):
        return isinstance(other, Line) and self.as_set() == other.as_set()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.as_set())

    def __repr__(self):
        return 'Line(%r)' % (list(self.coords()),)


def union_lines(left, right):
    """Union two optional :class:`Line` objects, sharing structure."""
    if left is None:
        return right
    if right is None:
        return left
    return left.union(right)


def _iter_data_pairs(stream):
    """Skip the bare header tuple of an ``iterwithlineage`` stream."""
    next(stream)  # header
    return stream


def require_tracked(table, role='table'):
    if not isinstance(table, TrackedTable):
        raise LineageLostError(
            '%s has no lineage information; wrap the original input with '
            'petl.lineage.tracked() and build the pipeline using only '
            'lineage-aware transforms' % role
        )
    return table


class TrackedTable(Table):
    """Base class for tables carrying row-level lineage.

    Subclasses implement :meth:`iterwithlineage`, which yields the header
    tuple first and then ``(row, line)`` pairs.  Plain table iteration
    delegates to the same generator, so data columns and values are always
    produced by exactly one code path.
    """

    def iterwithlineage(self):
        raise NotImplementedError

    def __iter__(self):
        stream = self.iterwithlineage()
        yield next(stream)  # header
        for row, _line in stream:
            yield row

    def lineage(self):
        """Return the :class:`Line` of each data row, aligned with iteration."""

        it = _iter_data_pairs(self.iterwithlineage())
        return [line for _row, line in it]

    def coords(self):
        """Return the provenance coordinate set of each data row."""

        return [line.as_set() if line is not None else frozenset()
                for line in self.lineage()]

    def explain(self, row_index):
        """Return the provenance coordinates of one 0-based output data row."""

        line = self.lineage()[row_index]
        if line is None:
            return frozenset()
        return line.as_set()


class TrackedSource(TrackedTable):
    """A tracked input table built from a plain petl table or row sequence.

    Rows are snapshotted once so the table, like every tracked table, can be
    iterated repeatedly.  Each physical data row receives an independent
    ordinal even when values repeat.
    """

    def __init__(self, table, source_id):
        it = iter(table)
        self._header = tuple(next(it))
        self._rows = tuple(tuple(row) for row in it)
        self.source_id = source_id
        self._lines = tuple(
            Line.leaf(SourceCoord(source_id, row_number))
            for row_number in range(1, len(self._rows) + 1)
        )

    @property
    def header(self):
        return self._header

    def iterwithlineage(self):
        yield self._header
        for row, line in zip(self._rows, self._lines):
            yield tuple(row), line


def tracked(table, source_id):
    """Attach a ``source_id`` to an input table and start lineage tracking.

    The input may be any iterable of a header tuple followed by data row
    tuples, including a plain petl table.  Re-wrapping a tracked table starts
    a fresh source with fresh physical row ordinals.
    """

    return TrackedSource(table, source_id)


class _Passthrough(TrackedTable):
    """Common machinery for transforms that preserve one upstream line per row."""

    def __init__(self, source):
        self.source = require_tracked(source)

    def _transform(self, header, rows):
        raise NotImplementedError

    def iterwithlineage(self):
        header = next(self.source.iterwithlineage())
        out_header, transform = self._transform(
            header,
            (row for row, _line in _iter_data_pairs(
                self.source.iterwithlineage())),
        )
        yield out_header
        upstream = _iter_data_pairs(self.source.iterwithlineage())
        for row, line in upstream:
            yield transform(row), line


class TrackedCut(_Passthrough):
    """Lineage-aware version of :func:`petl.transform.basics.cut`."""

    def __init__(self, source, spec, missing=None):
        super(TrackedCut, self).__init__(source)
        if len(spec) == 1 and isinstance(spec[0], (list, tuple)):
            spec = tuple(spec[0])
        self.spec = tuple(spec)
        self.missing = missing

    def _transform(self, header, rows):
        indices = asindices(header, self.spec)
        transform = rowgetter(*indices)
        out_header = transform(header)
        missing = self.missing

        def cut_row(row):
            try:
                return transform(row)
            except IndexError:
                return tuple(row[i] if i < len(row) else missing
                             for i in indices)

        return tuple(out_header), cut_row


def cut(table, *args, **kwargs):
    """Tracked equivalent of :func:`petl.cut`.

    Output columns and values match ``petl.cut`` exactly; each output row
    carries the provenance of its single upstream input row.
    """

    missing = kwargs.get('missing', None)
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        args = tuple(args[0])
    return TrackedCut(require_tracked(table, 'cut source'), args,
                      missing=missing)


class TrackedRename(_Passthrough):
    """Lineage-aware version of :func:`petl.transform.headers.rename`."""

    def __init__(self, source, spec, strict=True):
        super(TrackedRename, self).__init__(source)
        self.spec = dict(spec)
        self.strict = strict

    def _transform(self, header, rows):
        spec = self.spec
        flds = list(map(text_type, header))
        if self.strict:
            for selector in spec:
                if isinstance(selector, int):
                    if selector < 0 or selector >= len(header):
                        raise FieldSelectionError(selector)
                elif selector not in flds:
                    raise FieldSelectionError(selector)
        out_header = tuple(
            spec[i] if i in spec else spec[f] if f in spec else f
            for i, f in enumerate(flds)
        )
        return out_header, tuple


def rename(table, *args, **kwargs):
    """Tracked equivalent of :func:`petl.rename`.

    Only the header changes; data rows and their provenance pass through
    unchanged.
    """

    strict = kwargs.get('strict', True)
    if len(args) == 0:
        spec = {}
    elif len(args) == 1:
        spec = args[0]
    elif len(args) == 2:
        spec = {args[0]: args[1]}
    else:
        raise ArgumentError('expected at most two positional arguments')
    return TrackedRename(require_tracked(table, 'rename source'), spec,
                         strict=strict)


class TrackedSelect(TrackedTable):
    """Lineage-aware version of :func:`petl.transform.selects.select`."""

    def __init__(self, source, where=None, field=None, missing=None,
                 complement=False, trusted=True):
        self.source = require_tracked(source)
        self.field = field
        self.missing = missing
        self.complement = complement
        if isinstance(where, string_types):
            where = expr(where, trusted=trusted)
        elif where is not None:
            assert callable(where), 'condition must be string or callable'
        self.where = where

    def iterwithlineage(self):
        upstream = self.source.iterwithlineage()
        header = next(upstream)
        yield tuple(header)
        if self.field is None:
            flds = list(map(text_type, header))
            missing = self.missing
            where = self.where
            complement = self.complement
            for row, line in upstream:
                record = Record(row, flds, missing=missing)
                if bool(where(record)) != complement:
                    yield tuple(row), line
        else:
            indices = asindices(header, self.field)
            getv = operator.itemgetter(*indices)
            where = self.where
            missing = self.missing
            complement = self.complement
            for row, line in upstream:
                try:
                    value = getv(row)
                except IndexError:
                    value = missing
                if bool(where(value)) != complement:
                    yield tuple(row), line


def select(table, *args, **kwargs):
    """Tracked equivalent of :func:`petl.select`.

    Accepts either a row predicate (or expression string) or a
    ``field, predicate`` pair, plus ``complement`` and ``trusted`` keyword
    arguments with the same meaning as in petl.  Rejected rows are simply
    absent; retained rows keep their upstream provenance.
    """

    missing = kwargs.get('missing', None)
    complement = kwargs.get('complement', False)
    trusted = kwargs.get('trusted', True)
    source = require_tracked(table, 'select source')
    if len(args) == 0:
        raise ArgumentError('missing positional argument')
    if len(args) == 1:
        return TrackedSelect(source, where=args[0], missing=missing,
                             complement=complement, trusted=trusted)
    field, where = args
    assert callable(where), 'third argument must be callable'
    return TrackedSelect(source, where=where, field=field, missing=missing,
                         complement=complement, trusted=trusted)


class TrackedJoin(TrackedTable):
    """Lineage-aware merge join.

    Mirrors :class:`petl.transform.joins.JoinView`: rows are squared to the
    header width, stably sorted by the join key and merged with the relaxed
    :class:`~petl.comparison.Comparable` ordering.  Each emitted data row is
    paired with the union of the left and (where matched) right upstream
    lines.  Unmatched left rows in a left join carry only the left line.
    """

    def __init__(self, left, right, lkey, rkey, leftouter=False,
                 missing=None, lprefix=None, rprefix=None):
        self.left = require_tracked(left, 'left join input')
        self.right = require_tracked(right, 'right join input')
        self.lkey = lkey
        self.rkey = rkey
        self.leftouter = leftouter
        self.missing = missing
        self.lprefix = lprefix
        self.rprefix = rprefix

    @staticmethod
    def _square_and_index(source, key, missing):
        stream = source.iterwithlineage()
        header = next(stream)
        width = len(header)
        kind = asindices(header, key)
        getk = comparable_itemgetter(*kind)
        entries = []
        for row, line in stream:
            if len(row) < width:
                row = row + (missing,) * (width - len(row))
            elif len(row) > width:
                row = row[:width]
            entries.append((row, line))
        entries.sort(key=lambda entry: getk(entry[0]))
        return header, kind, getk, entries

    @staticmethod
    def _group(entries, getk):
        groups = []
        for key, group in itertools.groupby(entries, key=lambda e: getk(e[0])):
            groups.append((key, list(group)))
        return groups

    def iterwithlineage(self):
        missing = self.missing
        lhdr, lkind, lgetk, left_entries = self._square_and_index(
            self.left, self.lkey, missing)
        rhdr, rkind, rgetk, right_entries = self._square_and_index(
            self.right, self.rkey, missing)

        rvind = [i for i in range(len(rhdr)) if i not in set(rkind)]
        rgetv = rowgetter(*rvind)

        if self.lprefix is None:
            outhdr = list(lhdr)
        else:
            outhdr = [text_type(self.lprefix) + text_type(f) for f in lhdr]
        if self.rprefix is None:
            outhdr.extend(rgetv(rhdr))
        else:
            outhdr.extend([text_type(self.rprefix) + text_type(f)
                           for f in rgetv(rhdr)])
        yield tuple(outhdr)

        left_groups = iter(self._group(left_entries, lgetk))
        right_groups = iter(self._group(right_entries, rgetk))

        def emit_matched(lgroup, rgroup):
            # The right group is materialised once and shared by every left
            # row, just like iterjoin; lines are shared immutable nodes.
            for lrow, lline in lgroup:
                for rrow, rline in rgroup:
                    outrow = tuple(lrow) + tuple(rgetv(rrow))
                    yield outrow, union_lines(lline, rline)

        def emit_left_only(lgroup):
            for lrow, lline in lgroup:
                outrow = tuple(lrow) + (missing,) * len(rvind)
                yield outrow, lline

        # This mirrors petl.transform.joins.iterjoin control flow exactly,
        # including its treatment of None keys: Comparable(None) compares as
        # equal to itself and less than nothing, so the dangling-group checks
        # below must be reproduced rather than "improved".
        lkey, rkey = Comparable(None), Comparable(None)
        try:
            lkey, lgroup = next(left_groups)
            rkey, rgroup = next(right_groups)
            while True:
                if lkey < rkey:
                    if self.leftouter:
                        for item in emit_left_only(lgroup):
                            yield item
                    lkey, lgroup = next(left_groups)
                elif lkey > rkey:
                    rkey, rgroup = next(right_groups)
                else:
                    for item in emit_matched(lgroup, rgroup):
                        yield item
                    lkey, lgroup = next(left_groups)
                    rkey, rgroup = next(right_groups)
        except StopIteration:
            pass

        if self.leftouter:
            if lkey > rkey:
                for item in emit_left_only(lgroup):
                    yield item
            for _key, tail_group in left_groups:
                for item in emit_left_only(tail_group):
                    yield item


def innerjoin(left, right, key=None, lkey=None, rkey=None, **kwargs):
    """Tracked equivalent of :func:`petl.join` (inner join).

    Every result row carries the union of its left and right input lines;
    many-to-many matches share the immutable right-group subtree instead of
    copying history per result row.  Remaining keyword arguments match the
    petl signature (``missing``, ``lprefix``, ``rprefix`` ...).
    """

    lkey, rkey = keys_from_args(left, right, key, lkey, rkey)
    return TrackedJoin(left, right, lkey=lkey, rkey=rkey,
                       leftouter=False,
                       missing=kwargs.get('missing'),
                       lprefix=kwargs.get('lprefix'),
                       rprefix=kwargs.get('rprefix'))


def leftjoin(left, right, key=None, lkey=None, rkey=None, **kwargs):
    """Tracked equivalent of :func:`petl.leftjoin`.

    Matched rows carry the union of both sides; rows padded because the
    right side has no match carry the left line only.
    """

    lkey, rkey = keys_from_args(left, right, key, lkey, rkey)
    return TrackedJoin(left, right, lkey=lkey, rkey=rkey,
                       leftouter=True,
                       missing=kwargs.get('missing'),
                       lprefix=kwargs.get('lprefix'),
                       rprefix=kwargs.get('rprefix'))


def _normalise_key(key):
    # A one-element list/tuple key behaves like the bare field selector.
    if isinstance(key, (list, tuple)) and len(key) == 1:
        return key[0]
    return key


def _merge_group_lines(lines):
    merged = None
    for line in lines:
        merged = union_lines(merged, line)
    return merged


class TrackedAggregate(TrackedTable):
    """Lineage-aware version of :func:`petl.transform.reductions.aggregate`.

    Each output row carries the union of the provenance of every input row
    in its group.  Supports both the simple callable aggregation signature
    and the multi-field (dict/list/tuple) aggregation signature.
    """

    def __init__(self, source, key, aggregation=None, value=None,
                 presorted=False, field='value'):
        self.source = require_tracked(source, 'aggregate source')
        self.key = _normalise_key(key)
        self.aggregation = aggregation
        self.value = value
        self.presorted = presorted
        self.field = field

    def _snapshot(self):
        stream = self.source.iterwithlineage()
        header = next(stream)  # bare header tuple
        rows = []
        lines = []
        for row, line in stream:
            rows.append(tuple(row))
            lines.append(line)
        return tuple(header), rows, lines

    def _get_key_function(self, header):
        key = self.key
        if key is None:
            return lambda row: Comparable(None)
        if callable(key):
            flds = list(map(text_type, header))
            return lambda row: key(Record(row, flds))
        kindices = asindices(header, key)
        getk = comparable_itemgetter(*kindices)
        return lambda row: getk(row).inner

    def _groups(self, header, rows, lines):
        """Yield ``(out_key, group_rows, group_lines)`` in petl output order."""

        key = self.key
        if key is None:
            if rows:
                yield None, rows, lines
            return
        getk = self._get_key_function(header)
        if self.presorted:
            ordered = list(zip(rows, lines))
        else:
            if callable(key):
                # petl's own sort cannot sort by a callable key; match the
                # native failure rather than inventing a different behaviour.
                asindices(header, key)
            sort_getk = comparable_itemgetter(*asindices(header, key))
            ordered = sorted(zip(rows, lines), key=lambda pair: sort_getk(pair[0]))
        for group_key, group in itertools.groupby(
                ordered, key=lambda pair: getk(pair[0])):
            materialised = list(group)
            group_rows = [row for row, _line in materialised]
            group_lines = [line for _row, line in materialised]
            yield group_key, group_rows, group_lines

    def iterwithlineage(self):
        header, rows, lines = self._snapshot()
        key = self.key
        aggregation = self.aggregation
        value = self.value

        if callable(aggregation):
            yield self._simple_header()
            # Whole-table aggregation with no grouping key.
            if key is None:
                if aggregation == len:
                    out_value = len(rows)
                elif value is None:
                    flds = list(map(text_type, header))
                    out_value = aggregation(
                        Record(row, flds) for row in rows)
                else:
                    out_value = aggregation(self._value_stream(
                        header, rows, value))
                yield (out_value,), _merge_group_lines(lines)
                return

            effective_agg = aggregation
            if aggregation == len:
                effective_agg = lambda group: sum(1 for _ in group)
            for group_key, group_rows, group_lines in self._groups(
                    header, rows, lines):
                if value is None:
                    flds = list(map(text_type, header))
                    stream = (Record(row, flds) for row in group_rows)
                else:
                    stream = self._value_stream(header, group_rows, value)
                out_value = effective_agg(stream)
                out_row = self._prefix_key(group_key) + (out_value,)
                yield out_row, _merge_group_lines(group_lines)
            return

        # Multi-field aggregation (aggregation is None, list, tuple or dict).
        normalised = self._normalise_aggregations(aggregation)
        out_header = self._multi_header(normalised)
        yield tuple(out_header)
        if key is None:
            # petl's multi-aggregation emits no data row for an empty input,
            # unlike the simple callable form.
            group_stream = [(None, rows, lines)] if rows else []
        else:
            group_stream = list(self._groups(header, rows, lines))
        for group_key, group_rows, group_lines in group_stream:
            out_row = list(self._prefix_key(group_key))
            for out_field in normalised:
                src_field, aggfun = normalised[out_field]
                if src_field is None:
                    flds = list(map(text_type, header))
                    agg_input = [Record(row, flds) for row in group_rows]
                else:
                    agg_input = self._value_stream(
                        header, group_rows, src_field)
                out_row.append(aggfun(agg_input))
            yield tuple(out_row), _merge_group_lines(group_lines)

    @staticmethod
    def _value_stream(header, rows, value):
        indices = asindices(header, value)
        flds = list(map(text_type, header))
        getter = operator.itemgetter(*indices)
        for row in rows:
            # Routing through Record matches petl's rowgroupby(value=...):
            # short rows yield the missing marker instead of IndexError.
            yield getter(Record(row, flds))

    def _simple_header(self):
        key = self.key
        if isinstance(key, (list, tuple)):
            return tuple(key) + (self.field,)
        if callable(key):
            return ('key', self.field)
        if key is None:
            return (self.field,)
        return (key, self.field)

    def _prefix_key(self, group_key):
        key = self.key
        if isinstance(key, (list, tuple)):
            return tuple(group_key)
        if key is None:
            return ()
        return (group_key,)

    def _multi_header(self, normalised):
        key = self.key
        if isinstance(key, (list, tuple)):
            out = list(key)
        elif callable(key):
            out = ['key']
        elif key is None:
            out = []
        else:
            out = [key]
        out.extend(normalised)
        return tuple(out)

    @staticmethod
    def _normalise_aggregations(aggregation):
        if aggregation is None:
            aggregation = OrderedDict()
        elif isinstance(aggregation, (list, tuple)):
            ordered = OrderedDict()
            for triple in aggregation:
                ordered[triple[0]] = triple[1:]
            aggregation = ordered
        else:
            aggregation = OrderedDict(aggregation.items())
        for out_field in aggregation:
            spec = aggregation[out_field]
            if callable(spec):
                aggregation[out_field] = (None, spec)
            elif isinstance(spec, string_types):
                aggregation[out_field] = (spec, list)
            elif len(spec) == 1 and isinstance(spec[0], string_types):
                aggregation[out_field] = (spec[0], list)
            elif len(spec) == 1 and callable(spec[0]):
                aggregation[out_field] = (None, spec[0])
            elif len(spec) == 2:
                pass
            else:
                raise ArgumentError(
                    'invalid aggregation: %r, %r' % (out_field, spec))
        return aggregation


def aggregate(table, key, aggregation=None, value=None, presorted=False,
              buffersize=None, tempdir=None, cache=True, field='value'):
    """Tracked equivalent of :func:`petl.aggregate`.

    The output header and values follow petl's simple and multi-field
    aggregation rules; each output row's provenance is the union of all
    input rows in its group.  ``buffersize``, ``tempdir`` and ``cache`` are
    accepted for signature compatibility (tracked stages are in-memory).
    """

    return TrackedAggregate(table, key, aggregation=aggregation, value=value,
                            presorted=presorted, field=field)
