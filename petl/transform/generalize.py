# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import os
import math
from itertools import product
from tempfile import NamedTemporaryFile

from petl.compat import numeric_types, pickle
from petl.errors import FieldSelectionError, GeneralizationError
from petl.util.base import Table, asindices, wrap


__all__ = ['generalize', 'GeneralizeView']


DEFAULT_MAX_ROWS = 100000
DEFAULT_MAX_CANDIDATES = 1000000
NONE_POLICIES = ('keep', 'map', 'error')


def generalize(table, hierarchies, k, missing=None, none_policy='keep',
               max_rows=DEFAULT_MAX_ROWS,
               max_candidates=DEFAULT_MAX_CANDIDATES):
    """
    Generalize quasi-identifier columns so every non-empty equivalence class
    contains at least `k` rows, choosing the level combination with the
    smallest total loss. Returns a :class:`GeneralizeView` exposing the
    selected levels, the resulting group sizes and the generalized table.

    Each column uses one level for the whole batch. The caller supplies a
    finite hierarchy per column together with the loss of each level.
    E.g.::

        >>> import petl as etl
        >>> table = [['age', 'city'],
        ...          [25, 'Beijing'],
        ...          [27, 'Beijing'],
        ...          [41, 'Shanghai'],
        ...          [45, 'Shenzhen'],
        ...          [62, 'Shenzhen'],
        ...          [68, 'Shenzhen']]
        >>> hierarchies = {
        ...     'age': {
        ...         'levels': [
        ...             {'loss': 0, 'mapping': {
        ...                 25: 25, 27: 27, 41: 41, 45: 45, 62: 62, 68: 68}},
        ...             {'loss': 1, 'mapping': {
        ...                 25: 'under40', 27: 'under40', 41: '40to60',
        ...                 45: '40to60', 62: 'over60', 68: 'over60'}},
        ...             {'loss': 3, 'mapping': {
        ...                 'under40': 'any', '40to60': 'any',
        ...                 'over60': 'any'}},
        ...         ],
        ...     },
        ...     'city': {
        ...         'levels': [
        ...             {'loss': 0, 'mapping': {
        ...                 'Beijing': 'Beijing', 'Shanghai': 'Shanghai',
        ...                 'Shenzhen': 'Shenzhen'}},
        ...             {'loss': 1, 'mapping': {
        ...                 'Beijing': 'north', 'Shanghai': 'south',
        ...                 'Shenzhen': 'south'}},
        ...             {'loss': 2, 'mapping': {
        ...                 'north': 'any', 'south': 'any'}},
        ...         ],
        ...     },
        ... }
        >>> view = etl.generalize(table, hierarchies, k=2)
        >>> view.selected_levels
        {'age': 1, 'city': 1}
        >>> view.total_loss
        2
        >>> view.table.lookall()
        +-----------+---------+
        | age       | city    |
        +===========+=========+
        | 'under40' | 'north' |
        +-----------+---------+
        | 'under40' | 'north' |
        +-----------+---------+
        | '40to60'  | 'south' |
        +-----------+---------+
        | '40to60'  | 'south' |
        +-----------+---------+
        | 'over60'  | 'south' |
        +-----------+---------+
        | 'over60'  | 'south' |
        +-----------+---------+

    A hierarchy is ``{'levels': [level, ...]}`` where each level is
    ``{'loss': number, 'mapping': {value: generalized_value}}``. Level 0 is
    the most specific level; mappings are validated so each later level only
    merges labels produced by the previous level (true merging, no splitting
    or re-labelling), and all non-null keys and labels inside one mapping keep
    a consistent type boundary.

    The level-0 mapping defines the covered domain. Values appearing in the
    data but absent from that mapping raise an error: they must be given an
    explicit strategy rather than being silently dropped. ``None`` is handled
    explicitly through `none_policy`:

    * ``'keep'`` (default): ``None`` values stay ``None`` at every level;
    * ``'map'``: each mapping must contain a ``None`` key and ``None`` is
      generalized like any other value;
    * ``'error'``: any ``None`` in a quasi-identifier column raises an error.

    Duplicate rows count normally and no row is ever removed: the generalized
    table keeps the original row order, row count and all non-quasi-identifier
    fields unchanged.

    The source is read exactly once. Rows are staged locally in memory and
    reused for every count; if more than `max_rows` data rows are staged they
    spill to a local temporary pickle file that is re-read for later counts.
    If the number of level combinations exceeds `max_candidates`, or no
    combination reaches `k` anonymity, a :class:`GeneralizationError` is
    raised with the reason; a partial result is never reported as optimal.

    """

    return GeneralizeView(table, hierarchies, k, missing=missing,
                          none_policy=none_policy, max_rows=max_rows,
                          max_candidates=max_candidates)


Table.generalize = generalize


class GeneralizeView(Table):
    """View returned by :func:`generalize`; the search runs on first iteration."""

    def __init__(self, source, hierarchies, k, missing=None,
                 none_policy='keep', max_rows=DEFAULT_MAX_ROWS,
                 max_candidates=DEFAULT_MAX_CANDIDATES):
        self.source = source
        self.hierarchies = hierarchies
        self.k = k
        self.missing = missing
        self.none_policy = none_policy
        self.max_rows = max_rows
        self.max_candidates = max_candidates
        self._solution = None
        self._store = None

    def _ensure_solved(self):
        if self._solution is None:
            self._solution = _solve(self)
        return self._solution

    @property
    def selected_levels(self):
        """Mapping of quasi-identifier field name to the selected level index."""
        return self._ensure_solved().selected_levels

    @property
    def total_loss(self):
        """Sum of the per-column losses at the selected levels."""
        return self._ensure_solved().total_loss

    @property
    def groups(self):
        """Table of ``(quasi-identifier values..., count)`` group sizes."""
        return self._ensure_solved().groups

    @property
    def table(self):
        """Generalized table with the original rows, order and other fields."""
        return _GeneralizedTable(self)

    @property
    def spill_path(self):
        """Spill file path, or ``None`` when all rows stayed in memory."""
        if self._store is None:
            self._ensure_solved()
        return self._store.path

    def __iter__(self):
        return iter(self.table)

    def close(self):
        """Remove any local spill file. Safe to call more than once."""
        if self._store is not None:
            self._store.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class _Solution(object):

    def __init__(self, selected_levels, total_loss, groups, chain, header,
                 store, indices):
        self.selected_levels = selected_levels
        self.total_loss = total_loss
        self.groups = groups
        self.chain = chain
        self.header = header
        self.store = store
        self.indices = indices


class _GeneralizedTable(Table):

    def __init__(self, view):
        self.view = view

    def __iter__(self):
        solution = self.view._ensure_solved()
        header, store, indices, chain = (
            solution.header, solution.store, solution.indices, solution.chain)
        yield tuple(header)
        for row in store.rows():
            out = list(row)
            generalized = chain(tuple(row[index] for index in indices))
            for pos, index in enumerate(indices):
                out[index] = generalized[pos]
            yield tuple(out)


class _RowStore(object):
    """Stages source rows once; spills to a local pickle file past a row cap."""

    def __init__(self, max_rows):
        self.max_rows = max_rows
        self.rows_in_memory = []
        self.path = None
        self._append_file = None

    def add(self, row):
        if self._append_file is None:
            self.rows_in_memory.append(row)
            if len(self.rows_in_memory) > self.max_rows:
                self._spill()
        else:
            pickle.dump(row, self._append_file, protocol=-1)
            self._append_file.flush()

    def _spill(self):
        self._append_file = NamedTemporaryFile(delete=False, mode='wb')
        for row in self.rows_in_memory:
            pickle.dump(row, self._append_file, protocol=-1)
        self.rows_in_memory = []
        self.path = self._append_file.name

    def rows(self):
        if self._append_file is None:
            for row in self.rows_in_memory:
                yield row
            return
        self._append_file.flush()
        with open(self.path, 'rb') as handle:
            while True:
                try:
                    yield pickle.load(handle)
                except EOFError:
                    break

    def close(self):
        if self._append_file is not None:
            try:
                self._append_file.close()
            finally:
                if self.path is not None and os.path.exists(self.path):
                    os.remove(self.path)
                self._append_file = None
                self.path = None


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise GeneralizationError('%s must be a positive integer, got %r'
                                  % (name, value))
    return value


def _type_name(value):
    return type(value).__name__


def _check_type_boundary(mapping, context):
    key_type = None
    label_type = None
    for key, label in mapping.items():
        try:
            if key is not None:
                hash(key)
            hash(label)
        except TypeError:
            raise GeneralizationError(
                '%s keys and labels must all be hashable' % context)
        if key is not None:
            if key_type is None:
                key_type = type(key)
            elif type(key) is not key_type:
                raise GeneralizationError(
                    '%s mixes key types %s and %s'
                    % (context, _type_name(key_type), _type_name(key)))
        if label is not None:
            if label_type is None:
                label_type = type(label)
            elif type(label) is not label_type:
                raise GeneralizationError(
                    '%s mixes label types %s and %s'
                    % (context, _type_name(label_type),
                       _type_name(label)))
    return key_type, label_type


def _validate_hierarchy(spec, field, covered, none_seen, none_policy):
    if not isinstance(spec, dict) or 'levels' not in spec:
        raise GeneralizationError(
            "hierarchy for field %r must be a dict with a 'levels' entry"
            % field)
    levels = spec['levels']
    if not isinstance(levels, (list, tuple)) or not levels:
        raise GeneralizationError(
            'hierarchy for field %r needs at least one level' % field)

    normalized = []
    previous_labels = None
    previous_label_type = None
    for depth, level in enumerate(levels):
        context = 'field %r level %d' % (field, depth)
        if not isinstance(level, dict):
            raise GeneralizationError('%s must be a dict' % context)
        if 'mapping' not in level:
            raise GeneralizationError('%s is missing a mapping' % context)
        mapping = level['mapping']
        if not isinstance(mapping, dict) or not mapping:
            raise GeneralizationError('%s mapping must be a non-empty dict'
                                      % context)
        loss = level.get('loss', 0)
        if isinstance(loss, bool) or not isinstance(loss, numeric_types):
            raise GeneralizationError(
                '%s loss must be numeric, got %r' % (context, loss))
        if isinstance(loss, float) and not math.isfinite(loss):
            raise GeneralizationError(
                '%s loss must be finite, got %r' % (context, loss))

        key_type, label_type = _check_type_boundary(mapping, context)

        if none_policy == 'map' and None not in mapping:
            raise GeneralizationError(
                "%s: none_policy='map' requires a None key in every mapping"
                % context)
        if none_policy != 'map' and None in mapping:
            raise GeneralizationError(
                '%s: None keys are only allowed with none_policy=%r'
                % (context, 'map'))

        if depth == 0:
            domain = set(key for key in mapping if key is not None)
            uncovered = sorted((value for value in covered
                                if value is not None and value not in domain),
                               key=repr)
            if uncovered:
                raise GeneralizationError(
                    'field %r has values not covered by its level-0 mapping '
                    '(add an explicit mapping entry or remove the rows): %r'
                    % (field, uncovered))
            if any(key != label for key, label in mapping.items()
                   if key is not None):
                raise GeneralizationError(
                    'field %r level 0 must be the identity level: each '
                    'non-null key maps to itself' % field)
            previous_labels = set(mapping.values())
            previous_label_type = label_type
        else:
            keys = set(key for key in mapping if key is not None)
            missing = sorted((label for label in previous_labels
                              if label is not None and label not in keys),
                           key=repr)
            if missing:
                raise GeneralizationError(
                    '%s does not cover labels from the previous level: %r'
                    % (context, missing))
            extra = sorted((key for key in keys
                            if key not in previous_labels), key=repr)
            if extra:
                raise GeneralizationError(
                    '%s introduces labels absent from the previous level: %r'
                    % (context, extra))
            if key_type is not None and previous_label_type is not None and \
                    key_type is not previous_label_type:
                raise GeneralizationError(
                    '%s keys have type %s but the previous level produced '
                    'labels of type %s; levels must connect at a consistent '
                    'type boundary'
                    % (context, _type_name(key_type),
                       _type_name(previous_label_type)))
            labels = set(mapping.values())
            non_none_previous = set(label for label in previous_labels
                                    if label is not None)
            non_none_labels = set(label for key, label in mapping.items()
                                  if key is not None and label is not None)
            if len(non_none_labels) > len(non_none_previous):
                raise GeneralizationError(
                    '%s splits the previous level into more distinct '
                    'labels; hierarchies may only merge, never split'
                    % context)
            previous_labels = labels
            previous_label_type = label_type

        normalized.append({'loss': loss, 'mapping': dict(mapping)})

    if none_seen and none_policy == 'error':
        raise GeneralizationError(
            'field %r contains None but none_policy is %r'
            % (field, 'error'))
    return normalized


def _solve(view):
    k = _positive_int(view.k, 'k')
    max_rows = _positive_int(view.max_rows, 'max_rows')
    max_candidates = _positive_int(view.max_candidates, 'max_candidates')
    none_policy = view.none_policy
    if none_policy not in NONE_POLICIES:
        raise GeneralizationError(
            'none_policy must be one of %r, got %r'
            % (NONE_POLICIES, none_policy))
    hierarchies = view.hierarchies
    if not isinstance(hierarchies, dict) or not hierarchies:
        raise GeneralizationError(
            'hierarchies must be a non-empty dict of field -> hierarchy')

    source = iter(view.source)
    try:
        header = tuple(next(source))
    except StopIteration:
        raise GeneralizationError('source table has no header')

    fields = list(hierarchies.keys())
    try:
        indices = asindices(header, fields)
    except FieldSelectionError as exc:
        raise GeneralizationError(str(exc))
    store = _RowStore(max_rows)
    view._store = store
    try:
        covered = [set() for _ in fields]
        none_seen = [False] * len(fields)
        row_count = 0

        def getter(row):
            values = []
            for index in indices:
                values.append(row[index] if index < len(row)
                              else view.missing)
            return tuple(values)

        for row in source:
            store.add(tuple(row))
            row_count += 1
            values = getter(row)
            for pos, value in enumerate(values):
                if value is None:
                    none_seen[pos] = True
                else:
                    covered[pos].add(value)
        if row_count == 0:
            raise GeneralizationError(
                'source table has no data rows; no equivalence classes exist')

        columns = []
        candidate_count = 1
        for pos, field in enumerate(fields):
            levels = _validate_hierarchy(hierarchies[field], field,
                                         covered[pos], none_seen[pos],
                                         none_policy)
            columns.append(levels)
            candidate_count *= len(levels)
        if candidate_count > max_candidates:
            raise GeneralizationError(
                'candidate level combinations %d exceed max_candidates %d; '
                'refusing to report an unverified optimum'
                % (candidate_count, max_candidates))

        label_tables = []
        for levels in columns:
            per_depth = [levels[0]['mapping']]
            current = levels[0]['mapping']
            for level in levels[1:]:
                upper = level['mapping']
                composed = dict()
                for raw_value, base_label in current.items():
                    composed[raw_value] = upper[base_label]
                per_depth.append(composed)
                current = composed
            label_tables.append(per_depth)

        best = None
        ranges = [range(len(levels)) for levels in columns]
        for combo in product(*ranges):
            mappings = [label_tables[pos][combo[pos]]
                        for pos in range(len(fields))]

            def chain(raw_values, mappings=mappings):
                generalized = []
                for pos, value in enumerate(raw_values):
                    if value is None:
                        if none_policy == 'keep':
                            generalized.append(None)
                        else:
                            generalized.append(mappings[pos][None])
                    else:
                        generalized.append(mappings[pos][value])
                return tuple(generalized)

            counts = {}
            for row in store.rows():
                key = chain(getter(row))
                counts[key] = counts.get(key, 0) + 1

            if any(count < k for count in counts.values()):
                continue

            total_loss = sum(columns[pos][combo[pos]]['loss']
                             for pos in range(len(fields)))
            if best is None or total_loss < best.total_loss:
                ordered = sorted(
                    counts.items(),
                    key=lambda item: tuple(repr(part) for part in item[0]))
                group_rows = [tuple(fields) + ('count',)]
                group_rows.extend(list(key) + [count]
                                  for key, count in ordered)
                groups = wrap(group_rows)
                best = _Solution(dict(zip(fields, combo)), total_loss,
                                 groups, chain, header, store, indices)

        if best is None:
            raise GeneralizationError(
                'no feasible level combination reaches k=%d across %d rows; '
                'at least one equivalence class stays smaller than k and '
                'rows must not be deleted' % (k, row_count))
        return best
    except BaseException:
        store.close()
        view._store = None
        raise
