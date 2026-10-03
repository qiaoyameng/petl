from __future__ import absolute_import, print_function, division


import operator
from functools import partial

from petl.compat import text_type, binary_type, numeric_types

from petl.errors import SortError


class Comparable(object):
    """Wrapper class to allow for flexible comparison of objects of different
    types, preserving the relaxed sorting behaviour of Python 2 with
    additional flexibility to allow for comparison of arbitrary objects with
    the `None` value (for example, the date and time objects from the standard
    library cannot be directly compared with `None` in Python 2).

    """

    __slots__ = ['obj', 'inner']

    def __init__(self, obj):
        # store wrapped object unchanged
        self.inner = obj
        # handle lists and tuples
        if isinstance(obj, (list, tuple)):
            obj = tuple(Comparable(o) for o in obj)
        self.obj = obj

    def __lt__(self, other):

        # convenience
        obj = self.obj
        if isinstance(other, Comparable):
            other = other.obj

        # None < everything else
        if other is None:
            return False
        if obj is None:
            return True

        # numbers < everything else (except None)
        if isinstance(obj, numeric_types) \
                and not isinstance(other, numeric_types):
            return True
        if not isinstance(obj, numeric_types) \
                and isinstance(other, numeric_types):
            return False

        # binary < unicode
        if isinstance(obj, text_type) and isinstance(other, binary_type):
            return False
        if isinstance(obj, binary_type) and isinstance(other, text_type):
            return True

        try:
            # attempt native comparison
            return obj < other

        except TypeError:
            # fall back to comparing type names
            return _typestr(obj) < _typestr(other)

    def __eq__(self, other):
        if isinstance(other, Comparable):
            return self.obj == other.obj
        return self.obj == other

    def __le__(self, other):
        return self < other or self == other

    def __gt__(self, other):
        return not (self < other or self == other)

    def __ge__(self, other):
        return not (self < other)

    def __str__(self):
        return str(self.obj)

    def __unicode__(self):
        return text_type(self.obj)

    def __repr__(self):
        return 'Comparable(' + repr(self.obj) + ')'

    def __iter__(self, *args, **kwargs):
        return iter(self.obj, *args, **kwargs)

    def __len__(self):
        return len(self.obj)

    def __getitem__(self, item):
        return self.obj.__getitem__(item)


def _typestr(x):
    # attempt to preserve Python 2 name orderings
    if isinstance(x, binary_type):
        return 'str'
    if isinstance(x, text_type):
        return 'unicode'
    return type(x).__name__


def comparable_itemgetter(*args):
    getter = operator.itemgetter(*args)
    getter_with_default = _itemgetter_with_default(*args)

    def _getter_with_fallback(obj):
        try:
            return getter(obj)
        except (IndexError, KeyError):
            return getter_with_default(obj)
    g = lambda x: Comparable(_getter_with_fallback(x))
    return g


def _itemgetter_with_default(*args):
    """ itemgetter compatible with `operator.itemgetter` behavior, filling missing
    values with default instead of raising IndexError or KeyError """
    def _get_default(obj, item, default):
        try:
            return obj[item]
        except (IndexError, KeyError):
            return default
    if len(args) == 1:
        return partial(_get_default, item=args[0], default=None)
    return lambda obj: tuple(_get_default(obj, item=item, default=None) for item in args)


# ---------------------------------------------------------------------------
# Extended sort key support: per-field direction, null placement and explicit
# distinction between a short row (missing field) and an actual ``None`` value
# ---------------------------------------------------------------------------


class _Missing(object):
    """Sentinel marking a field that is absent from a short row (as opposed
    to a field that is present but holds the value ``None``)."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(_Missing, cls).__new__(cls)
        return cls._instance

    def __repr__(self):
        return 'MISSING'

    def __bool__(self):
        return False

    __nonzero__ = __bool__


MISSING = _Missing()


class FieldKey(object):
    """Comparison proxy for a single sort field.

    ``category`` is ``0`` for a missing field (short row), ``1`` for an
    actual ``None`` value and ``2`` for any other value. The caller supplies
    the ``null_band`` of the missing/None categories so that null placement
    (``nulls first``/``nulls last``) is respected independently of the
    direction used to compare ordinary values.
    """

    __slots__ = ['category', 'null_band', 'descending', 'value']

    def __init__(self, category, null_band, descending, value=None):
        self.category = category
        self.null_band = null_band
        self.descending = descending
        self.value = value

    def _band(self):
        if self.category < 2:
            return self.null_band
        return 2  # ordinary values always sort in the value band

    def __eq__(self, other):
        if not isinstance(other, FieldKey):
            return NotImplemented
        if self.category != other.category:
            return False
        if self.category < 2:
            # both missing or both actual None; direction/band irrelevant
            return True
        return self.value == other.value

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __lt__(self, other):
        if not isinstance(other, FieldKey):
            return NotImplemented
        band_self = self._band()
        band_other = other._band()
        if band_self != band_other:
            return band_self < band_other
        if self.category < 2:
            # same null band: MISSING and None do not order each other,
            # treat as equal so original input order is preserved
            return False
        if self.descending:
            return other.value < self.value
        return self.value < other.value

    def __le__(self, other):
        return self == other or self < other

    def __gt__(self, other):
        if not isinstance(other, FieldKey):
            return NotImplemented
        return other < self

    def __ge__(self, other):
        return self == other or other < self

    def __repr__(self):
        return 'FieldKey(%r, %r)' % (self.category, self.value)


class RowKey(object):
    """Lexicographic comparison proxy over a tuple of :class:`FieldKey`."""

    __slots__ = ['fields']

    def __init__(self, fields):
        self.fields = tuple(fields)

    def __eq__(self, other):
        if not isinstance(other, RowKey):
            return NotImplemented
        return self.fields == other.fields

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __lt__(self, other):
        if not isinstance(other, RowKey):
            return NotImplemented
        return self.fields < other.fields

    def __le__(self, other):
        return self == other or self < other

    def __gt__(self, other):
        if not isinstance(other, RowKey):
            return NotImplemented
        return other < self

    def __ge__(self, other):
        return self == other or other < self

    def __iter__(self):
        return iter(self.fields)

    def __repr__(self):
        return 'RowKey(%r)' % (self.fields,)


def sortable_itemgetter(*specs):
    """Build a key function implementing per-field sort semantics.

    Each ``spec`` is a ``(index, descending, nulls, missing)`` tuple where
    ``index`` is the zero-based field index, ``descending`` is a bool,
    ``nulls`` is one of ``'first'``/``'last'``/``'auto'`` and ``missing`` is
    one of ``'fill'``/``'raise'``/``'distinct'`` controlling how a short row
    (a field absent from the row) is handled.
    """

    prepared = []
    for index, descending, nulls, missing in specs:
        # null band for the MISSING (0) and None (1) categories
        if nulls == 'first':
            null_band = 0
        elif nulls == 'last':
            null_band = 4
        else:  # 'auto': leading when ascending, trailing when descending
            null_band = 4 if descending else 0
        prepared.append((index, descending, null_band, missing))

    def _get(obj):
        fields = []
        for index, descending, null_band, missing in prepared:
            value = MISSING
            try:
                value = obj[index]
            except (IndexError, KeyError):
                if missing == 'raise':
                    raise SortError(
                        'row is missing field at index %s' % index)
                elif missing == 'distinct':
                    fields.append(FieldKey(0, null_band, descending))
                    continue
                # 'fill': treat a missing field as an actual None value
                fields.append(FieldKey(1, null_band, descending))
                continue
            if value is None:
                fields.append(FieldKey(1, null_band, descending))
            else:
                fields.append(
                    FieldKey(2, null_band, descending, Comparable(value)))
        return RowKey(fields)

    return _get

