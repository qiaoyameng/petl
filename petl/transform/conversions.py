from __future__ import absolute_import, print_function, division

import traceback
from collections import deque
from tempfile import TemporaryFile

from petl.compat import next, integer_types, string_types, text_type, pickle

import petl.config as config
from petl.errors import ArgumentError, FieldSelectionError
from petl.util.base import Table, expr, fieldnames, Record
from petl.util.parsers import numparser


REJECTION_HEADER = ('row', 'field', 'value', 'error')
_DONE = object()
_NEED_DATA = object()


def _converters_from_args(args):
    converters = None
    if len(args) == 0:
        pass
    elif len(args) == 1:
        converters = args[0]
    elif len(args) > 1:
        converters = dict()
        field = args[0]
        if len(args) == 2:
            conv = args[1]
        else:
            conv = args[1:]
        if isinstance(field, (list, tuple)):
            for f in field:
                converters[f] = conv
        else:
            converters[field] = conv
    return converters


def _normalize_converters(converters):
    if converters is None:
        return dict()
    if isinstance(converters, dict):
        return converters
    if isinstance(converters, (tuple, list)):
        return dict([(i, v) for i, v in enumerate(converters)])
    raise ArgumentError('unexpected converters: %r' % converters)


def _build_converter_functions(flds, converters):
    converter_functions = dict()
    for k, c in converters.items():
        if not isinstance(k, integer_types):
            try:
                k = flds.index(k)
            except ValueError:
                raise FieldSelectionError(k)
        assert isinstance(k, int), 'expected integer, found %r' % k

        if callable(c):
            converter_functions[k] = c
        elif isinstance(c, string_types):
            converter_functions[k] = methodcaller(c)
        elif isinstance(c, (tuple, list)) and isinstance(c[0], string_types):
            methnm = c[0]
            methargs = c[1:]
            converter_functions[k] = methodcaller(methnm, *methargs)
        elif isinstance(c, dict):
            converter_functions[k] = dictconverter(c)
        elif c is None:
            pass
        else:
            raise ArgumentError(
                'unexpected converter specification on field %r: %r' % (k, c)
            )
    return converter_functions


def _prepare_where(where, trusted):
    if isinstance(where, string_types):
        where = expr(where, trusted=trusted)
    elif where is not None:
        assert callable(where), 'expected callable for "where" argument, ' \
                                'found %r' % where
    return where


def _format_exception(error):
    traceback_object = getattr(error, '__traceback__', None)
    try:
        return ''.join(
            traceback.format_exception(type(error), error, traceback_object)
        )
    finally:
        if hasattr(error, '__traceback__'):
            error.__traceback__ = None


def convert(table, *args, **kwargs):
    """Transform values under one or more fields via arbitrary functions, method
    invocations or dictionary translations. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar', 'baz'],
        ...           ['A', '2.4', 12],
        ...           ['B', '5.7', 34],
        ...           ['C', '1.2', 56]]
        >>> # using a built-in function:
        ... table2 = etl.convert(table1, 'bar', float)
        >>> table2
        +-----+-----+-----+
        | foo | bar | baz |
        +=====+=====+=====+
        | 'A' | 2.4 |  12 |
        +-----+-----+-----+
        | 'B' | 5.7 |  34 |
        +-----+-----+-----+
        | 'C' | 1.2 |  56 |
        +-----+-----+-----+

        >>> # using a lambda function::
        ... table3 = etl.convert(table1, 'baz', lambda v: v*2)
        >>> table3
        +-----+-------+-----+
        | foo | bar   | baz |
        +=====+=======+=====+
        | 'A' | '2.4' |  24 |
        +-----+-------+-----+
        | 'B' | '5.7' |  68 |
        +-----+-------+-----+
        | 'C' | '1.2' | 112 |
        +-----+-------+-----+

        >>> # a method of the data value can also be invoked by passing
        ... # the method name
        ... table4 = etl.convert(table1, 'foo', 'lower')
        >>> table4
        +-----+-------+-----+
        | foo | bar   | baz |
        +=====+=======+=====+
        | 'a' | '2.4' |  12 |
        +-----+-------+-----+
        | 'b' | '5.7' |  34 |
        +-----+-------+-----+
        | 'c' | '1.2' |  56 |
        +-----+-------+-----+

        >>> # arguments to the method invocation can also be given
        ... table5 = etl.convert(table1, 'foo', 'replace', 'A', 'AA')
        >>> table5
        +------+-------+-----+
        | foo  | bar   | baz |
        +======+=======+=====+
        | 'AA' | '2.4' |  12 |
        +------+-------+-----+
        | 'B'  | '5.7' |  34 |
        +------+-------+-----+
        | 'C'  | '1.2' |  56 |
        +------+-------+-----+

        >>> # values can also be translated via a dictionary
        ... table7 = etl.convert(table1, 'foo', {'A': 'Z', 'B': 'Y'})
        >>> table7
        +-----+-------+-----+
        | foo | bar   | baz |
        +=====+=======+=====+
        | 'Z' | '2.4' |  12 |
        +-----+-------+-----+
        | 'Y' | '5.7' |  34 |
        +-----+-------+-----+
        | 'C' | '1.2' |  56 |
        +-----+-------+-----+

        >>> # the same conversion can be applied to multiple fields
        ... table8 = etl.convert(table1, ('foo', 'bar', 'baz'), str)
        >>> table8
        +-----+-------+------+
        | foo | bar   | baz  |
        +=====+=======+======+
        | 'A' | '2.4' | '12' |
        +-----+-------+------+
        | 'B' | '5.7' | '34' |
        +-----+-------+------+
        | 'C' | '1.2' | '56' |
        +-----+-------+------+

        >>> # multiple conversions can be specified at the same time
        ... table9 = etl.convert(table1, {'foo': 'lower',
        ...                               'bar': float,
        ...                               'baz': lambda v: v * 2})
        >>> table9
        +-----+-----+-----+
        | foo | bar | baz |
        +=====+=====+=====+
        | 'a' | 2.4 |  24 |
        +-----+-----+-----+
        | 'b' | 5.7 |  68 |
        +-----+-----+-----+
        | 'c' | 1.2 | 112 |
        +-----+-----+-----+

        >>> # ...or alternatively via a list
        ... table10 = etl.convert(table1, ['lower', float, lambda v: v*2])
        >>> table10
        +-----+-----+-----+
        | foo | bar | baz |
        +=====+=====+=====+
        | 'a' | 2.4 |  24 |
        +-----+-----+-----+
        | 'b' | 5.7 |  68 |
        +-----+-----+-----+
        | 'c' | 1.2 | 112 |
        +-----+-----+-----+

        >>> # conversion can be conditional
        ... table11 = etl.convert(table1, 'baz', lambda v: v * 2,
        ...                       where=lambda r: r.foo == 'B')
        >>> table11
        +-----+-------+-----+
        | foo | bar   | baz |
        +=====+=======+=====+
        | 'A' | '2.4' |  12 |
        +-----+-------+-----+
        | 'B' | '5.7' |  68 |
        +-----+-------+-----+
        | 'C' | '1.2' |  56 |
        +-----+-------+-----+

        >>> # conversion can access other values from the same row
        ... table12 = etl.convert(table1, 'baz',
        ...                       lambda v, row: v * float(row.bar),
        ...                       pass_row=True)
        >>> table12
        +-----+-------+--------------------+
        | foo | bar   | baz                |
        +=====+=======+====================+
        | 'A' | '2.4' | 28.799999999999997 |
        +-----+-------+--------------------+
        | 'B' | '5.7' |              193.8 |
        +-----+-------+--------------------+
        | 'C' | '1.2' |               67.2 |
        +-----+-------+--------------------+
        >>> # conversion can use a custom function
        >>> def my_func(val, row):
        ...     return float(row.bar) + row.baz
        ... 
        >>> table13 = etl.convert(table1, 'foo', my_func, pass_row=True)
        >>> table13
        +------+-------+-----+
        | foo  | bar   | baz |
        +======+=======+=====+
        | 14.4 | '2.4' |  12 |
        +------+-------+-----+
        | 39.7 | '5.7' |  34 |
        +------+-------+-----+
        | 57.2 | '1.2' |  56 |
        +------+-------+-----+

    Note that either field names or indexes can be given.

    The ``where`` keyword argument can be given with a callable or expression
    which is evaluated on each row and which should return True if the
    conversion should be applied on that row, else False.

    The ``pass_row`` keyword argument can be given, which if True will mean
    that both the value and the containing row will be passed as
    arguments to the conversion function (so, i.e., the conversion function
    should accept two arguments).

    The row supports attribute, field-name and zero-based index access, as
    described under :func:`petl.util.base.records`. Use field-name access for
    column names containing spaces or punctuation, which cannot be written
    as attributes. For example::

        >>> orders = [['Unit Price', 'Quantity', 'Total'],
        ...           [5, 2, None], [3, 4, None]]
        >>> totals = etl.convert(orders, 'Total',
        ...                      lambda value, row: row['Unit Price'] * row['Quantity'],
        ...                      pass_row=True)
        >>> list(totals)
        [('Unit Price', 'Quantity', 'Total'), (5, 2, 10), (3, 4, 12)]

    The same conversion can use indexes instead of field names::

        >>> totals = etl.convert(orders, 'Total',
        ...                      lambda value, row: row[0] * row[1],
        ...                      pass_row=True)
        >>> list(totals)
        [('Unit Price', 'Quantity', 'Total'), (5, 2, 10), (3, 4, 12)]

    Return the replacement value from the conversion function; the row is a
    read-only record, so do not assign to ``row['Total']``. The input table
    remains unchanged::

        >>> orders
        [['Unit Price', 'Quantity', 'Total'], [5, 2, None], [3, 4, None]]

    When multiple fields are converted in a single call, the conversions
    are independent of each other. Each conversion sees the original row::

        >>> # multiple conversions do not affect each other
        ... table13 = etl.convert(table1, {
        ...                           "foo": lambda foo, row: row.bar,
        ...                           "bar": lambda bar, row: row.foo,
        ...                       }, pass_row=True)
        >>> table13
        +-------+-----+-----+
        | foo   | bar | baz |
        +=======+=====+=====+
        | '2.4' | 'A' |  12 |
        +-------+-----+-----+
        | '5.7' | 'B' |  34 |
        +-------+-----+-----+
        | '1.2' | 'C' |  56 |
        +-------+-----+-----+

    Also accepts `failonerror` and `errorvalue` keyword arguments,
    documented under :func:`petl.config.failonerror`

    The ``trusted`` keyword argument can be used to specify whether the
    expression is trusted. See `:func:`petl.util.base.expr` for details.

    """

    converters = _converters_from_args(args)
    return FieldConvertView(table, converters, **kwargs)


Table.convert = convert


def biconvert(table, *args, **kwargs):
    """Convert rows and separate successful rows from rejected rows.

    Return a :class:`ConvertSplitResult`. ``accepted`` is a table containing
    converted rows; ``rejected`` is a table with the header
    ``('row', 'field', 'value', 'error')``. If any selected field raises, the
    entire original row is rejected and no partial row is accepted. Row
    numbers start at one for the first data row.

    Each output can only be iterated once. Reading one output lazily pulls
    and converts source rows as needed, buffering rows destined for the other
    output. At most ``buffersize`` rows per output remain in memory; extra
    rows are spooled to temporary files. Call ``close()`` on either output to
    abandon that output, or call :meth:`ConvertSplitResult.close` (or use the
    result as a context manager) to release all temporary state.
    """

    converters = _converters_from_args(args)
    where = kwargs.pop('where', None)
    pass_row = kwargs.pop('pass_row', False)
    trusted = kwargs.pop('trusted', True)
    buffersize = kwargs.pop('buffersize', 1000)
    tempdir = kwargs.pop('tempdir', None)
    if kwargs:
        raise ArgumentError('unexpected keyword argument: %r'
                            % next(iter(kwargs)))
    if buffersize is not None and buffersize < 1:
        raise ArgumentError('buffersize must be a positive integer or None')
    splitter = _FieldConvertSplitter(
        table, converters, where=where, pass_row=pass_row,
        trusted=trusted, buffersize=buffersize, tempdir=tempdir
    )
    return ConvertSplitResult(splitter.accepted, splitter.rejected, splitter)


convert_split = biconvert
Table.biconvert = biconvert
Table.convert_split = biconvert


class ConvertSplitResult(tuple):

    def __new__(cls, accepted, rejected, splitter):
        return super(ConvertSplitResult, cls).__new__(cls,
                                                      (accepted, rejected))

    def __init__(self, accepted, rejected, splitter):
        self._splitter = splitter

    @property
    def accepted(self):
        return self[0]

    @property
    def rejected(self):
        return self[1]

    def close(self):
        self._splitter.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def __repr__(self):
        return '<ConvertSplitResult accepted rejected>'


def convertall(table, *args, **kwargs):
    """
    Convenience function to convert all fields in the table using a common
    function or mapping. See also :func:`convert`.

    The ``where`` keyword argument can be given with a callable or expression
    which is evaluated on each row and which should return True if the
    conversion should be applied on that row, else False.

    """

    # TODO don't read the data twice!
    return convert(table, fieldnames(table), *args, **kwargs)


Table.convertall = convertall


def replace(table, field, a, b, **kwargs):
    """
    Convenience function to replace all occurrences of `a` with `b` under the
    given field. See also :func:`convert`.

    The ``where`` keyword argument can be given with a callable or expression
    which is evaluated on each row and which should return True if the
    conversion should be applied on that row, else False.

    """

    return convert(table, field, {a: b}, **kwargs)


Table.replace = replace


def replaceall(table, a, b, **kwargs):
    """
    Convenience function to replace all instances of `a` with `b` under all
    fields. See also :func:`convertall`.

    The ``where`` keyword argument can be given with a callable or expression
    which is evaluated on each row and which should return True if the
    conversion should be applied on that row, else False.

    """

    return convertall(table, {a: b}, **kwargs)


Table.replaceall = replaceall


def update(table, field, value, **kwargs):
    """
    Convenience function to convert a field to a fixed value. Accepts the
    ``where`` keyword argument. See also :func:`convert`.

    """

    return convert(table, field, lambda v: value, **kwargs)


Table.update = update


def convertnumbers(table, strict=False, **kwargs):
    """
    Convenience function to convert all field values to numbers where
    possible. E.g.::

        >>> import petl as etl
        >>> table1 = [['foo', 'bar', 'baz', 'quux'],
        ...           ['1', '3.0', '9+3j', 'aaa'],
        ...           ['2', '1.3', '7+2j', None]]
        >>> table2 = etl.convertnumbers(table1)
        >>> table2
        +-----+-----+--------+-------+
        | foo | bar | baz    | quux  |
        +=====+=====+========+=======+
        |   1 | 3.0 | (9+3j) | 'aaa' |
        +-----+-----+--------+-------+
        |   2 | 1.3 | (7+2j) | None  |
        +-----+-----+--------+-------+

    """

    return convertall(table, numparser(strict), **kwargs)


Table.convertnumbers = convertnumbers


class FieldConvertView(Table):

    def __init__(self, source, converters=None, failonerror=None,
                 errorvalue=None, where=None, pass_row=False, trusted=True):
        self.source = source
        self.converters = _normalize_converters(converters)
        self.failonerror = (config.failonerror if failonerror is None
                                else failonerror)
        self.errorvalue = errorvalue
        self.where = where
        self.pass_row = pass_row
        self.trusted = trusted

    def __iter__(self):
        return iterfieldconvert(self.source, self.converters, self.failonerror,
                                self.errorvalue, self.where, self.pass_row, self.trusted)

    def __setitem__(self, key, value):
        self.converters[key] = value


class _SpooledRowQueue(object):

    def __init__(self, buffersize, tempdir):
        self.buffersize = buffersize
        self.tempdir = tempdir
        self.memory = deque()
        self.file = None
        self.read_position = 0
        self.write_position = 0
        self.finished = False
        self.abandoned = False

    def append(self, row):
        if self.abandoned:
            return
        if self.buffersize is None or len(self.memory) < self.buffersize:
            if self.read_position == self.write_position:
                self.memory.append(row)
                return
        self._ensure_file()
        self.file.seek(self.write_position)
        pickle.dump(row, self.file, protocol=-1)
        self.file.flush()
        self.write_position = self.file.tell()

    def popleft(self):
        if self.memory:
            row = self.memory.popleft()
            self._reset_file_if_caught_up()
            return row
        if self.read_position < self.write_position:
            self.file.seek(self.read_position)
            row = pickle.load(self.file)
            self.read_position = self.file.tell()
            self._reset_file_if_caught_up()
            return row
        if self.finished:
            return _DONE
        return _NEED_DATA

    def finish(self):
        self.finished = True

    def abandon(self):
        self.abandoned = True
        self.close()

    def close(self):
        self.memory.clear()
        if self.file is not None:
            self.file.close()
            self.file = None
        self.read_position = 0
        self.write_position = 0

    def _ensure_file(self):
        if self.file is None:
            self.file = TemporaryFile(mode='wb+', dir=self.tempdir)

    def _reset_file_if_caught_up(self):
        if self.file is not None and self.read_position == self.write_position:
            self.file.seek(0)
            self.file.truncate()
            self.read_position = 0
            self.write_position = 0


class _ConvertOutput(Table):

    def __init__(self, splitter, accepted):
        self.splitter = splitter
        self.accepted = accepted

    def __iter__(self):
        generator = self.splitter.iter_output(self.accepted)
        self.splitter.begin_output(self.accepted, generator)
        return generator

    def __len__(self):
        raise TypeError('split conversion outputs are one-shot iterables')

    def __repr__(self):
        return '<accepted split conversion table>' if self.accepted \
            else '<rejected split conversion table>'

    def close(self):
        self.splitter.abandon_output(self.accepted)


class _FieldConvertSplitter(object):

    def __init__(self, source, converters, where, pass_row, trusted,
                 buffersize, tempdir):
        self.source = source
        self.converters = _normalize_converters(converters)
        self.where = where
        self.pass_row = pass_row
        self.trusted = trusted
        self.buffersize = buffersize
        self.tempdir = tempdir
        self.accepted = _ConvertOutput(self, True)
        self.rejected = _ConvertOutput(self, False)
        self.queues = {
            True: _SpooledRowQueue(buffersize, tempdir),
            False: _SpooledRowQueue(buffersize, tempdir),
        }
        self.states = {True: 'waiting', False: 'waiting'}
        self.source_iterator = None
        self.started = False
        self.source_done = False
        self.closed = False
        self.header = ()
        self.fields = []
        self.converter_functions = {}
        self.row_number = 0
        self.iterators = {True: None, False: None}

    def __del__(self):
        self.close()

    def begin_output(self, accepted, generator=None):
        if self.closed:
            raise RuntimeError('conversion result is closed')
        if self.states[accepted] != 'waiting':
            raise RuntimeError('each conversion output can be read once')
        self.states[accepted] = 'active'
        self.iterators[accepted] = generator

    def iter_output(self, accepted):
        self._start()
        try:
            yield tuple(self.header) if accepted else REJECTION_HEADER
            while True:
                if self.states[accepted] == 'abandoned':
                    return
                item = self._pull(accepted)
                if item is _DONE:
                    break
                yield item
            self.states[accepted] = 'done'
            self._auto_close()
        except GeneratorExit:
            self._abandon(accepted)
            self._auto_close()
            raise
        except BaseException:
            self.close()
            raise
        finally:
            self.iterators[accepted] = None

    def close(self):
        if self.closed:
            return
        self.closed = True
        for queue in self.queues.values():
            queue.close()
        iterator = self.source_iterator
        if iterator is not None and hasattr(iterator, 'close'):
            try:
                iterator.close()
            except Exception:
                pass

    def _start(self):
        if self.started:
            return
        self.source_iterator = iter(self.source)
        try:
            self.header = next(self.source_iterator)
        except StopIteration:
            self.header = ()
            self.fields = []
        else:
            self.fields = list(map(text_type, self.header))
        self.converter_functions = _build_converter_functions(
            self.fields, self.converters
        )
        self.where = _prepare_where(self.where, self.trusted)
        self.started = True

    def _pull(self, wanted):
        queue = self.queues[wanted]
        item = queue.popleft()
        if item is not _NEED_DATA:
            return item

        while not self.source_done:
            try:
                row = next(self.source_iterator)
            except StopIteration:
                self.source_done = True
                break
            self.row_number += 1
            target, output_row = self._convert_row(row)
            self.queues[target].append(output_row)
            item = queue.popleft()
            if item is not _NEED_DATA:
                return item

        self._finish_active_queues()
        return queue.popleft()

    def _convert_row(self, row):
        record = Record(row, self.fields) \
            if self.pass_row or self.where is not None else row
        if self.where is not None and not self.where(record):
            return True, tuple(record)

        output_row = list(record)
        for index, value in enumerate(record):
            if index not in self.converter_functions:
                continue
            converter = self.converter_functions[index]
            try:
                if self.pass_row:
                    output_row[index] = converter(value, record)
                else:
                    output_row[index] = converter(value)
            except Exception as error:
                field = self.header[index] if index < len(self.header) else index
                return False, (
                    self.row_number, field, value,
                    _format_exception(error)
                )
        return True, tuple(output_row)

    def _finish_active_queues(self):
        for accepted, state in self.states.items():
            if state != 'abandoned':
                self.queues[accepted].finish()

    def _abandon(self, accepted):
        if self.states[accepted] == 'active':
            self.states[accepted] = 'abandoned'
            self.queues[accepted].abandon()

    def abandon_output(self, accepted):
        if self.states[accepted] in ('waiting', 'active'):
            self.states[accepted] = 'abandoned'
            self.queues[accepted].abandon()
            iterator = self.iterators[accepted]
            if iterator is not None:
                iterator.close()
            self._auto_close()

    def _auto_close(self):
        if all(state in ('done', 'abandoned')
               for state in self.states.values()):
            self.close()


def iterfieldconvert(source, converters, failonerror, errorvalue, where,
                     pass_row, trusted):

    # grab the fields in the source table
    it = iter(source)
    try:
        hdr = next(it)
        flds = list(map(text_type, hdr))
        yield tuple(hdr)  # these are not modified
    except StopIteration:
        hdr = flds = []  # converters will fail selecting a field

    converter_functions = _build_converter_functions(flds, converters)

    # define a function to transform a value
    def transform_value(i, v, *args):
        if i not in converter_functions:
            # no converter defined on this field, return value as-is
            return v
        else:
            try:
                return converter_functions[i](v, *args)
            except Exception as e:
                if failonerror == 'inline':
                    return e
                elif failonerror:
                    raise e
                else:
                    return errorvalue

    # define a function to transform a row
    if pass_row:
        def transform_row(_row):
            return tuple(transform_value(i, v, _row)
                         for i, v in enumerate(_row))
    else:
        def transform_row(_row):
            return tuple(transform_value(i, v)
                         for i, v in enumerate(_row))

    where = _prepare_where(where, trusted=trusted)

    # prepare iterator
    if pass_row or where:
        # wrap rows as records
        it = (Record(row, flds) for row in it)

    # construct the data rows
    if where is None:
        # simple case, transform all rows
        for row in it:
            yield transform_row(row)
    else:
        # conditionally transform rows
        for row in it:
            if where(row):
                yield transform_row(row)
            else:
                yield row


def methodcaller(nm, *args):
    return lambda v: getattr(v, nm)(*args)


def dictconverter(d):
    def conv(v):
        try:
            if v in d:
                return d[v]
            else:
                return v
        except TypeError:
            # value is not hashable
            return v
    return conv


def format(table, field, fmt, **kwargs):
    """
    Convenience function to format all values in the given `field` using the
    `fmt` format string.

    The ``where`` keyword argument can be given with a callable or expression
    which is evaluated on each row and which should return True if the
    conversion should be applied on that row, else False.

    """

    conv = lambda v: fmt.format(v)
    return convert(table, field, conv, **kwargs)


Table.format = format


def formatall(table, fmt, **kwargs):
    """
    Convenience function to format all values in all fields using the
    `fmt` format string.

    The ``where`` keyword argument can be given with a callable or expression
    which is evaluated on each row and which should return True if the
    conversion should be applied on that row, else False.

    """

    conv = lambda v: fmt.format(v)
    return convertall(table, conv, **kwargs)


Table.formatall = formatall


def interpolate(table, field, fmt, **kwargs):
    """
    Convenience function to interpolate all values in the given `field` using
    the `fmt` string.

    The ``where`` keyword argument can be given with a callable or expression
    which is evaluated on each row and which should return True if the
    conversion should be applied on that row, else False.

    """

    conv = lambda v: fmt % v
    return convert(table, field, conv, **kwargs)


Table.interpolate = interpolate


def interpolateall(table, fmt, **kwargs):
    """
    Convenience function to interpolate all values in all fields using
    the `fmt` string.

    The ``where`` keyword argument can be given with a callable or expression
    which is evaluated on each row and which should return True if the
    conversion should be applied on that row, else False.

    """

    conv = lambda v: fmt % v
    return convertall(table, conv, **kwargs)


Table.interpolateall = interpolateall
