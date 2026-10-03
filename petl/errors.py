from __future__ import absolute_import, print_function, division


class DuplicateKeyError(Exception):

    def __init__(self, key):
        self.key = key

    def __str__(self):
        return 'duplicate key: %r' % (self.key,)


class FieldSelectionError(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return 'selection is not a field or valid field index: %r' % (self.value,)


class ArgumentError(Exception):

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return 'argument error: %s' % self.message


class AggregationError(Exception):

    def __init__(self, phase, key, field, error):
        self.phase = phase
        self.key = key
        self.field = field
        self.metric = field
        self.error = error

    def __str__(self):
        return ('aggregation %s failed for key %r and field %r: %s'
                % (self.phase, self.key, self.field, self.error))
