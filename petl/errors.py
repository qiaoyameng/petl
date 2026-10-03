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

    def __init__(self, key, metric, phase, cause=None):
        self.key = key
        self.metric = metric
        self.phase = phase
        self.cause = cause

    def __str__(self):
        message = '%s failed for key %r and metric %r' % (
            self.phase, self.key, self.metric
        )
        if self.cause is not None:
            message += ': %s' % self.cause
        return message
