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


class ChangeEventError(Exception):
    """Raised when an ordered change event stream is malformed."""

    def __init__(self, message, position=None, txn=None, op=None):
        self.message = message
        # zero-based index of the offending event in the source stream
        self.position = position
        self.txn = txn
        self.op = op

    def __str__(self):
        return ('change event error at position %r (txn=%r, op=%r): %s'
                % (self.position, self.txn, self.op, self.message))
