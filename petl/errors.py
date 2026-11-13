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
    """Base class for malformed ordered change event streams."""

    def __init__(self, position, message):
        self.position = position
        self.message = message

    def __str__(self):
        return 'change event at position %s: %s' % (self.position,
                                                     self.message)


class TransactionBoundaryError(ChangeEventError):
    """Raised when begin/commit/rollback markers are duplicated or misplaced."""


class ChangeStreamError(ChangeEventError):
    """Raised for other malformed ordered change event streams."""


class ChangeConsistencyError(ChangeEventError):
    """Raised when a before-image does not match the visible row state."""


class UnclosedTransactionError(ChangeEventError):
    """Raised when the event stream ends while a transaction is still open."""


class ChangeSourceError(ChangeEventError):
    """Raised when the event source itself fails mid-stream.

    The original exception is preserved as ``__cause__``.
    """
