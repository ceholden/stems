"""Tests for :py:mod:`stems.logging_`
"""
import logging

from stems import logging_


# ----------------------------------------------------------------------------
# setup_logger
def test_setup_logger_1():
    log = logging_.setup_logger(replace_handler=True)
    assert log.name == 'stems'
    assert len(log.handlers) == 1
    assert log.handlers[0].formatter._fmt == logging_.DEFAULT_LOG_FORMAT
    assert log.handlers[0].formatter.datefmt == logging_.DEFAULT_LOG_DATE_FORMAT
    assert log.level == logging.INFO


def test_setup_logger_2():
    # test initializing with existing logger
    log = logging.getLogger('stems')
    log_ = logging_.setup_logger(log, replace_handler=True)
    assert log == log_


def test_setup_logger_3():
    # Add a handler
    log_ = logging.getLogger('asdf')
    handler_ = logging.StreamHandler()
    log_.addHandler(handler_)

    # Should still exist
    log = logging_.setup_logger('asdf', replace_handler=False)
    assert handler_ in log.handlers
    assert len(log.handlers) == 2
