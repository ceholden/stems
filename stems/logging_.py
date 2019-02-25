"""Some sensible defaults for logging

Designed for applications or interactive use -- not importing in library code.
The 'stems' logger is setup for library use with a NullHandler in
``stems/__init__.py``.
"""
import logging

DEFAULT_LOG_FORMAT = ' '.join(
    ('%(asctime)s',
     '%(levelname)s',
     '%(lineno)s',
     '%(module)s.%(funcName)s',
     '%(message)s', )
)
DEFAULT_LOG_DATE_FORMAT = '%H:%M:%S'


def setup_logger(name='stems',
                 fmt=DEFAULT_LOG_FORMAT,
                 datefmt=DEFAULT_LOG_DATE_FORMAT,
                 level=logging.INFO,
                 handler=None,
                 replace_handler=True):
    """ Setup and return a logger with formatter and handler

    Parameters
    ----------
    name : str, optional
        Name of logger. Defaults to 'stems' package module level
    fmt : str, optional
        Format string for ``logging.Formatter``
    datefmt : str, optional
        Date format string for ``logging.Formatter``
    level : int or str, optional
        Level for logging passed to ``logging.Logger.setLevel``
    handler : logging.Handler or None, optional
        Specify a handler to use. Defaults to a newly created
        ``logging.StreamHandler``.
    replace_handler : bool, optional
        Replace all existing handlers (otherwise appends handler)

    Returns
    -------
    logging.Logger
        Configured logger
    """
    formatter = logging.Formatter(fmt, datefmt)
    handler = handler or logging.StreamHandler()
    handler.setFormatter(formatter)

    log = logging.getLogger(name)
    if replace_handler:
        log.handlers = [handler]
    else:
        log.addHandler(handler)
    log.addHandler(handler)
    log.setLevel(level)

    return log
