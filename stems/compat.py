""" External library compatibility
"""
# cytoolz over toolz
try:
    import cytoolz as toolz
except ImportError:
    import toolz


def requires_module(module):
    """ Decorator to dependency check at function import time

    Parameters
    ----------
    module : str
        Module required for the function

    Returns
    -------
    callable
        Returns decorated function if dependency is importable,
        otherwise returns a dummy-function that will immediately
        raise an ImportError
    """
    import functools, importlib

    has_module = True
    try:
        mod = importlib.import_module(module)
    except ImportError as e:
        has_module = False
        err = str(e)

    def decorator(f):
        @functools.wraps(f)
        def inner(*args, **kwds):
            raise ImportError(f'Function "{f.__name__}" requires module '
                              f'"{module}", but it is not available: {err}')

        if has_module:
            return f
        else:
            return inner

    return decorator
