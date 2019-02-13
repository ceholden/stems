""" External library compatibility
"""
# cytoolz over toolz
try:
    import cytoolz as toolz
except ImportError:
    import toolz
