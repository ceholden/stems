""" Tests for :py:mod:`stems.utils`
"""
import collections
from functools import singledispatch
from pathlib import Path

import numpy as np
import pytest

from stems import utils
from stems.compat import toolz


# ----------------------------------------------------------------------------
# to_number
def test_to_number_1():
    with pytest.raises(ValueError):
        utils.to_number('asdf')
    assert utils.to_number('5') == 5
    assert utils.to_number('5.') == 5.


# ----------------------------------------------------------------------------
# list_like
@pytest.mark.parametrize(('obj', 'ans', ), (
    ('str', False),
    (tuple('str'), True),
    (list('str'), True),
    (np.array([5, 4]), True),
    ({}, False),
    (object, False),
    (123, False)
))
def test_list_like(obj, ans):
    assert utils.list_like(obj) == ans


# ----------------------------------------------------------------------------
# squeeze
@pytest.mark.parametrize(('obj', 'ans'), (
    ([0], 0),
    ([1, 2], [1, 2]),
    ([[1]], [1])
))
def test_squeeze(obj, ans):
    assert utils.squeeze(obj) == ans


# ----------------------------------------------------------------------------
# concat_lists
@pytest.mark.parametrize(('obj', 'ans', ), (
    ([0, 1], [0, 1]),
    ([0, [1]], [0, 1]),
    ([0, [1], [['abc']]], [0, 1, ['abc']]),
    ([0, [1], [[1]], ['abc']], [0, 1, [1], 'abc'])
))
def test_concat_lists_1(obj, ans):
    assert list(utils.concat_lists(obj)) == ans


# ----------------------------------------------------------------------------
# flatten_lists
@pytest.mark.parametrize(('obj', 'ans', ), (
    ([0, 1], [0, 1]),
    ([0, [1]], [0, 1]),
    ([0, [1], ['abc']], [0, 1, 'abc']),
    ([0, [1], [[1]], ['abc']], [0, 1, 1, 'abc'])
))
def test_flatten_lists_1(obj, ans):
    assert list(utils.flatten_lists(obj)) == ans


# ----------------------------------------------------------------------------
# update_nested 
def test_update_nested():
    d = {
        'a': {
            'a': 'A',
            'b': 'B'
        },
        'b': 'B',
        'c': {
            'c': 'C'
        }
    }
    o = {
        'a': {
            'b': 'BEE'
        },
        'c': {
            'd': 'DEE'
        }
    }
    ans = utils.update_nested(d, o)
    assert ans is not d  # new dict
    assert ans['a']['b'] == 'BEE'
    assert ans['a']['a'] == 'A'
    assert ans['b'] == 'B'
    assert ans['c']['c'] == 'C'
    assert ans['c']['d'] == 'DEE'


# ----------------------------------------------------------------------------
# FrozenKeyDict
def test_FrozenKeyDict_init():
    d = {'a': 5, 'b': 6}
    # keywords
    d_ = utils.FrozenKeyDict(a=5, b=6)
    assert d_ == d
    # keyword arguments
    d_ = utils.FrozenKeyDict(**d)
    assert d_ == d
    # dict
    d_ = utils.FrozenKeyDict(d)
    assert d_ == d


def test_FrozenKeyDict_1():
    d = utils.FrozenKeyDict(a=5, b='bee')
    assert isinstance(d, collections.abc.MutableMapping)
    assert d['a'] == 5 and d['b'] == 'bee'
    assert len(d) == 2
    assert sorted([k for k in d]) == ['a', 'b']
    repr_ = repr(d)
    assert repr_.startswith('FrozenKeyDict')
    assert repr(collections.OrderedDict(d)) in repr_

    d['a'] = 'five'
    assert d['a'] == 'five'

    with pytest.raises(KeyError, match=r'Cannot create new keys.*'):
        d['c'] = 6

    with pytest.raises(KeyError, match=r'Cannot delete keys'):
        del d['a']
    assert 'a' in d

    d2 = d.copy()
    assert d2 is not d
    assert d2 == d


# ---------------------------------------------------------------------------
# cached_property
def test_cached_property():
    # Test class -> creates new list if not cached
    class A(object):
        @utils.cached_property
        def a(self):
            # should be same list
            return [5, 10, 15]
        @property
        def b(self):
            return [5, 10, 15]

    test = A()
    a = test.a
    assert a == test.a
    assert a is test.a
    assert a == test.b
    assert a is not test.b


# ---------------------------------------------------------------------------
# register_multi_singledispatch
def test_register_multi_singledispatch():
    # test function
    @singledispatch
    def test(a):
        raise TypeError(f'No such function for type {type(a)}')
    @utils.register_multi_singledispatch(test, (int, float, str))
    def _test_scalar(a):
        return 'scalar'
    @test.register(list)
    def _test_list(a):
        return 'list'
    @utils.register_multi_singledispatch(test, (set, tuple))
    def _test_immut(a):
        return 'immut'

    assert len(test.registry) == 7

    assert test(5) == 'scalar'
    assert test(5.0) == 'scalar'
    assert test('five') == 'scalar'
    assert test([]) == 'list'
    assert test(tuple()) == 'immut'
    assert test(set()) == 'immut'


def test_register_multi_singledispatch_fail_1():
    def f():
        return 0
    with pytest.raises(TypeError, match=r'Function must be dispatchable.*'):
        @utils.register_multi_singledispatch(f, int)
        def nope():
            return 0


# ----------------------------------------------------------------------------
# np_promote_all_types
@pytest.mark.parametrize(('types', 'ans', ), (
    ((np.float64, np.float32, np.bool, ), np.float64),
    ((np.float64, np.int32, np.bool, ), np.float64),
    ((np.float32, np.float16, np.bool, ), np.float32),
    ((np.uint32, np.int32, np.byte, ), np.int64),
))
def test_np_promote_all_types(types, ans):
    test = utils.np_promote_all_types(*types)
    assert test == ans


# ----------------------------------------------------------------------------
# dtype_info
def _dtype_info_eq(a, b):
    if type(a) == type(b):
        attrs = [attr for attr in dir(a) if not attr.startswith('_')]
        if all([getattr(a, attr) == getattr(b, attr) for attr in attrs]):
            return True
    return False


def test_dtype_info():
    a = np.arange(10)
    # "Good" usage
    for dt in (np.int32, np.float32, np.byte):
        dt = np.dtype(dt)
        test = utils.dtype_info(a.astype(dt))
        ans = np.iinfo(dt) if dt.kind == 'i' else np.finfo(dt)
        assert _dtype_info_eq(test, ans)


def test_dtype_info_errors():
    a = np.arange(10)
    # Exceptions
    for dt in (np.bool, np.complex, np.datetime64):
        dt = np.dtype(dt)
        with pytest.raises(TypeError, match=r'Only valid for NumPy.*'):
            utils.dtype_info(a.astype(dt))


# ============================================================================
# FILE HELPERS
# ============================================================================
# find
def test_find(tmpdir):
    # Setup
    files = [
        tmpdir.join('no', 'test.txt'),
        tmpdir.join('asdf', 'asdf.yaml'),
        tmpdir.join('no', 'asdf', 'nope', 'uh.txt')
    ]
    for f in files:
        p = [p for p in f.parts() if not p.exists()][:-1]
        for p_ in p:
            p_.mkdir()
        f.write('test')

    # Find files -- no regex
    ans = utils.find(str(tmpdir), '*.txt', regex=False)
    assert len(ans) == 2
    ans = utils.find(str(tmpdir), 'asdf*', regex=False)
    assert len(ans) == 1

    # Find -- regex
    ans = utils.find(str(tmpdir), '^asdf$')
    assert len(ans) == 0
    ans = utils.find(str(tmpdir), '.*txt')
    assert len(ans) == 2


# renamed_upon_completion
@pytest.mark.parametrize('kwds', (
    {'prefix': '', 'suffix': '.tmp'},
    {'prefix': '_', 'suffix': ''},
    {'prefix': '', 'suffix': ''}
))
def test_renamed_upon_completion_1(tmpdir, kwds):
    d = tmpdir.mkdir('results')
    dest = str(d.join('some_result.nc'))

    kwds_ = toolz.merge(kwds, {'tmpdir': None})
    with utils.renamed_upon_completion(dest, **kwds_) as tmpfile:
        # Should be a str
        assert isinstance(tmpfile, str)
        # Should start/end according to args
        p = Path(tmpfile)
        if kwds_['prefix']:
            assert p.name.startswith(kwds_['prefix'])
        if kwds_['suffix']:
            assert p.name.endswith(kwds_['suffix'])
        # Should be in the same directory as the destination ``d``
        assert p.parent == Path(str(d))
        # Write to the tmpfile
        with open(tmpfile, 'w') as f:
            f.write('hello')
    # Should exist at destination now
    assert Path(str(dest)).exists()
# ------------------------------------------------------------------------------
# renamed_upon_completion
@pytest.mark.parametrize('kwds', (
    {'prefix': '', 'suffix': '.tmp'},
    {'prefix': '_', 'suffix': ''},
    {'prefix': '', 'suffix': ''}
))
def test_renamed_upon_completion_1(tmpdir, kwds):
    d = tmpdir.mkdir('results')
    dest = str(d.join('some_result.nc'))

    kwds_ = toolz.merge(kwds, {'tmpdir': None})
    with utils.renamed_upon_completion(dest, **kwds_) as tmpfile:
        # Should be a str
        assert isinstance(tmpfile, str)
        # Should start/end according to args
        p = Path(tmpfile)
        if kwds_['prefix']:
            assert p.name.startswith(kwds_['prefix'])
        if kwds_['suffix']:
            assert p.name.endswith(kwds_['suffix'])
        # Should be in the same directory as the destination ``d``
        assert p.parent == Path(str(d))
        # Write to the tmpfile
        with open(tmpfile, 'w') as f:
            f.write('hello')
    # Should exist at destination now
    assert Path(str(dest)).exists()


def test_renamed_upon_completion_tmpdir(tmpdir):
    d = tmpdir.mkdir('results')
    dest = str(d.join('some_results.nc'))

    # Should also work as a Path
    tmpdir_ = Path(str(tmpdir))
    with utils.renamed_upon_completion(dest, tmpdir=tmpdir_) as tmpfile:
        p = Path(tmpfile)
        # The tmpfile should be ``tmpdir``, not ``d``
        assert p.parent == tmpdir_
        with open(tmpfile, 'w') as f:
            f.write('hello')
    # Should exist at destination now
    assert Path(dest).exists()


# ============================================================================
# IMPORT / MODULE HELPERS 
# ============================================================================
# find_subclasses
def test_find_subclasses():
    class A(object):
        pass
    class B(A):
        pass
    class C(B):
        pass
    class D(B):
        pass

    subcls = utils.find_subclasses(A)
    assert all([cls_ in subcls for cls_ in (B, C, D)])


# ----------------------------------------------------------------------------
# import_object
def test_import_object_1():
    import collections
    test = utils.import_object('collections.abc.Sequence')
    assert test is collections.abc.Sequence


def test_import_object_exception():
    with pytest.raises(ImportError):
        utils.import_object('not.a.module')
