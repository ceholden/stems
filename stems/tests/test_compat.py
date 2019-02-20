""" Tests for :py:mod:`stems.compat`
"""
import pytest

from stems import compat


# =============================================================================
# requires_module
@pytest.mark.parametrize('module', ('numpy', 'pandas', 'os', ))
def test_requires_module_pass(module):
    # Create test function
    @compat.requires_module(module)
    def test():
        return 42

    assert test() == 42



@pytest.mark.parametrize('module', ('asdf', 'not_a_module', ))
def test_requires_module_fail(module):
    # Create test function
    @compat.requires_module(module)
    def test():
        return 42

    match = rf'Function "test" requires module "{module}".*'
    with pytest.raises(ImportError, match=match):
        test()
