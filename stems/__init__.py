# -*- coding: utf-8 -*-
""" stems
"""
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
__author__ = """Chris Holden"""
__email__ = 'ceholden@gmail.com'

__package__ = 'stems'

__docs__ = 'https://ceholden.github.io/{pkg}/'.format(pkg=__package__)
__docs_version__ = '%s/%s' % (__docs__, __version__)
__repo_url__ = 'https://github.com/ceholden/{pkg}'.format(pkg=__package__)
__repo_issues__ = '/'.join([__repo_url__, 'issues'])


# See: http://docs.python-guide.org/en/latest/writing/logging/
import logging  # noqa
from logging import NullHandler as _NullHandler
logging.getLogger(__name__).addHandler(_NullHandler())
