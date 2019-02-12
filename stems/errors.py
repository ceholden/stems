""" Custom errors/warnings for STEMS
"""
from . import (__docs_version__,
               __repo_issues__,
               __version__)


class TODO(NotImplementedError):

    _help = (f'This version ("{__version__}") has not yet implemented this '
             f'functionality. Please visit {__repo_issues__} and copy and '
             f'paste the following message if you would like this feature '
             f'to be supported.')

    def __init__(self, msg, *args, **kwds):
        msg = f'{self._help} : "{msg}"'
        super(NotImplementedError, self).__init__(msg, *args[1:], **kwds)
