"""Main group of commands for STEMS CLI
"""
import logging
from pkg_resources import iter_entry_points

import click
import click_plugins

import stems
from stems.logging_ import setup_logger

from . import options

_context = dict(
    token_normalize_func=lambda x: x.lower(),
    help_option_names=['--help', '-h']
)


@click_plugins.with_plugins(ep for ep in
                            iter_entry_points('stems.cli'))
@click.group(help='STEMS command line interface', context_settings=_context)
@click.version_option(stems.__version__)
@options.opt_verbose
@options.opt_quiet
@options.opt_scheduler
@options.opt_nprocs
@options.opt_nthreads
@click.pass_context
def main(ctx, verbose, quiet, scheduler, nprocs, nthreads):
    """ Spatio-temporal Tools for Earth Monitoring Science

    Home: https://github.com/ceholden/stems
    Docs: https://ceholden.github.io/stems/
    """
    verbosity = verbose - quiet
    log_level = max(logging.DEBUG, logging.WARNING - verbosity * 10)
    logger = setup_logger('stems', level=log_level)

    # check debug level since could be expensive to get info
    if scheduler is not None and log_level == logging.DEBUG:
        from stems.executor import executor_info
        info = executor_info(scheduler)
        for i in info:
            logger.debug(i)

    ctx.obj = {}
    ctx.obj['logger'] = logger
    ctx.obj['client'] = scheduler
