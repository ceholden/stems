""" STEMS CLI components
"""
import datetime as dt
import logging
import re

import click

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
_RE_NUMBER = re.compile(r'[-+]?\d*\.\d+|\d+')
_TYPE_FILE = click.Path(exists=True, readable=True,
                        dir_okay=False, resolve_path=True)
_KEY_DATE_FORMAT = 'date_format'


# ============================================================================
# CALLBACKS
def cb_dict(ctx, param, value):
    """ Call back for dict style arguments (e.g., KEY=VALUE)
    """
    d = {}
    for val in value or []:
        if '=' not in val:
            raise click.BadParameter(
                f'Must specify "{param}" as KEY=VALUE ({value} given)'
            )
        else:
            k, v = val.split('=', 1)
            d[k] = v
    return d


def cb_time(ctx, param, value):
    """ Callback for parsing to a ``datetime`` with ``opt_date_format``
    """
    if value is None:
        return value

    _format = ctx.params[_KEY_DATE_FORMAT]
    try:
        time = dt.datetime.strptime(value, _format)
    except KeyError:
        raise click.ClickException(
            f'Need to use `--{_KEY_DATE_FORMAT}` when using `cb_time`.')
    except Exception as e:
        raise click.BadParameter(
            f'Cannot parse "{value}" to date with format "{_format}"')
    else:
        return time


def cb_bounds(ctx, param, value):
    """ Callback to create a BoundingBox
    """
    if value is None:
        return value

    try:
        bbox = _RE_NUMBER.findall(value)
        if len(bbox) != 4:
            raise ValueError(
                f'Did not parse 4 numbers from input "{value}" (got "{bbox}")')
        bbox = [float(i) for i in bbox]
    except Exception as e:
        raise click.BadParameter(
            f'Cannot parse "{value}" to a BoundingBox: {e}')
    else:
        from rasterio.coords import BoundingBox
        return BoundingBox(*bbox)


def cb_executor(ctx, param, value):
    """ Callback for returning a Distributed client
    """
    # TODO: can we find if there's no subcommand, and just skip?
    from stems.executor import setup_executor
    nprocs = ctx.params.get('nprocs', None)
    nthreads = ctx.params.get('nthreads', None)

    # Parse address
    if value:
        client = setup_executor(address=value)
    elif nprocs or nthreads:
        client = setup_executor(n_workers=nprocs,
                                threads_per_worker=nthreads or 1)
    else:
        client = None

    return client


def close_scheduler():
    """ Close all clients/clusters on exit using ``ctx.call_on_close``
    """
    ctx = click.get_current_context()
    client = ctx.obj.get('client', None)
    if client is not None:
        try:
            client.close(timeout=10)
        except Exception as e:
            logger.debug(f'Exception closing executor client: {e}')
        else:
            logger.debug('Closed executor client')
    else:
        logger.debug('No client to close')


# ============================================================================
# ARGS
arg_config_file = click.argument('config', nargs=1, type=_TYPE_FILE)
arg_job_id = click.argument('job_id', nargs=1, type=click.INT)
arg_job_total = click.argument('job_total', nargs=1, type=click.INT)


# ============================================================================
# OPTIONS
opt_verbose = click.option('--verbose', '-v', count=True, help='Be verbose')

opt_quiet = click.option('--quiet', '-q', count=True, help='Be quiet')

opt_bounds = click.option(
    '--bounds', default=None, callback=cb_bounds,
    help='BoundingBox : left, bottom, right, top'
)
opt_date_format = click.option(
    '--%s' % _KEY_DATE_FORMAT, default='%Y-%m-%d',
    show_default=True, is_eager=True,
    help='Format string for dates'
)
opt_nodata = click.option(
    '--nodata', '--ndv', 'nodata',
    default=-9999, type=float,
    show_default=True, help='NoDataValue'
)

# Dask Distributed Executor / Client
opt_nprocs = click.option(
    '--nprocs', default=None,
    show_default=True, is_eager=True, type=click.INT,
    help='Number of workers to create'
)
opt_nthreads = click.option(
    '--nthreads', default=None,
    show_default=True, is_eager=True, type=click.INT,
    help='Number of threads per worker'
)
opt_scheduler = click.option(
    '--scheduler', default=None, show_default=True,
    callback=cb_executor,
    help='Scheduler address. Otherwise spins up `LocalCluster`'
)
