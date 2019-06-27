""" STEMS CLI components
"""
import datetime as dt
import logging
import re

import click

from stems.executor import setup_backend, setup_executor

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


def _param_name_job_id_total(param):
    if isinstance(param, click.Option):
        return '"--job_id"', '"--job_total"'
    else:
        return '"job_id"'.upper(), '"job_total"'.upper()


def _cb_job_id(ctx, param, value):
    s_job_id, s_job_total = _param_name_job_id_total(param)
    if value is None:
        return 1
    elif value <= 0:
        raise click.BadParameter(f'{s_job_id} must be at least 1 (>=1)')
    else:
        total = ctx.params.get('job_total', 1)
        if value > total:
            raise click.BadParameter(f'{s_job_id} is larger than {s_job_total} '
                                     f'({value} > {total})')
        else:
            return value


def _cb_job_total(ctx, param, value):
    s_job_id, s_job_total = _param_name_job_id_total(param)
    if value <= 0:
        raise click.BadParameter(f'{s_job_total} must be at least 1')
    else:
        return value


# ============================================================================
# ARGS
arg_config_file = click.argument('config', nargs=1, type=_TYPE_FILE)

arg_job_id = click.argument('job_id', nargs=1, callback=_cb_job_id,
                            type=click.INT)
arg_job_total = click.argument('job_total', nargs=1, callback=_cb_job_total,
                               type=click.INT)


# ============================================================================
# OPTIONS

# Job ID / Total as options
opt_job_id = click.option('--job_id', type=click.INT, callback=_cb_job_id,
                          help='Job ID (out of ``--job_total`` workers)')
opt_job_total = click.option('--job_total', type=click.INT, is_eager=True,
                             callback=_cb_job_total,
                             default=1, show_default=True,
                             help='Total number of jobs running this script')


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

# =============================================================================
# Dask Distributed Executor / Client
EXECUTORS = ('sync', 'threads', 'processes',  'distributed', )


def cb_executor(ctx, param, value):
    """ Callback for returning a Distributed client

    TODO
    ----
    * Can we find if there's no subcommand, and just skip?
    """
    executor, workers_or_ip = value

    # Handle requests for threads/processes
    client = None
    if executor in ('threads', 'processes', 'sync', ):
        setup_backend(executor, workers_or_ip)
    elif executor == 'distributed':
        try:
            workers = int(workers_or_ip)
        except ValueError:
            address = workers_or_ip
            workers = None
        else:
            address = None
        client = setup_executor(address=address, n_workers=workers)

    if client:
        def close_scheduler():
            if client is not None:
                try:
                    client.close(timeout=10)
                except Exception as e:
                    logger.debug(f'Exception closing executor client: {e}')
                else:
                    logger.debug('Closed executor client')
            else:
                logger.debug('No client to close')
        ctx.call_on_close(close_scheduler)

    return client


opt_executor = click.option(
    '--executor', type=(click.Choice(EXECUTORS), str),
    default=('sync', None), show_default=True,
    callback=cb_executor,
    help=(
        'Configure parallel processing options for Dask locally ("sync", '
        '"threads", or "processes") or using Distributed ("distributed"). '
        'Must provide either worker count or scheduler address (ip:port).'
    )
)
