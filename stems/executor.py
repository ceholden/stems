""" Dask/Distributed related helpers
"""
import logging
import socket

logger = logging.getLogger(__name__)


def setup_backend(scheduler, workers=None):
    """ Setup Dask to use threads or processes
    """
    import dask

    if scheduler in ('sync', 'single-threaded', 'synchronous', ):
        logger.debug('Using the synchronous/single-threaded backend')
        dask.config.set(scheduler='synchronous')
    elif scheduler == 'threads':
        if workers:
            from multiprocessing.pool import ThreadPool
            pool = ThreadPool(int(workers))
        else:
            pool = None
        logger.debug(f'Using `threads` backend (n={workers or "auto"})')
        dask.config.set(scheduler='threads', pool=pool)
    elif scheduler == 'processes':
        if workers:
            from multiprocessing import Pool
            pool = Pool(int(workers))
        else:
            pool = None
        logger.debug(f'Using `processes` backend (n={workers or "auto"})')
        dask.config.set(scheduler='processes', pool=pool)
    else:
        raise KeyError(f'Unsupported scheduler "{scheduler}"')


def setup_executor(address=None, n_workers=None, threads_per_worker=1, **kwds):
    """ Setup a Dask distributed cluster scheduler client

    Parameters
    ----------
    address : str, optional
        This can be the address of a ``Scheduler`` server, like a string
        ``'127.0.0.1:8786'``. If ``None``, sets up a ``LocalCluster``
    n_workers : int, optional
        Number of workers. Only used if setting up a ``LocalCluster``
    threads_per_worker : int, optional
        Number of threads per worker
    kwds
        Additional options passed to :py:func:`distributed.Client`

    Returns
    -------
    distributed.Client
        Distributed compute client
    """
    import distributed
    try:
        client = distributed.Client(address=address,
                                    n_workers=n_workers,
                                    threads_per_worker=threads_per_worker,
                                    **kwds)
    except Exception as e:
        logger.exception('Could not start `distributed` cluster')
        raise
    else:
        return client


def executor_info(client, ip=True, bokeh=True, stats=True):
    """ Return a list of strings with info for a scheduler

    Parameters
    ----------
    client : distributed.client.Client
        Scheduler (e.g., from ``client.scheduler``)
    ip : bool, optional
        Include scheduler IP
    bokeh : bool, optional
        Include Bokeh visualization IP
    stats : bool, optional
        Include stats on cluster (ncores, memory, etc)

    Returns
    -------
    list[str]
        Cluster information items
    """
    import distributed
    info = client.scheduler_info()

    if client.scheduler is not None:
        ip_str = client.scheduler.address
    else:
        ip_str = ''

    if info and 'bokeh' in info['services']:
        protocol, rest = client.scheduler.address.split('://')
        port = info['services']['bokeh']
        if protocol == 'inproc':  # process/thread server
            host = 'localhost'
        else:
            host = rest.split(':')[0]
        bokeh_str = 'http://{host}:{port}/status'.format(host=host, port=port)
    else:
        bokeh_str = ''

    if info:
        workers = len(info['workers'])
        cores = sum(w.get('ncores', w.get('nthreads', 1))
                    for w in info['workers'].values())
        memory = sum(w['memory_limit'] for w in info['workers'].values())
        memory = distributed.utils.format_bytes(memory)
    else:
        workers, cores, memory = '', '', ''

    infos = []
    if ip:
        infos.append('Scheduler: {0}'.format(ip_str))
        infos.append('Host: {0}'.format(socket.gethostname()))
    if bokeh:
        infos.append('Bokeh: {0}'.format(bokeh_str))
    if stats:
        infos.append('Workers: {0}'.format(workers))
        infos.append('Cores: {0}'.format(cores))
        infos.append('Memory: {0}'.format(memory))

    return infos
