"""Tests for :py:mod:`plants.executor`
"""
import distributed
import pytest

from stems import executor


# =============================================================================
# setup_executor
def test_setup_executor_process(n_workers=2, threads_per_worker=4):
    client = executor.setup_executor(n_workers=n_workers,
                                     threads_per_worker=threads_per_worker)
    ncores = client.ncores()

    assert len(ncores) == n_workers
    assert sum(ncores.values()) == n_workers * threads_per_worker

    client.close()


def test_setup_executor_distributed(n_workers=1, threads_per_worker=2):
    cluster = distributed.LocalCluster(n_workers=n_workers,
                                       threads_per_worker=threads_per_worker)
    client = distributed.Client(cluster)
    address = cluster.scheduler.address

    test = executor.setup_executor(address)

    assert test.scheduler.address == cluster.scheduler.address
    assert client.scheduler_info() == test.scheduler_info()

    test.close()
    cluster.close()
    client.close()


# =============================================================================
# executor_info
def test_executor_info(n_workers=2, threads_per_worker=4):
    client = executor.setup_executor(n_workers=n_workers,
                                     threads_per_worker=threads_per_worker)
    ncores = client.ncores()

    infos = executor.executor_info(client, True, True, True)
    assert len(infos) == 6
    assert str(len(ncores)) in infos[3]
    assert str(sum(ncores.values())) in infos[4]

    infos = executor.executor_info(client, False, True, True)
    assert len(infos) == 4
    assert 'Bokeh' in infos[0]

    infos = executor.executor_info(client, False, False, True)
    assert len(infos) == 3
    assert 'Workers' in infos[0]

    client.close()
