""" Data and functions related to the Landsat family of satellites
"""
from pathlib import Path

import yaml

_HERE = Path(__file__).parent


def _load_data(f):
    with open(f) as src:
        return yaml.safe_load(src)


LANDSAT_C01_QAQC_FILE = _HERE.joinpath('landsat_qaqc_c01.yml')
LANDSAT_C01_QAQC_DATA = _load_data(LANDSAT_C01_QAQC_FILE)
