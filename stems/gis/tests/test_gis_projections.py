""" Tests for :py:mod:`stems.gis.projections`
"""
from osgeo import osr
from rasterio.crs import CRS
import pytest

from stems.gis import projections


WKT_WGS84 = """
GEOGCS["WGS 84",
    DATUM["WGS_1984",
        SPHEROID["WGS 84",6378137,298.257223563,
            AUTHORITY["EPSG","7030"]],
        AUTHORITY["EPSG","6326"]],
    PRIMEM["Greenwich",0,
        AUTHORITY["EPSG","8901"]],
    UNIT["degree",0.01745329251994328,
        AUTHORITY["EPSG","9122"]],
    AUTHORITY["EPSG","4326"]]
"""
WKT_LAEA_NA = """
PROJCS["Lambert Azimuthal Equal Area - North America",
    GEOGCS["WGS 84",
        DATUM["WGS_1984",
            SPHEROID["WGS 84",6378137,298.257223563,
                AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.0174532925199433,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4326"]],
    PROJECTION["Lambert_Azimuthal_Equal_Area"],
    PARAMETER["latitude_of_center",45],
    PARAMETER["longitude_of_center",100],
    PARAMETER["false_easting",0],
    PARAMETER["false_northing",0],
    UNIT["Meter",1]]
"""
WKT_AEA_NA = """
PROJCS["Alber's Equal Area Conic - North America",
    GEOGCS["WGS 84",
        DATUM["WGS_1984",
            SPHEROID["WGS 84",6378137,298.257223563,
                AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.0174532925199433,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4326"]],
    PROJECTION["Albers_Conic_Equal_Area"],
    PARAMETER["standard_parallel_1",29.5],
    PARAMETER["standard_parallel_2",45.5],
    PARAMETER["latitude_of_center",23],
    PARAMETER["longitude_of_center",-96],
    PARAMETER["false_easting",0],
    PARAMETER["false_northing",0],
    UNIT["Meter",1]]
"""

PROJ_WKT = [WKT_WGS84, WKT_LAEA_NA, WKT_AEA_NA]
PROJ_NAMES = ['latitude_longitude',
              'lambert_azimuthal_equal_area',
              'albers_conic_equal_area']

PROJ_EPSG = [4326, 3857, 32619]
PROJ_LONGNAMES = ['WGS 84',
                  'WGS 84 / Pseudo-Mercator',
                  'WGS 84 / UTM zone 19N']


# ============================================================================
# EPSG code
@pytest.mark.parametrize('code', [5070, 4326, 32619])
def test_epsg_code(code):
    # test rasterio
    crs_ = CRS.from_epsg(code)
    test = projections.epsg_code(crs_)
    assert test == f'epsg:{code}'
    # test osr
    crs_ = osr.SpatialReference()
    crs_.ImportFromEPSG(code)
    test = projections.epsg_code(crs_)
    assert test == f'epsg:{code}'


@pytest.mark.parametrize('wkt', [WKT_LAEA_NA, WKT_AEA_NA])
def test_epsg_code_fail(wkt):
    crs_ = CRS.from_wkt(wkt)
    with pytest.raises(ValueError, match=r'Cannot decode.*'):
        projections.epsg_code(crs_)


# =============================================================================
# CF info / parameters
@pytest.mark.parametrize(('wkt', 'name', ), zip(PROJ_WKT, PROJ_NAMES))
def test_cf_crs_name(wkt, name):
    crs_ = CRS.from_string(wkt)
    test = projections.cf_crs_name(crs_)
    assert test == name


@pytest.mark.parametrize(('code', 'longname', ),
                         zip(PROJ_EPSG, PROJ_LONGNAMES))
def test_cf_crs_longname(code, longname):
    crs_ = CRS.from_epsg(code)
    test = projections.cf_crs_longname(crs_)
    assert test == longname


def test_cf_crs_longname_custom():
    crs_ = CRS.from_wkt(WKT_AEA_NA)
    test = projections.cf_crs_longname(crs_)
    assert test == "Alber's Equal Area Conic - North America"
