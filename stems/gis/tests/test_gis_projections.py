""" Tests for :py:mod:`stems.gis.projections`
"""
from osgeo import osr
from rasterio.crs import CRS
import pytest

from stems.gis import projections
from stems.gis.tests import data

PROJ_EPSG = [4326, 3857, 32619]


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


@pytest.mark.parametrize('wkt', [
    data.EXAMPLE_LAEA_NA['wkt'],
    data.EXAMPLE_AEA_NA['wkt'],
])
def test_epsg_code_fail(wkt):
    crs_ = CRS.from_wkt(wkt)
    with pytest.raises(ValueError, match=r'Cannot decode.*'):
        projections.epsg_code(crs_)


def test_crs_longname(example_crs):
    crs_ = CRS.from_wkt(example_crs['wkt'])
    test = projections.crs_longname(crs_)
    assert test == example_crs['longname']


# =============================================================================
# CF info / parameters
# =============================================================================

# -----------------------------------------------------------------------------
# cf_crs_name
def test_cf_crs_name(example_crs):
    crs_ = CRS.from_string(example_crs['wkt'])
    test = projections.cf_crs_name(crs_)
    assert test == example_crs['cf_name']


# -----------------------------------------------------------------------------
# cf_crs_attrs
def test_cf_crs_attrs_1():
    # WGS84
    crs = CRS.from_epsg(4326)
    test = projections.cf_crs_attrs(crs)

    assert test['horizontal_datum_name'] == 'WGS_1984'
    assert test['reference_ellipsoid_name'] == 'WGS 84'
    assert 'towgs84' not in test
    assert test['prime_meridian_name'] == 'Greenwich'


def test_cf_crs_attrs_2():
    # NAD83 / CONUS AEA
    crs = CRS.from_epsg(5070)
    test = projections.cf_crs_attrs(crs)

    assert test['horizontal_datum_name'] == 'North_American_Datum_1983'
    assert test['reference_ellipsoid_name'] == 'GRS 1980'
    assert test['towgs84'] == '0'
    assert test['prime_meridian_name'] == 'Greenwich'


# -----------------------------------------------------------------------------
# cf_proj_params
@pytest.mark.parametrize('lat_0', [-30, 0, 30])
@pytest.mark.parametrize('lon_0', [-45, 0, 45])
def test_cf_proj_params_LAEA(lon_0, lat_0):
    # Lambert Azimuthal Equal Area
    proj4 = (f"+proj=laea +lat_0={lat_0} +lon_0={lon_0} +x_0=0 +y_0=0 "
             f"+datum=WGS84 +units=m +no_defs ")
    crs = CRS.from_string(proj4)
    cf_attrs = projections.cf_proj_params(crs)
    assert cf_attrs['longitude_of_projection_origin'] == lon_0
    assert cf_attrs['latitude_of_projection_origin'] == lat_0


@pytest.mark.parametrize('lon_0', [-45, 0, 45])
@pytest.mark.parametrize(('lat_0', 'lat_1', 'lat_2', ), [
    (-30, -40, -20, ),
    (-20, 0, 20, ),
    (20, 30, 40, )
])
def test_cf_proj_params_AEA(lon_0, lat_0, lat_1, lat_2):
    # Alber's Conic Equal Area
    proj4 = (f"+proj=aea +lat_1={lat_1} +lat_2={lat_2} +lat_0={lat_0} "
             f"+lon_0={lon_0} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs ")
    crs = CRS.from_string(proj4)
    cf_attrs = projections.cf_proj_params(crs)
    assert cf_attrs['longitude_of_central_meridian'] == lon_0
    assert cf_attrs['latitude_of_projection_origin'] == lat_0
    assert cf_attrs['standard_parallel'] == (lat_1, lat_2)


# -----------------------------------------------------------------------------
# cf_ellps_params
def test_cf_ellps_params_WGS84():
    crs = CRS.from_epsg(4326)
    test = projections.cf_ellps_params(crs)

    assert test['semi_major_axis'] == 6378137.0
    assert abs(test['semi_minor_axis'] - 6356752.314245179) < 1e7
    assert abs(test['inverse_flattening'] - 298.257223563) < 1e7


def test_cf_ellps_params_NAD83():
    crs = CRS.from_epsg(5070)
    test = projections.cf_ellps_params(crs)

    assert test['semi_major_axis'] == 6378137.0
    assert abs(test['semi_minor_axis'] - 6356752.314140356) < 1e7
    assert abs(test['inverse_flattening'] - 298.257222101) < 1e7


# ----------------------------------------------------------------------------
# cf_xy_coord_names
@pytest.mark.parametrize(('crs', 'ans'), (
    (CRS.from_epsg(4326), ('longitude', 'latitude')),
    (CRS.from_epsg(5070), ('x', 'y'))
))
def test_cf_xy_coord_names(crs, ans):
    test = projections.cf_xy_coord_names(crs)
    assert test == ans
