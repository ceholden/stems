""" Tests for :py:mod:`stems.masking`
"""
import dask.array as da
import numpy as np
import pytest
import xarray as xr

from stems import masking


# -----------------------------------------------------------------------------
# checkbit
def test_checkbit(landsat8_pixelqa_values, landsat8_pixelqa_info):
    for (label, quality), values in landsat8_pixelqa_values.items():
        info = landsat8_pixelqa_info[label]
        offset = info['offset']
        width = info['width']
        value = info.get('values', {}).get(quality, None)
        unpacked = masking.checkbit(np.array(values),
                                    offset,
                                    width=width,
                                    value=value)
        assert unpacked.all()


# -----------------------------------------------------------------------------
# bitpack_to_coding
def test_bitpack_to_coding():
    qaqc = np.array([
        [1, 1, 2722, 2724, 7808],  # fill, fill, terrain, clear, snow
        [1, 2720, 6900, 6904, 6908],  # fill, clear, cloud, cloud, cloud
        [2724, 2732, 3012, 3008, 6856]  # clear, clear, shadow, shadow, cirrus
    ])
    # In this example, we're going for Landsat ARD style codings
    # (can't reproduce 100% since C01 pixel_qa doesn't store water info)
    coding = {
        1: [(0, 1, 1)],  # fill
        5: [(4, 1, 1)],  # cloud
        3: [(7, 2, 2)],  # shadow
        4: [(9, 2, 2)],  # snow
        8: [(11, 2, 2)],  # cirrus
        10: [(1, 1, 1)]  # terrain
    }
    truth = np.array([
        [1, 1, 10, 0, 4],
        [1, 0, 5, 5, 5],
        [0, 0, 3, 3, 8]
    ])
    ans = masking.bitpack_to_coding(qaqc, coding, fill=0)
    np.testing.assert_equal(ans, truth)


# =============================================================================
# See:
# https://www.usgs.gov/land-resources/nli/landsat/landsat-collection-1-level-1-quality-assessment-band
@pytest.fixture
def landsat8_pixelqa_info(request):
    from stems.sensors import landsat
    return landsat.LANDSAT_C01_QAQC_DATA['OLI']


@pytest.fixture
def landsat8_pixelqa_values(request):
    pixel_qa = {}  # (label, quality/confidence) : values
    pixel_qa[('fill', None)] = [1]
    pixel_qa[('terrain_occl', None)] = [2, 2722]
    pixel_qa[('radiometric_sat', '1-2')] = [
        2724, 2756, 2804, 2980, 3012, 3748, 3780
    ]
    pixel_qa[('radiometric_sat', '3-4')] = [
        2728, 2760, 2808, 2984, 3016, 3752, 3784,
        6824, 6856, 6904, 7080, 7112, 7848, 7880
    ]
    pixel_qa[('radiometric_sat', '5+')] = [
        2732, 2764, 2812, 2988, 3020, 3756, 3788,
        6828, 6860, 6908, 7084, 7116, 7852, 7884
    ]
    pixel_qa[('cloud', None)] = [
        2800, 2804, 2808, 2812, 6896, 6900, 6904, 6908
    ]
    pixel_qa[('cloud_confidence', 'low')] = [
        2720, 2722, 2724, 2728, 2732, 2976, 2980, 2984,
        3744, 3748, 3752, 3756, 6816, 6820, 6824, 6828,
        7072, 7076, 7080, 7084, 7840, 7844, 7848, 7852
    ]
    pixel_qa[('cloud_confidence', 'medium')] = [
        2752, 2756, 2760, 2764, 3008, 3012, 3016, 3020,
        3776, 3780, 3784, 3788, 6848, 6852, 6856, 6860,
        7104, 7108, 7112, 7116, 7872, 7876, 7880, 7884
    ]
    pixel_qa[('cloud_confidence', 'high')] = [
        2800, 2804, 2808, 2812, 6896, 6900, 6904, 6908
    ]
    pixel_qa[('snow_ice_confidence', 'high')] = [
        3744, 3748, 3752, 3756, 3776, 3780, 3784, 3788,
        7840, 7844, 7848, 7852, 7872, 7876, 7880, 7884
    ]
    pixel_qa[('cirrus_confidence', 'low')] = [
        2720, 2722, 2724, 2728, 2732, 2752, 2760, 2764,
        2800, 2804, 2808, 2812, 2976, 2980, 2984, 2988,
        3008, 3012, 3016, 3020, 3744, 3748, 3756, 3780,
        3784, 3788
    ]
    pixel_qa[('cirrus_confidence', 'high')] = [
        6816, 6820, 6828, 6848, 6852, 6856, 6880, 6896,
        6900, 6904, 6908, 7072, 7076, 7080, 7084, 7104,
        7108, 7112, 7116, 7840, 7844, 7848, 7852, 7872,
        7876, 7880, 7884
    ]

    return pixel_qa
