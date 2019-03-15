""" Create VRTs
"""
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
import logging
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom

from osgeo import gdal
import rasterio
from rasterio.coords import BoundingBox
from rasterio.dtypes import _gdal_typename
from rasterio.transform import rowcol
import six

from ..gis.geom import (bounds_transform_union,
                        calculate_src_window,
                        calculate_dst_window)
from ..utils import cached_property, list_like, relative_to

gdal.UseExceptions()

logger = logging.getLogger(__name__)


_NOBANDS_NOPROPS_ERROR_MSG = (
    'Cannot determine dataset properties without storing any '
    'bands. Add some via ``VRTDataset.add_band``'
)


class VRTDataset(object):
    """ Create a VRT from a band in one or more datasets

    Parameters
    ----------
    separate : bool, optional
        Put input bands in separate, stacked bands in the output

    """
    def __init__(self, separate=True):
        self.separate = separate
        self.root = ET.Element('VRTDataset')
        self._bands = defaultdict(list)

    @property
    def bands(self):
        """dict[int, list[VRTSourceBand]]: Bands organized by output VRT band
        """
        # Ensure is returned ordered by VRT bidx (the key)
        # Also ensure we don't return empty list of bands
        return OrderedDict((k, self._bands[k]) for k in sorted(self._bands)
                           if self._bands[k])

    @property
    def transform(self):
        """Affine: Affine transform for VRTDataset
        """
        _, transform, _ = self._get_bounds_transform()
        return transform

    @property
    def bounds(self):
        """BoundingBox: Bounding box of VRTDataset
        """
        bounds, _, _ = self._get_bounds_transform()
        return bounds

    @property
    def width(self):
        """ int: Number of columns
        """
        return self.shape[1]

    @property
    def height(self):
        """ int: Number of rows
        """
        return self.shape[0]

    @property
    def shape(self):
        """tuple[int, int]: Number of rows and columns
        """
        _, _, shape = self._get_bounds_transform()
        return shape

    @property
    def count(self):
        """int: Number of output bands in VRT
        """
        return len(self.bands)

    @property
    def crs(self):
        """CRS: VRTDataset coordinate reference system
        """
        if not self.bands:
            raise ValueError(_NOBANDS_NOPROPS_ERROR_MSG)
        bands_list = self._bands_to_list()
        return bands_list[0][1].crs

    @classmethod
    def from_bands(cls, paths, separate=True, bidx=1, **kwds):
        """
        Parameters
        ----------
        paths : str or list[str]
            List of paths to open as datasets
        separate : bool, optional
            Put input bands in separate, stacked bands in the output
        bidx : int, list[int], optional
            Band indices of `datasets` to include. If ``int``, all
            ``paths`` will use this band ``bidx``. Otherwise, pass a
            list of band indices for each path in ``paths``. Defaults
            to ``1``.
        kwds : dict
            Keywords to pass to ``VRTSourceBand()`` for each band. Pass
            a list or tuple as a value to specify different values
            for each band.

        Returns
        -------
        VRTDataset
            VRTDataset initialized from input paths
        """
        if isinstance(paths, six.string_types):
            paths = (paths, )
        if isinstance(bidx, int):
            bidx = (bidx, ) * len(paths)
        assert len(paths) == len(bidx)

        if kwds:
            for k, v in kwds.items():
                if not list_like(v):
                    logger.debug('Found scalar for "{0}" keyword. Duplicating '
                                 'for each band')
                    kwds[k] = (v, ) * len(paths)
                assert len(kwds[k]) == len(paths)

        vrt = cls()
        for i, (path, bidx_) in enumerate(zip(paths, bidx)):
            _kwds = {k: v[i] for k, v in kwds.items()}
            vrt.add_band(path, bidx_, **_kwds)

        return vrt

    def add_band(self, path, src_bidx=1, vrt_bidx=None, **band_kwds):
        """ Add a band to VRT dataset

        Parameters
        ----------
        path : str
            List of paths to open as datasets
        src_bidx : int, optional
            Band indices of `datasets` to include. Defaults to ``1``
        vrt_bidx : int or None
            Destination band in VRT for new band. Only used if
            ``self.separate`` is True
        band_kwds : dict
            Additional keyword arguments passed onto :py:class:`VRTSourceBand`

        Returns
        -------
        vrt_bidx: int
            VRT band index

        Raises
        ------
        ValueError
            Raised if vrt_bidx is invalid
        """
        # Handle non-specified vrt_bidx
        n_band = len(self.bands)
        if self.separate:
            vrt_bidx = n_band + 1 if vrt_bidx is None else vrt_bidx
        else:
            if vrt_bidx is not None and vrt_bidx != 1:
                raise ValueError('`vrt_bidx` must be `1` if not stacking into '
                                 'separate bands (see `self.separate`)')
            vrt_bidx = 1
        if vrt_bidx <= 0:
            raise ValueError('`vrt_bid` must be greater than 0')

        vrtband = VRTSourceBand(path, src_bidx, **band_kwds)

        # Validate if not 1st band
        if n_band > 0:
            self._validate(vrtband)

        # Append
        self._bands[vrt_bidx].append(vrtband)

        return vrt_bidx

    def write(self, path=None, relative=False):
        """ Save VRT XML data to a filename

        Parameters
        ----------
        path : str, optional
            Save VRT to this filename. If ``None``, returns
            the XML text
        relative : bool, optional
            Reference VRT sources relative to the VRT

        Returns
        -------
        str
            Filename
        """
        if relative:
            relative = str(path)

        xml_ele = _make_vrt_element(self.shape[1], self.shape[0],
                                    self.transform,
                                    self.crs,
                                    self.bands,
                                    relative_to_vrt=relative)
        xmlstr = _make_vrt_str(xml_ele)

        if path is not None:
            with open(str(path), 'w') as fid:
                fid.write(xmlstr)
            return path
        else:
            return xmlstr

    def _bands_to_list(self):
        # () -> [(vrt_bidx, VRTSourceBand), ...]
        return list([(k, band) for k in self.bands for band in self.bands[k]])

    def _validate(self, test_band):
        # Validate suitability of newly added band
        if test_band.crs != self.crs:
            raise ValueError('All bands must have same ``crs``')

    def _get_bounds_transform(self):
        if not self.bands:
            raise ValueError(_NOBANDS_NOPROPS_ERROR_MSG)
        bands_list = self._bands_to_list()
        bounds_ = list([b.bounds for idx, b in bands_list])
        transforms_ = list([b.transform for idx, b in bands_list])
        return bounds_transform_union(bounds_, transforms_[0])


class VRTSourceBand(object):
    """ A VRT band originating from some other file

    Note that all properties on this object return information used
    for VRT XML generation, but not XML elements (e.g., returns the
    SubElement name and this new element's text value).

    Parameters
    ----------
    path : str
        Filename of dataset containing band
    src_bidx : int
        Source band index (begins on 1)
    description : str, optional
        Override band description
    nodata : float or int, optional
        Override NoDataValue
    keep_open : bool, optional
        Keep dataset open
    """
    def __init__(self, path, src_bidx,
                 description=None,
                 nodata=None,
                 keep_open=False):
        self.path = path
        self.src_bidx = src_bidx
        self._desc = description
        self._ndv = nodata
        self.keep_open = keep_open
        self._ds = None

    @contextmanager
    def open(self):
        self.start()
        yield self._ds
        if not self.keep_open:
            self._ds = None

    def start(self):
        """ Open dataset, if closed
        """
        if getattr(self._ds, 'closed', True):
            logger.debug('Opening dataset for VRTSourceBand')
            self._ds = rasterio.open(str(self.path), 'r')

    def close(self):
        """ Close dataset reference, if open
        """
        if self._ds is not None:
            self._ds.close()
            self._ds = None

    @cached_property
    def crs(self):
        with self.open() as ds:
            return ds.crs

    @cached_property
    def transform(self):
        with self.open() as ds:
            return ds.transform

    @cached_property
    def bounds(self):
        with self.open() as ds:
            return ds.bounds

    @cached_property
    def width(self):
        with self.open() as ds:
            return ds.width

    @cached_property
    def height(self):
        with self.open() as ds:
            return ds.height

    @cached_property
    def shape(self):
        with self.open() as ds:
            return (self.height, self.width)

    @cached_property
    def dtype(self):
        with self.open() as ds:
            return ds.dtypes[self.src_bidx - 1]

    @cached_property
    def blockxsize(self):
        with self.open() as ds:
            return ds.block_shapes[self.src_bidx - 1][1]

    @cached_property
    def blockysize(self):
        with self.open() as ds:
            return ds.block_shapes[self.src_bidx - 1][0]

    @cached_property
    def nodata(self):
        if self._ndv is not None:
            return self._ndv
        else:
            with self.open() as ds:
                return ds.nodatavals[self.src_bidx - 1]

    @cached_property
    def description(self):
        if self._desc is not None:
            return self._desc
        else:
            with self.open() as ds:
                return ds.descriptions[self.src_bidx - 1]

    @cached_property
    def colorinterp(self):
        with self.open() as ds:
            return ds.colorinterp[self.src_bidx - 1]


# ----------------------------------------------------------------------------
# XML
def _make_vrt_str(root):
    root_str = ET.tostring(root)
    return (minidom
            .parseString(root_str)
            .toprettyxml(indent='    '))


def _make_vrt_element(vrt_width, vrt_height, vrt_transform, vrt_crs, vrt_bands,
                      relative_to_vrt=None):
    """Return VRT as XML Element

    Parameters
    ----------
    vrt_width : int
        Number of columns
    vrt_height : int
        Number of rows
    vrt_transform : Affine
        Transform of output VRT
    vrt_crs : CRS
        CRS of output VRT
    vrt_bands : dict[int, list[VRTSourceBand]]
        VRTSourceBand information, organized by output VRT bidx (e.g.,
        ``{1: [VRTSourceBand], 2: [VRTSourceBand]}`` if separate, or
        ``{1: [VRTSourceBand, VRTSourceBand]}`` if mosaicing)
    relative_to_vrt : str or Path
        Reference VRT sources relative to the VRT at this location

    Returns
    -------
    xml.etree.ElementTree
        XML element tree with XML information
    """
    # Needed, but since we can get from other inputs calculate to save space
    vrt_bounds = BoundingBox(vrt_transform.c,
                             vrt_transform.f + vrt_transform.e * vrt_height,
                             vrt_transform.c + vrt_transform.a * vrt_width,
                             vrt_transform.f)

    # Create root element
    root = _make_vrt_root(vrt_width, vrt_height, vrt_transform, vrt_crs)

    # Create destination VRT bands
    xml_bands = {}
    for vrt_bidx in sorted(vrt_bands):
        # Only create if there are bands to put in
        if not vrt_bands[vrt_bidx]:
            continue

        # For now, first band sets some metadata, like description and NDV
        # TODO: Allow overrides to pass to `_make_band`!
        ex = vrt_bands[vrt_bidx][0]
        with ex.open():  # ensure open the whole time
            xml_band = _make_band(root, ex, vrt_bidx)
        xml_bands[vrt_bidx] = xml_band

    # Add sources for each output VRT band
    for vrt_bidx, xml_band in xml_bands.items():
        for src_band in vrt_bands[vrt_bidx]:
            with src_band.open():  # ensure open the entire time
                _make_source(xml_band, src_band,
                             vrt_bounds, vrt_transform,
                             relative_to_vrt=relative_to_vrt)

    return root


def _make_vrt_root(width, height, transform, crs):
    root = ET.Element('VRTDataset')
    root.set('rasterXSize', str(width))
    root.set('rasterYSize', str(height))
    _make_geotransform(root, transform)
    _make_crs(root, crs)

    return root


def _make_crs(root, crs):
    ele = _make_subelement(root, 'SRS', crs.wkt)
    return ele


def _make_geotransform(root, transform, precision=9):
    # Output VRT tranform
    gt_str = list(str(round(n, precision)) for n in transform.to_gdal())
    ele = _make_subelement(root, 'GeoTransform', ', '.join(gt_str))
    return ele


def _make_band(root, source_band, vrt_bidx,
               description=None, vrt_ndv=None):
    # Create <VRTRasterBand>
    band = ET.SubElement(root, 'VRTRasterBand')
    band.set('dataType', _gdal_typename(source_band.dtype))
    band.set('band', str(vrt_bidx))

    # Optional subelements
    # TODO: ColorTable, GDALRasterAttributeTable, UnitType,
    #       Offset, Scale, CategoryNames
    colorinterp = gdal.GetColorInterpretationName(
        source_band.colorinterp.value)
    _make_subelement(band, 'ColorInterp', colorinterp)

    # Output band NDV defaults to source NDV
    vrt_ndv = vrt_ndv if vrt_ndv is not None else source_band.nodata
    if vrt_ndv is not None:
        _make_subelement(band, 'NoDataValue', str(vrt_ndv))

    description = description or source_band.description
    if description:
        band.set('Description', description)

    return band


def _make_source(xml_band, source_band, vrt_bounds, vrt_transform,
                 relative_to_vrt=None):
    source = ET.SubElement(xml_band, 'ComplexSource')

    _make_source_path(source, source_band.path,
                      relative_to_vrt=relative_to_vrt)
    _make_source_band(source, source_band.src_bidx)
    _make_source_props(source, source_band)

    # SrcRect and DstRect
    _make_src_rect(source, source_band.bounds, source_band.transform,
                   vrt_bounds)
    _make_dst_rect(source, source_band.bounds, vrt_transform)

    # NODATA
    if source_band.nodata is not None:
        _make_subelement(source, 'NODATA', str(source_band.nodata))

    return source


def _make_source_path(xml_parent, path,
                      relative_to_vrt=None):
    ele = ET.SubElement(xml_parent, 'SourceFilename')
    if relative_to_vrt:
        path = relative_to(path, relative_to_vrt)
        ele.set('relativeToVRT', '1')
    else:
        path = Path(path).absolute()
    ele.text = str(path)
    return ele


def _make_source_band(xml_parent, src_bidx):
    # Creates <SourceBand> ... a number ... <SourceBand/> tag
    return _make_subelement(xml_parent, 'SourceBand', src_bidx)


def _make_source_props(xml_parent, source_band):
    """Creates <SourceProperties ... /> tag

    Parameters
    ----------
    xml_parent : xml.etree.Element.SubElement
        Parent XML element, like a "ComplexSource" or "SimpleSource"
    source_band : VRTSourceBand
        The source band

    Returns
    -------
    xml.etree.Element.SubElement
        "SourceProperties" subelement
    """
    ele = ET.SubElement(xml_parent, 'SourceProperties')

    ele.set('RasterXSize', str(source_band.width))
    ele.set('RasterYSize', str(source_band.height))
    ele.set('DataType', _gdal_typename(source_band.dtype))
    ele.set('BlockXSize', str(source_band.blockxsize))
    ele.set('BlockYSize', str(source_band.blockysize))

    return ele


def _make_src_rect(xml_parent, src_bounds, src_transform, dst_bounds):
    win, _ = calculate_src_window(src_bounds, src_transform, dst_bounds)

    ele = ET.SubElement(xml_parent, 'SrcRect')
    ele.set('xOff', str(win.col_off))
    ele.set('yOff', str(win.row_off))
    ele.set('xSize', str(win.width))
    ele.set('ySize', str(win.height))

    return ele


def _make_dst_rect(xml_parent, src_bounds, dst_transform):
    win = calculate_dst_window(src_bounds, dst_transform)

    ele = ET.SubElement(xml_parent, 'DstRect')
    ele.set('xOff', str(win.col_off))
    ele.set('yOff', str(win.row_off))
    ele.set('xSize', str(win.width))
    ele.set('ySize', str(win.height))

    return ele


def _make_subelement(root, name, text):
    sub = ET.SubElement(root, name)
    sub.text = str(text)
    return sub
