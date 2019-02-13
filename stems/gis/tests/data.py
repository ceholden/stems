""" Test data for :py:mod:`stems.gis`
"""
from textwrap import dedent


# ============================================================================
# Projections

# Geographic
EXAMPLE_WGS84 = {
    "cf_name": "latitude_longitude",
    "longname": "WGS 84",
    "epsg": 4326,
    "wkt": dedent("""
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
    """)
}
EXAMPLE_WEBMERCATOR = {
    "cf_name": "Mercator_1SP",
    "longname": "WGS 84 / Pseudo-Mercator",
	"epsg": 3857,
    "wkt": dedent("""
		PROJCS["WGS 84 / Pseudo-Mercator",
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
			PROJECTION["Mercator_1SP"],
			PARAMETER["central_meridian",0],
			PARAMETER["scale_factor",1],
			PARAMETER["false_easting",0],
			PARAMETER["false_northing",0],
			UNIT["metre",1,
				AUTHORITY["EPSG","9001"]],
			AXIS["X",EAST],
			AXIS["Y",NORTH],
			EXTENSION["PROJ4","+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +wktext +no_defs"],
			AUTHORITY["EPSG","3857"]]
    """)
}

# Projected
EXAMPLE_LAEA_NA = {
    "cf_name": "lambert_azimuthal_equal_area",
    "longname": "Lambert Azimuthal Equal Area - North America",
	"epsg": None,
    "wkt": dedent("""
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
    """)
}
EXAMPLE_AEA_NA = {
    "cf_name": "albers_conic_equal_area",
    "longname": "Alber's Equal Area Conic - North America",
    "epsg": None,
    "wkt": dedent("""
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
    """)
}
