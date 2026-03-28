import sunpy.map
from sunpy.coordinates.frames import HeliographicStonyhurst

from astropy import units as u
from astropy.coordinates import SkyCoord


def reproject_instrument(
    sunpy_map: sunpy.map.Map,
    radius: u.Quantity = 1 * u.au,
    scale: u.Quantity = u.Quantity([0.5, 0.5], u.arcsec / u.pixel),
):
    """
    Reproject a `sunpy.map.Map` to a new observer location with a specified radius and plate scale.

    Parameters:
    -----------
    sunpy_map : `sunpy.map.Map`
        The input map to be reprojected.

    radius : `astropy.units.quantity.Quantity`, optional
        The radius at which the new observer is located. Default is 1 astronomical unit (1*u.au).

    scale : `astropy.units.quantity.Quantity`, optional
        The plate scale of the output map. Default is [0.5, 0.5] arcsec per pixel.

    Returns:
    --------
    out_map : `sunpy.map.Map`
        The reprojected map with the specified observer location, radius, and plate scale.

    """

    original_observer = sunpy_map.reference_coordinate.frame.observer

    new_observer = SkyCoord(
        0 * u.arcsec,
        0 * u.arcsec,
        frame="helioprojective",
        rsun=sunpy_map.reference_coordinate.rsun,
        obstime=original_observer.obstime,
        observer=HeliographicStonyhurst(
            original_observer.lon,
            original_observer.lat,
            radius,
            obstime=original_observer.obstime,
            rsun=original_observer.rsun,
        ),
    )

    out_header = sunpy.map.make_fitswcs_header(
        sunpy_map.data.shape,
        new_observer,
        scale=scale,
    )

    out_map = sunpy_map.reproject_to(out_header)

    return out_map
