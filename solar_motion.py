import numpy as np
from dates import *

"""
---------------------------------
Apparent solar motion in the sky
---------------------------------

To calculate the Sun's posotion in the sky at a given time and location, we need:

(1) Shape and orientation of Earth's orbit
    - Eccentricity, semi-major axis, argument of periapsis
(2) Phase of Earth's orbital motion
    - Mean anomaly at given epoch, sidereal orbital period
(3) Orientation of Earth's rotational axis
    - Time of equinox, Earth's axial tilt
(4) Phase of Earth's rotation
    - Location of subsolar point at time of equinox (lat = 0)
    - Rotation rate
"""
# (0) Units of time
DAY             = 86400             # [s]
SYNODIC_DAY     = 1                 # synodic day [d]

# (1) Earth orbital elements; Epoch = 2451545.0 (2000-Jan-1.5, J2000)
ECCENTRICITY    = 0.0167086
SEMI_MAJOR_AXIS = 1.49598023e11     # [m]
ARG_PERIAPSIS   = 114.20783         # [deg]

# (2) Earth's orbital motion
EPOCH           = 2451545.0
MEAN_ANOMALY    = 358.617           # [deg]
ORBITAL_PERIOD  = 365.256363004     # sidereal year [d]

# (3) Earth's rotational axis; https://hpiers.obspm.fr/eop-pc/models/constants.html
ROT_TILT        = 23.4392811        # relative to orbital plane [deg]
EQUINOX_DATE    = calendar2jd("2020-03-20 03:49:00.0") # UTC; March 2020 equinox

# (4) Earth's rotation
LAT_SUBSOLAR    = 0                 # [deg]
EQ_SOLAR_NOON   = calendar2jd("2020-03-20 12:07:00.0") # UTC; Solar noon following equinox at lon = 0
#EQ_MEAN_NOON    = round(EQ_SOLAR_NOON)
LON_SUBSOLAR    = (EQ_SOLAR_NOON - EQUINOX_DATE) / SYNODIC_DAY * 360 # longitude of subsolar point at EQUINOX_DATE

def lon_subsolar(calendar_date):
    """
    Longitude of subsolar point at a given time

    Parameters
    ----------
    calendar_date: str
        Calendar date in the form "YYYY-MM-DD HH:MM:SS.S"

    Returns
    -------
    lon: float
        Longitude of the subsolar point at the given time [deg]
    """
    Y, M, D, H, MIN, S = calendar_split(calendar_date)

    utc = H + MIN / 60 + S / 3600 # UTC in decimal hours since 00:00:00 UTC on the given date [h]
    d = ymd2d(Y, M, D)
    EOT = equation_of_time(Y, d)
    
    return -15 * (utc - 12 + EOT / 60)

#LON_SUBSOLAR    = lon_subsolar(jd2calendar(EQUINOX_DATE)) # calculated using equation of time

SIDEREAL_DAY    = 86164.0905 / DAY  # sidereal day [d]
ROT_RATE        = 2 * np.pi / (SIDEREAL_DAY * DAY) # Earth rotation rate (sidereal) [rad/s]

# Other parameters
EARTH_RADIUS    = 6.371e6 # [m]; assume sphere

def ecc_anomaly(e, M, tolerance=0.002):
    """
    Solve for the eccentric anomaly E given eccentricity e and mean anomaly M

    Parameters
    ----------
    e: float
        Orbital eccentricity [rad]
    M: float
        Mean anomaly [deg]
    tolerance: float
        Maximum fractional error allowed in E; default = 0.002
    
    Returns
    -------
    E: float
        Eccentric anomaly [deg]
    """
    e_deg = e * 180 / np.pi # e [deg]
    E  = M + e_deg * np.sin(M * np.pi / 180) # initial guess for E [deg]
    dE = 1

    while np.abs(dE/E) > tolerance:
        dM = M - (E - e_deg * np.sin(E * np.pi / 180))
        dE = dM / (1 - e * np.cos(E * np.pi / 180))
        E += dE

    return E # [deg]

def sun_vector(jd):
    """
    Position of the Sun in geocentric coordinates at a given time

    Parameters
    ----------
    jd: float
        Julian date

    Returns
    -------
    sun_vector: 1d array
        Vector pointing at the Sun from Earth's center [m] 
    """
    time_since_epoch = jd - EPOCH # [d]
    M = (MEAN_ANOMALY + 360 / ORBITAL_PERIOD * time_since_epoch) % 360 # mean anomaly at jd [deg]

    E = ecc_anomaly(ECCENTRICITY, M) * np.pi / 180 # eccentric anomaly [rad], 0 < E < 2*pi
    nu = 2 * np.arctan2(np.sqrt(1 + ECCENTRICITY) * np.sin(E / 2), \
                        np.sqrt(1 - ECCENTRICITY) * np.cos(E / 2)) # true anomaly [rad], 0 < nu < 2*pi

    # heliocentric coordinates [m] in orbital plane, z = 0
    r = SEMI_MAJOR_AXIS * (1 - ECCENTRICITY * np.cos(E))
    x = r * np.cos(nu) # = a * (np.cos(E) - e)
    y = r * np.sin(nu) # = a * (1 - e**2)**0.5 * np.sin(E)

    return np.array([-x, -y, 0]) # [m]

def rot_axis(march_equinox=EQUINOX_DATE):
    """
    Find Earth's rotational axis (ignoring precession) given time of equinox

    Parameters
    ----------
    march_equinox: float
        A reference March equinox in Julian date

    Returns
    -------
    axis: 1d array
        Unit vector of Earth's rotational axis
    """
    xhat = np.array([1, 0, 0]) # reference direction on orbital plane
    zhat = np.array([0, 0, 1]) # normal vector to orbital plane

    sun = unit_vec(sun_vector(march_equinox)) # in the orbital plane, orthogonal to Earth's axis at equinox
    theta = rotation_angle(sun, xhat, axis=zhat) # angle between Earth-Sun vector and reference direction
    
    # rotate zhat CW around Earth-Sun vector (radially inward) by ROT_TILT to get axis
    D = Rz(theta) # CCW around z
    R = Rx(ROT_TILT / 180 * np.pi) # CCW around x

    return np.squeeze(np.array(D.T * R.T * D * np.matrix(zhat).T))

def zenith(jd, lat, lon):
    """
    Find unit vector pointing directly overhead at given location and time

    Parameters
    ----------
    jd: float
        Julian date
    lat, lon: float
        Latitude and longitude of location on Earth [deg]

    Returns
    -------
    zenith: 1d array
        Unit vector pointing at zenith
    """
    elapsed_time = jd - EQUINOX_DATE # time elapsed since reference date [d]
    xhat = np.array([1, 0, 0]) # reference direction in orbital plane
    zhat = np.array([0, 0, 1]) # normal vector to the orbital plane
    zenith_0 = unit_vec(sun_vector(EQUINOX_DATE)) # subsolar zenith at equinox aligns with Earth-Sun vector

    lat_diff = (lat - LAT_SUBSOLAR) / 180 * np.pi # *** note that this is a CW angle since colat = 90 - lat
    lon_diff = (lon - LON_SUBSOLAR) / 180 * np.pi # CCW angle
    lon_change = elapsed_time * DAY * ROT_RATE # CCW change in longitude since reference date [rad]

    # rotate zenith_0 CW around y = cross(rot_axis, zenith_0) by lat_diff
    # then rotate CCW around rot_axis by lon_diff + lon_change
    theta = rotation_angle(zenith_0, xhat, axis=zhat) # angle between initial zenith and reference direction
    D = Rz(theta) # CCW around z
    R_axis = Rx(ROT_TILT / 180 * np.pi) # CCW around x
    R_lat = Ry(-lat_diff) # CCW by |lat_diff| around y
    R_lon = Rz(lon_diff + lon_change) # CCW around z

    return np.squeeze(np.array(D.T * R_axis.T * R_lon * R_lat.T * R_axis * D * np.matrix(zenith_0).T))

def sun_location(jd, lat, lon):
    """
    Find altitude and azimuthal angles of the Sun at given time and location

    Parameters
    ----------
    jd: float
        Julian date
    lat, lon: float
        Latitude and longitude of location on Earth [deg]

    Returns
    -------
    altitude, azimuth: float
        Altitude and azimuthal angles of the Sun [deg]
    """
    axis = rot_axis() # Earth's axis
    zhat = zenith(jd, lat, lon) # zenith
    sun = unit_vec(sun_vector(jd) - EARTH_RADIUS * zhat) # unit vector pointing at the Sun

    altitude = 90 - (np.arccos(np.dot(sun, zhat)) / np.pi * 180) # -90 < altitude < 90

    east = unit_vec(np.cross(axis, zhat)) # East direction
    north = unit_vec(np.cross(zhat, east)) # North direction
    sun_proj = unit_vec(sun - zhat * np.dot(sun, zhat)) # azimuthal vector of the Sun

    azimuth = rotation_angle(sun_proj, north, axis=zhat) * 180 / np.pi # 0 < azimuth < 360

    return altitude, azimuth

def rotation_angle(A, B, axis=None):
    """
    Find the CCW rotation angle between vectors A and B 

    Parameters
    ----------
    A, B: 1d array
        3d vectors
    axis: 1d array
        Axis of rotation; must be either unit_vec(cross(A, B)) or unit_vec(cross(B, A))

    Returns
    -------
    theta: float
        CCW rotation angle between A and B [rad]; 0 < theta < 2*pi
    """
    normal = unit_vec(np.cross(A, B)) # default axis
    if axis is None:
        axis = normal
    else:
        axis = unit_vec(axis)
        assert np.isclose(np.abs(np.dot(axis, normal)), 1)
    
    dot = np.dot(A, B) # |A||B|cos(theta)
    det = np.linalg.det(np.array([axis, A, B])) # signed |A||B|sin(theta)

    return (np.arctan2(det, dot) + 2 * np.pi) % (2 * np.pi) # rotate A CCW around axis by theta to get B

# angle needed to rotate a to b CCW
def rotation_angle_(a, b, axis=2):
    ref = np.zeros(3)
    ref[axis] = 1
    normal = unit_vec(np.cross(a, b))
    dot = np.dot(normal, ref)
    if dot >= 0:
        theta = np.arccos(np.dot(a, b)) # [0, pi]
        dot = 10
    else:
        theta = 2 * np.pi - np.arccos(np.dot(a, b))
        dot = -10

    return theta

# CCW rotation around x-axis
def Rx(theta):
    R = np.matrix([[1, 0            ,  0            ],
                   [0, np.cos(theta), -np.sin(theta)],
                   [0, np.sin(theta),  np.cos(theta)]])

    return R

# CCW rotation around y-axis
def Ry(theta):
    R = np.matrix([[np.cos(theta), 0, -np.sin(theta)],
                   [0            , 1,  0            ],
                   [np.sin(theta), 0,  np.cos(theta)]])

    return R

# CCW rotation around z-axis
def Rz(theta):
    R = np.matrix([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta),  np.cos(theta), 0],
                   [0            ,  0            , 1]])

    return R

def unit_vec(vec):
    return vec / np.linalg.norm(vec)
