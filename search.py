from solar_motion import *
from dates import *
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

def find_twilight(calendar_date, lat, lon, rise=False, N=100):
    """
    Find sunrise/sunset time and location on a given day

    Parameters
    ----------
    calendar_date: str
        Local calendar date in the form "YYYY-MM-DD"
    lat, lon: float
        Latitude and longitude of observing location [deg]
    rise: boolean
        If True, find sunrise; otherwise find sunset
    N: int
        Number of points sampled within the 12 hour window

    Returns
    -------
    sunset: tuple
        Julian date and the Sun's altitude and azimuth at sunset

    Notes
    -----
    - Sunset is defined as the moment in the evening when the upper 
      edge of the sun is in line with the horizon at 0 deg.
    - Sunrise is defined as the moment in the morning when the upper 
      edge of the sun is in line with the horizon at 0 deg.
    """
    tz = round(lon / (360 / 24))
    H = 12 - tz # UTC hour at local noon
    jd_local_noon = calendar2jd("%s %02d:00:00.0"%(calendar_date, H))
    
    if rise: # find previous midnight
        jd_local_midnight = jd_local_noon - 0.5
    else: # find next midnight
        jd_local_midnight = jd_local_noon + 0.5

    time = np.linspace(*sorted([jd_local_midnight, jd_local_noon]), N)

    alt = np.zeros_like(time) # altitude of the upper edge of the sun
    azi = np.zeros_like(time)

    for i, t in enumerate(time):
        alt_, azi[i], angular = sun_location(t, lat, lon)
        alt[i] = alt_ + angular

    alt_interp = interp1d(time, alt, kind='cubic')
    azi_interp = interp1d(time, azi, kind='cubic')

    sol = root_scalar(alt_interp, bracket=[time[0], time[-1]])

    if not sol.converged:
        raise ValueError("No sunset on %s."%calendar_date)

    return sol.root, alt_interp(sol.root), azi_interp(sol.root)

def find_obs_window(heading, begin_date, end_date, lat, lon, rise=False, tol=0.5, N=100):
    """
    Find time window during which the sun rises/sets in the desired heading

    Parameters
    ----------
    heading: float
        Desired heading of the sun [deg]
    begin_date, end_date: str
        Beginning and end of the search range in the form "YYYY-MM-DD"
    lat, lon: float
        Latitude and longitude of observing location [deg]
    rise: boolean
        If True, find sunrise; otherwise find sunset
    tol: float
        Tolerated difference in the heading [deg]; default = 0.5
    N: int
        Number of points sampled within the 12 hour window

    Returns
    -------
    window: list
        List of sunrises/sunsets during the observation window; includes 
        local sunrise/sunset time (standard time), altitude, and azimuth
    """
    Y0, M0, D0_ = np.array(begin_date.split("-")).astype(int)
    D0 = ymd2d(Y0, M0, D0_)
    Y1, M1, D1_ = np.array(end_date.split("-")).astype(int)
    D1 = ymd2d(Y1, M1, D1_)

    window = []
    tz, tz_label = time_zone(lon)

    while Y0 < Y1 or (Y0 == Y1 and D0 <= D1):
        M, D = yd2md(Y0, D0)
        calendar_date = "%04d-%02d-%02d"%(Y0, M, D)
        jd, alt, azi = find_twilight(calendar_date, lat, lon, rise=rise, N=N)
        if np.abs(heading - azi) < tol:
            window.append(("%s %s"%(jd2calendar(jd + tz / 24), tz_label), alt, azi))

        D0 += 1
        if D0 > 364 + int(is_leap(Y0)):
            D0 %= 364 + int(is_leap(Y0))
            Y0 += 1

    return window

# San Diego, CA
LAT = 32.85950
LON = -117.21240
# Scripps Pier heading: 289 deg (WNW)
# Downtown and Salk heading: 270 deg (W)

BEGIN = "2025-01-01" # local date
END = "2025-12-31"

if __name__ == '__main__':
    observing_dates = find_obs_window(289, BEGIN, END, LAT, LON, N=30)
    for d in observing_dates:
        print("%s; Alt. = %.4g, Azi. = %.4g"%d)
