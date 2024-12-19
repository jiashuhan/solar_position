from solar_motion import *
from dates import *
from search import *
import numpy as np

xhat = np.array([1, 0, 0]) # reference direction on the orbital plane
yhat = np.array([0, 1, 0])
zhat = np.array([0, 0, 1]) # normal vector to the orbital plane

# Rotations
assert np.isclose(rotation_angle(xhat, yhat, axis=zhat) * 180 / np.pi, 90)
assert np.isclose(rotation_angle(yhat, xhat, axis=zhat) * 180 / np.pi, 270)
assert np.isclose(rotation_angle(yhat, xhat, axis=-zhat) * 180 / np.pi, 90)

# Locations of equinoxes and solstices
march_equinox_2020 = EQUINOX_DATE
jun_solstice_2020 = calendar2jd("2020-06-20 21:43:00.0")
sep_equinox_2020 = calendar2jd("2020-09-22 13:31:00.0")
dec_solstice_2020 = calendar2jd("2020-12-21 10:02:00.0")

s1 = unit_vec(sun_vector(march_equinox_2020)[0])
s2 = unit_vec(sun_vector(jun_solstice_2020)[0])
s3 = unit_vec(sun_vector(sep_equinox_2020)[0])
s4 = unit_vec(sun_vector(dec_solstice_2020)[0])
assert np.isclose(np.dot(s1, s3), -1) # equinoxes opposite to each other
assert np.isclose(np.dot(s2, s4), -1) # solstices opposite to each other

sep_equinox_2024 = calendar2jd("2024-09-22 12:43:00.0")
dec_solstice_2024 = calendar2jd("2024-12-21 09:21:00.0")

s5 = unit_vec(sun_vector(sep_equinox_2024)[0])
s6 = unit_vec(sun_vector(dec_solstice_2024)[0])
assert np.isclose(np.dot(s3, s5), 1) # equinox direction unchanged
assert np.isclose(np.dot(s4, s6), 1) # solstice direction unchanged

# Orientation of Earth's rotational axis
axis = unit_vec(rot_axis())
axis_proj = unit_vec(axis - zhat * np.dot(zhat, axis)) # Earth's rotational axis projected onto the orbital plane
assert np.isclose(np.dot(axis_proj, s2), 1) # Earth's axis tilts toward the Sun on June solstice
assert np.isclose(np.dot(axis_proj, s4), -1) # Earth's axis tilts away from the Sun on December solstice

# Location of the zenith 
z1 = zenith(EQUINOX_DATE, 0, LON_SUBSOLAR)
assert np.isclose(np.dot(s1, z1), 1) # Zenith of subsolar point matches the Sun's location

zs = zenith(EQUINOX_DATE, -90, LON_SUBSOLAR) # should be constant wrt date and lon
zn = zenith(EQUINOX_DATE, 90, LON_SUBSOLAR)
assert np.isclose(np.dot(zn, axis), 1) and np.isclose(np.dot(zs, axis), -1) # Poles align with the rotational axis

# San Diego, CA
LAT = 32.85950
LON = -117.21240

jd = calendar2jd("2024-12-17 19:45:19") # UT time of solar noon at San Diego (number obtained from Stellarium)

test_func = lambda x: np.minimum(np.abs(x), 360 - np.abs(x))
san_diego_noon = np.array(sun_location(jd, LAT, LON)[:2])
san_diego_noon_stellarium = np.array([33.774861, 180.00075])
san_diego_noon_plus_12 = np.array(sun_location(jd+0.5, LAT, LON)[:2])
san_diego_noon_plus_12_stellarium = np.array([-80.54219, 359.66142])
assert max(test_func(san_diego_noon - san_diego_noon_stellarium) / san_diego_noon_stellarium) < 1e-2
assert max(test_func(san_diego_noon_plus_12 - san_diego_noon_plus_12_stellarium) / san_diego_noon_plus_12_stellarium) < 1e-2

equator_noon = np.array(sun_location(jd, 0, LON)[:2])
equator_noon_stellarium = np.array([66.61778, 180.00156])
equator_noon_plus_12 = np.array(sun_location(jd+0.5, 0, LON)[:2])
equator_noon_plus_12_stellarium = np.array([-66.59947, 180.14008])
assert max(test_func(equator_noon - equator_noon_stellarium) / equator_noon_stellarium) < 1e-2
assert max(test_func(equator_noon_plus_12 - equator_noon_plus_12_stellarium) / equator_noon_plus_12_stellarium) < 1e-2

south_noon = np.array(sun_location(jd, -LAT, LON)[:2])
south_noon_stellarium = np.array([80.53133, 359.99625])
south_noon_plus_12 = np.array(sun_location(jd+0.5, -LAT, LON)[:2])
south_noon_plus_12_stellarium = np.array([-33.74108, 180.06692])
assert max(test_func(south_noon - south_noon_stellarium) / south_noon_stellarium) < 1e-2
assert max(test_func(south_noon_plus_12 - south_noon_plus_12_stellarium) / south_noon_plus_12_stellarium) < 1e-2

# Sunset predictions; # actual numbers from https://www.calculatorsoup.com/calculators/time/sunrise_sunset.php
sunset_time, sunset_alt, sunset_azi = find_twilight("2000-01-01", LAT, LON, rise=False, N=30)
tz, tz_label = time_zone(LON)
print("Predicted: %s %s; Alt. = %.4g, Azi. = %.4g"%(jd2calendar(sunset_time + tz / 24), tz_label, sunset_alt, sunset_azi))
print("Actual: 2000-01-01 16:53 UTC-8")

sunset_time, sunset_alt, sunset_azi = find_twilight("2020-03-20", LAT, LON, rise=False, N=30)
tz, tz_label = time_zone(LON)
print("Predicted: %s %s; Alt. = %.4g, Azi. = %.4g"%(jd2calendar(sunset_time + tz / 24), tz_label, sunset_alt, sunset_azi))
print("Actual: 2020-03-20 18:00 UTC-8")

sunset_time, sunset_alt, sunset_azi = find_twilight("2020-03-21", LAT, LON, rise=False, N=30)
tz, tz_label = time_zone(LON)
print("Predicted: %s %s; Alt. = %.4g, Azi. = %.4g"%(jd2calendar(sunset_time + tz / 24), tz_label, sunset_alt, sunset_azi))
print("Actual: 2020-03-21 18:01 UTC-8")

sunset_time, sunset_alt, sunset_azi = find_twilight("2024-08-09", LAT, LON, rise=False, N=30)
tz, tz_label = time_zone(LON)
print("Predicted: %s %s; Alt. = %.4g, Azi. = %.4g"%(jd2calendar(sunset_time + tz / 24), tz_label, sunset_alt, sunset_azi))
print("Actual: 2024-08-09 18:39 UTC-8")

sunset_time, sunset_alt, sunset_azi = find_twilight("3000-01-01", LAT, LON, rise=False, N=30)
tz, tz_label = time_zone(LON)
print("Predicted: %s %s; Alt. = %.4g, Azi. = %.4g"%(jd2calendar(sunset_time + tz / 24), tz_label, sunset_alt, sunset_azi))
print("Actual: 3000-01-01 16:51 UTC-8")

# Time variation of solar position
import matplotlib.pyplot as plt

LAT = ROT_TILT * 0.7 # near solstice, azimuth doesn't reach 360 degrees in the tropical area
N = 2
dates = np.linspace(jd, jd + N, N * 1000)
N_alt = np.zeros_like(dates)
N_azi = np.zeros_like(dates)

S_alt = np.zeros_like(dates)
S_azi = np.zeros_like(dates)

for i, d in enumerate(dates):
    a, b, _ = sun_location(d, LAT, LON)
    N_alt[i] = a
    N_azi[i] = b

    a, b, _ = sun_location(d, -LAT, LON)
    S_alt[i] = a
    S_azi[i] = b

plt.plot(dates, N_alt, color='C0', linestyle='-')
plt.plot(dates, N_azi, color='C1', linestyle='-')

plt.plot(dates, S_alt, color='C2', linestyle='--')
plt.plot(dates, S_azi, color='C3', linestyle='--')
utc_noon = np.floor(dates[0]) + 1 # mean noon
utc_minus_8_noon = utc_noon + 8 / 24
plt.axvline(utc_minus_8_noon, color='k', linestyle='--')
plt.axhline(0, color='k', linestyle=':')
plt.show()
