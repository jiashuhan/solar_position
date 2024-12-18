from solar_motion import *
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
jun_solstice_2020 = 2459021.404861
sep_equinox_2020 = 2459115.063194
dec_solstice_2020 = 2459204.918056

s1 = unit_vec(sun_vector(march_equinox_2020))
s2 = unit_vec(sun_vector(jun_solstice_2020))
s3 = unit_vec(sun_vector(sep_equinox_2020))
s4 = unit_vec(sun_vector(dec_solstice_2020))
assert np.isclose(np.dot(s1, s3), -1) # equinoxes opposite to each other
assert np.isclose(np.dot(s2, s4), -1) # solstices opposite to each other

sep_equinox_2024 = 2460576.0715277777
dec_solstice_2024 = 2460665.8881944446

s5 = unit_vec(sun_vector(sep_equinox_2024))
s6 = unit_vec(sun_vector(dec_solstice_2024))
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
LAT = 32.866
LON = -117.254

jd = 2460661.166667
print(sun_location(jd+0.16, LAT, LON)) # EXPECT +33.7
print(sun_location(jd+0.16+0.5, LAT, LON)) # EXPECT -80.5 

print(sun_location(jd+0.16, 0, LON)) # EXPECT +66.5
print(sun_location(jd+0.16+0.5, 0, LON)) # EXPECT -66.5

print(sun_location(jd+0.16, -LAT, LON)) # EXPECT +80.5
print(sun_location(jd+0.16+0.5, -LAT, LON)) # EXPECT -33.7

# Azimuth should look like sawtooth at the poles near solstice
# Azimuth should look smooth at equator near solstice but is not sinusoidal
import matplotlib.pyplot as plt

#LAT = ROT_TILT * 0 # near solstice, azimuth doesn't reach 360 degrees in the tropical area
N = 2
dates = np.linspace(jd, jd + N, N * 1000)
N_alt = np.zeros_like(dates)
N_azi = np.zeros_like(dates)

S_alt = np.zeros_like(dates)
S_azi = np.zeros_like(dates)

for i, d in enumerate(dates):
    a, b = sun_location(d, LAT, LON)
    N_alt[i] = a
    N_azi[i] = b

    a, b = sun_location(d, -LAT, LON)
    S_alt[i] = a
    S_azi[i] = b

plt.plot(dates, N_alt, color='C0', linestyle='-')
plt.plot(dates, N_azi, color='C1', linestyle='-')

plt.plot(dates, S_alt, color='C2', linestyle='--')
plt.plot(dates, S_azi, color='C3', linestyle='--')
utc_noon = np.floor(dates[0]) + 1 
utc_minus_8_noon = utc_noon + 8 / 24
plt.axvline(utc_minus_8_noon, color='k', linestyle='--')
plt.axhline(0, color='k', linestyle=':')
plt.show()
