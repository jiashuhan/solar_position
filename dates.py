import numpy as np

def equation_of_time(y, d):
    """
    Approximate form of the equation of time 

    Parameters
    ----------
    y: int
        Year in AD format
    d: int
        Number of days since January 1 of the year

    Returns
    -------
    eot: float
        Difference between the apparent solar time and the mean solar time on the given date [min]
    """
    D = 6.24004077 + 0.01720197 * (365.25 * (y - 2000) + d)
    return -7.659 * np.sin(D) + 9.863 * np.sin(2 * D + 3.5932) # [min]

def is_leap(y):
    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)

def ymd2d(y, m, d): # find number of days since January 1 of a given year
    months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    if is_leap(y):
        months[1] += 1

    assert m >= 1 and m <= 12 and d >= 1 and d <= months[m - 1]

    return sum(months[:m - 1]) + d - 1 # a number between 0 and 364 (or 365 in a leap year)

def yd2md(y, d): # inverse of ymd2d
    months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    if is_leap(y):
        months[1] += 1

    assert d >= 0 and (d < 365 or (d == 365 and months[1] == 29))

    m = 1
    while d > months[m - 1]:
        d -= months[m - 1]
        m += 1

    if d == months[m - 1]:
        d = 1
        m += 1
    else:
        d += 1

    return m, d

def num_leap(y1, y2): # Number of leap years between y1 and y2, including y1 but not y2
    if y1 == y2:
        return 0

    assert y2 > y1

    mod4_y1 = y1 % 4
    mod100_y1 = y1 % 100
    mod400_y1 = y1 % 400

    mod4_y2 = y2 % 4
    mod100_y2 = y2 % 100
    mod400_y2 = y2 % 400

    N_julian_leap = (y2 - (y1 - mod4_y1)) // 4 - int(mod4_y2 == 0) + int(mod4_y1 == 0) # number of Julian leap years in [y1, y2)
    N_100 = (y2 - (y1 - mod100_y1)) // 100 - int(mod100_y2 == 0) + int(mod100_y1 == 0)
    N_400 = (y2 - (y1 - mod400_y1)) // 400 - int(mod400_y2 == 0) + int(mod400_y1 == 0)
    N_extra_leap = N_100 - N_400 # number of Julian leap years that are not Gregorian leap years in [y1, y2)

    return N_julian_leap - N_extra_leap

def calendar2jd(calendar_date):
    """
    Convert calendar date (after October 15, 1582 AD) to Julian date

    Parameters
    ----------
    calendar_date: str
        Calendar date in the form "YYYY-MM-DD"

    Returns
    -------
    jd: float
        Julian date corresponding to 12:00:00 UTC of the input calendar date

    Notes
    -----
    - First day of Julian date: January 1, 4713 BC 12:00:00 UT1 -> JD 0.0
    - Last day of Julian calendar: October 4, 1582 AD 12:00:00 UT1 -> JD 2299160.0
    - First day of Gregorian calendar: October 15, 1582 AD 12:00:00 UT1 -> JD 2299161.0
    """
    Y, M, D = np.array(calendar_date.split("-")).astype(int)
    assert Y > 1582 or (Y == 1582 and (M > 10 or (M == 10 and D >= 15))), "Must be later than October 15, 1582 AD."

    N_days = ymd2d(Y, M, D) # days since January 1, year Y
    days_in_1582 = ymd2d(1582, 10, 15) # days already passed in Gregorian 1582 (starts 10 days earlier than Julian 1582)
    N_year = Y - 1582 # number of full years since January 1, 1582 (Gregorian)
    N_leap = num_leap(1582, Y) # number of leap days in between

    jd = 2299161.0 + N_year * 365 + N_leap + N_days - days_in_1582

    return jd

def jd2calendar(jd):
    """
    Convert Julian date to calendar date (after October 15, 1582 AD)

    Parameters
    ----------
    jd: float
        Julian date

    Returns
    -------
    calendar_date: str
        Calendar date in the form "YYYY-MM-DD HH:MM:SS.S"
    """
    gregorian0 = 2299160.5 # October 15, 1582 AD 00:00:00 UT1
    assert jd >= gregorian0, "Must be later than JD 2299160.5 / October 15, 1582 AD 00:00:00.000."

    time_since = jd - gregorian0
    hms = time_since % 1 # [d]
    days_since = round(time_since - hms)

    days_in_1582 = ymd2d(1582, 10, 15) # days already passed in Gregorian 1582 (starts 10 days earlier than Julian 1582)

    # try to find number of leap days until target year by guessing target year
    N_year = (days_since + days_in_1582) // (365 + 1/4 - 3/400) # number of full average Gregorian years since January 1, 1582
    year_guess = 1582 + N_year # may underestimate by slightly more than a year if close to beginning of calendar year
    year_guess_is_leap = is_leap(year_guess)
    
    N_leap = num_leap(1582, year_guess) # number of leap days until guessed year
    jd_guess = gregorian0 - days_in_1582 + N_year * 365 + N_leap # Julian date of January 1 00:00:00 UT1 of guessed year
    diff = jd - hms - jd_guess # number of days underestimated, plus days in target year
    
    if diff >= 365 + int(year_guess_is_leap): # underestimated year
        year = year_guess + 1
        diff -= 365 + int(year_guess_is_leap)
    else:
        year = year_guess

    month, day = yd2md(year, diff)

    UT = hms * 86400 # [s]
    hour = UT // 3600
    minute = (UT % 3600) // 60
    second = UT % 60

    return "%04d-%02d-%02d %02d:%02d:%04.1f"%(year, month, day, hour, minute, second)

def calendar_split(calendar_date):
    YMD, HMS = calendar_date.split(" ")
    Y, M, D = YMD.split("-")
    H, MIN, S = HMS.split(":")

    return int(Y), int(M), int(D), int(H), int(MIN), float(S)

if __name__ == '__main__':
    assert ymd2d(2020, 1, 1) == 0
    assert ymd2d(2020, 1, 31) == 30
    assert ymd2d(2020, 2, 1) == 31
    assert ymd2d(2020, 3, 1) == 60
    assert ymd2d(2020, 12, 31) == 365
    assert ymd2d(2021, 3, 1) == 59
    assert ymd2d(2021, 12, 31) == 364
    assert ymd2d(2000, 3, 1) == 60
    assert ymd2d(2100, 3, 1) == 59

    assert yd2md(2020, 0) == (1, 1)
    assert yd2md(2020, 30) == (1, 31)
    assert yd2md(2020, 31) == (2, 1)
    assert yd2md(2020, 60) == (3, 1)
    assert yd2md(2020, 365) == (12, 31)
    assert yd2md(2021, 59) == (3, 1)
    assert yd2md(2021, 364) == (12, 31)
    assert yd2md(2000, 60) == (3, 1)
    assert yd2md(2100, 59) == (3, 1)

    assert num_leap(2020, 2021) == 1
    assert num_leap(2020, 2040) == 5
    assert num_leap(2020, 2042) == 6
    assert num_leap(2021, 2022) == 0
    assert num_leap(2021, 2040) == 4
    assert num_leap(2021, 2042) == 5
    assert num_leap(1997, 2005) == 2
    assert num_leap(1896, 2005) == 27

    assert calendar2jd("1582-10-15") == 2299161.0
    assert calendar2jd("1582-10-16") == 2299162.0
    assert calendar2jd("1582-12-31") == 2299238.0
    assert calendar2jd("2020-01-01") == 2458850.0
    assert calendar2jd("2024-02-29") == 2460370.0

    assert jd2calendar(2299161.0) == "1582-10-15 12:00:00.0"
    assert jd2calendar(2299162.0) == "1582-10-16 12:00:00.0"
    assert jd2calendar(2299238.0) == "1582-12-31 12:00:00.0"
    assert jd2calendar(2458850.0) == "2020-01-01 12:00:00.0"
    assert jd2calendar(2460370.0) == "2024-02-29 12:00:00.0"
    assert jd2calendar(2460370.5) == "2024-03-01 00:00:00.0"
    assert jd2calendar(2460371.037896) == "2024-03-01 12:54:34.2"
    assert jd2calendar(2460663.330794) == "2024-12-18 19:56:20.6"
    assert jd2calendar(2632242.123173) == "2494-09-24 14:57:22.1"
