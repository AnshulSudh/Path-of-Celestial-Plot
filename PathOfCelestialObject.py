"""
PathOfCelestialObject

Requirements:
 - skyfield
 - numpy
 - matplotlib
 - local hip_main.dat (Hipparcos catalog)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from math import degrees
from skyfield.api import load, Topos, Star, Angle, utc
from skyfield.data import hipparcos
from skyfield import almanac

# =========================================================
# DATA LOADING
# =========================================================

HIP_FILE = "hip_main.dat"
if not os.path.exists(HIP_FILE):
    raise FileNotFoundError("Place hip_main.dat in the script directory.")

eph = load("de421.bsp")
ts = load.timescale()

with load.open(HIP_FILE) as f:
    hip_data = hipparcos.load_dataframe(f)

NAME_TO_HIP = {
    "polaris": 11767,
    "sirius": 32349,
    "vega": 91262,
    "altair": 97649,
    "deneb": 102098,
    "betelgeuse": 27989,
}

# =========================================================
# HELPERS
# =========================================================

def star_from_hip(hip_id):
    row = hip_data.loc[hip_id]
    return Star(
        ra=Angle(degrees=float(row["ra_degrees"])),
        dec=Angle(degrees=float(row["dec_degrees"])),
        parallax_mas=float(row.get("parallax_mas", 0.0) or 0.0),
        ra_mas_per_year=float(row.get("ra_mas_per_year", 0.0) or 0.0),
        dec_mas_per_year=float(row.get("dec_mas_per_year", 0.0) or 0.0),
    )

def signed_az_diff(a2, a1):
    return (a2 - a1 + 180.0) % 360.0 - 180.0

def unwrap_az(az):
    az = np.asarray(az, float)
    out = az.copy()
    for i in range(1, len(az)):
        out[i] = out[i-1] + signed_az_diff(az[i], out[i-1])
    return out

def build_day(date_str, mode):
    d = datetime.fromisoformat(date_str).replace(tzinfo=utc)
    if mode == "noon":
        start = d.replace(hour=12, minute=0, second=0)
    elif mode == "midnight":
        start = d.replace(hour=0, minute=0, second=0)
    else:
        raise ValueError("day_mode must be 'noon' or 'midnight'")
    return start, start + timedelta(days=1)

# =========================================================
# NUMERICAL DIAGNOSTICS
# =========================================================

def circular_stats_deg(az_deg):
    """Circular mean, Rayleigh R, circular std (deg)."""
    az = np.deg2rad(az_deg % 360.0)
    C = np.mean(np.cos(az))
    S = np.mean(np.sin(az))
    R = np.hypot(C, S)
    mean = (np.degrees(np.arctan2(S, C)) + 360.0) % 360.0
    circ_std = np.degrees(np.sqrt(-2.0 * np.log(np.clip(R, 1e-12, 1.0))))
    return mean, R, circ_std


def angular_speeds_deg_per_hr(times_dt, az_deg, alt_deg):
    """Instantaneous angular speed on the sky."""
    az_un = unwrap_az(az_deg)
    alt = np.asarray(alt_deg)

    dAz = np.diff(az_un)
    dAlt = np.diff(alt)

    dt_hr = np.array(
        [(times_dt[i+1] - times_dt[i]).total_seconds() / 3600
         for i in range(len(times_dt)-1)]
    )

    mean_alt = (alt[:-1] + alt[1:]) / 2.0
    ds = np.sqrt((dAz * np.cos(np.deg2rad(mean_alt)))**2 + dAlt**2)

    speed = np.where(dt_hr > 0, ds / dt_hr, np.nan)

    mean_speed = np.nanmean(speed)
    imax = np.nanargmax(speed)
    imin = np.nanargmin(speed)

    return (
        speed,
        mean_speed,
        speed[imax], times_dt[imax+1],
        speed[imin], times_dt[imin+1]
    )


def compute_curvature(times_dt, az_deg, alt_deg):
    """
    Parametric curvature of az-alt track.
    """
    if len(az_deg) < 5:
        return np.nan, np.nan, None, None, None

    x = unwrap_az(az_deg)
    y = np.asarray(alt_deg)

    t0 = times_dt[0]
    T = np.array([(t - t0).total_seconds() for t in times_dt])

    dx = np.gradient(x, T)
    dy = np.gradient(y, T)
    d2x = np.gradient(dx, T)
    d2y = np.gradient(dy, T)

    denom = (dx*dx + dy*dy)**1.5
    with np.errstate(divide='ignore', invalid='ignore'):
        K = np.abs(dx*d2y - dy*d2x) / denom

    if np.all(np.isnan(K)):
        return np.nan, np.nan, None, None, None

    imax = np.nanargmax(K)
    return (
        float(np.nanmean(K)),
        float(K[imax]),
        times_dt[imax],
        float(x[imax] % 360),
        float(y[imax])
    )

def analemma_arc_length(az_un, alt):
    """Approximate arc length of analemma in degrees."""
    az_un = np.asarray(az_un, dtype=float)
    alt = np.asarray(alt, dtype=float)

    if len(az_un) < 2:
        return 0.0

    dAz = np.diff(az_un)
    dAlt = np.diff(alt)
    mean_alt = 0.5 * (alt[:-1] + alt[1:])

    ds = np.sqrt(
        (dAz * np.cos(np.deg2rad(mean_alt)))**2 +
        dAlt**2
    )

    return float(np.nansum(ds))


def analemma_curvature(days, az_un, alt):
    days = np.asarray(days, dtype=float)
    az_un = np.asarray(az_un, dtype=float)
    alt = np.asarray(alt, dtype=float)

    if len(days) < 5:
        return np.nan, np.nan, None

    dx = np.gradient(az_un, days)
    dy = np.gradient(alt, days)
    d2x = np.gradient(dx, days)
    d2y = np.gradient(dy, days)

    denom = (dx*dx + dy*dy)**1.5
    with np.errstate(divide='ignore', invalid='ignore'):
        K = np.abs(dx*d2y - dy*d2x) / denom

    if np.all(np.isnan(K)):
        return np.nan, np.nan, None

    imax = int(np.nanargmax(K))
    return float(np.nanmean(K)), float(K[imax]), imax

def analemma_loop_asymmetry(alt):
    alt = np.asarray(alt, dtype=float)

    mid = np.median(alt)
    upper = alt[alt >= mid]
    lower = alt[alt < mid]

    if len(upper) == 0 or len(lower) == 0:
        return np.nan

    return float(np.ptp(upper) - np.ptp(lower))

def analemma_orientation(az_un, alt):
    az_un = np.asarray(az_un, dtype=float)
    alt = np.asarray(alt, dtype=float)

    X = np.column_stack([
        az_un - np.mean(az_un),
        alt - np.mean(alt)
    ])

    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    direction = Vt[0]

    return float(np.degrees(np.arctan2(direction[1], direction[0])))

# =========================================================
# FOURIER ANALYSIS (ANALemma)
# =========================================================

def fourier_decompose(signal):
    """
    Compute Fourier frequencies, coefficients, and power spectrum.
    """
    signal = np.asarray(signal, dtype=float)
    signal = signal - np.nanmean(signal)

    F = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(signal))

    power = np.abs(F)**2
    return freq, F, power


def dominant_harmonics(freq, power, n=5):
    """
    Return n strongest positive-frequency harmonics.
    """
    freq = np.asarray(freq)
    power = np.asarray(power)

    mask = freq > 0
    freq = freq[mask]
    power = power[mask]

    idx = np.argsort(power)[-n:][::-1]
    return freq[idx], power[idx]


def analemma_fourier_diagnostics(az_un, alt):
    """
    Fourier diagnostics for analemma:
    - dominant harmonics in altitude and azimuth
    """
    az_un = np.asarray(az_un, dtype=float)
    alt = np.asarray(alt, dtype=float)

    f_alt, _, P_alt = fourier_decompose(alt)
    f_az,  _, P_az  = fourier_decompose(az_un)

    fA, pA = dominant_harmonics(f_alt, P_alt)
    fZ, pZ = dominant_harmonics(f_az,  P_az)

    return {
        "altitude": list(zip(fA, pA)),
        "azimuth": list(zip(fZ, pZ)),
    }



# =========================================================
# SKY PATH PLOT
# =========================================================

def plot_celestial_path(
    object_name,
    latitude,
    longitude,
    date_str,
    sample_minutes=5,
    day_mode="noon",
    pad_deg=10,
    figsize=(18,12),
):
    start_dt, end_dt = build_day(date_str, day_mode)

    times_dt = []
    t = start_dt
    while t <= end_dt:
        times_dt.append(t)
        t += timedelta(minutes=sample_minutes)

    times = ts.utc(times_dt)

    topos = Topos(latitude_degrees=latitude, longitude_degrees=longitude)
    observer = eph["earth"] + topos

    key = object_name.lower()
    planet_map = {
        "sun": "sun",
        "mercury": "mercury",
        "venus": "venus",
        "earth": "earth",
        "moon": "moon",
        "mars": "mars",
    
        # Outer planets (barycenters ONLY in DE421)
        "jupiter": "jupiter barycenter",
        "saturn": "saturn barycenter",
        "uranus": "uranus barycenter",
        "neptune": "neptune barycenter",
        "pluto": "pluto barycenter",
    
        # System level
        "solar system barycenter": "solar system barycenter",
        }

    if key in planet_map:
        target = eph[planet_map[key]]
    else:
        hip = NAME_TO_HIP.get(key)
        if hip is None:
            raise ValueError("Unknown object name.")
        target = star_from_hip(hip)

    azs, alts, t_vis = [], [], []

    for ti, tdt in zip(times, times_dt):
        alt, az, _ = observer.at(ti).observe(target).apparent().altaz()
        if alt.degrees > 0:
            azs.append(az.degrees)
            alts.append(alt.degrees)
            t_vis.append(tdt)

    if len(azs) < 2:
        print("Object never above horizon.")
        return

    azs = np.array(azs)
    alts = np.array(alts)
    az_un = unwrap_az(azs)
    
    # =========================================================
    # DIAGNOSTICS
    # =========================================================
    
    visible_samples = len(azs)
    
    dt_seconds = np.array(
        [(t_vis[i+1] - t_vis[i]).total_seconds()
         for i in range(len(t_vis)-1)]
    )
    visible_duration_hr = dt_seconds.sum() / 3600.0 if len(dt_seconds) else 0.0
    
    # Circular azimuth statistics
    mean_az, R, circ_std = circular_stats_deg(azs)
    
    # Altitude statistics
    alt_min = float(np.min(alts))
    alt_max = float(np.max(alts))
    alt_mean = float(np.mean(alts))
    
    # Angular speed
    (
        speeds,
        mean_speed,
        max_speed, max_speed_time,
        min_speed, min_speed_time
    ) = angular_speeds_deg_per_hr(t_vis, azs, alts)
    
    # Curvature
    K_mean, K_max, K_time, K_az, K_alt = compute_curvature(t_vis, azs, alts)
    
    # Arc length
    if len(azs) > 1:
        dAz = np.diff(az_un)
        dAlt = np.diff(alts)
        mean_alt = (alts[:-1] + alts[1:]) / 2
        arc_length = np.sum(
            np.sqrt((dAz * np.cos(np.deg2rad(mean_alt)))**2 + dAlt**2)
        )
    else:
        arc_length = 0.0

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(az_un, alts, c=np.linspace(0, 1, len(alts)),
                    cmap="gist_rainbow", s=22)
    ax.plot(az_un, alts, alpha=0.4)

    left = az_un.min() - pad_deg
    right = az_un.max() + pad_deg

    ax.set_xlim(left, right)
    ax.set_ylim(0, min(90, alts.max() + 5))
    ax.set_xlabel("Azimuth (deg, unwrapped)")
    ax.set_ylabel("Altitude (deg)")

    xticks = np.arange(np.floor(left / 5) * 5, np.ceil(right / 5) * 5 + 1, 5)
    ax.set_xticks(xticks)

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title(
        f"{object_name} — {date_str} ({day_mode}→{day_mode}, UTC)\n"
        f"lat {latitude:.2f}°, lon {longitude:.2f}°"
    )

    for az, alt, t in zip(az_un, alts, t_vis):
        if t.minute == 0:
            ax.text(az, alt + 0.6, f"{t.hour:02d}:00",
                    fontsize=8, ha="center", color="orange")

    plt.colorbar(sc, ax=ax, label="Time progression")
    plt.tight_layout()    
    plt.show()
    
    print("=" * 100)
    print("| === NUMERICAL DIAGNOSTICS (2D az-alt trajectory) ===")
    print("|----------------------------------------------------")
    print(f"| Visible samples: {visible_samples}")
    print(f"| Visible duration: {visible_duration_hr:.4f} hours")
    print("|----------------------------------------------------")
    print("| Azimuth (circular):")
    print(f"|   Mean azimuth: {mean_az:.6f}°")
    print(f"|   Rayleigh R: {R:.6f}  (variance={1-R:.6f})")
    print(f"|   Circular std: {circ_std:.6f}°")
    print("|----------------------------------------------------")
    print("| Altitude (deg):")
    print(f"|   min={alt_min:.6f}, max={alt_max:.6f}, mean={alt_mean:.6f}")
    print("|----------------------------------------------------")
    print("| Angular speed (deg/hr):")
    print(f"|   mean={mean_speed:.6f}")
    print(f"|   max={max_speed:.6f} at {max_speed_time}")
    print(f"|   min={min_speed:.6f} at {min_speed_time}")
    print("|----------------------------------------------------")
    print("| Curvature:")
    if K_time:
        print(f"|   mean={K_mean:.6e}")
        print(f"|   max={K_max:.6e} at {K_time}")
        print(f"|   at max curvature: Az={K_az:.3f}°, Alt={K_alt:.3f}°")
    else:
        print("|   not well-defined")
    print("|----------------------------------------------------")
    print(f"| Arc length: {arc_length:.6f}°")
    print("| === end diagnostics ===")
    print("=" * 100)
    

# =========================================================
# ANALemma
# =========================================================

def plot_analemma(object_name, latitude, longitude, year, sample_time_utc=(12, 0), period=365):
    topos = Topos(latitude_degrees=latitude, longitude_degrees=longitude)
    observer = eph["earth"] + topos

    key = object_name.lower()
    planet_map = {
        "sun": "sun",
        "mercury": "mercury",
        "venus": "venus",
        "earth": "earth",
        "moon": "moon",
        "mars": "mars",
    
        # Outer planets (barycenters ONLY in DE421)
        "jupiter": "jupiter barycenter",
        "saturn": "saturn barycenter",
        "uranus": "uranus barycenter",
        "neptune": "neptune barycenter",
        "pluto": "pluto barycenter",
    
        # System level
        "solar system barycenter": "solar system barycenter",
        }

    if key in planet_map:
        target = eph[planet_map[key]]
    else:
        hip = NAME_TO_HIP.get(key)
        if hip is None:
            raise ValueError("Unknown object name.")
        target = star_from_hip(hip)

    azs, alts = [], []

    for i in range(period+1):
        d = datetime(year, 1, 1, tzinfo=utc) + timedelta(days=i)
        t = ts.utc(d.year, d.month, d.day, sample_time_utc[0], sample_time_utc[1])
        alt, az, _ = observer.at(t).observe(target).apparent().altaz()
        azs.append(az.degrees)
        alts.append(alt.degrees)

    az_un = unwrap_az(azs)

    fig, ax = plt.subplots(figsize=(18, 12))
    sc = ax.scatter(az_un, alts, c=np.arange(len(alts)), cmap="gist_rainbow", s=22)
    ax.plot(az_un, alts, alpha=0.4)

    ax.set_xlabel("Azimuth (deg, unwrapped)")
    ax.set_ylabel("Altitude (deg)")
    ax.set_title(f"{object_name} Analemma — {year}")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.colorbar(sc, ax=ax, label="Days Since the Start ")
    plt.tight_layout()
    plt.show()
    
    # ================== ANALemma DIAGNOSTICS ==================
    days = np.arange(len(alts))
    arc_len = analemma_arc_length(az_un, alts)
    
    K_mean, K_max, K_day = analemma_curvature(days, az_un, alts)
    
    orientation = analemma_orientation(az_un, alts)
    asymmetry = analemma_loop_asymmetry(alts)
    
    print("=" * 70)
    print("ANALemma numerical diagnostics")
    print(f"Object: {object_name}")
    print(f"Altitude range: {np.ptp(alts):.3f} deg")
    print(f"Azimuth range:  {np.ptp(az_un):.3f} deg")
    print(f"Arc length:     {arc_len:.3f} deg")
    print(f"Orientation:    {orientation:.2f} deg (principal axis)")
    print(f"Loop asymmetry: {asymmetry:.3f} deg")
    if K_day is not None:
        print(f"Mean curvature: {K_mean:.6e}")
        print(f"Max curvature:  {K_max:.6e} on day {K_day}")
    print("=" * 70)
    
    # ================== FOURIER ANALemma DIAGNOSTICS ==================
    fourier = analemma_fourier_diagnostics(az_un, alts)
    
    print("-" * 70)
    print("Fourier diagnostics (dominant harmonics)")
    print("Altitude:")
    for f, p in fourier["altitude"]:
        print(f"  freq={f:.4f} cycles/day   power={p:.3e}")
    
    print("Azimuth:")
    for f, p in fourier["azimuth"]:
        print(f"  freq={f:.4f} cycles/day   power={p:.3e}")



# =========================================================
# EXAMPLES
# =========================================================

if __name__ == "__main__":
    lat, lon = 51.1079, 17.0385  # Wrocław
    #lat, lon = 18.5246, 73.8786  # Pune
    
    date = "2025-12-31" # YYYY-MM-DD

    #plot_celestial_path("Altair", lat, lon, date, day_mode="midnight")
    #plot_celestial_path("Sirius", lat, lon, date, day_mode="noon")
    #plot_celestial_path("Polaris", lat, lon, date, day_mode="noon")
    #plot_celestial_path("Venus", lat, lon, date, day_mode="midnight")
    #plot_celestial_path("Venus", lat, lon, date, day_mode="midnight")
    #plot_celestial_path("Jupiter", lat, lon, date, day_mode="noon")

    #plot_analemma("Sun", lat, lon, 2025, sample_time_utc=(5, 0))
    #plot_analemma("Venus", lat, lon, 2025, sample_time_utc=(12, 0), period=1000)
    plot_analemma("Mars", lat, lon, 1900, sample_time_utc=(12, 0), period=10000)
    #plot_analemma("Moon", lat, lon, 2025, sample_time_utc=(14, 0), period=30)
    #plot_analemma("Jupiter", lat, lon, 2025, sample_time_utc=(12, 0), period=5000)
