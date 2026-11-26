"""WGS84 ellipsoidal Earth gravity, directional curvature transport, and 3D ECI J2 dynamics.

CDR Rev 3.1 architecture:
  - Stage 1 RTLS ENU frame:
      Uses WGS84 Somigliana normal gravity `gamma(phi0, h)` strictly along `-Z_ENU`
      (which embeds static Earth centrifugal force so the geodetic plumb-line is strictly vertical
      with zero artificial ~1.7 mg sideways drift) plus directional radii of curvature `(R_M, R_N)`
      for horizontal transport `a_z = v_x^2 / (R_N + h) + v_y^2 / (R_M + h)`.
  - Stage 2 700 km Orbit Insertion (3D ECI + J2):
      Transforms `(pos_enu, vel_enu)` at separation into exact WGS84 ECEF -> ECI (including
      `Omega_E x r_ecef`), applies J2-compensated circularization speed, and evaluates the
      700 +/- 15 km criterion via 1-orbit RK4 mean orbital radius `r_mean - R_mean` and
      mean eccentricity `e_mean < 0.005` (eliminating polar oblateness `a - b = 21.4 km` bias).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin, sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray

WGS84_A_M = 6_378_137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B_M = WGS84_A_M * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
WGS84_R_MEAN_M = 6_371_008.8
WGS84_J2 = 1.08262668e-3
WGS84_MU_M3_S2 = 3.986004418e14
WGS84_OMEGA_RAD_S = 7.292115e-5

# Official WGS84 Somigliana normal gravity formula constants
WGS84_GAMMA_EQUATOR_M_S2 = 9.7803253359
WGS84_SOMIGLIANA_K = 0.00193185265241


def wgs84_radii_of_curvature(latitude_deg: float) -> tuple[float, float]:
    """Return meridional & prime-vertical curvature radii `(R_M, R_N)` [m] at geodetic latitude."""
    sin_phi = sin(radians(float(latitude_deg)))
    w_sq = 1.0 - WGS84_E2 * sin_phi * sin_phi
    w = sqrt(w_sq)
    r_n = WGS84_A_M / w
    r_m = WGS84_A_M * (1.0 - WGS84_E2) / (w_sq * w)
    return r_m, r_n


def wgs84_normal_gravity(latitude_deg: float, altitude_m: float) -> float:
    """WGS84 Somigliana normal gravity [m/s^2] along local geodetic `-Z_ENU`."""
    sin_phi = sin(radians(float(latitude_deg)))
    sin2 = sin_phi * sin_phi
    w = sqrt(1.0 - WGS84_E2 * sin2)
    gamma_0 = WGS84_GAMMA_EQUATOR_M_S2 * (1.0 + WGS84_SOMIGLIANA_K * sin2) / w
    h = float(altitude_m) if altitude_m > 0.0 else 0.0
    r_m, r_n = wgs84_radii_of_curvature(latitude_deg)
    r_eff = sqrt(r_m * r_n)
    ratio = r_eff / (r_eff + h)
    return gamma_0 * ratio * ratio


def wgs84_coriolis_and_transport_enu(
    *,
    altitude_m: float,
    vel_enu_m_s: ArrayLike,
    latitude_deg: float,
    coriolis_enabled: bool = True,
    planar_coriolis_only: bool = False,
) -> NDArray[np.float64]:
    """Return ENU inertial correction [m/s^2]: directional `(R_N, R_M)` transport + 3D Coriolis."""
    v = np.asarray(vel_enu_m_s, dtype=np.float64).reshape(3)
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    h = float(altitude_m) if altitude_m > 0.0 else 0.0
    r_m, r_n = wgs84_radii_of_curvature(latitude_deg)

    ax = 0.0
    ay = 0.0
    az = (vx * vx) / (r_n + h) + (vy * vy) / (r_m + h)

    if coriolis_enabled:
        lat = radians(float(latitude_deg))
        om_y = WGS84_OMEGA_RAD_S * cos(lat)
        om_z = 0.0 if planar_coriolis_only else WGS84_OMEGA_RAD_S * sin(lat)
        ax += -2.0 * (om_y * vz - om_z * vy)
        ay += -2.0 * (om_z * vx)
        az += 2.0 * (om_y * vx)

    return np.array([ax, ay, az], dtype=np.float64)


def enu_to_ecef_rotmat(latitude_deg: float, longitude_deg: float = 0.0) -> NDArray[np.float64]:
    """Return 3x3 orthogonal matrix `R_ecef_enu` mapping ENU vectors to ECEF."""
    lat = radians(float(latitude_deg))
    lon = radians(float(longitude_deg))
    slat, clat = sin(lat), cos(lat)
    slon, clon = sin(lon), cos(lon)
    return np.array(
        [
            [-slon, -slat * clon, clat * clon],
            [clon, -slat * slon, clat * slon],
            [0.0, clat, slat],
        ],
        dtype=np.float64,
    )


def geodetic_to_ecef(
    latitude_deg: float, longitude_deg: float = 0.0, altitude_m: float = 0.0
) -> NDArray[np.float64]:
    """Convert geodetic `(lat_deg, lon_deg, alt_m)` on WGS84 ellipsoid to ECEF [m]."""
    lat = radians(float(latitude_deg))
    lon = radians(float(longitude_deg))
    slat, clat = sin(lat), cos(lat)
    slon, clon = sin(lon), cos(lon)
    _, r_n = wgs84_radii_of_curvature(latitude_deg)
    h = float(altitude_m)
    return np.array(
        [
            (r_n + h) * clat * clon,
            (r_n + h) * clat * slon,
            (r_n * (1.0 - WGS84_E2) + h) * slat,
        ],
        dtype=np.float64,
    )


def enu_to_eci_state(
    pos_enu_m: ArrayLike,
    vel_enu_m_s: ArrayLike,
    latitude_deg: float,
    longitude_deg: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Map local launch-pad ENU `(pos, vel)` to inertial ECI `(r_eci, v_eci)` at epoch GMST=0."""
    p_enu = np.asarray(pos_enu_m, dtype=np.float64).reshape(3)
    v_enu = np.asarray(vel_enu_m_s, dtype=np.float64).reshape(3)
    r0_ecef = geodetic_to_ecef(latitude_deg, longitude_deg, 0.0)
    rot = enu_to_ecef_rotmat(latitude_deg, longitude_deg)
    r_eci = r0_ecef + rot @ p_enu
    omega_vec = np.array([0.0, 0.0, WGS84_OMEGA_RAD_S], dtype=np.float64)
    v_eci = rot @ v_enu + np.cross(omega_vec, r_eci)
    return r_eci, v_eci


def eci_j2_gravity(r_eci_m: ArrayLike) -> NDArray[np.float64]:
    """3D ECI gravitational acceleration [m/s^2] (central potential + WGS84 J2 zonal harmonic)."""
    r_vec = np.asarray(r_eci_m, dtype=np.float64).reshape(3)
    x, y, z = float(r_vec[0]), float(r_vec[1]), float(r_vec[2])
    r2 = x * x + y * y + z * z
    r = sqrt(r2)
    if r <= 1.0:
        return np.zeros(3, dtype=np.float64)
    r3 = r2 * r
    r5 = r3 * r2
    mu_r3 = -WGS84_MU_M3_S2 / r3
    z2_r2 = (z * z) / r2
    j2_fac = -1.5 * WGS84_J2 * WGS84_MU_M3_S2 * (WGS84_A_M * WGS84_A_M) / r5
    ax = mu_r3 * x + j2_fac * x * (1.0 - 5.0 * z2_r2)
    ay = mu_r3 * y + j2_fac * y * (1.0 - 5.0 * z2_r2)
    az = mu_r3 * z + j2_fac * z * (3.0 - 5.0 * z2_r2)
    return np.array([ax, ay, az], dtype=np.float64)


def j2_circular_speed(radius_m: float, sin_geocentric_lat: float) -> float:
    """J2-compensated circular orbital speed [m/s] across equatorial, inclined, and polar orbits."""
    r = max(float(radius_m), 1.0)
    s2 = float(sin_geocentric_lat) ** 2
    corr = 1.0 + 1.5 * WGS84_J2 * ((WGS84_A_M / r) ** 2) * (1.0 - 3.0 * s2)
    return sqrt(max(0.0, (WGS84_MU_M3_S2 / r) * corr))


def rk4_step_eci_j2(
    r_eci: NDArray[np.float64],
    v_eci: NDArray[np.float64],
    dt_s: float,
    thrust_accel_eci: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Advance `(r_eci, v_eci)` by `dt_s` with 4th-order Runge-Kutta under ECI J2 gravity."""
    a_ext = (
        np.zeros(3, dtype=np.float64)
        if thrust_accel_eci is None
        else np.asarray(thrust_accel_eci, dtype=np.float64).reshape(3)
    )
    k1_r = v_eci
    k1_v = eci_j2_gravity(r_eci) + a_ext

    r2 = r_eci + 0.5 * dt_s * k1_r
    v2 = v_eci + 0.5 * dt_s * k1_v
    k2_r = v2
    k2_v = eci_j2_gravity(r2) + a_ext

    r3 = r_eci + 0.5 * dt_s * k2_r
    v3 = v_eci + 0.5 * dt_s * k2_v
    k3_r = v3
    k3_v = eci_j2_gravity(r3) + a_ext

    r4 = r_eci + dt_s * k3_r
    v4 = v_eci + dt_s * k3_v
    k4_r = v4
    k4_v = eci_j2_gravity(r4) + a_ext

    r_next = r_eci + (dt_s / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)
    v_next = v_eci + (dt_s / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    return r_next, v_next


@dataclass(frozen=True, slots=True)
class J2OrbitAudit:
    mean_radius_m: float
    min_radius_m: float
    max_radius_m: float
    mean_altitude_km: float
    min_altitude_km: float
    max_altitude_km: float
    mean_eccentricity: float
    orbital_period_s: float


def audit_j2_orbit_one_rev(
    r_eci0: ArrayLike,
    v_eci0: ArrayLike,
    *,
    dt_s: float = 10.0,
) -> J2OrbitAudit:
    """Propagate 1 full J2 orbital revolution via RK4 and audit mean radius and eccentricity."""
    r = np.asarray(r_eci0, dtype=np.float64).reshape(3).copy()
    v = np.asarray(v_eci0, dtype=np.float64).reshape(3).copy()
    r0_mag = float(np.linalg.norm(r))
    v0_sq = float(v @ v)
    eps = 0.5 * v0_sq - WGS84_MU_M3_S2 / max(r0_mag, 1.0)
    if eps >= -1.0:
        period_s = 6_000.0
    else:
        a_est = -WGS84_MU_M3_S2 / (2.0 * eps)
        period_s = 2.0 * np.pi * sqrt(max(a_est, WGS84_R_MEAN_M) ** 3 / WGS84_MU_M3_S2)
    n_steps = max(60, int(round(period_s / dt_s)))
    step = period_s / n_steps

    r_sum = 0.0
    r_min = r0_mag
    r_max = r0_mag
    for _ in range(n_steps):
        r, v = rk4_step_eci_j2(r, v, step)
        rm = float(np.linalg.norm(r))
        r_sum += rm
        if rm < r_min:
            r_min = rm
        if rm > r_max:
            r_max = rm

    r_mean = r_sum / n_steps
    e_mean = (r_max - r_min) / max(r_max + r_min, 1.0)
    return J2OrbitAudit(
        mean_radius_m=r_mean,
        min_radius_m=r_min,
        max_radius_m=r_max,
        mean_altitude_km=(r_mean - WGS84_R_MEAN_M) / 1000.0,
        min_altitude_km=(r_min - WGS84_R_MEAN_M) / 1000.0,
        max_altitude_km=(r_max - WGS84_R_MEAN_M) / 1000.0,
        mean_eccentricity=e_mean,
        orbital_period_s=period_s,
    )
