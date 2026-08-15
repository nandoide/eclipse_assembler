#!/usr/bin/env python3
"""
================================================================================
NASA JPL HORIZONS API ECLIPSE ENGINE (2026 - 2036)
================================================================================
Queries EXCLUSIVELY the official NASA JPL Horizons REST API (JSON format):
  https://ssd.jpl.nasa.gov/api/horizons.api

Astronomical Implementation Details:
  1. Syzygies & Node Crossings: Dynamically discovers New Moon (Solar) and Full Moon
     (Lunar) candidate dates across the requested year range (2026-2036).
  2. Terrestrial Time (TT) Sampling: Queries Horizons in TT timescale ('TT YYYY-MM-DD HH:MM')
     centered at integer hour t0 = round(t_syzygy).
  3. Fundamental Besselian Plane: Derives (x, y, d, l1, l2, mu, tan_f1, tan_f2) using
     IAU/Chauvenet rigorous geometric transformations and fits 3rd-order polynomials.
  4. Lunar Eclipse Contacts: Computes geocentric shadow timings (P1, U1, U2, MAX, U3, U4, P4).
  5. Master Database: Compiles and persists into '030_db/eclipses_db.json'.

Authors: Fernando (nandoide) & Antigravity (Google Deepmind)
================================================================================
"""

import os
import sys
import json
import time
import math
import datetime
import urllib.request
import urllib.parse
import numpy as np


HORIZONS_API_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"

# Fundamental IAU physical constants
A_EARTH_KM = 6378.137          # Earth equatorial radius (WGS-84)
R_SUN_KM   = 696340.0          # Solar radius
R_MOON_KM  = 1737.4            # Lunar radius
AU_TO_KM   = 149597870.7       # 1 Astronomical Unit in km


# ─────────────────────────────────────────────────────────────────────────────
# 1. HORIZONS API JSON CLIENT (Terrestrial Time TT)
# ─────────────────────────────────────────────────────────────────────────────

def query_horizons_tt(body_code: str, date_str: str, t0_int: float, span_hours: float = 3.0, step_min: int = 5):
    """
    Queries NASA JPL Horizons REST API in Terrestrial Time (TT) timescale.
    """
    dt_base = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    t_start = dt_base + datetime.timedelta(hours=t0_int - span_hours)
    t_stop  = dt_base + datetime.timedelta(hours=t0_int + span_hours)

    start_str = f"TT {t_start.strftime('%Y-%m-%d %H:%M')}"
    stop_str  = t_stop.strftime('%Y-%m-%d %H:%M')

    params = {
        'format': 'json',
        'COMMAND': f"'{body_code}'",
        'OBJ_DATA': 'NO',
        'MAKE_EPHEM': 'YES',
        'EPHEM_TYPE': 'OBSERVER',
        'CENTER': "'500@399'",  # Geocentric
        'START_TIME': f"'{start_str}'",
        'STOP_TIME': f"'{stop_str}'",
        'STEP_SIZE': f"'{step_min}m'",
        'QUANTITIES': "'1,20'"  # 1=RA/Dec, 20=Range (Delta)
    }

    url = HORIZONS_API_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={'User-Agent': 'eclipse_assembler/3.0 (NASA Horizons Client)'})
    
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        result_text = data.get('result', '')
        
        lines = result_text.split('\n')
        in_data = False
        parsed_points = []
        
        for line in lines:
            if '$$SOE' in line:
                in_data = True
                continue
            if '$$EOE' in line:
                break
            if in_data and len(line.strip()) > 0:
                parts = line.strip().split()
                if len(parts) >= 8:
                    ra_h, ra_m, ra_s = float(parts[2]), float(parts[3]), float(parts[4])
                    dec_d, dec_m, dec_s = float(parts[5]), float(parts[6]), float(parts[7])
                    ra_deg = (ra_h + ra_m / 60.0 + ra_s / 3600.0) * 15.0
                    sign = -1.0 if parts[5].startswith('-') else 1.0
                    dec_deg = sign * (abs(dec_d) + dec_m / 60.0 + dec_s / 3600.0)
                    dist_au = float(parts[8]) if len(parts) > 8 else 1.0
                    
                    parsed_points.append({
                        'datetime_str': parts[0] + ' ' + parts[1],
                        'ra_deg': ra_deg,
                        'dec_deg': dec_deg,
                        'dist_au': dist_au
                    })
        return parsed_points


# ─────────────────────────────────────────────────────────────────────────────
# 2. DYNAMIC CANDIDATE DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def get_candidate_syzygies(start_year: int = 2026, start_month: int = 8, end_year: int = 2036) -> list:
    """
    Calculates candidate New Moon and Full Moon syzygies near orbital nodes.
    """
    candidates = []
    ref_date = datetime.date(2000, 1, 6)
    d_start = (datetime.date(start_year, start_month, 1) - ref_date).days
    d_end   = (datetime.date(end_year, 12, 31) - ref_date).days
    
    k_min = int(d_start / 29.530588861) - 1
    k_max = int(d_end / 29.530588861) + 2

    for k_int in range(k_min, k_max):
        for is_full in [False, True]:
            k_val = k_int + (0.5 if is_full else 0.0)
            T = k_val / 1236.85
            T2, T3, T4 = T*T, T*T*T, T*T*T*T

            jde = 2451550.09766 + 29.530588861 * k_val + 0.0001337 * T2 - 0.000000150 * T3 + 0.00000000073 * T4
            M = 2.5534 + 29.10535670 * k_val - 0.0000014 * T2 - 0.00000011 * T3
            Mp = 201.5643 + 385.81693528 * k_val + 0.0107582 * T2 + 0.00001238 * T3 - 0.000000058 * T4
            F = 160.7108 + 390.67050284 * k_val - 0.0016118 * T2 - 0.00000227 * T3 + 0.000000011 * T4
            Om = 124.7746 - 1.56375588 * k_val + 0.0020672 * T2 + 0.00000215 * T3
            E = 1.0 - 0.002516 * T - 0.0000074 * T2

            M_r, Mp_r, F_r, Om_r = [math.radians(x % 360) for x in (M, Mp, F, Om)]

            if not is_full:
                corr = - 0.40720 * math.sin(Mp_r) + 0.17241 * E * math.sin(M_r) \
                       + 0.01608 * math.sin(2 * Mp_r) + 0.01039 * math.sin(2 * F_r) \
                       + 0.00739 * E * math.sin(Mp_r - M_r) - 0.00514 * E * math.sin(Mp_r + M_r) \
                       + 0.00208 * E * E * math.sin(2 * M_r)
            else:
                corr = - 0.40614 * math.sin(Mp_r) + 0.17302 * E * math.sin(M_r) \
                       + 0.01614 * math.sin(2 * Mp_r) + 0.01043 * math.sin(2 * F_r) \
                       + 0.00734 * E * math.sin(Mp_r - M_r) - 0.00515 * E * math.sin(Mp_r + M_r) \
                       + 0.00209 * E * E * math.sin(2 * M_r)

            jde_true = jde + corr
            F1 = F - 0.02665 * math.sin(Om_r)
            f_dist = min(F1 % 180.0, 180.0 - (F1 % 180.0))

            limit = 18.0 if not is_full else 20.0
            if f_dist <= limit:
                jd = jde_true + 0.5
                z = int(jd)
                f = jd - z
                alpha = int((z - 1867216.25) / 36524.25)
                a = z + 1 + alpha - int(alpha / 4)
                b = a + 1524
                c = int((b - 122.1) / 365.25)
                d = int(365.25 * c)
                e = int((b - d) / 30.6001)
                day = b - d - int(30.6001 * e) + f
                month = e - 1 if e < 14 else e - 13
                year = c - 4716 if month > 2 else c - 4715
                day_int = int(day)
                sec_total = (day - day_int) * 86400.0
                hour = int(sec_total // 3600)
                minute = int((sec_total % 3600) // 60)

                dt = datetime.datetime(year, month, day_int, hour, minute)
                if (year > start_year or (year == start_year and month >= start_month)) and year <= end_year:
                    saros_num = int(k_val / 223.0) % 180
                    # t0 in Besselian elements is standard integer hour
                    t0_int = float(round(hour + minute / 60.0))
                    candidates.append({
                        'date': dt.strftime('%Y-%m-%d'),
                        'dt_utc': dt,
                        'cat': 'lunar' if is_full else 'solar',
                        't0': t0_int,
                        'saros': saros_num,
                        'delta_t': round(71.0 + (year - 2024) * 0.9, 1)
                    })

    candidates.sort(key=lambda x: x['date'])
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# 3. BESSELIAN ELEMENTS IN TERRESTRIAL TIME
# ─────────────────────────────────────────────────────────────────────────────

def process_solar_eclipse_via_horizons(candidate: dict) -> dict:
    """
    Derives standard IAU Besselian polynomial elements from NASA JPL Horizons TT ephemeris.
    """
    date_str = candidate["date"]
    t0_int = float(candidate["t0"])
    delta_t_s = float(candidate.get("delta_t", 71.4))

    moon_pts = query_horizons_tt('301', date_str, t0_int, span_hours=3.0, step_min=5)
    sun_pts  = query_horizons_tt('10',  date_str, t0_int, span_hours=3.0, step_min=5)

    if not moon_pts or not sun_pts or len(moon_pts) != len(sun_pts):
        raise RuntimeError(f"Horizons API points mismatch for solar eclipse on {date_str}")

    N = len(moon_pts)
    t_hours = np.linspace(-3.0, 3.0, N)

    xs, ys, ds, l1s, l2s, mus, tan_f1s, tan_f2s = [], [], [], [], [], [], [], []
    dt_base = datetime.datetime.strptime(date_str, "%Y-%m-%d")

    for i in range(N):
        m = moon_pts[i]
        s = sun_pts[i]
        t = t_hours[i]

        as_rad = math.radians(s['ra_deg'])
        ds_rad = math.radians(s['dec_deg'])
        am_rad = math.radians(m['ra_deg'])
        dm_rad = math.radians(m['dec_deg'])

        r_sun_km = s['dist_au'] * AU_TO_KM
        r_moon_km = m['dist_au'] * AU_TO_KM

        # Fundamental plane coordinates (IAU Standard)
        x = (r_moon_km / A_EARTH_KM) * math.cos(dm_rad) * math.sin(am_rad - as_rad)
        y = (r_moon_km / A_EARTH_KM) * (math.sin(dm_rad) * math.cos(ds_rad) - math.cos(dm_rad) * math.sin(ds_rad) * math.cos(am_rad - as_rad))
        z = (r_moon_km / A_EARTH_KM) * (math.sin(dm_rad) * math.sin(ds_rad) + math.cos(dm_rad) * math.cos(ds_rad) * math.cos(am_rad - as_rad))

        tan_f1 = (R_SUN_KM + R_MOON_KM) / (r_sun_km - r_moon_km)
        tan_f2 = (R_SUN_KM - R_MOON_KM) / (r_sun_km - r_moon_km)

        l1 = z * tan_f1 + (R_MOON_KM / A_EARTH_KM)
        
        sd_moon = math.asin(R_MOON_KM / r_moon_km)
        sd_sun  = math.asin(R_SUN_KM / r_sun_km)
        is_total = (sd_moon >= sd_sun)
        l2_sign = -1.0 if is_total else 1.0
        l2 = z * tan_f2 + l2_sign * (R_MOON_KM / A_EARTH_KM)

        # Greenwich Hour Angle of shadow axis: mu = GMST(t_UT) - as_rad
        cur_dt_ut = dt_base + datetime.timedelta(hours=t0_int + t - delta_t_s / 3600.0)
        d_days_ut = (cur_dt_ut - datetime.datetime(2000, 1, 1, 12, 0)).total_seconds() / 86400.0
        gmst_deg = (280.46061837 + 360.98564736629 * d_days_ut) % 360.0
        mu_deg = (gmst_deg - s['ra_deg']) % 360.0

        xs.append(x)
        ys.append(y)
        ds.append(math.degrees(ds_rad))
        l1s.append(l1)
        l2s.append(l2)
        mus.append(mu_deg)
        tan_f1s.append(tan_f1)
        tan_f2s.append(tan_f2)

    mus_unwrapped = np.unwrap(np.radians(mus))
    mus_deg_unwrapped = np.degrees(mus_unwrapped)

    p_x  = [float(c) for c in np.polyfit(t_hours, xs, 3)[::-1]]
    p_y  = [float(c) for c in np.polyfit(t_hours, ys, 3)[::-1]]
    p_d  = [float(c) for c in np.polyfit(t_hours, ds, 3)[::-1]]
    p_l1 = [float(c) for c in np.polyfit(t_hours, l1s, 3)[::-1]]
    p_l2 = [float(c) for c in np.polyfit(t_hours, l2s, 3)[::-1]]
    p_mu = [float(c) for c in np.polyfit(t_hours, mus_deg_unwrapped, 3)[::-1]]
    p_mu[0] = p_mu[0] % 360.0

    min_dist = min(math.sqrt(xs[i]**2 + ys[i]**2) for i in range(N))
    if min_dist > 1.55:
        etype = "Partial"
    elif is_total:
        etype = "Total"
    else:
        etype = "Annular"

    return {
        "name": f"{etype} Solar Eclipse of {date_str[:4]} {dt_base.strftime('%b %d')}",
        "type": etype,
        "category": "solar",
        "t0": t0_int,
        "delta_t": delta_t_s,
        "saros": candidate.get("saros", 0),
        "polynomials": {
            "x": p_x,
            "y": p_y,
            "d": p_d,
            "l1": p_l1,
            "l2": p_l2,
            "mu": p_mu,
            "tan_f1": float(np.mean(tan_f1s)),
            "tan_f2": float(np.mean(tan_f2s))
        }
    }


def process_lunar_eclipse_via_horizons(candidate: dict) -> dict:
    """
    Computes geocentric shadow timings from NASA JPL Horizons TT ephemeris.
    """
    date_str = candidate["date"]
    t0_int = float(candidate["t0"])
    delta_t_s = float(candidate.get("delta_t", 71.5))

    moon_pts = query_horizons_tt('301', date_str, t0_int, span_hours=4.0, step_min=5)
    sun_pts  = query_horizons_tt('10',  date_str, t0_int, span_hours=4.0, step_min=5)

    if not moon_pts or not sun_pts:
        raise RuntimeError(f"Horizons points missing for lunar eclipse on {date_str}")

    dt_base = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    N = len(moon_pts)
    t_hours = np.linspace(-4.0, 4.0, N)

    dists_from_shadow_center = []
    r_umbra_list = []
    r_penumbra_list = []

    for i in range(N):
        m = moon_pts[i]
        s = sun_pts[i]

        ra_opp = (s['ra_deg'] + 180.0) % 360.0
        dec_opp = -s['dec_deg']

        dra = (m['ra_deg'] - ra_opp) * math.cos(math.radians((m['dec_deg'] + dec_opp) / 2.0))
        ddec = m['dec_deg'] - dec_opp
        sep_deg = math.sqrt(dra**2 + ddec**2)

        r_moon_km = m['dist_au'] * AU_TO_KM
        pi_moon_deg = math.degrees(math.asin(A_EARTH_KM / r_moon_km))
        pi_sun_deg  = math.degrees(math.asin(A_EARTH_KM / (s['dist_au'] * AU_TO_KM)))
        sd_sun_deg  = math.degrees(math.asin(R_SUN_KM / (s['dist_au'] * AU_TO_KM)))
        sd_moon_deg = math.degrees(math.asin(R_MOON_KM / r_moon_km))

        r_umbra_deg    = 1.02 * (pi_moon_deg + pi_sun_deg - sd_sun_deg)
        r_penumbra_deg = 1.02 * (pi_moon_deg + pi_sun_deg + sd_sun_deg)

        dists_from_shadow_center.append(sep_deg)
        r_umbra_list.append(r_umbra_deg)
        r_penumbra_list.append(r_penumbra_deg)

    dists = np.array(dists_from_shadow_center)
    r_u   = np.array(r_umbra_list)
    r_p   = np.array(r_penumbra_list)

    idx_max = np.argmin(dists)
    t_max_h = t_hours[idx_max]

    def to_iso(t_h):
        dt_ut = dt_base + datetime.timedelta(hours=t0_int + t_h - delta_t_s / 3600.0)
        return dt_ut.strftime("%Y-%m-%dT%H:%M:%SZ")

    contacts = {"MAX": to_iso(t_max_h)}

    sd_m = 0.25
    in_pen = np.where(dists <= (r_p + sd_m))[0]
    if len(in_pen) > 0:
        contacts["P1"] = to_iso(t_hours[in_pen[0]])
        contacts["P4"] = to_iso(t_hours[in_pen[-1]])

    in_u = np.where(dists <= (r_u + sd_m))[0]
    if len(in_u) > 0:
        contacts["U1"] = to_iso(t_hours[in_u[0]])
        contacts["U4"] = to_iso(t_hours[in_u[-1]])

    in_tot = np.where(dists <= (r_u - sd_m))[0]
    if len(in_tot) > 0:
        contacts["U2"] = to_iso(t_hours[in_tot[0]])
        contacts["U3"] = to_iso(t_hours[in_tot[-1]])
        etype = "Total"
    elif len(in_u) > 0:
        etype = "Partial"
    else:
        etype = "Penumbral"

    return {
        "name": f"{etype} Lunar Eclipse of {date_str[:4]} {dt_base.strftime('%b %d')}",
        "type": etype,
        "category": "lunar",
        "delta_t": delta_t_s,
        "saros": candidate.get("saros", 0),
        "contacts_utc": contacts
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. MASTER BUILDER (100% HORIZONS API)
# ─────────────────────────────────────────────────────────────────────────────

def build_database_from_horizons(output_path: str = "030_db/eclipses_db.json", start_year: int = 2026, end_year: int = 2036):
    """
    100% NASA JPL Horizons API engine in Terrestrial Time.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_path = output_path if os.path.isabs(output_path) else os.path.join(repo_root, output_path)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    print("=" * 65)
    print(f"QUERYING NASA JPL HORIZONS REST API (TT TIMESCALE) ({start_year} - {end_year})")
    print("=" * 65)

    candidates = get_candidate_syzygies(start_year=start_year, start_month=8, end_year=end_year)
    print(f"  • Found {len(candidates)} candidate syzygies near orbital nodes.")

    solar_db = {}
    lunar_db = {}

    for cand in candidates:
        d = cand["date"]
        cat = cand["cat"]
        print(f"  • [{cat.upper()}] Querying Horizons (TT) for {d}...", end="", flush=True)

        try:
            if cat == "solar":
                data = process_solar_eclipse_via_horizons(cand)
                solar_db[d] = data
            else:
                data = process_lunar_eclipse_via_horizons(cand)
                lunar_db[d] = data
            print(f" [OK: {data['type']}]")
        except Exception as e:
            print(f" [SKIP: {e}]")

        time.sleep(0.12)

    master_db = {
        "_metadata": {
            "description": f"Universal Solar & Lunar Eclipse Ephemeris Database ({start_year} - {end_year})",
            "source": "NASA JPL Solar System Dynamics Horizons REST API (https://ssd.jpl.nasa.gov/api/horizons.api)",
            "time_span": f"{start_year}-08 to {end_year}-12",
            "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "generator": "020_src/fetch_eclipses_horizons.py",
            "events_count": len(solar_db) + len(lunar_db),
            "version": "3.1"
        },
        "solar_eclipses": solar_db,
        "lunar_eclipses": lunar_db
    }

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(master_db, f, indent=2)

    sz_kb = os.path.getsize(target_path) / 1024.0
    print("\n" + "=" * 65)
    print(f"SUCCESS: NASA Horizons Database compiled ({sz_kb:.1f} KB, {len(solar_db)} Solar, {len(lunar_db)} Lunar)!")
    print(f"  Location: {target_path}")
    print("=" * 65)
    return master_db


if __name__ == "__main__":
    build_database_from_horizons()
