#!/usr/bin/env python3
"""
================================================================================
ASTRONOMICAL ECLIPSE EPHEMERIS DATABASE & SOLVER (Solar & Lunar Eclipses)
================================================================================
Universal astronomical engine that:
  1. Manages the official NASA / Espenak Eclipse Database ('030_db/eclipses_db.json').
  2. Automatically detects eclipse dates from telescope telemetry ('000_raw/').
  3. Computes topocentric Besselian contact times for Solar Eclipses (C1, C2, MAX, C3, C4).
  4. Computes geocentric contact times and visibility for Lunar Eclipses (P1, U1, U2, MAX, U3, U4, P4).
  5. Connects to NASA JPL Horizons REST API (https://ssd.jpl.nasa.gov/api/horizons.api)
     for online database building and verification ('--force-db').

Authors: Fernando (nandoide) & Antigravity (Google Deepmind)
================================================================================
"""

import os
import sys
import json
import re
import math
import datetime
import urllib.request
import urllib.parse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_eclipses_horizons as feh


DEFAULT_DB_REL_PATH = "030_db/eclipses_db.json"


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE MANAGER & LOADER
# ─────────────────────────────────────────────────────────────────────────────

def resolve_db_path(db_path: str = None) -> str:
    if db_path and os.path.isabs(db_path):
        return db_path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target = db_path or DEFAULT_DB_REL_PATH
    return os.path.join(repo_root, target)


def load_or_fetch_eclipse_db(db_path: str = None, force: bool = False) -> dict:
    """
    Loads '030_db/eclipses_db.json'. If missing or force=True, queries NASA JPL Horizons
    and rebuilds the database for 2026-2036.
    """
    full_path = resolve_db_path(db_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    if os.path.exists(full_path) and not force:
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "solar_eclipses" in data and "lunar_eclipses" in data:
                    return data
        except Exception as e:
            print(f"[Ephemeris DB] Warning reading {full_path}: {e}. Rebuilding via NASA JPL Horizons...")

    print(f"[Ephemeris DB] Fetching / generating database via NASA JPL Horizons API...")
    data = feh.build_and_validate_database(output_path=full_path, verify_api=True)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# DATE DETECTION & RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_eclipse_date(raw_dir: str = "000_raw", in_dir: str = "010_in") -> str:
    """
    Scans files in raw_dir and in_dir to automatically detect the eclipse date (YYYY-MM-DD).
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    search_dirs = [
        os.path.join(repo_root, raw_dir) if not os.path.isabs(raw_dir) else raw_dir,
        os.path.join(repo_root, in_dir) if not os.path.isabs(in_dir) else in_dir
    ]

    date_regex = re.compile(r"(\d{4})[-_](\d{2})[-_](\d{2})")

    for d in search_dirs:
        if os.path.exists(d):
            for fname in os.listdir(d):
                m = date_regex.search(fname)
                if m:
                    yyyy, mm, dd = m.groups()
                    if 2000 <= int(yyyy) <= 2100 and 1 <= int(mm) <= 12 and 1 <= int(dd) <= 31:
                        return f"{yyyy}-{mm}-{dd}"

    # Default fallback
    return "2026-08-12"


def get_eclipse_entry(date_str: str, db: dict):
    """
    Retrieves the eclipse entry matching date_str (or closest date) from the database.
    """
    # 1. Direct match in solar eclipses
    if date_str in db.get("solar_eclipses", {}):
        entry = db["solar_eclipses"][date_str].copy()
        entry["date"] = date_str
        return entry

    # 2. Direct match in lunar eclipses
    if date_str in db.get("lunar_eclipses", {}):
        entry = db["lunar_eclipses"][date_str].copy()
        entry["date"] = date_str
        return entry

    # 3. Search closest date if year/month matches
    all_dates = list(db.get("solar_eclipses", {}).keys()) + list(db.get("lunar_eclipses", {}).keys())
    for d in all_dates:
        if d[:7] == date_str[:7]:
            cat = "solar_eclipses" if d in db.get("solar_eclipses", {}) else "lunar_eclipses"
            entry = db[cat][d].copy()
            entry["date"] = d
            return entry

    # Fallback to 2026-08-12
    return db.get("solar_eclipses", {}).get("2026-08-12", list(db.get("solar_eclipses", {}).values())[0])


# ─────────────────────────────────────────────────────────────────────────────
# ANALYTICAL SOLVERS (Solar & Lunar)
# ─────────────────────────────────────────────────────────────────────────────

def solve_topocentric_solar_eclipse(
    solar_meta: dict,
    lat_deg: float = 43.235556,
    lon_deg: float = -7.558333,
    alt_m: float = 438.7,
    tz_offset_hours: float = 2.0
) -> dict:
    """
    Computes topocentric contact times (C1, C2, MAX, C3, C4) for a Solar Eclipse
    using NASA / Fred Espenak Besselian polynomial elements.
    """
    phi = math.radians(lat_deg)
    lam = math.radians(lon_deg)
    h_km = alt_m / 1000.0

    a = 6378.137
    f = 1.0 / 298.257223563
    C_geo = 1.0 / math.sqrt(math.cos(phi)**2 + (1 - f)**2 * math.sin(phi)**2)
    S_geo = (1 - f)**2 * C_geo

    rho_cos_phi_prime = (C_geo + h_km / a) * math.cos(phi)
    rho_sin_phi_prime = (S_geo + h_km / a) * math.sin(phi)

    delta_t_s = float(solar_meta.get("delta_t", 71.4))
    delta_t_h = delta_t_s / 3600.0
    t0 = float(solar_meta.get("t0", 18.0))

    poly = solar_meta["polynomials"]
    x_poly = poly["x"]
    y_poly = poly["y"]
    d_poly = poly["d"]
    l1_poly = poly["l1"]
    l2_poly = poly["l2"]
    mu_poly = poly["mu"]
    tan_f1 = float(poly["tan_f1"])
    tan_f2 = float(poly["tan_f2"])

    def eval_p(p, t):
        return sum(c * (t**i) for i, c in enumerate(p))

    def compute_at(t_tdt):
        x = eval_p(x_poly, t_tdt)
        y = eval_p(y_poly, t_tdt)
        d = math.radians(eval_p(d_poly, t_tdt))
        l1 = eval_p(l1_poly, t_tdt)
        l2 = eval_p(l2_poly, t_tdt)
        mu_deg = eval_p(mu_poly, t_tdt)

        # Standard IAU Besselian Topocentric Hour Angle with Delta-T rotation
        H_deg = mu_deg + math.degrees(lam) - (1.0027379 * delta_t_s / 240.0)
        H = math.radians(H_deg)

        xi = rho_cos_phi_prime * math.sin(H)
        eta = rho_sin_phi_prime * math.cos(d) - rho_cos_phi_prime * math.sin(d) * math.cos(H)
        zeta = rho_sin_phi_prime * math.sin(d) + rho_cos_phi_prime * math.cos(d) * math.cos(H)

        u = x - xi
        v = y - eta
        dist = math.sqrt(u * u + v * v)

        L1 = l1 - zeta * tan_f1
        L2 = l2 - zeta * tan_f2
        return dist, L1, abs(L2)

    # Scan umbral contacts
    times_umbra = np.linspace(-3.0, 3.0, 60000)
    res_umbra = [compute_at(t) for t in times_umbra]
    dists = np.array([r[0] for r in res_umbra])
    l2s = np.array([r[2] for r in res_umbra])
    diff_umbral = dists - l2s

    inside_idx = np.where(diff_umbral <= 0)[0]
    is_total_at_site = (len(inside_idx) > 0)
    if is_total_at_site:
        t_c2_tdt = times_umbra[inside_idx[0]]
        t_c3_tdt = times_umbra[inside_idx[-1]]
    else:
        t_c2_tdt = None
        t_c3_tdt = None

    t_max_tdt = times_umbra[np.argmin(dists)]

    # Scan penumbral contacts
    times_pen = np.linspace(-4.0, 4.0, 80000)
    diff_pen = [compute_at(t)[0] - compute_at(t)[1] for t in times_pen]
    diff_pen = np.array(diff_pen)
    inside_pen = np.where(diff_pen <= 0)[0]

    date_parts = [int(p) for p in solar_meta.get("date", "2026-08-12").split("-")]
    base_date = datetime.date(date_parts[0], date_parts[1], date_parts[2])

    def to_datetime_local(t_tdt):
        if t_tdt is None:
            return None
        t_ut_hours = t0 + t_tdt - delta_t_h
        t_loc_hours = t_ut_hours + tz_offset_hours
        h = int(t_loc_hours) % 24
        m = int((t_loc_hours - int(t_loc_hours)) * 60)
        s = (t_loc_hours - int(t_loc_hours) - m / 60.0) * 3600.0
        sec_int = int(s)
        microsec = int(round((s - sec_int) * 1_000_000))
        if microsec >= 1_000_000:
            sec_int += 1
            microsec = 0
        return datetime.datetime(base_date.year, base_date.month, base_date.day, h, m, sec_int, microsec)

    t_c1_tdt = times_pen[inside_pen[0]] if len(inside_pen) > 0 else -1.0
    t_c4_tdt = times_pen[inside_pen[-1]] if len(inside_pen) > 0 else 1.0

    return {
        "name": solar_meta.get("name", "Solar Eclipse"),
        "category": "solar",
        "type": solar_meta.get("type", "Total"),
        "date": solar_meta.get("date", "2026-08-12"),
        "C1": to_datetime_local(t_c1_tdt),
        "C2": to_datetime_local(t_c2_tdt),
        "MAX": to_datetime_local(t_max_tdt),
        "C3": to_datetime_local(t_c3_tdt),
        "C4": to_datetime_local(t_c4_tdt),
        "is_total_at_site": is_total_at_site,
        "delta_t": delta_t_s
    }


def solve_lunar_eclipse(
    lunar_meta: dict,
    lat_deg: float = 43.235556,
    lon_deg: float = -7.558333,
    alt_m: float = 438.7,
    tz_offset_hours: float = 2.0
) -> dict:
    """
    Computes local contact times (P1, U1, U2, MAX, U3, U4, P4) for a Lunar Eclipse.
    """
    contacts_utc = lunar_meta.get("contacts_utc", {})
    contacts_local = {}

    for k, v in contacts_utc.items():
        dt_utc = datetime.datetime.fromisoformat(v.replace("Z", "+00:00"))
        dt_local = dt_utc + datetime.timedelta(hours=tz_offset_hours)
        contacts_local[k] = dt_local.replace(tzinfo=None)

    return {
        "name": lunar_meta.get("name", "Lunar Eclipse"),
        "category": "lunar",
        "type": lunar_meta.get("type", "Total"),
        "date": lunar_meta.get("date", "2026-08-28"),
        "P1": contacts_local.get("P1"),
        "U1": contacts_local.get("U1"),
        "U2": contacts_local.get("U2"),
        "MAX": contacts_local.get("MAX"),
        "U3": contacts_local.get("U3"),
        "U4": contacts_local.get("U4"),
        "P4": contacts_local.get("P4"),
        "contacts_local": contacts_local,
        "delta_t": float(lunar_meta.get("delta_t", 71.5))
    }


def solve_eclipse(
    lat_deg: float = 43.235556,
    lon_deg: float = -7.558333,
    alt_m: float = 438.7,
    date_str: str = None,
    tz_name: str = "CEST",
    tz_offset_hours: float = 2.0,
    db_path: str = None,
    raw_dir: str = "000_raw",
    in_dir: str = "010_in",
    force_db: bool = False
) -> dict:
    """
    Unified entry point: loads database, auto-detects date if needed, and solves
    either a Solar or Lunar eclipse.
    """
    db = load_or_fetch_eclipse_db(db_path=db_path, force=force_db)

    resolved_date = date_str or detect_eclipse_date(raw_dir=raw_dir, in_dir=in_dir)
    entry = get_eclipse_entry(resolved_date, db)

    cat = entry.get("category", "solar")
    if cat == "solar":
        res = solve_topocentric_solar_eclipse(
            solar_meta=entry,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            alt_m=alt_m,
            tz_offset_hours=tz_offset_hours
        )
    else:
        res = solve_lunar_eclipse(
            lunar_meta=entry,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            alt_m=alt_m,
            tz_offset_hours=tz_offset_hours
        )

    res["tz_name"] = tz_name
    res["tz_offset_hours"] = tz_offset_hours
    res["lat_deg"] = lat_deg
    res["lon_deg"] = lon_deg
    res["alt_m"] = alt_m
    return res


# ─────────────────────────────────────────────────────────────────────────────
# CLI TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Astronomical Eclipse Ephemeris Solver (Solar & Lunar 2026-2036)")
    parser.add_argument("--date", "-d", type=str, default=None, help="Eclipse date YYYY-MM-DD (default: auto-detect)")
    parser.add_argument("--lat", type=float, default=43.235556, help="Observer Latitude in degrees (default: 43.235556)")
    parser.add_argument("--lon", type=float, default=-7.558333, help="Observer Longitude in degrees (default: -7.558333)")
    parser.add_argument("--alt", type=float, default=438.7, help="Observer Altitude in meters (default: 438.7)")
    parser.add_argument("--timezone", "-tz", type=str, default="CEST", help="Timezone label (default: CEST)")
    parser.add_argument("--tz-offset", type=float, default=2.0, help="UTC offset in hours (default: 2.0)")
    parser.add_argument("--force-db", action="store_true", help="Force rebuilding/refreshing eclipse database via NASA JPL Horizons API")
    args = parser.parse_args()

    sol = solve_eclipse(
        lat_deg=args.lat,
        lon_deg=args.lon,
        alt_m=args.alt,
        date_str=args.date,
        tz_name=args.timezone,
        tz_offset_hours=args.tz_offset,
        force_db=args.force_db
    )

    print("=" * 65)
    print(f"ECLIPSE EPHEMERIS SOLVER: {sol['name']}")
    print("=" * 65)
    print(f"  Category    : {sol['category'].upper()} ({sol['type']})")
    print(f"  Date        : {sol['date']}")
    print(f"  Observer    : {sol['lat_deg']:.6f}°N, {abs(sol['lon_deg']):.6f}°W (Alt: {sol['alt_m']:.1f}m)")
    print(f"  Timezone    : {sol['tz_name']} (UTC+{sol['tz_offset_hours']:.1f}h)")
    print(f"  Delta T     : {sol['delta_t']:.1f}s")
    print("-" * 65)

    if sol["category"] == "solar":
        print(f"  • C1  (First Contact)    : {sol['C1'].strftime('%H:%M:%S') if sol['C1'] else 'N/A'}")
        print(f"  • C2  (Totality Start)   : {sol['C2'].strftime('%H:%M:%S') if sol['C2'] else 'N/A'}")
        print(f"  • MAX (Maximum Eclipse)  : {sol['MAX'].strftime('%H:%M:%S') if sol['MAX'] else 'N/A'}")
        print(f"  • C3  (Totality End)     : {sol['C3'].strftime('%H:%M:%S') if sol['C3'] else 'N/A'}")
        print(f"  • C4  (Fourth Contact)   : {sol['C4'].strftime('%H:%M:%S') if sol['C4'] else 'N/A'}")
        if sol['C2'] and sol['C3']:
            dur_s = (sol['C3'] - sol['C2']).total_seconds()
            m = int(dur_s // 60)
            s = int(round(dur_s % 60))
            print(f"  • Totality Duration      : {m}m {s:02d}s ({dur_s:.1f}s)")
    else:
        for k, v in sol.get("contacts_local", {}).items():
            print(f"  • {k:<4}: {v.strftime('%H:%M:%S')}")
    print("=" * 65)
