"""Report a detected victim to the Round-1 grader.

Each call appends one JSON line to $KIT_REPORT_PATH (default /student/output/reports.jsonl),
so reports written before a timeout or crash still count.

    from victim_report import report_person
    report_person(north_m=-42.0, east_m=-61.5, sim_s=f.sim, confidence=0.81)
    report_person(lat=37.41210, lon=-121.99920)

Coordinates: north_m/east_m are metres from the takeoff point (same frame as LOCAL_POSITION_NED
x/y), or lat/lon in degrees. Pass sim_s (PX4 time_boot_ms / 1000) so late reports can be rejected.
"""
import json
import math
import os
from pathlib import Path

DEFAULT_PATH = '/student/output/reports.jsonl'
HOME_LAT = 37.412173
HOME_LON = -121.998878
METERS_PER_DEG_LAT = 111319.49
METERS_PER_DEG_LON = 111319.49 * math.cos(math.radians(HOME_LAT))


def report_person(north_m=None, east_m=None, lat=None, lon=None, sim_s=None, confidence=None, frame_id=None, sim_time_s=None):
    if (north_m is None or east_m is None) and (lat is None or lon is None):
        raise ValueError('need north_m+east_m or lat+lon')
    if sim_s is None and sim_time_s is not None:
        sim_s = sim_time_s
    if north_m is not None and east_m is not None:
        if lat is None:
            lat = HOME_LAT + (float(north_m) / METERS_PER_DEG_LAT)
        if lon is None:
            lon = HOME_LON + (float(east_m) / METERS_PER_DEG_LON)
    elif lat is not None and lon is not None:
        if north_m is None:
            north_m = (float(lat) - HOME_LAT) * METERS_PER_DEG_LAT
        if east_m is None:
            east_m = (float(lon) - HOME_LON) * METERS_PER_DEG_LON
    item = {k: v for k, v in dict(north_m=north_m, east_m=east_m, lat=lat, lon=lon, sim_s=sim_s,
                                  confidence=confidence, frame_id=frame_id).items() if v is not None}
    path = Path(os.environ.get('KIT_REPORT_PATH') or
                (DEFAULT_PATH if Path('/student').is_dir() else 'output/reports.jsonl'))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as f:
        f.write(json.dumps(item, default=float) + '\n')
        f.flush()
        os.fsync(f.fileno())
    return item
