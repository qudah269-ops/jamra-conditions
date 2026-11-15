"""التوقّع من الظروف: موديل بيتعلّم الظروف، بلا تسريب بين السنين (من JAMRA tests/test_judge.py)."""
import math
from datetime import date

import numpy as np
import pandas as pd

import jamra_conditions.model as cn

CELLS = cn.area_cells()[:6]


def synth(seed=1):
    """طقس 2021-11 → 2025 لست خلايا، والحريق بيصير بس بالأيام الحارة الجافة."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2021-11-01", "2025-12-31").date
    w, f = [], []
    for c in CELLS:
        doy = np.array([d.timetuple().tm_yday for d in days])
        t = 25 + 12 * np.sin(2 * np.pi * (doy - 110) / 365) + rng.normal(0, 3, len(days))
        rh = np.clip(60 - (t - 25) * 2 + rng.normal(0, 8, len(days)), 3, 100)
        rain = np.where(rng.random(len(days)) < 0.1, rng.gamma(2, 5, len(days)), 0.0)
        w.append(pd.DataFrame({"date": [str(d) for d in days], "temperature_2m_max": t, "relative_humidity_2m_min": rh,
                               "wind_speed_10m_max": rng.uniform(5, 30, len(days)), "precipitation_sum": rain, "region": c}))
        fire = (t > 33) & (rh < 25) & (rng.random(len(days)) < 0.6)
        lat, lon = cn.h3.cell_to_latlng(c)
        f.append(pd.DataFrame({"region": c, "date": np.array(days)[fire], "n_viirs": 3, "n_modis": 1,
                               "first_utc": 1030, "lat": lat, "lon": lon}))
    return pd.concat(w, ignore_index=True), pd.concat(f, ignore_index=True)


def test_model_learns_conditions_and_replays_without_leak():
    w, f = synth()
    m = cn.Conditions(w, f)
    assert m.eval["auc"]["model"] > 0.85 > m.eval["auc"]["climate"] - 0.2
    o = m.outlook(date(2024, 7, 20), bbox=(-180, -90, 180, 90))
    assert o["mode"] == "replay" and o["hit"]["fires"] >= 0
    t = m.table
    assert t["p_oof"].notna().all()                       # كل يوم بيتوقّعه موديل ما شاف سنته
    hot = t[(t.tmax > 35) & (t.rhmin < 20)]["p_oof"].mean()
    cold = t[t.tmax < 20]["p_oof"].mean()
    assert hot > 5 * cold
    r = o["regions"][0]
    assert all(a["date"] < "2024-07-20" for a in r["analogs"])   # الحالات السابقة من الماضي بس
    assert r["analogs"] and r["analogs"][0]["sat_time"] == "13:30"   # 10:30 UTC → 13:30 بالأردن


def test_ffdi_matches_mcarthur_scalar():
    # JAMRA weather.py's scalar McArthur formula, inlined: the array version must agree with it
    df = max(0.3, min(10.0, 10.0 * math.exp(-4.0 / 25.0)))
    want = 2.0 * math.exp(-0.450 + 0.987 * math.log(df) - 0.0345 * 15 + 0.0338 * 35 + 0.0234 * 20)
    assert abs(cn.ffdi(np.array([35.0]), np.array([15.0]), np.array([20.0]), 4.0)[0] - want) < 1e-9


def test_peak_hours_picks_hot_dry_afternoon():
    times = [f"2026-07-01T{h:02d}:00" for h in range(24)]
    t = [20 + (10 if 13 <= h <= 15 else 0) for h in range(24)]
    rh = [50 - (30 if 13 <= h <= 15 else 0) for h in range(24)]
    assert [x["hour"] for x in cn.peak_hours(times, t, rh, [10] * 24, 0.0)] == ["13:00", "14:00", "15:00"]
