"""
jamra_conditions.model (JAMRA conditions.py) — توقّع الحريق من الظروف

فريق TFC · NASA Space Apps 2026

الفكرة: كل يوم صار فيه حريق من 2021-11 لآخر 2025، الأقمار سجّلته (6 مليون كشف)، وOpen-Meteo
بيعطينا الطقس بنفس اليوم بنفس المنطقة. فبنتعلّم: بأي ظروف (حرارة، رطوبة، ريح، أيام بلا مطر،
الشهر) بتشتعل كل منطقة. وبعدين لأي يوم جاي (توقّع الطقس) أو ماضي (إعادة)، بنعطي:
  · احتمال الحريق لكل منطقة،
  · أخطر الساعات من الطقس بالساعة،
  · حالات سابقة بنفس الظروف: متى صارت، الساعة اللي شافها فيها القمر، ووين.

الوحدة: خلية H3 بدقة 4 (حوالي 40 كم) على الأردن وجواره: 118 خلية، ولكل خلية طقسها.
جرّبنا مناطق الطقس الواسعة (دقة 2) أول: المنطقة بتغطّي شمال الأردن وفلسطين ولبنان مع بعض،
فبيطلع فيها حريق تقريباً كل يوم، وهاد ما بيوجّه درون.
الحريق باليوم = أي كشف نباتي (type 0) بثقة غير منخفضة من MODIS أو VIIRS بالخلية.

الطقس التاريخي: `jamra-conditions build` بينزّله مرة وحدة لـ data/weather_cells.parquet.

⚠ الساعة بالحالات السابقة هي ساعة مرور القمر، مش ساعة الاشتعال: الأقمار القطبية بتمرّ مرتين
  باليوم بس. عشان هيك «أخطر الساعات» جاية من الطقس بالساعة، مش من الأقمار.
"""
from __future__ import annotations

import json
import math
import urllib.request

import numpy as np
import pandas as pd
from h3.api import basic_int as h3

FEATS = ["tmax", "rhmin", "wind", "rain", "rain20", "dry", "ffdi", "msin", "mcos", "clim", "prev1", "nb_prev1"]
# prev1 / nb_prev1: a vegetation fire in this cell / a neighbouring cell the day before. Measured year-out
# (JAMRA experiments/conditions_v2.py): top-10% capture 74.2% → 75.6%, AUC 0.933 → 0.937, better in every year 2021–2025.
RES = 4
AREA = (34.7, 29.0, 39.5, 33.6)   # lon0, lat0, lon1, lat1: الأردن وجواره
START, END = "2021-11-01", "2025-12-31"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
TZ_H = 3                     # الأردن: UTC+3 طول السنة من 2022
FORECAST = "https://api.open-meteo.com/v1/forecast"
DAILY = "temperature_2m_max,relative_humidity_2m_min,wind_speed_10m_max,precipitation_sum"
HOURLY = "temperature_2m,relative_humidity_2m,wind_speed_10m"


def region(lat, lon):
    return h3.latlng_to_cell(lat, lon, RES)


def area_cells(bbox=AREA):
    lon0, lat0, lon1, lat1 = bbox
    return sorted(h3.polygon_to_cells(h3.LatLngPoly([(lat0, lon0), (lat0, lon1), (lat1, lon1), (lat1, lon0)]), RES))


def _get(url, timeout=120, tries=4):
    import time
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.load(r)
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(10 * (k + 1))


def _multi(base, cells, extra):
    ll = [h3.cell_to_latlng(c) for c in cells]
    d = _get(f"{base}?latitude={','.join(f'{a:.3f}' for a, _ in ll)}"
             f"&longitude={','.join(f'{b:.3f}' for _, b in ll)}&{extra}")
    return d if isinstance(d, list) else [d]


def build_weather(path, cells=None, batch=6, pause=30):
    """طقس يومي 2021-11 → 2025 لكل خلية (Open-Meteo archive، بلا مفتاح). بيكمّل من وين وقف."""
    import os
    import time
    cells = cells or area_cells()
    have = pd.read_parquet(path) if os.path.exists(path) else None
    done = set(have["region"]) if have is not None else set()
    rows = [have] if have is not None else []
    todo = [c for c in cells if c not in done]
    for i in range(0, len(todo), batch):
        part = todo[i:i + batch]
        for c, d in zip(part, _multi(ARCHIVE, part, f"start_date={START}&end_date={END}&daily={DAILY}&timezone=GMT")):
            rows.append(pd.DataFrame(d["daily"]).rename(columns={"time": "date"}).assign(region=c))
        pd.concat(rows, ignore_index=True).to_parquet(path, index=False)
        print(f"{min(i + batch, len(todo))}/{len(todo)}", flush=True)
        time.sleep(pause)   # Open-Meteo بيحسب الطلب حسب حجمه: أسرع من هيك بيرجّع 429
    return pd.concat(rows, ignore_index=True)


def ffdi(t, rh, wind, rain20):
    """McArthur FFDI بنفس معادلة weather.py، على مصفوفات."""
    df = np.clip(10.0 * np.exp(-np.maximum(rain20, 0) / 25.0), 0.3, 10.0)
    return 2.0 * np.exp(-0.450 + 0.987 * np.log(df) - 0.0345 * rh + 0.0338 * t + 0.0234 * wind)


def fire_days(points_glob, start=START, bbox=AREA):
    """خلية × يوم من كشوفات الأقمار: عدد MODIS وVIIRS، أول مرور (UTC)، وأسخن نقطة."""
    import duckdb
    con = duckdb.connect()
    try:
        con.sql("LOAD h3")
    except duckdb.Error:  # first run on this machine, same as FireCal's db.connect
        con.sql("INSTALL h3 FROM community")
        con.sql("LOAD h3")
    return con.sql(f"""
        select h3_cell_to_parent(h3_fine, {RES})::bigint as region, acq_date::date as date,
               sum((instrument = 'VIIRS')::int) as n_viirs, sum((instrument = 'MODIS')::int) as n_modis,
               min(acq_time) as first_utc, arg_max(lat, frp) as lat, arg_max(lon, frp) as lon
        from read_parquet('{points_glob}', hive_partitioning = 1)
        where type = 0 and conf_class <> 'low' and acq_date >= '{start}'
              and lon between {bbox[0] - 0.5} and {bbox[2] + 0.5} and lat between {bbox[1] - 0.5} and {bbox[3] + 0.5}
        group by 1, 2""").df()


def static_spots(points_glob, bbox=AREA):
    """Where the archive saw only non-vegetation heat (FIRMS type 2 static land sources, 3 offshore): gas flares,
    factories. Live detections lack the type, so near these spots they must not count as yesterday's fire."""
    import duckdb
    return {(round(a, 2), round(b, 2)) for a, b in duckdb.sql(f"""
        select distinct round(lat, 2), round(lon, 2) from read_parquet('{points_glob}', hive_partitioning = 1)
        where type in (2, 3) and lon between {bbox[0] - 0.5} and {bbox[2] + 0.5}
              and lat between {bbox[1] - 0.5} and {bbox[3] + 0.5}""").fetchall()}


def burned_regions(dets, static=frozenset()):
    """Cells with a live vegetation-like detection: polar sensors only (the model was trained on MODIS/VIIRS),
    nothing within ~1 km (0.01°) of a known static heat source."""
    out = set()
    for d in dets:
        if "Meteosat" in d.get("sensor", ""):
            continue
        a, b = round(d["lat"], 2), round(d["lon"], 2)
        if any((round(a + i / 100, 2), round(b + j / 100, 2)) in static for i in (-1, 0, 1) for j in (-1, 0, 1)):
            continue
        out.add(region(d["lat"], d["lon"]))
    return out


def add_persistence(t, burned=None):
    """prev1 / nb_prev1 from the table's own fire column (history), or from `burned` cells (live day)."""
    if burned is not None:
        t["prev1"] = t["region"].map(lambda r: int(int(r) in burned))
        t["nb_prev1"] = t["region"].map(lambda r: int(any(n in burned for n in h3.grid_disk(int(r), 1) if n != int(r))))
        return t
    fired = set(zip(t.loc[t["fire"] == 1, "region"].astype(int), t.loc[t["fire"] == 1, "date"]))
    yday = (pd.to_datetime(t["date"]) - pd.Timedelta(days=1)).dt.date
    ring = {r: [n for n in h3.grid_disk(int(r), 1) if n != int(r)] for r in t["region"].unique()}
    t["prev1"] = [int((int(r), d) in fired) for r, d in zip(t["region"], yday)]
    t["nb_prev1"] = [int(any((n, d) in fired for n in ring[r])) for r, d in zip(t["region"], yday)]
    return t


def add_weather_features(w):
    """weather_daily (أو توقّع Open-Meteo بنفس الأعمدة) → ميزات الظروف."""
    w = w.rename(columns={"temperature_2m_max": "tmax", "relative_humidity_2m_min": "rhmin",
                          "wind_speed_10m_max": "wind", "precipitation_sum": "rain"}).copy()
    w["date"] = pd.to_datetime(w["date"]).dt.date
    w = w.sort_values(["region", "date"]).reset_index(drop=True)
    w[["tmax", "rhmin", "wind", "rain"]] = w[["tmax", "rhmin", "wind", "rain"]].astype(float)
    w["rain"] = w["rain"].fillna(0.0)
    w["rain20"] = w.groupby("region")["rain"].transform(lambda s: s.rolling(20, min_periods=1).sum())
    w["rain5"] = w.groupby("region")["rain"].transform(lambda s: s.rolling(5, min_periods=1).mean())   # the cause model's input
    wet = (w["rain"] > 1.0).astype(int).groupby(w["region"]).cumsum()
    w["dry"] = w.groupby([w["region"], wet]).cumcount().clip(upper=60)   # أيام من آخر مطر > 1 مم
    w["ffdi"] = ffdi(w["tmax"], w["rhmin"], w["wind"], w["rain20"])
    m = pd.to_datetime(w["date"]).dt.month
    w["month"] = m
    w["msin"], w["mcos"] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)
    return w


def _local_hhmm(first_utc):
    if first_utc is None or (isinstance(first_utc, float) and math.isnan(first_utc)):
        return None
    mins = (int(first_utc) // 100 * 60 + int(first_utc) % 100 + TZ_H * 60) % 1440
    return f"{mins // 60:02d}:{mins % 60:02d}"


class Conditions:
    """يتعلّم من الماضي. القياس والإعادة «سنة برّا»: كل سنة بيتوقّعها موديل ما شافها، فلا الأرقام ولا
    الإعادة فيها غش. والتوقّع الحي بيستعمل موديل متدرّب على كل السنين."""

    def __init__(self, weather_daily, fires):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self._hgb = HistGradientBoostingClassifier
        w = add_weather_features(weather_daily)
        f = fires.copy()
        f["date"] = pd.to_datetime(f["date"]).dt.date
        t = w.merge(f, on=["region", "date"], how="left")
        t[["n_viirs", "n_modis"]] = t[["n_viirs", "n_modis"]].fillna(0)
        t["fire"] = ((t["n_viirs"] + t["n_modis"]) > 0).astype(int)
        t = add_persistence(t)
        t["year"] = pd.to_datetime(t["date"]).dt.year
        t["p_oof"], t["clim_oof"] = np.nan, np.nan
        for y in sorted(t["year"].unique()):
            tr, te = t[t["year"] != y], t[t["year"] == y]
            clim, clim_month = self._climatology(tr)           # من باقي السنين بس: بدون تسريب
            te = self._with_clim(te, clim, clim_month)
            t.loc[te.index, "clim_oof"] = te["clim"]
            if tr["fire"].nunique() == 2:
                t.loc[te.index, "p_oof"] = self._fit(self._with_clim(tr, clim, clim_month)).predict_proba(te[FEATS])[:, 1]
        self.table = t
        self.eval = self._evaluate(t.dropna(subset=["p_oof"]))
        self.clim, self.clim_month = self._climatology(t)
        self.model = self._fit(self._with_clim(t))
        hist = t[t["fire"] == 1]
        self.scale = t[["tmax", "rhmin", "wind", "rain20"]].std().replace(0, 1)
        self.history = hist

    # ── التدريب والقياس ──────────────────────────────────────
    @staticmethod
    def _climatology(t):
        return t.groupby(["region", "month"])["fire"].mean(), t.groupby("month")["fire"].mean()

    def _with_clim(self, t, clim=None, clim_month=None):
        clim = self.clim if clim is None else clim
        clim_month = self.clim_month if clim_month is None else clim_month
        key = pd.MultiIndex.from_arrays([t["region"], t["month"]])
        c = pd.Series(clim.reindex(key).to_numpy(), index=t.index)
        return t.assign(clim=c.fillna(t["month"].map(clim_month)).fillna(0.0))

    def _fit(self, t):
        return self._hgb(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=7).fit(
            t[FEATS], t["fire"])

    @staticmethod
    def _evaluate(t):
        from sklearn.metrics import roc_auc_score
        y = t["fire"].to_numpy()
        if len(t) == 0 or len(set(y)) < 2:
            return None

        def top10(score):  # من أيام الحريق الحقيقية، كم واحد وقع بأعلى 10% من التوقعات
            k = max(1, int(len(score) * 0.10))
            return round(float(y[np.argsort(-np.asarray(score))[:k]].sum() / y.sum()), 3)

        def auc(score):
            return round(float(roc_auc_score(y, score)), 3)
        return {"method": "leave-one-year-out", "span": f"{t['date'].min()} → {t['date'].max()}",
                "cell_days": int(len(t)), "fire_days": int(y.sum()), "base_rate": round(float(y.mean()), 4),
                "auc": {"model": auc(t["p_oof"]), "ffdi": auc(t["ffdi"]), "climate": auc(t["clim_oof"])},
                "top10_capture": {"model": top10(t["p_oof"]), "ffdi": top10(t["ffdi"]),
                                  "climate": top10(t["clim_oof"]), "chance": 0.10}}

    # ── التوقع ───────────────────────────────────────────────
    def predict(self, rows):
        """صفوف فيها ميزات الطقس (add_weather_features) → احتمال الحريق."""
        r = self._with_clim(rows)
        return self.model.predict_proba(r[FEATS])[:, 1]

    def analogs(self, row, k=4):
        """حالات حريق سابقة (قبل هاد اليوم) بنفس الخلية وبظروف قريبة (± شهر)، الأقرب أولاً."""
        h = self.history[(self.history["region"] == row["region"]) & (self.history["date"] < row["date"])]
        h = h[((h["month"] - row["month"]) % 12).isin([0, 1, 11])]
        if h.empty:
            return []
        cols = ["tmax", "rhmin", "wind", "rain20"]
        d = (((h[cols] - row[cols].astype(float)) / self.scale) ** 2).sum(axis=1) ** 0.5
        out = []
        for i in d.nsmallest(k).index:
            a = h.loc[i]
            out.append({"date": str(a["date"]), "sat_time": _local_hhmm(a["first_utc"]),
                        "lat": round(float(a["lat"]), 4), "lon": round(float(a["lon"]), 4),
                        "n_viirs": int(a["n_viirs"]), "n_modis": int(a["n_modis"]),
                        "tmax": round(float(a["tmax"]), 1), "rhmin": int(a["rhmin"]),
                        "wind": round(float(a["wind"]), 1), "dry": int(a["dry"]), "similarity_d": round(float(d[i]), 2)})
        return out

    def regions_in(self, bbox):
        """الخلايا اللي مركزها جوّا المستطيل [lon0, lat0, lon1, lat1]."""
        lon0, lat0, lon1, lat1 = bbox
        return [c for c in sorted(self.table["region"].unique())
                if lat0 <= h3.cell_to_latlng(int(c))[0] <= lat1 and lon0 <= h3.cell_to_latlng(int(c))[1] <= lon1]

    def outlook(self, day, bbox=AREA, fetch=None, top=10, burned=None):
        """يوم بالماضي → إعادة من الطقس المحفوظ مع شو صار فعلاً. يوم جاي → توقّع Open-Meteo بالساعة.
        `cells` لكل الخلايا (للخريطة)، و`regions` أعلى `top` بالتفاصيل والحالات السابقة.
        `burned`: خلايا فيها حريق حي بآخر 24 ساعة (burned_regions)، بتصير «حريق مبارح» لليوم الجاي.
        shortcut: بعد بكرة كمان بياخد نفس آخر 24 ساعة؛ أدق شي بيكون لليوم وبكرة."""
        regs = self.regions_in(bbox)
        past = self.table[(self.table["date"] == day) & self.table["region"].isin(regs)]
        if not past.empty:
            rows, mode, hourly = past.reset_index(drop=True), "replay", {}
            p = rows["p_oof"].fillna(rows["clim_oof"]).to_numpy()      # موديل ما شاف هالسنة
        else:
            rows, hourly = (fetch or fetch_forecast)(regs, day)
            mode = "live"
            if rows is None or rows.empty:
                return {"date": str(day), "mode": "unavailable", "regions": [], "cells": [], "eval": self.eval}
            rows = add_persistence(rows, set(burned or ()))
            p = self.predict(rows)
        order = np.argsort(-p)
        cells = [{"region": str(int(rows.at[i, "region"])),
                  "lat": round(h3.cell_to_latlng(int(rows.at[i, "region"]))[0], 3),
                  "lon": round(h3.cell_to_latlng(int(rows.at[i, "region"]))[1], 3),
                  "prob": round(float(p[i]), 3)} for i in order]
        out = []
        for i in order[:top]:
            row = rows.loc[i]
            item = {**cells[len(out)], "ffdi": round(float(row["ffdi"]), 1),
                    "tmax": round(float(row["tmax"]), 1), "rhmin": int(row["rhmin"]),
                    "wind": round(float(row["wind"]), 1), "rain20": round(float(row["rain20"]), 1),
                    "rain": round(float(row["rain"]), 1), "rain5": round(float(row["rain5"]), 2),
                    "dry": int(row["dry"]), "peak_hours": hourly.get(int(row["region"]), []),
                    "analogs": self.analogs(row)}
            if mode == "replay":
                item["actual"] = {"fire": bool(row["fire"]), "n_viirs": int(row["n_viirs"]),
                                  "n_modis": int(row["n_modis"]), "sat_time": _local_hhmm(row.get("first_utc"))}
            out.append(item)
        res = {"date": str(day), "mode": mode, "regions": out, "cells": cells, "eval": self.eval}
        if mode == "replay":   # كم خلية من اللي اشتعلت فعلاً كانت بأعلى `top` توقّع
            fired = rows["fire"].to_numpy().astype(bool)
            res["hit"] = {"fires": int(fired.sum()), "in_top": int(fired[order[:top]].sum()), "top": int(min(top, len(rows)))}
        return res


def peak_hours(times, t, rh, wind, rain20, k=3):
    """أعلى k ساعات خطورة (FFDI بالساعة) بهاد اليوم، مرتّبة بالوقت."""
    f = ffdi(np.asarray(t, float), np.asarray(rh, float), np.asarray(wind, float), rain20)
    idx = sorted(np.argsort(-f)[:k])
    return [{"hour": times[i][-5:], "ffdi": round(float(f[i]), 1)} for i in idx]


def fetch_forecast(regions, day):
    """Open-Meteo (مجاني بلا مفتاح): 20 يوم مضت + 7 جاية، يومي وبالساعة، بطلب واحد لكل الخلايا."""
    if not regions:
        return None, {}
    try:
        data = _multi(FORECAST, regions, f"daily={DAILY}&hourly={HOURLY}&past_days=20&forecast_days=7&timezone=auto",)
    except Exception:
        return None, {}
    w = add_weather_features(pd.concat(
        [pd.DataFrame(d["daily"]).rename(columns={"time": "date"}).assign(region=reg) for reg, d in zip(regions, data)],
        ignore_index=True))
    w = w[w["date"] == day].dropna(subset=["tmax", "rhmin", "wind"]).reset_index(drop=True)
    hourly, r20 = {}, dict(zip(w["region"], w["rain20"]))
    for reg, d in zip(regions, data):
        h = d["hourly"]
        sel = [i for i, s_ in enumerate(h["time"]) if s_.startswith(str(day)) and h["temperature_2m"][i] is not None]
        if sel and reg in r20:
            hourly[reg] = peak_hours([h["time"][i] for i in sel], [h["temperature_2m"][i] for i in sel],
                                     [h["relative_humidity_2m"][i] for i in sel],
                                     [h["wind_speed_10m"][i] for i in sel], float(r20[reg]))
    return w, hourly


def main(argv=None):
    """build: download daily weather per cell (Open-Meteo archive, ~15 min, resumes).
    eval:  train with leave-one-year-out and print the honest scores."""
    import argparse
    p = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", default="data/weather_cells.parquet")
    e = sub.add_parser("eval")
    e.add_argument("--weather", default="data/weather_cells.parquet")
    e.add_argument("--points", required=True, help="FireCal points parquet glob, e.g. 'data_real/points/**/*.parquet'")
    a = p.parse_args(argv)
    if a.cmd == "build":
        import os
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        print(len(build_weather(a.out)), "rows ->", a.out)
    else:
        m = Conditions(pd.read_parquet(a.weather), fire_days(a.points))
        print(json.dumps(m.eval, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
