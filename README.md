# jamra-conditions — fire probability from the day's weather

Team TFC · NASA Space Apps 2026 · part of [JAMRA](https://jamra.tfs.ink).

[العربية ↓](#العربية)

## What it does
For every H3 cell (resolution 4, ~40 km; 118 cells over Jordan and its neighbours) and every day, it gives the
probability that satellites see a vegetation fire there, from the weather of that day:

- **Features:** max temperature, min humidity, max wind, rain, 20-day rain, days since rain, McArthur FFDI,
  month (sin/cos) and the cell's historical fire rate for that month.
- **Model:** scikit-learn `HistGradientBoostingClassifier`.
- **Honest scoring:** leave-one-year-out — every day is scored by a model that never saw its year, and the
  climatology feature is also rebuilt without that year.
- **Outlook:** a past day → *replay* against what really happened; a day up to 7 days ahead → live forecast from
  Open-Meteo, with the 3 most dangerous hours (hourly FFDI) and **analog cases**: earlier fires in the same cell
  under similar weather (date, satellite overpass time, place).

A fire-day = any vegetation detection (`type 0`, confidence not low) from MODIS or VIIRS in the cell.

## Results (2021-11 → 2025, leave-one-year-out)
| | AUC | share of fire-days in the top 10 % of predictions |
|---|---|---|
| model | **0.933** | **74 %** (×7.4 chance) |
| climatology alone | 0.933 | 73 % |
| FFDI alone | 0.38 | — |

Replay of 2024-06-17 (Ajloun/Burqush fire): 3 of Jordan's 4 fires were in the top 8 cells.

**Read this honestly:** climatology alone scores about the same. Most fires in this region are agricultural burning
at fixed places and seasons, not hot dry days, so the weather adds little on top of "where and when it usually burns".

## Install
```bash
uv sync
uv run pytest -q
```

## Minimal example
```python
from datetime import date
import pandas as pd
import jamra_conditions as jc

weather = pd.read_parquet("data/weather_cells.parquet")              # jamra-conditions build
fires = jc.fire_days("../firecal/data_real/points/**/*.parquet")     # FireCal ingest output
m = jc.Conditions(weather, fires)                                   # ~20 s
print(m.eval)                                                       # AUC + top-10 % capture
o = m.outlook(date(2024, 6, 17))                                    # replay; a future date -> live forecast
for r in o["regions"][:3]:
    print(r["lat"], r["lon"], r["prob"], r["actual"])
```
CLI: `uv run jamra-conditions build` (weather) and `uv run jamra-conditions eval --points '<glob>'`.

## Data
No data is committed. `scripts/get_data.sh` downloads the daily weather (Open-Meteo archive, no key, ~15 min,
resumes after a break) and, with the firecal repo next to this one, the NASA FIRMS archives ingested to Parquet.

Sources: **NASA FIRMS** (MODIS C6.1, VIIRS S-NPP / NOAA-20), **Open-Meteo** archive (ERA5) and forecast API.

## Honest limits
- Weather adds little over climatology here (see above); FFDI alone is worse than chance on this region's fires.
- The time in analog cases is the satellite **overpass** time, not the ignition time: polar orbiters pass twice a day.
- One small region (Jordan and neighbours), four years of weather. Other regions need their own weather build.
- Local time assumes UTC+3 (Jordan, all year since 2022).
- The live forecast needs network access to Open-Meteo; if it fails, `outlook` returns `mode: "unavailable"`.

---

<div dir="rtl">

## العربية

# jamra-conditions — احتمال الحريق من طقس اليوم

### ماذا يفعل
لكل خلية H3 (دقة 4، نحو 40 كم؛ 118 خلية على الأردن وجواره) ولكل يوم، يعطي احتمال أن تسجّل الأقمار حريقاً نباتياً فيها،
من طقس ذلك اليوم: الحرارة، الرطوبة، الريح، المطر، مطر 20 يوماً، أيام بلا مطر، FFDI، الشهر، والمعدّل التاريخي للخلية.
القياس «سنة برّا»: كل يوم يتوقّعه نموذج لم يرَ سنته.
ليوم ماضٍ يعطي **إعادة** مع ما حدث فعلاً، وليوم قادم (حتى 7 أيام) **توقّعاً حياً** من Open-Meteo مع أخطر 3 ساعات
و**حالات سابقة** بظروف مشابهة.

### النتائج (2021-11 → 2025)
| | AUC | من أيام الحريق ضمن أعلى 10% |
|---|---|---|
| النموذج | **0.933** | **74%** (×7.4 الحظ) |
| المناخ وحده | 0.933 | 73% |
| FFDI وحده | 0.38 | — |

إعادة يوم 17-06-2024: 3 من 4 حرائق في الأردن كانت ضمن أعلى 8 أماكن.

**بصراحة:** المناخ وحده يعطي تقريباً نفس الرقم. معظم حرائق المنطقة حرق زراعي بأماكن ومواسم ثابتة، فالطقس يضيف قليلاً.

### التثبيت
```bash
uv sync
uv run pytest -q
```

### البيانات
لا نرفع بيانات. `scripts/get_data.sh` ينزّل الطقس اليومي من Open-Meteo (بلا مفتاح)، ومع مستودع firecal بجانبه
ينزّل أرشيف NASA FIRMS ويحوّله إلى Parquet.
المصادر: NASA FIRMS، وOpen-Meteo (ERA5 والتوقّع).

### حدود بصراحة
- الطقس يضيف قليلاً فوق المناخ، وFFDI وحده أسوأ من الحظ على حرائق هذه المنطقة.
- الساعة في الحالات السابقة هي ساعة مرور القمر لا ساعة الاشتعال.
- منطقة صغيرة وأربع سنوات طقس فقط.
- التوقيت المحلي مفترض UTC+3.
- التوقّع الحي يحتاج اتصالاً بـ Open-Meteo.

</div>
