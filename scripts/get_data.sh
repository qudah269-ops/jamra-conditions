#!/bin/sh
# 1) Daily weather 2021-11 -> 2025 for 118 H3 cells from Open-Meteo (free, no key; ~15 min, resumes).
# 2) Fire detections: NASA FIRMS archives ingested by FireCal into Parquet points
#    (needs the firecal repo next to this one, or FIRECAL_DIR=...).
set -eu
uv sync -q
uv run jamra-conditions build --out data/weather_cells.parquet
FIRECAL="${FIRECAL_DIR:-../firecal}"
[ -d "$FIRECAL" ] || { echo "weather done; for the fire points clone the firecal repo next to this one or set FIRECAL_DIR"; exit 0; }
POINTS="$(cd "$FIRECAL" && pwd)/data_real/points"
(cd "$FIRECAL" && uv sync -q && uv run python scripts/download_firms.py --out data/raw_firms \
   --countries Jordan Palestine Israel Lebanon Syria Iraq --years 2021 2022 2023 2024 2025 \
 && uv run firecal --data data_real ingest "data/raw_firms/**/*.csv")
echo "done. evaluate with:"
echo "  uv run jamra-conditions eval --points '$POINTS/**/*.parquet'"
