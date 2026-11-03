"""Download public play-by-play files once into the shared data cache (outside the repository)."""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

CACHE = Path(os.environ.get("SCALEPROBIT_DATA", Path.home() / "projects" / "oss" / "_data"))
SDV = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download/pbp"
SOURCES = {
    "nba_espn": lambda y: f"{SDV}/espn_nba_pbp/play_by_play_{y}.parquet",
    "wnba_espn": lambda y: f"{SDV}/espn_wnba_pbp/play_by_play_{y}.parquet",
    "nfl_pbp": lambda y: f"{NFLVERSE}/play_by_play_{y}.parquet",
}


def fetch(source: str, season: int) -> Path:
    """Return the cached path of one season's file, downloading it if missing."""
    path = CACHE / source / f"pbp_{season}.parquet"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(SOURCES[source](season), tmp)
        tmp.rename(path)
    return path
