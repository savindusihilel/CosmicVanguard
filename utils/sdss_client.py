"""
SDSS SkyServer DR18 async client for live galaxy data queries.

Provides search by objID/coordinates, random sampling, cone search,
and main sequence background data from the MPA-JHU catalog.
"""

import asyncio
import math
import time
import urllib.parse
from collections import OrderedDict
from typing import Optional

import httpx

SKYSERVER_BASE = "https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch"

# Shared SQL fragments
_GALAXY_COLUMNS = """
    p.objID, p.ra, p.dec,
    p.modelMag_u AS u, p.modelMag_g AS g, p.modelMag_r AS r,
    p.modelMag_i AS i, p.modelMag_z AS z_mag,
    s.z AS redshift, s.zErr,
    gse.lgm_tot_p50 AS true_mass,
    gse.sfr_tot_p50 AS true_sfr
""".strip()

_GALAXY_JOIN = """
    FROM PhotoObjAll AS p
    JOIN SpecObjAll AS s ON p.objID = s.bestObjID
    LEFT JOIN galSpecExtra AS gse ON s.specObjID = gse.specObjID
""".strip()

_GALAXY_WHERE = "WHERE s.class = 'GALAXY' AND s.zWarning = 0"


def compute_absolute_magnitude(r_mag: float, redshift: float) -> float:
    """Approximate absolute magnitude M_r — must match main.py exactly."""
    c = 3e5   # km/s
    H0 = 70   # km/s/Mpc
    d_mpc = (c / H0) * redshift
    d_pc = d_mpc * 1e6
    if d_pc <= 0:
        return -20.0
    return r_mag - 5 * math.log10(d_pc) + 5


def _assemble_features(row: dict) -> list:
    """
    Build the 10-dimensional feature vector expected by the PINN.
    Order: [u, g, r, i, z, g-r, u-g, r-i, Mr, redshift]
    """
    u = float(row["u"])
    g = float(row["g"])
    r = float(row["r"])
    i = float(row["i"])
    z = float(row["z_mag"])
    redshift = float(row["redshift"])
    Mr = compute_absolute_magnitude(r, redshift)
    return [u, g, r, i, z, g - r, u - g, r - i, Mr, redshift]


def _safe_float(val, lo=-999, hi=999):
    """Convert to float or return None if outside valid range."""
    if val is None:
        return None
    try:
        v = float(val)
        if v <= lo or v > hi:
            return None
        return v
    except (TypeError, ValueError):
        return None


def _parse_galaxy_row(row: dict) -> Optional[dict]:
    """Convert a raw SDSS SQL result row into a galaxy object."""
    try:
        features = _assemble_features(row)
    except (TypeError, ValueError, KeyError):
        return None

    true_mass = _safe_float(row.get("true_mass"), lo=0, hi=15)
    true_sfr = _safe_float(row.get("true_sfr"), lo=-99, hi=5)

    return {
        "objID": str(row["objID"]),
        "ra": float(row["ra"]),
        "dec": float(row["dec"]),
        "u": float(row["u"]),
        "g": float(row["g"]),
        "r": float(row["r"]),
        "i": float(row["i"]),
        "z_mag": float(row["z_mag"]),
        "redshift": float(row["redshift"]),
        "features": features,
        "true_mass": true_mass,
        "true_sfr": true_sfr,
        "source": "SDSS DR18",
    }


# ---------------------------------------------------------------------------
# Simple TTL + LRU cache
# ---------------------------------------------------------------------------

class _TTLCache:
    """In-memory LRU cache with per-entry TTL."""

    def __init__(self, maxsize: int = 500, ttl_seconds: int = 3600):
        self._maxsize = maxsize
        self._ttl = ttl_seconds
        self._store: OrderedDict = OrderedDict()

    def get(self, key: str):
        if key in self._store:
            ts, value = self._store[key]
            if time.time() - ts < self._ttl:
                self._store.move_to_end(key)
                return value
            else:
                del self._store[key]
        return None

    def set(self, key: str, value):
        if key in self._store:
            del self._store[key]
        self._store[key] = (time.time(), value)
        while len(self._store) > self._maxsize:
            self._store.popitem(last=False)


# ---------------------------------------------------------------------------
# SDSS Client
# ---------------------------------------------------------------------------

class SDSSClient:
    """Async client for SDSS SkyServer DR18 SQL queries."""

    def __init__(self):
        self._cache = _TTLCache(maxsize=500, ttl_seconds=3600)
        self._ms_cache = _TTLCache(maxsize=1, ttl_seconds=3600)
        self._last_request_time = 0.0
        self._rate_limit_delay = 1.0  # seconds between requests

    # ---- internal helpers ----

    async def _rate_limit(self):
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._rate_limit_delay:
            await asyncio.sleep(self._rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    async def _execute_query(self, sql: str, retries: int = 3) -> list:
        """Execute SQL against SkyServer with exponential-backoff retry."""
        await self._rate_limit()

        params = {"cmd": sql, "format": "json"}

        for attempt in range(retries):
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.get(SKYSERVER_BASE, params=params)

                if resp.status_code == 200:
                    data = resp.json()
                    # SkyServer returns [{"TableName":..., "Rows":[...]}]
                    if isinstance(data, list) and len(data) > 0:
                        first = data[0]
                        if isinstance(first, dict):
                            return first.get("Rows", [])
                        return data
                    if isinstance(data, dict):
                        return data.get("Rows", [])
                    return []

                if resp.status_code >= 500:
                    wait = (2 ** attempt) * 0.5
                    await asyncio.sleep(wait)
                    continue

                # 4xx — don't retry
                return []

            except (httpx.TimeoutException, httpx.ConnectError):
                if attempt < retries - 1:
                    wait = (2 ** attempt) * 0.5
                    await asyncio.sleep(wait)
                    continue
                raise

        return []

    # ---- public API ----

    async def search(self, query: str) -> Optional[dict]:
        """
        Search for a galaxy by objID or 'RA,Dec' string.
        Returns a single galaxy dict or None.
        """
        query = query.strip()

        cached = self._cache.get(f"search:{query}")
        if cached is not None:
            return cached

        if "," in query:
            # RA,Dec — cone search with tiny radius
            parts = query.split(",")
            try:
                ra = float(parts[0].strip())
                dec = float(parts[1].strip())
            except (ValueError, IndexError):
                return None

            sql = f"""
                SELECT TOP 1 {_GALAXY_COLUMNS}
                FROM dbo.fGetNearbyObjEq({ra}, {dec}, 0.5) AS nb
                JOIN PhotoObjAll AS p ON nb.objID = p.objID
                JOIN SpecObjAll AS s ON p.objID = s.bestObjID
                LEFT JOIN galSpecExtra AS gse ON s.specObjID = gse.specObjID
                {_GALAXY_WHERE}
            """
        else:
            try:
                obj_id = int(query)
            except ValueError:
                return None

            sql = f"""
                SELECT TOP 1 {_GALAXY_COLUMNS}
                {_GALAXY_JOIN}
                {_GALAXY_WHERE}
                AND p.objID = {obj_id}
            """

        rows = await self._execute_query(sql)
        if not rows:
            return None

        galaxy = _parse_galaxy_row(rows[0])
        if galaxy:
            self._cache.set(f"search:{query}", galaxy)
        return galaxy

    async def random(
        self, n: int = 20, z_min: float = 0.01, z_max: float = 0.3
    ) -> list:
        """Fetch n random galaxies within a redshift range."""
        n = min(max(n, 1), 100)

        sql = f"""
            SELECT TOP {n} {_GALAXY_COLUMNS}
            {_GALAXY_JOIN}
            {_GALAXY_WHERE}
            AND s.z BETWEEN {z_min} AND {z_max}
            AND p.modelMag_r BETWEEN 14 AND 21
            ORDER BY NEWID()
        """

        rows = await self._execute_query(sql)
        galaxies = []
        for row in rows:
            g = _parse_galaxy_row(row)
            if g:
                galaxies.append(g)
        return galaxies

    async def cone_search(
        self, ra: float, dec: float, radius_arcmin: float = 5.0, n: int = 50
    ) -> list:
        """Cone search around RA/Dec. Deduplicates by objID (smallest zErr)."""
        radius_arcmin = min(max(radius_arcmin, 0.1), 10.0)
        n = min(max(n, 1), 100)

        sql = f"""
            SELECT TOP {n}
                p.objID, p.ra, p.dec,
                p.modelMag_u AS u, p.modelMag_g AS g, p.modelMag_r AS r,
                p.modelMag_i AS i, p.modelMag_z AS z_mag,
                s.z AS redshift, s.zErr,
                gse.lgm_tot_p50 AS true_mass,
                gse.sfr_tot_p50 AS true_sfr
            FROM dbo.fGetNearbyObjEq({ra}, {dec}, {radius_arcmin}) AS nb
            JOIN PhotoObjAll AS p ON nb.objID = p.objID
            JOIN SpecObjAll AS s ON p.objID = s.bestObjID
            LEFT JOIN galSpecExtra AS gse ON s.specObjID = gse.specObjID
            {_GALAXY_WHERE}
        """

        rows = await self._execute_query(sql)

        # Deduplicate by objID — keep the entry with smallest zErr
        seen: dict = {}
        for row in rows:
            oid = str(row.get("objID", ""))
            try:
                z_err = float(row.get("zErr", 999))
            except (TypeError, ValueError):
                z_err = 999.0
            if oid not in seen or z_err < seen[oid][1]:
                seen[oid] = (row, z_err)

        galaxies = []
        for row, _ in seen.values():
            g = _parse_galaxy_row(row)
            if g:
                galaxies.append(g)
        return galaxies

    async def get_main_sequence(self) -> dict:
        """
        Fetch main sequence (mass vs SFR) from MPA-JHU catalog.
        Cached server-side for 1 hour.
        """
        cached = self._ms_cache.get("main_sequence")
        if cached is not None:
            return cached

        sql = """
            SELECT TOP 1500
                gse.lgm_tot_p50 AS mass,
                gse.sfr_tot_p50 AS sfr
            FROM galSpecExtra AS gse
            JOIN SpecObjAll AS s ON gse.specObjID = s.specObjID
            WHERE s.class = 'GALAXY' AND s.zWarning = 0
                AND gse.lgm_tot_p50 > 0
                AND gse.sfr_tot_p50 > -99
                AND s.z BETWEEN 0.02 AND 0.15
            ORDER BY NEWID()
        """

        rows = await self._execute_query(sql)
        mass_list, sfr_list = [], []
        for row in rows:
            try:
                m = float(row["mass"])
                s = float(row["sfr"])
                if 0 < m < 15 and -10 < s < 5:
                    mass_list.append(round(m, 4))
                    sfr_list.append(round(s, 4))
            except (TypeError, ValueError, KeyError):
                continue

        result = {
            "mass": mass_list,
            "sfr": sfr_list,
            "source": "MPA-JHU DR8 (via DR18)",
        }
        if mass_list:
            self._ms_cache.set("main_sequence", result)
        return result
