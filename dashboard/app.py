
from __future__ import annotations

import asyncio
import html
import json
import os
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import parse_qs, quote, urlencode

import asyncpg
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials


DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
DASHBOARD_USERNAME = os.getenv("DASHBOARD_USERNAME", "admin").strip()
DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "").strip()
ALLIANCE_NAME = os.getenv("TRACKER_ALLIANCE_NAME", "ROYAL SOLDIERS").strip()
TRACKER_SERVER = os.getenv("TRACKER_SERVER", "SK1").strip()
TRACKER_API_BASE = os.getenv("TRACKER_API_BASE", "https://api.gge-tracker.com/api/v1").rstrip("/")
TRACKER_SYNC_INTERVAL_SECONDS = int(os.getenv("TRACKER_SYNC_INTERVAL_SECONDS", "1800"))
TRACKER_USER_AGENT = os.getenv(
    "TRACKER_USER_AGENT",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
)

PERIODS: dict[str, str] = {
    "24h": "24h",
    "7d": "7 dní",
    "30d": "30 dní",
    "all": "Celé obdobie",
}
POWER_PERIODS: dict[str, tuple[str, timedelta]] = {
    "24h": ("24h", timedelta(hours=24)),
    "7d": ("7 dní", timedelta(days=7)),
    "30d": ("30 dní", timedelta(days=30)),
}
EVENT_TYPES: dict[str, dict[str, str]] = {
    "nomadi": {
        "title": "Nomádi",
        "nav": "Nomádi",
        "tracker_key": "player_event_nomad_history",
        "emoji": "🏕️",
    },
    "samuraji": {
        "title": "Samuraji",
        "nav": "Samuraji",
        "tracker_key": "player_event_samurai_history",
        "emoji": "⛩️",
    },
    "cudzinci": {
        "title": "Cudzinci",
        "nav": "Cudzinci",
        "tracker_key": "player_event_war_realms_history",
        "emoji": "🛡️",
    },
    "vrany": {
        "title": "Vrany",
        "nav": "Vrany",
        "tracker_key": "player_event_bloodcrow_history",
        "emoji": "🩸",
    },
}
ADMIN_REPORTS_PER_PAGE = 10

security = HTTPBasic()
pool: asyncpg.Pool | None = None
tracker_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool, tracker_task
    if DATABASE_URL:
        pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
        await ensure_dashboard_tables()
        tracker_task = asyncio.create_task(tracker_sync_loop())
    yield
    if tracker_task is not None:
        tracker_task.cancel()
        try:
            await tracker_task
        except asyncio.CancelledError:
            pass
        tracker_task = None
    if pool is not None:
        await pool.close()
        pool = None


app = FastAPI(title="GGE Report Dashboard", lifespan=lifespan)


def require_auth(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    if not DASHBOARD_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Set DASHBOARD_PASSWORD in Railway Variables first.",
        )
    username_ok = secrets.compare_digest(credentials.username, DASHBOARD_USERNAME)
    password_ok = secrets.compare_digest(credentials.password, DASHBOARD_PASSWORD)
    if not (username_ok and password_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


def ensure_pool() -> asyncpg.Pool:
    if pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DATABASE_URL is missing or database connection is not ready.",
        )
    return pool


async def ensure_dashboard_tables() -> None:
    if pool is None:
        return
    async with pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS discord_members (
                guild_id BIGINT NOT NULL,
                user_id BIGINT NOT NULL,
                display_name TEXT NOT NULL,
                username TEXT,
                is_bot BOOLEAN NOT NULL DEFAULT FALSE,
                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (guild_id, user_id)
            )
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tracker_players (
                server TEXT NOT NULL,
                player_id BIGINT NOT NULL,
                player_name TEXT NOT NULL,
                alliance_id BIGINT,
                alliance_name TEXT NOT NULL,
                alliance_rank INTEGER,
                level INTEGER,
                legendary_level INTEGER,
                might_current BIGINT NOT NULL DEFAULT 0,
                might_all_time BIGINT NOT NULL DEFAULT 0,
                loot_current BIGINT NOT NULL DEFAULT 0,
                honor BIGINT NOT NULL DEFAULT 0,
                tracker_updated_at TIMESTAMPTZ,
                synced_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (server, player_id)
            )
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tracker_power_snapshots (
                server TEXT NOT NULL,
                player_id BIGINT NOT NULL,
                player_name TEXT NOT NULL,
                alliance_name TEXT NOT NULL,
                might_points BIGINT NOT NULL,
                recorded_at TIMESTAMPTZ NOT NULL,
                source TEXT NOT NULL DEFAULT 'tracker',
                PRIMARY KEY (server, player_id, recorded_at)
            )
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS tracker_power_snapshots_player_time_idx
            ON tracker_power_snapshots (server, player_id, recorded_at DESC)
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tracker_loot_snapshots (
                server TEXT NOT NULL,
                player_id BIGINT NOT NULL,
                player_name TEXT NOT NULL,
                alliance_name TEXT NOT NULL,
                loot_points BIGINT NOT NULL,
                recorded_at TIMESTAMPTZ NOT NULL,
                source TEXT NOT NULL DEFAULT 'tracker',
                PRIMARY KEY (server, player_id, recorded_at)
            )
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS tracker_loot_snapshots_player_time_idx
            ON tracker_loot_snapshots (server, player_id, recorded_at DESC)
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tracker_event_snapshots (
                server TEXT NOT NULL,
                event_key TEXT NOT NULL,
                player_id BIGINT NOT NULL,
                player_name TEXT NOT NULL,
                alliance_name TEXT NOT NULL,
                event_points BIGINT NOT NULL,
                recorded_at TIMESTAMPTZ NOT NULL,
                source TEXT NOT NULL DEFAULT 'tracker',
                PRIMARY KEY (server, event_key, player_id, recorded_at)
            )
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS tracker_event_snapshots_event_player_time_idx
            ON tracker_event_snapshots (server, event_key, player_id, recorded_at DESC)
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tracker_event_occurrences (
                server TEXT NOT NULL,
                event_key TEXT NOT NULL,
                started_at TIMESTAMPTZ NOT NULL,
                ended_at TIMESTAMPTZ NOT NULL,
                sample_player_id BIGINT,
                sample_points BIGINT NOT NULL DEFAULT 0,
                synced_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (server, event_key, started_at, ended_at)
            )
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS tracker_event_occurrences_event_time_idx
            ON tracker_event_occurrences (server, event_key, started_at DESC)
            """
        )


def tracker_json(path: str) -> Any:
    url = f"{TRACKER_API_BASE}{path}"
    headers = {
        "User-Agent": TRACKER_USER_AGENT,
        "Accept": "application/json",
        "Referer": "https://docs.gge-tracker.com/",
        "gge-server": TRACKER_SERVER,
    }
    req = urlrequest.Request(url, headers=headers)
    with urlrequest.urlopen(req, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


async def tracker_json_async(path: str) -> Any:
    return await asyncio.to_thread(tracker_json, path)


def parse_tracker_time(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


async def sync_tracker_power() -> dict[str, int]:
    """Fetch ROYAL SOLDIERS from GGE Tracker and store current + historic power/loot points."""
    db = ensure_pool()
    alliance_path = f"/alliances/name/{quote(ALLIANCE_NAME)}"
    alliance = await tracker_json_async(alliance_path)
    alliance_id = to_int(alliance.get("alliance_id"))
    if not alliance_id:
        raise RuntimeError(f"GGE Tracker did not return alliance_id for {ALLIANCE_NAME!r}.")

    detail = await tracker_json_async(f"/alliances/id/{alliance_id}")
    players = detail.get("players") or []
    history: list[dict[str, Any]] = []
    loot_history: list[dict[str, Any]] = []
    event_histories: dict[str, list[dict[str, Any]]] = {
        config["tracker_key"]: [] for config in EVENT_TYPES.values()
    }
    event_occurrences: dict[str, list[dict[str, Any]]] = {
        config["tracker_key"]: [] for config in EVENT_TYPES.values()
    }
    try:
        stats = await tracker_json_async(f"/statistics/alliance/{alliance_id}")
        points = stats.get("points") or {}
        history = points.get("player_might_history") or []
        loot_history = points.get("player_loot_history") or []
        for event_key in event_histories:
            event_histories[event_key] = points.get(event_key) or []
    except (urlerror.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"[DASHBOARD] Tracker history sync failed: {type(exc).__name__}: {exc}")

    sample_player_id = next((to_int(player.get("player_id")) for player in players if to_int(player.get("player_id"))), 0)
    if sample_player_id:
        for event_key in event_occurrences:
            try:
                occurrence_data = await tracker_json_async(
                    f"/statistics/player/{sample_player_id}/{event_key}/occurrences"
                )
                event_occurrences[event_key] = occurrence_data.get("occurrences") or []
            except (urlerror.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
                print(
                    f"[DASHBOARD] Tracker occurrence sync failed for {event_key}: "
                    f"{type(exc).__name__}: {exc}"
                )

    current_time = utc_now()
    async with db.acquire() as conn:
        async with conn.transaction():
            for player in players:
                player_id = to_int(player.get("player_id"))
                if not player_id:
                    continue
                tracker_updated_at = parse_tracker_time(player.get("updated_at"))
                await conn.execute(
                    """
                    INSERT INTO tracker_players (
                        server, player_id, player_name, alliance_id, alliance_name,
                        alliance_rank, level, legendary_level, might_current, might_all_time,
                        loot_current, honor, tracker_updated_at, synced_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, NOW())
                    ON CONFLICT (server, player_id) DO UPDATE SET
                        player_name = EXCLUDED.player_name,
                        alliance_id = EXCLUDED.alliance_id,
                        alliance_name = EXCLUDED.alliance_name,
                        alliance_rank = EXCLUDED.alliance_rank,
                        level = EXCLUDED.level,
                        legendary_level = EXCLUDED.legendary_level,
                        might_current = EXCLUDED.might_current,
                        might_all_time = EXCLUDED.might_all_time,
                        loot_current = EXCLUDED.loot_current,
                        honor = EXCLUDED.honor,
                        tracker_updated_at = EXCLUDED.tracker_updated_at,
                        synced_at = NOW()
                    """,
                    TRACKER_SERVER,
                    player_id,
                    str(player.get("player_name") or player_id),
                    alliance_id,
                    detail.get("alliance_name") or ALLIANCE_NAME,
                    to_int(player.get("alliance_rank"), -1),
                    to_int(player.get("level")),
                    to_int(player.get("legendary_level")),
                    to_int(player.get("might_current")),
                    to_int(player.get("might_all_time")),
                    to_int(player.get("loot_current")),
                    to_int(player.get("honor")),
                    tracker_updated_at,
                )
                await conn.execute(
                    """
                    INSERT INTO tracker_power_snapshots (
                        server, player_id, player_name, alliance_name, might_points, recorded_at, source
                    ) VALUES ($1, $2, $3, $4, $5, $6, 'current')
                    ON CONFLICT (server, player_id, recorded_at) DO NOTHING
                    """,
                    TRACKER_SERVER,
                    player_id,
                    str(player.get("player_name") or player_id),
                    detail.get("alliance_name") or ALLIANCE_NAME,
                    to_int(player.get("might_current")),
                    tracker_updated_at or current_time,
                )
                await conn.execute(
                    """
                    INSERT INTO tracker_loot_snapshots (
                        server, player_id, player_name, alliance_name, loot_points, recorded_at, source
                    ) VALUES ($1, $2, $3, $4, $5, $6, 'current')
                    ON CONFLICT (server, player_id, recorded_at) DO NOTHING
                    """,
                    TRACKER_SERVER,
                    player_id,
                    str(player.get("player_name") or player_id),
                    detail.get("alliance_name") or ALLIANCE_NAME,
                    to_int(player.get("loot_current")),
                    tracker_updated_at or current_time,
                )

            for point in history:
                player_id = to_int(point.get("player_id"))
                recorded_at = parse_tracker_time(point.get("date"))
                might_points = to_int(point.get("point"), -1)
                if not player_id or recorded_at is None or might_points < 0:
                    continue
                await conn.execute(
                    """
                    INSERT INTO tracker_power_snapshots (
                        server, player_id, player_name, alliance_name, might_points, recorded_at, source
                    ) VALUES (
                        $1, $2,
                        COALESCE((SELECT player_name FROM tracker_players WHERE server = $1 AND player_id = $2), $3),
                        $4, $5, $6, 'history'
                    )
                    ON CONFLICT (server, player_id, recorded_at) DO UPDATE SET
                        might_points = EXCLUDED.might_points,
                        player_name = EXCLUDED.player_name,
                        alliance_name = EXCLUDED.alliance_name,
                        source = EXCLUDED.source
                    """,
                    TRACKER_SERVER,
                    player_id,
                    str(player_id),
                    detail.get("alliance_name") or ALLIANCE_NAME,
                    might_points,
                    recorded_at,
                )

            for point in loot_history:
                player_id = to_int(point.get("player_id"))
                recorded_at = parse_tracker_time(point.get("date"))
                loot_points = to_int(point.get("point"), -1)
                if not player_id or recorded_at is None or loot_points < 0:
                    continue
                await conn.execute(
                    """
                    INSERT INTO tracker_loot_snapshots (
                        server, player_id, player_name, alliance_name, loot_points, recorded_at, source
                    ) VALUES (
                        $1, $2,
                        COALESCE((SELECT player_name FROM tracker_players WHERE server = $1 AND player_id = $2), $3),
                        $4, $5, $6, 'history'
                    )
                    ON CONFLICT (server, player_id, recorded_at) DO UPDATE SET
                        loot_points = EXCLUDED.loot_points,
                        player_name = EXCLUDED.player_name,
                        alliance_name = EXCLUDED.alliance_name,
                        source = EXCLUDED.source
                    """,
                    TRACKER_SERVER,
                    player_id,
                    str(player_id),
                    detail.get("alliance_name") or ALLIANCE_NAME,
                    loot_points,
                    recorded_at,
                )

            for event_key, event_history in event_histories.items():
                for point in event_history:
                    player_id = to_int(point.get("player_id"))
                    recorded_at = parse_tracker_time(point.get("date"))
                    event_points = to_int(point.get("point"), -1)
                    if not player_id or recorded_at is None or event_points < 0:
                        continue
                    await conn.execute(
                        """
                        INSERT INTO tracker_event_snapshots (
                            server, event_key, player_id, player_name, alliance_name,
                            event_points, recorded_at, source
                        ) VALUES (
                            $1, $2, $3,
                            COALESCE((SELECT player_name FROM tracker_players WHERE server = $1 AND player_id = $3), $4),
                            $5, $6, $7, 'history'
                        )
                        ON CONFLICT (server, event_key, player_id, recorded_at) DO UPDATE SET
                            event_points = EXCLUDED.event_points,
                            player_name = EXCLUDED.player_name,
                            alliance_name = EXCLUDED.alliance_name,
                            source = EXCLUDED.source
                        """,
                        TRACKER_SERVER,
                        event_key,
                        player_id,
                        str(player_id),
                        detail.get("alliance_name") or ALLIANCE_NAME,
                        event_points,
                        recorded_at,
                    )

            for event_key, occurrences in event_occurrences.items():
                for occurrence in occurrences:
                    started_at = parse_tracker_time(occurrence.get("started_at"))
                    ended_at = parse_tracker_time(occurrence.get("ended_at"))
                    if started_at is None or ended_at is None:
                        continue
                    await conn.execute(
                        """
                        INSERT INTO tracker_event_occurrences (
                            server, event_key, started_at, ended_at,
                            sample_player_id, sample_points, synced_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, NOW())
                        ON CONFLICT (server, event_key, started_at, ended_at) DO UPDATE SET
                            sample_player_id = EXCLUDED.sample_player_id,
                            sample_points = EXCLUDED.sample_points,
                            synced_at = NOW()
                        """,
                        TRACKER_SERVER,
                        event_key,
                        started_at,
                        ended_at,
                        sample_player_id,
                        to_int(occurrence.get("point")),
                    )
    return {
        "players": len(players),
        "history": len(history),
        "loot_history": len(loot_history),
        "event_history": sum(len(points) for points in event_histories.values()),
        "event_occurrences": sum(len(points) for points in event_occurrences.values()),
    }


async def tracker_sync_loop() -> None:
    while True:
        try:
            result = await sync_tracker_power()
            print(
                f"[DASHBOARD] Synced GGE Tracker power: {result['players']} players, "
                f"{result['history']} power history points, "
                f"{result.get('loot_history', 0)} loot history points, "
                f"{result.get('event_history', 0)} event history points, "
                f"{result.get('event_occurrences', 0)} event occurrences"
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"[DASHBOARD] Tracker sync failed: {type(exc).__name__}: {exc}")
        await asyncio.sleep(max(300, TRACKER_SYNC_INTERVAL_SECONDS))


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def fmt_number(value: int | None) -> str:
    return f"{int(value or 0):,}".replace(",", " ")


def fmt_ratio(losses: int | None, kills: int | None) -> str:
    losses = int(losses or 0)
    kills = int(kills or 0)
    if losses == 0:
        return "1:∞" if kills else "1:1.00"
    return f"1:{kills / losses:.2f}"


def ratio_score(losses: int | None, kills: int | None) -> float:
    losses = int(losses or 0)
    kills = int(kills or 0)
    if losses <= 0:
        return float(kills) if kills else 1.0
    return kills / losses


def period_label(period: str, date_from: str | None = None, date_to: str | None = None) -> str:
    if date_from or date_to:
        if date_from and date_to:
            return f"{date_from} – {date_to}"
        return f"od {date_from}" if date_from else f"do {date_to}"
    return PERIODS.get(period, PERIODS["all"])


def get_filters(request: Request) -> tuple[str, str | None, str | None]:
    period = request.query_params.get("period", "all")
    if period not in PERIODS:
        period = "all"
    return period, request.query_params.get("from") or None, request.query_params.get("to") or None


def add_period_filter(
    clauses: list[str],
    params: list[Any],
    *,
    period: str = "all",
    date_from: str | None = None,
    date_to: str | None = None,
) -> None:
    if date_from:
        params.append(date_from)
        clauses.append(f"created_at >= ${len(params)}::date")
    if date_to:
        params.append(date_to)
        clauses.append(f"created_at < (${len(params)}::date + INTERVAL '1 day')")
    if date_from or date_to:
        return
    if period == "24h":
        clauses.append("created_at >= NOW() - INTERVAL '24 hours'")
    elif period == "7d":
        clauses.append("created_at >= NOW() - INTERVAL '7 days'")
    elif period == "30d":
        clauses.append("created_at >= NOW() - INTERVAL '30 days'")


def where_sql(clauses: list[str]) -> str:
    return "WHERE " + " AND ".join(clauses) if clauses else ""


def query_string(period: str, date_from: str | None, date_to: str | None, **extra: str) -> str:
    query: dict[str, str] = {"period": period}
    if date_from:
        query["from"] = date_from
    if date_to:
        query["to"] = date_to
    query.update({key: value for key, value in extra.items() if value})
    return urlencode(query)


def sort_header(
    label: str,
    *,
    base: str,
    params: dict[str, str],
    key: str,
    active_sort: str,
    direction: str,
    numeric: bool = False,
) -> str:
    next_dir = "asc" if active_sort == key and direction == "desc" else "desc"
    merged = {**params, "sort": key, "dir": next_dir}
    arrow = " ↓" if active_sort == key and direction == "desc" else " ↑" if active_sort == key else ""
    active = " active" if active_sort == key else ""
    th_class = ' class="num"' if numeric else ""
    return (
        f"<th{th_class}>"
        f'<a class="sort-link{active}" href="{esc(base)}?{urlencode(merged)}">{esc(label)}{esc(arrow)}</a>'
        "</th>"
    )


def parse_form_body(raw: bytes) -> dict[str, str]:
    parsed = parse_qs(raw.decode("utf-8"), keep_blank_values=True)
    return {key: values[-1].strip() for key, values in parsed.items()}


async def fetch_dashboard_data(*, period: str, date_from: str | None, date_to: str | None) -> dict[str, Any]:
    db = ensure_pool()
    clauses: list[str] = []
    params: list[Any] = []
    add_period_filter(clauses, params, period=period, date_from=date_from, date_to=date_to)
    where = where_sql(clauses)

    async with db.acquire() as conn:
        alliance = await conn.fetchrow(
            f"""
            SELECT COUNT(*) AS report_count,
                   COALESCE(SUM(own_losses), 0) AS total_losses,
                   COALESCE(SUM(enemy_kills), 0) AS total_kills,
                   COUNT(DISTINCT player_id) AS player_count
            FROM battle_reports
            {where}
            """,
            *params,
        )
        leaderboard = await conn.fetch(
            f"""
            SELECT player_id,
                   MAX(player_name) AS player_name,
                   COUNT(*) AS report_count,
                   COALESCE(SUM(own_losses), 0) AS total_losses,
                   COALESCE(SUM(enemy_kills), 0) AS total_kills
            FROM battle_reports
            {where}
            GROUP BY player_id
            ORDER BY total_kills DESC, total_losses ASC, report_count DESC, player_name ASC
            LIMIT 50
            """,
            *params,
        )
        recent = await conn.fetch(
            """
            SELECT id, guild_id, message_id, attachment_id, player_id, player_name,
                   own_losses, enemy_kills, created_at
            FROM battle_reports
            ORDER BY created_at DESC
            LIMIT 20
            """
        )
        chart = await conn.fetch(
            f"""
            SELECT created_at::date AS day,
                   COUNT(*) AS report_count,
                   COALESCE(SUM(own_losses), 0) AS total_losses,
                   COALESCE(SUM(enemy_kills), 0) AS total_kills
            FROM battle_reports
            {where}
            GROUP BY created_at::date
            ORDER BY day ASC
            LIMIT 90
            """,
            *params,
        )
        blacklist_count = await conn.fetchval("SELECT COUNT(*) FROM battle_report_blacklist")

    return {
        "alliance": dict(alliance),
        "leaderboard": [dict(row) for row in leaderboard],
        "recent": [dict(row) for row in recent],
        "chart": [dict(row) for row in chart],
        "blacklist_count": int(blacklist_count or 0),
    }


async def fetch_player_data(
    *, player_id: int, period: str, date_from: str | None, date_to: str | None
) -> dict[str, Any]:
    db = ensure_pool()
    clauses = ["player_id = $1"]
    params: list[Any] = [player_id]
    add_period_filter(clauses, params, period=period, date_from=date_from, date_to=date_to)
    where = where_sql(clauses)
    async with db.acquire() as conn:
        summary = await conn.fetchrow(
            f"""
            SELECT MAX(player_name) AS player_name,
                   COUNT(*) AS report_count,
                   COALESCE(SUM(own_losses), 0) AS total_losses,
                   COALESCE(SUM(enemy_kills), 0) AS total_kills
            FROM battle_reports
            {where}
            """,
            *params,
        )
        reports = await conn.fetch(
            f"""
            SELECT id, guild_id, message_id, attachment_id, player_id, player_name,
                   own_losses, enemy_kills, created_at
            FROM battle_reports
            {where}
            ORDER BY created_at DESC
            LIMIT 50
            """,
            *params,
        )
        chart = await conn.fetch(
            f"""
            SELECT created_at::date AS day,
                   COUNT(*) AS report_count,
                   COALESCE(SUM(own_losses), 0) AS total_losses,
                   COALESCE(SUM(enemy_kills), 0) AS total_kills
            FROM battle_reports
            {where}
            GROUP BY created_at::date
            ORDER BY day ASC
            LIMIT 90
            """,
            *params,
        )
    return {"summary": dict(summary), "reports": [dict(row) for row in reports], "chart": [dict(row) for row in chart]}


async def fetch_admin_data(*, page: int = 1, report_limit: int = ADMIN_REPORTS_PER_PAGE) -> dict[str, Any]:
    db = ensure_pool()
    page = max(page, 1)
    report_limit = max(1, report_limit)
    offset = (page - 1) * report_limit
    async with db.acquire() as conn:
        reports = await conn.fetch(
            """
            SELECT id, guild_id, message_id, attachment_id, player_id, player_name,
                   own_losses, enemy_kills, created_at
            FROM battle_reports
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
            """,
            report_limit,
            offset,
        )
        report_count = await conn.fetchval("SELECT COUNT(*) FROM battle_reports")
        players = await conn.fetch(
            """
            WITH report_totals AS (
                SELECT guild_id, player_id, MAX(player_name) AS player_name,
                       COUNT(*) AS report_count,
                       COALESCE(SUM(own_losses), 0) AS total_losses,
                       COALESCE(SUM(enemy_kills), 0) AS total_kills
                FROM battle_reports
                GROUP BY guild_id, player_id
            )
            SELECT
                COALESCE(dm.guild_id, rt.guild_id) AS guild_id,
                COALESCE(dm.user_id, rt.player_id) AS player_id,
                COALESCE(NULLIF(dm.display_name, ''), rt.player_name, dm.username, rt.player_id::text) AS player_name,
                COALESCE(rt.report_count, 0) AS report_count,
                COALESCE(rt.total_losses, 0) AS total_losses,
                COALESCE(rt.total_kills, 0) AS total_kills,
                COALESCE(dm.is_active, TRUE) AS is_active
            FROM discord_members dm
            FULL OUTER JOIN report_totals rt
              ON rt.guild_id = dm.guild_id AND rt.player_id = dm.user_id
            WHERE COALESCE(dm.is_bot, FALSE) = FALSE
              AND COALESCE(dm.is_active, TRUE) = TRUE
            ORDER BY LOWER(COALESCE(NULLIF(dm.display_name, ''), rt.player_name, dm.username, rt.player_id::text)) ASC
            LIMIT 500
            """
        )
        blacklist = await conn.fetch(
            """
            SELECT guild_id, own_losses, enemy_kills, source_message_id,
                   blacklisted_by_name, created_at
            FROM battle_report_blacklist
            ORDER BY created_at DESC
            LIMIT 100
            """
        )
    return {
        "reports": [dict(row) for row in reports],
        "report_count": int(report_count or 0),
        "page": page,
        "per_page": report_limit,
        "players": [dict(row) for row in players],
        "blacklist": [dict(row) for row in blacklist],
    }


def layout(title: str, body: str, *, active: str = "dashboard") -> str:
    nav = [
        ("dashboard", "/", "Dashboard"),
        ("might", "/?tab=moc", "Moc"),
        ("loot", "/?tab=rabovanie", "Rabovanie"),
        ("admin", "/admin", "Admin"),
    ]
    nav_html = "".join(
        f'<a class="nav-link {"active" if key == active else ""}" href="{href}">{label}</a>'
        for key, href, label in nav
    )
    event_active = active in EVENT_TYPES
    event_links = "".join(
        f'<a class="nav-sub-link {"active" if slug == active else ""}" href="/?tab={esc(slug)}"><span>{esc(config["emoji"])}</span>{esc(config["nav"])}</a>'
        for slug, config in EVENT_TYPES.items()
    )
    events_nav = f'<details class="nav-group" {"open" if event_active else ""}><summary class="nav-link {"active" if event_active else ""}">⚔️ Eventy <span class="nav-chevron">⌄</span></summary><div class="nav-menu">{event_links}</div></details>'
    nav_html = nav_html.replace(f'<a class="nav-link {"active" if active == "admin" else ""}" href="/admin">Admin</a>', events_nav + f'<a class="nav-link {"active" if active == "admin" else ""}" href="/admin">Admin</a>')
    return f"""<!doctype html>
<html lang="sk">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(title)}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #080a0f; --panel: rgba(20,24,35,.88); --panel2: rgba(30,36,52,.94);
      --line: rgba(255,255,255,.10); --text: #f7f3e8; --muted: #9ca3af;
      --gold: #f5c451; --green: #7dd87d; --red: #ff7575; --shadow: 0 18px 50px rgba(0,0,0,.38);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin:0; min-height:100vh; font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top left, rgba(245,196,81,.18), transparent 34rem),
                  radial-gradient(circle at 70% 10%, rgba(130,182,255,.11), transparent 28rem),
                  linear-gradient(180deg, #0b0e15, #080a0f 45%, #06070b);
      color:var(--text);
    }}
    a {{ color:inherit; }}
    .shell {{ width:min(1480px, calc(100% - 48px)); margin:0 auto; padding:24px 0 48px; }}
    .app-frame {{ display:grid; grid-template-columns:232px minmax(0,1fr); gap:28px; align-items:start; }}
    .sidebar {{ position:sticky; top:24px; display:flex; flex-direction:column; gap:24px; padding:18px 14px; min-height:calc(100vh - 72px); background:rgba(14,18,28,.86); border:1px solid var(--line); border-radius:24px; box-shadow:var(--shadow); }}
    .workspace {{ min-width:0; }}
    .topbar {{ display:flex; justify-content:space-between; align-items:center; gap:16px; position:sticky; top:0; z-index:10; margin-bottom:24px; padding:10px 4px 16px; backdrop-filter:blur(18px); }}
    .topbar-context {{ display:grid; gap:4px; }}
    .topbar-context strong {{ font-size:18px; letter-spacing:-.02em; }}
    .eyebrow {{ color:var(--gold); font-size:11px; font-weight:900; letter-spacing:.14em; text-transform:uppercase; }}
    .brand {{ display:flex; align-items:center; gap:12px; text-decoration:none; }}
    .crest {{ width:42px; height:42px; border-radius:12px; display:grid; place-items:center; background:linear-gradient(135deg,#f5c451,#7b4a12); color:#1a1205; box-shadow:var(--shadow); font-size:24px; }}
    .brand strong {{ display:block; font-size:17px; }} .brand span span {{ color:var(--muted); font-size:13px; }}
    .nav,.periods,.actions,.mini-form {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; }}
    .sidebar .nav {{ display:grid; gap:6px; align-content:start; }}
    .nav-link,.btn,.tab {{ border:1px solid var(--line); background:rgba(255,255,255,.04); color:var(--text); text-decoration:none; border-radius:12px; padding:10px 13px; font-weight:800; font-size:14px; }}
    .nav-link,.btn,.tab {{ transition:background .18s ease, border-color .18s ease, transform .18s ease; }}
    .nav-link:hover,.btn:hover,.tab:hover {{ border-color:rgba(245,196,81,.42); transform:translateY(-1px); }}
    .nav-link.active,.btn.primary,.tab.active {{ background:linear-gradient(135deg,var(--gold),#e19b31); color:#1d1405; border-color:rgba(245,196,81,.65); }}
    .nav-group {{ position:relative; width:100%; }}
    .nav-group summary {{ cursor:pointer; list-style:none; user-select:none; }}
    .nav-group summary::-webkit-details-marker {{ display:none; }}
    .nav-chevron {{ display:inline-block; margin-left:4px; font-size:16px; transition:transform .2s ease; }}
    .nav-group[open] .nav-chevron {{ transform:rotate(180deg); }}
    .nav-menu {{ display:grid; gap:4px; margin:5px 0 2px 10px; padding-left:8px; border-left:1px solid rgba(245,196,81,.25); }}
    .nav-sub-link {{ display:flex; align-items:center; gap:9px; padding:10px 11px; border-radius:10px; color:var(--muted); text-decoration:none; font-size:14px; font-weight:800; }}
    .nav-sub-link:hover,.nav-sub-link.active {{ color:var(--text); background:rgba(245,196,81,.14); }}
    .nav-sub-link.active {{ color:var(--gold); }}
    .hero {{ display:grid; grid-template-columns:1.2fr .8fr; gap:18px; align-items:stretch; margin:8px 0 18px; }}
    .hero-card,.panel,.card {{ background:linear-gradient(180deg,var(--panel2),var(--panel)); border:1px solid var(--line); border-radius:22px; box-shadow:var(--shadow); }}
    .hero-card {{ padding:24px; overflow:hidden; position:relative; }}
    h1 {{ margin:0; font-size:clamp(30px,5vw,54px); line-height:1.02; letter-spacing:-.04em; }}
    h2 {{ margin:0 0 12px; font-size:20px; }} h3 {{ margin:0 0 10px; font-size:16px; color:var(--gold); }}
    .subtitle {{ color:var(--muted); margin-top:10px; max-width:720px; font-size:15px; line-height:1.5; }}
    .periods {{ margin-top:18px; }} .tab {{ color:var(--muted); }} .tab.active {{ color:#1d1405; }}
    .filters {{ display:grid; gap:10px; align-content:start; padding:20px; }}
    label {{ display:grid; gap:6px; color:var(--muted); font-size:13px; font-weight:800; }}
    input,select {{ width:100%; border:1px solid var(--line); background:rgba(0,0,0,.26); color:var(--text); border-radius:12px; padding:10px 11px; font:inherit; }}
    button {{ cursor:pointer; }} .danger {{ color:#fff; background:rgba(255,80,80,.18); border-color:rgba(255,80,80,.35); }}
    .cards {{ display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:12px; margin-bottom:18px; }}
    .card {{ padding:16px; min-height:105px; }} .card span {{ color:var(--muted); font-size:13px; font-weight:800; }} .card strong {{ display:block; margin-top:10px; font-size:clamp(22px,3vw,31px); white-space:nowrap; }}
    .grid,.admin-grid {{ display:grid; grid-template-columns:1.35fr .9fr; gap:18px; align-items:start; }}
    .panel {{ overflow:hidden; }} .panel-head {{ padding:18px 18px 0; display:flex; justify-content:space-between; gap:12px; align-items:center; }} .panel-body {{ padding:18px; }}
    table {{ width:100%; border-collapse:collapse; }} th,td {{ padding:12px 14px; border-top:1px solid var(--line); text-align:left; white-space:nowrap; }}
    th {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.08em; }} td.num,th.num {{ text-align:right; }} tr:hover td {{ background:rgba(255,255,255,.025); }}
    .sort-link {{ color:var(--muted); text-decoration:none; display:inline-flex; align-items:center; gap:5px; }}
    .sort-link.active {{ color:var(--gold); }}
    .sort-link:hover {{ color:var(--text); }}
    .rank {{ width:34px; height:34px; border-radius:11px; display:inline-grid; place-items:center; background:rgba(255,255,255,.06); font-weight:900; }} .rank.top {{ background:linear-gradient(135deg,var(--gold),#a76c18); color:#1b1204; }}
    .player-link {{ color:var(--text); text-decoration:none; font-weight:900; }} .player-link:hover {{ color:var(--gold); }}
    .pill {{ display:inline-flex; align-items:center; gap:6px; border:1px solid var(--line); border-radius:999px; padding:5px 9px; color:var(--muted); background:rgba(255,255,255,.035); font-size:12px; font-weight:800; }}
    .good {{ color:var(--green); }} .bad {{ color:var(--red); }} .empty {{ color:var(--muted); text-align:center!important; padding:28px; }}
    .bars {{ display:flex; align-items:end; gap:6px; height:190px; padding-top:8px; }} .bar-wrap {{ flex:1; display:grid; align-content:end; gap:4px; min-width:8px; }}
    .bar {{ border-radius:9px 9px 3px 3px; min-height:3px; background:linear-gradient(180deg,var(--gold),#8b5b19); }} .bar.loss {{ background:linear-gradient(180deg,#ff7777,#7f2424); opacity:.72; }}
    .bar-label {{ color:var(--muted); font-size:10px; writing-mode:vertical-rl; transform:rotate(180deg); justify-self:center; max-height:44px; overflow:hidden; }}
    .recent-list {{ display:grid; gap:10px; }} .report-item {{ display:grid; grid-template-columns:1fr auto; gap:10px; border-top:1px solid var(--line); padding:12px 0; }} .report-item:first-child {{ border-top:0; }}
    .report-meta {{ color:var(--muted); font-size:13px; margin-top:3px; }} .mini-form input {{ width:150px; padding:8px 9px; border-radius:10px; font-size:13px; }}
    .notice {{ margin:0 0 16px; padding:12px 14px; border-radius:14px; background:rgba(125,216,125,.12); border:1px solid rgba(125,216,125,.25); color:#caffe0; }}
    details.panel {{ padding:0; }} summary.panel-head {{ cursor:pointer; list-style:none; }} summary.panel-head::-webkit-details-marker {{ display:none; }}
    .pagination {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; justify-content:flex-end; padding:0 18px 18px; }}
    .page-link {{ min-width:38px; text-align:center; border:1px solid var(--line); background:rgba(255,255,255,.04); color:var(--text); text-decoration:none; border-radius:12px; padding:8px 11px; font-weight:900; }}
    .page-link.active {{ background:linear-gradient(135deg,var(--gold),#e19b31); color:#1d1405; border-color:rgba(245,196,81,.65); }}
    footer {{ color:var(--muted); margin-top:22px; font-size:13px; }}
    @media (min-width:1051px) {{
      .shell {{ width:min(1480px, calc(100% - 64px)); padding-top:26px; }}
      .brand {{ padding:4px 7px; }}
      .brand strong {{ font-size:18px; letter-spacing:.01em; }}
      .nav {{ justify-content:flex-start; gap:9px; }}
      .nav-link {{ padding:11px 13px; }}
      .hero-card {{ padding:28px; }}
      .panel-head {{ padding:20px 22px 0; }}
      th,td {{ padding:14px 16px; }}
      .cards {{ gap:14px; }}
    }}
    @media (max-width:1050px) {{ .app-frame {{ display:block; }} .sidebar {{ position:static; min-height:0; margin-bottom:18px; padding:14px; }} .sidebar .nav {{ display:flex; }} .sidebar .nav-link {{ border-radius:999px; }} .nav-menu {{ position:absolute; left:0; top:calc(100% + 6px); margin:0; padding:7px; min-width:190px; background:rgba(20,24,35,.98); border:1px solid var(--line); border-radius:16px; box-shadow:var(--shadow); z-index:20; }} .hero,.grid,.admin-grid {{ grid-template-columns:1fr; }} .cards {{ grid-template-columns:repeat(3,minmax(0,1fr)); }} }}
    @media (max-width:680px) {{ .shell {{ width:min(100% - 18px,1320px); padding-top:12px; }} .topbar {{ align-items:flex-start; flex-direction:column; }} .cards {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .panel {{ overflow-x:auto; }} th,td {{ padding:10px; }} .report-item {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <main class="shell"><div class="app-frame">
    <aside class="sidebar">
      <a class="brand" href="/"><span class="crest">♛</span><span><strong>ROYAL SOLDIERS</strong><span>Reporty · ratio · power</span></span></a>
      <nav class="nav">{nav_html}</nav>
    </aside>
    <section class="workspace">
      <header class="topbar"><div class="topbar-context"><span class="eyebrow">Royal Soldiers · GGE tracker</span><strong>{esc(title)}</strong></div><span class="pill">read-only</span></header>
      {body}
      <footer>Public mód je read-only. Admin akcie sú chránené heslom a zapisujú priamo do rovnakej databázy ako Discord bot.</footer>
    </section>
  </div></main>
  </main>
  <script>
    document.addEventListener("submit", function (event) {{
      const form = event.target;
      if (!(form instanceof HTMLFormElement)) return;
      const message = form.dataset.confirm;
      if (message && !window.confirm(message)) {{
        event.preventDefault();
      }}
    }});
  </script>
</body>
</html>"""


def render_cards(alliance: dict[str, Any], blacklist_count: int) -> str:
    total_losses = int(alliance["total_losses"] or 0)
    total_kills = int(alliance["total_kills"] or 0)
    cards = [
        ("Reporty", fmt_number(alliance["report_count"])),
        ("Aktívni hráči", fmt_number(alliance["player_count"])),
        ("Killy", fmt_number(total_kills)),
        ("Straty", fmt_number(total_losses)),
        ("Ratio", fmt_ratio(total_losses, total_kills)),
        ("Blacklist", fmt_number(blacklist_count)),
    ]
    return "".join(f'<section class="card"><span>{esc(label)}</span><strong>{esc(value)}</strong></section>' for label, value in cards)


def render_period_tabs(period: str, date_from: str | None, date_to: str | None, base: str = "/") -> str:
    return "".join(
        f'<a class="tab {"active" if key == period and not (date_from or date_to) else ""}" href="{base}?period={key}">{esc(label)}</a>'
        for key, label in PERIODS.items()
    )


def render_filter_form(period: str, date_from: str | None, date_to: str | None, base: str = "/") -> str:
    options = "".join(f'<option value="{key}" {"selected" if key == period else ""}>{esc(label)}</option>' for key, label in PERIODS.items())
    return f"""
    <form class="filters hero-card" action="{esc(base)}" method="get">
      <h3>Vlastný filter</h3>
      <label>Obdobie<select name="period">{options}</select></label>
      <label>Od dátumu<input type="date" name="from" value="{esc(date_from or '')}"></label>
      <label>Do dátumu<input type="date" name="to" value="{esc(date_to or '')}"></label>
      <button class="btn primary" type="submit">Použiť filter</button>
    </form>
    """


def render_chart(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<div class="empty">Zatiaľ žiadne dáta pre graf.</div>'
    max_value = max(max(int(row["total_kills"] or 0), int(row["total_losses"] or 0)) for row in rows) or 1
    bars = []
    for row in rows[-30:]:
        kills = int(row["total_kills"] or 0)
        losses = int(row["total_losses"] or 0)
        day = row["day"].strftime("%d.%m.") if hasattr(row["day"], "strftime") else str(row["day"])
        bars.append(
            f"""
            <div class="bar-wrap" title="{esc(day)} · killy {fmt_number(kills)} · straty {fmt_number(losses)}">
              <div class="bar" style="height:{max(3, round((kills / max_value) * 170))}px"></div>
              <div class="bar loss" style="height:{max(3, round((losses / max_value) * 170))}px"></div>
              <span class="bar-label">{esc(day)}</span>
            </div>
            """
        )
    return f'<div class="bars">{"".join(bars)}</div><div class="report-meta">Zlatá = killy, červená = straty.</div>'


def power_period(request: Request) -> str:
    period = request.query_params.get("period", "24h")
    return period if period in POWER_PERIODS else "24h"


def power_sort(request: Request) -> tuple[str, str]:
    sort = request.query_params.get("sort", "moc")
    direction = request.query_params.get("dir", "desc")
    if sort not in {"meno", "moc", "zmena", "level", "update"}:
        sort = "moc"
    if direction not in {"asc", "desc"}:
        direction = "desc"
    return sort, direction


def table_sort(request: Request, allowed: set[str], default: str = "score") -> tuple[str, str]:
    sort = request.query_params.get("sort", default)
    direction = request.query_params.get("dir", "desc")
    if sort not in allowed:
        sort = default
    if direction not in {"asc", "desc"}:
        direction = "desc"
    return sort, direction


async def fetch_power_data(period: str, sort: str = "moc", direction: str = "desc") -> dict[str, Any]:
    db = ensure_pool()
    _, interval = POWER_PERIODS[period]
    async with db.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                p.player_id,
                p.player_name,
                p.level,
                p.legendary_level,
                p.alliance_rank,
                p.might_current,
                p.might_all_time,
                p.tracker_updated_at,
                p.synced_at,
                COALESCE(before_cutoff.might_points, first_available.might_points, p.might_current) AS baseline_might,
                COALESCE(before_cutoff.recorded_at, first_available.recorded_at, p.tracker_updated_at, p.synced_at) AS baseline_at
            FROM tracker_players p
            LEFT JOIN LATERAL (
                SELECT might_points, recorded_at
                FROM tracker_power_snapshots s
                WHERE s.server = p.server
                  AND s.player_id = p.player_id
                  AND s.recorded_at <= NOW() - ($2::interval)
                ORDER BY s.recorded_at DESC
                LIMIT 1
            ) before_cutoff ON TRUE
            LEFT JOIN LATERAL (
                SELECT might_points, recorded_at
                FROM tracker_power_snapshots s
                WHERE s.server = p.server
                  AND s.player_id = p.player_id
                ORDER BY s.recorded_at ASC
                LIMIT 1
            ) first_available ON TRUE
            WHERE p.server = $1 AND p.alliance_name = $3
            ORDER BY p.might_current DESC, LOWER(p.player_name) ASC
            """,
            TRACKER_SERVER,
            interval,
            ALLIANCE_NAME,
        )
        last_sync = await conn.fetchval(
            """
            SELECT MAX(synced_at)
            FROM tracker_players
            WHERE server = $1 AND alliance_name = $2
            """,
            TRACKER_SERVER,
            ALLIANCE_NAME,
        )
    return {
        "players": [dict(row) for row in rows],
        "last_sync": last_sync,
        "period": period,
        "sort": sort,
        "direction": direction,
    }


async def fetch_power_player_data(player_id: int, period: str) -> dict[str, Any]:
    db = ensure_pool()
    _, interval = POWER_PERIODS[period]
    async with db.acquire() as conn:
        player = await conn.fetchrow(
            """
            SELECT *
            FROM tracker_players
            WHERE server = $1 AND player_id = $2
            """,
            TRACKER_SERVER,
            player_id,
        )
        history = await conn.fetch(
            """
            SELECT recorded_at, might_points
            FROM tracker_power_snapshots
            WHERE server = $1
              AND player_id = $2
              AND recorded_at >= NOW() - ($3::interval)
            ORDER BY recorded_at ASC
            LIMIT 500
            """,
            TRACKER_SERVER,
            player_id,
            interval,
        )
    if player is None:
        raise HTTPException(status_code=404, detail="Power hráč nebol nájdený.")
    return {"player": dict(player), "history": [dict(row) for row in history], "period": period}


def fmt_signed(value: int) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{fmt_number(value)}"


def fmt_dt(value: Any) -> str:
    return value.strftime("%d.%m. %H:%M") if hasattr(value, "strftime") else "—"


def render_power_tabs(
    period: str,
    player_id: int | None = None,
    sort: str = "moc",
    direction: str = "desc",
) -> str:
    def href_for(key: str) -> str:
        params = {"tab": "moc", "period": key, "sort": sort, "dir": direction}
        if player_id is not None:
            params["power_player_id"] = str(player_id)
        return f"/?{urlencode(params)}"

    return "".join(
        f'<a class="tab {"active" if key == period else ""}" href="{href_for(key)}">{esc(label)}</a>'
        for key, (label, _) in POWER_PERIODS.items()
    )


def render_power_sort_controls(period: str, sort: str, direction: str) -> str:
    options = [
        ("moc", "desc", "Moc ↓"),
        ("moc", "asc", "Moc ↑"),
        ("zmena", "desc", "Zmena ↓"),
        ("zmena", "asc", "Zmena ↑"),
    ]
    links = []
    for sort_key, dir_key, label in options:
        active = sort == sort_key and direction == dir_key
        href = f"/?{urlencode({'tab': 'moc', 'period': period, 'sort': sort_key, 'dir': dir_key})}"
        links.append(f'<a class="tab {"active" if active else ""}" href="{href}">{esc(label)}</a>')
    return "".join(links)


def render_power_chart(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<div class="empty">Zatiaľ nemáme historické power dáta pre toto obdobie.</div>'
    sample = rows[-60:]
    values = [int(row["might_points"] or 0) for row in sample]
    min_value = min(values)
    max_value = max(values)
    span = max(max_value - min_value, 1)
    bars = []
    for row in sample:
        value = int(row["might_points"] or 0)
        recorded_at = row["recorded_at"].strftime("%d.%m. %H:%M")
        bars.append(
            f"""
            <div class="bar-wrap" title="{esc(recorded_at)} · power {fmt_number(value)}">
              <div class="bar" style="height:{max(4, round(((value - min_value) / span) * 170))}px"></div>
              <span class="bar-label">{esc(row["recorded_at"].strftime("%d.%m."))}</span>
            </div>
            """
        )
    return f'<div class="bars">{"".join(bars)}</div><div class="report-meta">Graf je škálovaný medzi najnižšou a najvyššou hodnotou v zobrazenom období.</div>'


def render_power_page(data: dict[str, Any]) -> str:
    period = data["period"]
    sort = data.get("sort", "moc")
    direction = data.get("direction", "desc")
    period_label_text = POWER_PERIODS[period][0]
    rows = []
    total_might = 0
    total_delta = 0
    prepared_rows = []
    for row in data["players"]:
        current = int(row["might_current"] or 0)
        baseline = int(row["baseline_might"] or current)
        delta = current - baseline
        total_might += current
        total_delta += delta
        prepared_rows.append({**row, "_current": current, "_delta": delta})

    sort_field = {
        "meno": "player_name",
        "level": "legendary_level",
        "update": "tracker_updated_at",
        "zmena": "_delta",
    }.get(sort, "_current")
    prepared_rows.sort(
        key=lambda row: (
            str(row[sort_field]).lower()
            if sort == "meno"
            else row[sort_field] or row["synced_at"] or datetime.min.replace(tzinfo=timezone.utc)
            if sort == "update"
            else int(row[sort_field] or 0),
            str(row["player_name"]).lower(),
        ),
        reverse=direction == "desc",
    )

    for index, row in enumerate(prepared_rows, start=1):
        current = int(row["_current"])
        delta = int(row["_delta"])
        delta_class = "good" if delta > 0 else "bad" if delta < 0 else ""
        updated_at = row["tracker_updated_at"] or row["synced_at"]
        updated_label = updated_at.strftime("%d.%m. %H:%M") if hasattr(updated_at, "strftime") else "—"
        rows.append(
            "<tr>"
            f"<td>{index}</td>"
            f'<td><a class="player-link" href="/?{urlencode({"tab": "moc", "power_player_id": str(int(row["player_id"])), "period": period})}">{esc(row["player_name"])}</a></td>'
            f'<td class="num">{fmt_number(current)}</td>'
            f'<td class="num {delta_class}">{fmt_signed(delta)}</td>'
            f'<td class="num">{int(row["level"] or 0)}/{int(row["legendary_level"] or 0)}</td>'
            f"<td>{esc(updated_label)}</td>"
            "</tr>"
        )
    body_rows = "\n".join(rows) or '<tr><td colspan="6" class="empty">Zatiaľ nemám dáta z GGE Trackeru. Skús refresh o chvíľu.</td></tr>'
    last_sync = data["last_sync"].strftime("%d.%m. %H:%M") if hasattr(data["last_sync"], "strftime") else "zatiaľ bez syncu"
    cards = [
        ("Členovia", fmt_number(len(data["players"]))),
        ("Power spolu", fmt_number(total_might)),
        (f"Zmena {period_label_text}", fmt_signed(total_delta)),
    ]
    cards_html = "".join(f'<section class="card"><span>{esc(label)}</span><strong>{esc(value)}</strong></section>' for label, value in cards)
    body = f"""
    <section class="hero">
      <div class="hero-card">
        <h1>Moc · ROYAL SOLDIERS</h1>
        <p class="subtitle">Mená a moc členov z GGE Trackeru. Dáta sa pravidelne aktualizujú automaticky; posledný sync: <strong>{esc(last_sync)}</strong>.</p>
        <div class="periods">{render_power_tabs(period, sort=sort, direction=direction)}</div>
      </div>
      <div class="filters hero-card">
        <h3>Zdroj</h3>
        <p class="subtitle" style="margin:0">Server: <strong>{esc(TRACKER_SERVER)}</strong><br>Aliancia: <strong>{esc(ALLIANCE_NAME)}</strong><br>Interval syncu: približne každých {fmt_number(TRACKER_SYNC_INTERVAL_SECONDS // 60)} min.</p>
      </div>
    </section>
    <section class="cards" style="grid-template-columns:repeat(3,minmax(0,1fr))">{cards_html}</section>
    <section class="panel"><div class="panel-head"><h2>Členovia podľa power</h2><span class="pill">{esc(period_label_text)}</span></div>
      <table><thead><tr><th>#</th>{sort_header("Hráč", base="/", params={"tab": "moc", "period": period}, key="meno", active_sort=sort, direction=direction)}{sort_header("Power", base="/", params={"tab": "moc", "period": period}, key="moc", active_sort=sort, direction=direction, numeric=True)}{sort_header("Zmena", base="/", params={"tab": "moc", "period": period}, key="zmena", active_sort=sort, direction=direction, numeric=True)}{sort_header("Level", base="/", params={"tab": "moc", "period": period}, key="level", active_sort=sort, direction=direction, numeric=True)}{sort_header("Update", base="/", params={"tab": "moc", "period": period}, key="update", active_sort=sort, direction=direction)}</tr></thead><tbody>{body_rows}</tbody></table>
    </section>
    """
    return layout("Moc · ROYAL SOLDIERS", body, active="might")


def render_power_player_page(player_id: int, data: dict[str, Any]) -> str:
    period = data["period"]
    player = data["player"]
    history = data["history"]
    current = int(player["might_current"] or 0)
    first = int(history[0]["might_points"]) if history else current
    delta = current - first
    cards = [
        ("Aktuálny power", fmt_number(current)),
        (f"Zmena {POWER_PERIODS[period][0]}", fmt_signed(delta)),
        ("All-time max", fmt_number(int(player["might_all_time"] or 0))),
    ]
    cards_html = "".join(f'<section class="card"><span>{esc(label)}</span><strong>{esc(value)}</strong></section>' for label, value in cards)
    points = []
    for row in history[-30:]:
        points.append(
            f"<tr><td>{esc(row['recorded_at'].strftime('%d.%m. %H:%M'))}</td><td class='num'>{fmt_number(int(row['might_points'] or 0))}</td></tr>"
        )
    history_rows = "\n".join(points) or '<tr><td colspan="2" class="empty">Zatiaľ žiadna história.</td></tr>'
    body = f"""
    <section class="hero">
      <div class="hero-card">
        <a class="tab" href="/?tab=moc&period={period}">← Späť na Moc</a>
        <h1 style="margin-top:16px">{esc(player["player_name"])}</h1>
        <p class="subtitle">Vývoj moci hráča podľa dát z GGE Trackeru. Player ID: <code>{int(player_id)}</code></p>
        <div class="periods">{render_power_tabs(period, int(player_id))}</div>
      </div>
      <div class="panel"><div class="panel-head"><h2>Trend power</h2></div><div class="panel-body">{render_power_chart(history)}</div></div>
    </section>
    <section class="cards" style="grid-template-columns:repeat(3,minmax(0,1fr))">{cards_html}</section>
    <section class="panel"><div class="panel-head"><h2>Posledné body</h2></div><table><thead><tr><th>Čas</th><th class="num">Power</th></tr></thead><tbody>{history_rows}</tbody></table></section>
    """
    return layout(f"{player['player_name']} · Moc", body, active="might")


async def fetch_loot_data(sort: str = "this_week", direction: str = "desc") -> dict[str, Any]:
    db = ensure_pool()
    async with db.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                p.player_id,
                p.player_name,
                p.level,
                p.legendary_level,
                p.loot_current AS this_week_loot,
                COALESCE(last_week.loot_points, 0) AS last_week_loot,
                p.tracker_updated_at,
                p.synced_at
            FROM tracker_players p
            LEFT JOIN LATERAL (
                SELECT loot_points, recorded_at
                FROM tracker_loot_snapshots s
                WHERE s.server = p.server
                  AND s.player_id = p.player_id
                  AND s.recorded_at >= date_trunc('week', NOW()) - INTERVAL '7 days'
                  AND s.recorded_at < date_trunc('week', NOW())
                ORDER BY s.recorded_at DESC
                LIMIT 1
            ) last_week ON TRUE
            WHERE p.server = $1 AND p.alliance_name = $2
            ORDER BY p.loot_current DESC, LOWER(p.player_name) ASC
            """,
            TRACKER_SERVER,
            ALLIANCE_NAME,
        )
        last_sync = await conn.fetchval(
            """
            SELECT MAX(synced_at)
            FROM tracker_players
            WHERE server = $1 AND alliance_name = $2
            """,
            TRACKER_SERVER,
            ALLIANCE_NAME,
        )
    return {"players": [dict(row) for row in rows], "last_sync": last_sync, "sort": sort, "direction": direction}


def render_loot_page(data: dict[str, Any]) -> str:
    rows = []
    total_this_week = 0
    sort = data.get("sort", "this_week")
    direction = data.get("direction", "desc")
    prepared_rows = []
    for row in data["players"]:
        this_week = int(row["this_week_loot"] or 0)
        total_this_week += this_week
        prepared_rows.append({**row, "_this_week": this_week})

    sort_field = {
        "meno": "player_name",
        "level": "legendary_level",
        "update": "tracker_updated_at",
    }.get(sort, "_this_week")
    prepared_rows.sort(
        key=lambda row: (
            str(row[sort_field]).lower()
            if sort == "meno"
            else row[sort_field] or row["synced_at"] or datetime.min.replace(tzinfo=timezone.utc)
            if sort == "update"
            else int(row[sort_field] or 0),
            str(row["player_name"]).lower(),
        ),
        reverse=direction == "desc",
    )

    for index, row in enumerate(prepared_rows, start=1):
        this_week = int(row["_this_week"])
        updated_at = row["tracker_updated_at"] or row["synced_at"]
        updated_label = updated_at.strftime("%d.%m. %H:%M") if hasattr(updated_at, "strftime") else "—"
        rows.append(
            "<tr>"
            f"<td>{index}</td>"
            f"<td>{esc(row['player_name'])}</td>"
            f'<td class="num">{fmt_number(this_week)}</td>'
            f'<td class="num">{int(row["level"] or 0)}/{int(row["legendary_level"] or 0)}</td>'
            f"<td>{esc(updated_label)}</td>"
            "</tr>"
        )
    body_rows = "\n".join(rows) or '<tr><td colspan="5" class="empty">Zatiaľ nemám dáta o rabovaní z GGE Trackeru. Skús refresh o chvíľu.</td></tr>'
    last_sync = data["last_sync"].strftime("%d.%m. %H:%M") if hasattr(data["last_sync"], "strftime") else "zatiaľ bez syncu"
    cards = [
        ("Členovia", fmt_number(len(data["players"]))),
        ("Aktuálne rabovanie", fmt_number(total_this_week)),
    ]
    cards_html = "".join(f'<section class="card"><span>{esc(label)}</span><strong>{esc(value)}</strong></section>' for label, value in cards)
    body = f"""
    <section class="hero">
      <div class="hero-card">
        <h1>Rabovanie · ROYAL SOLDIERS</h1>
        <p class="subtitle">Aktuálne rabovanie členov podľa GGE Trackeru. Beriem priamo hodnotu <strong>loot_current</strong>. Posledný sync: <strong>{esc(last_sync)}</strong>.</p>
      </div>
      <div class="filters hero-card">
        <h3>Zdroj</h3>
        <p class="subtitle" style="margin:0">Server: <strong>{esc(TRACKER_SERVER)}</strong><br>Aliancia: <strong>{esc(ALLIANCE_NAME)}</strong><br>Interval syncu: približne každých {fmt_number(TRACKER_SYNC_INTERVAL_SECONDS // 60)} min.</p>
      </div>
    </section>
    <section class="cards" style="grid-template-columns:repeat(2,minmax(0,1fr))">{cards_html}</section>
    <section class="panel"><div class="panel-head"><h2>Rabovanie podľa hráčov</h2><span class="pill">aktuálny tracker stav</span></div>
      <table><thead><tr><th>#</th>{sort_header("Hráč", base="/", params={"tab": "rabovanie"}, key="meno", active_sort=sort, direction=direction)}{sort_header("Rabovanie", base="/", params={"tab": "rabovanie"}, key="this_week", active_sort=sort, direction=direction, numeric=True)}{sort_header("Level", base="/", params={"tab": "rabovanie"}, key="level", active_sort=sort, direction=direction, numeric=True)}{sort_header("Update", base="/", params={"tab": "rabovanie"}, key="update", active_sort=sort, direction=direction)}</tr></thead><tbody>{body_rows}</tbody></table>
    </section>
    """
    return layout("Rabovanie · ROYAL SOLDIERS", body, active="loot")


async def fetch_live_event_fallback(event_slug: str, players: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not players:
        return None
    event_config = EVENT_TYPES[event_slug]
    tracker_key = event_config["tracker_key"]
    sample_player_id = int(players[0]["player_id"])
    try:
        alliance = await tracker_json_async(f"/alliances/name/{quote(ALLIANCE_NAME)}")
        alliance_id = to_int(alliance.get("alliance_id"))
        if not alliance_id:
            return None
        stats = await tracker_json_async(f"/statistics/alliance/{alliance_id}")
        occurrence_data = await tracker_json_async(
            f"/statistics/player/{sample_player_id}/{tracker_key}/occurrences"
        )
    except (urlerror.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"[DASHBOARD] Live event fallback failed for {tracker_key}: {type(exc).__name__}: {exc}")
        return None

    parsed_occurrences: list[tuple[datetime, datetime]] = []
    for occurrence in occurrence_data.get("occurrences") or []:
        started_at = parse_tracker_time(occurrence.get("started_at"))
        ended_at = parse_tracker_time(occurrence.get("ended_at"))
        if started_at is not None and ended_at is not None:
            parsed_occurrences.append((started_at, ended_at))
    parsed_occurrences.sort(key=lambda item: item[0])
    if not parsed_occurrences:
        return None

    current_start, current_end = parsed_occurrences[-1]
    previous_start, previous_end = parsed_occurrences[-2] if len(parsed_occurrences) > 1 else (None, None)
    current_scores: dict[int, tuple[int, datetime | None]] = {}
    previous_scores: dict[int, int] = {}
    for point in (stats.get("points") or {}).get(tracker_key) or []:
        player_id = to_int(point.get("player_id"))
        recorded_at = parse_tracker_time(point.get("date"))
        score = to_int(point.get("point"), -1)
        if not player_id or recorded_at is None or score < 0:
            continue
        if current_start <= recorded_at < current_end:
            existing_score, _ = current_scores.get(player_id, (0, None))
            if score >= existing_score:
                current_scores[player_id] = (score, recorded_at)
        if previous_start is not None and previous_end is not None and previous_start <= recorded_at < previous_end:
            previous_scores[player_id] = max(previous_scores.get(player_id, 0), score)

    live_players = []
    for player in players:
        player_id = int(player["player_id"])
        score, current_at = current_scores.get(player_id, (0, None))
        live_players.append(
            {
                **player,
                "current_score": score,
                "previous_score": previous_scores.get(player_id, 0),
                "current_at": current_at,
                "event_starts_at": current_start,
                "event_ends_at": current_end,
                "previous_starts_at": previous_start,
                "previous_ends_at": previous_end,
            }
        )
    return {
        "players": live_players,
        "last_sync": utc_now(),
        "event_slug": event_slug,
        "event": event_config,
        "live_fallback": True,
    }


async def fetch_event_data(event_slug: str, sort: str = "score", direction: str = "desc") -> dict[str, Any]:
    event_config = EVENT_TYPES[event_slug]
    tracker_key = event_config["tracker_key"]
    db = ensure_pool()
    async with db.acquire() as conn:
        rows = await conn.fetch(
            """
            WITH current_window AS (
                SELECT started_at AS starts_at, ended_at AS ends_at
                FROM tracker_event_occurrences
                WHERE server = $1 AND event_key = $2
                ORDER BY started_at DESC
                LIMIT 1
            ),
            previous_window AS (
                SELECT started_at AS starts_at, ended_at AS ends_at
                FROM tracker_event_occurrences
                WHERE server = $1
                  AND event_key = $2
                  AND started_at < (SELECT starts_at FROM current_window)
                ORDER BY started_at DESC
                LIMIT 1
            ),
            current_period AS (
                SELECT server, event_key, player_id,
                       MAX(event_points) AS current_score,
                       MAX(recorded_at) AS current_at
                FROM tracker_event_snapshots
                WHERE server = $1
                  AND event_key = $2
                  AND recorded_at >= (SELECT starts_at FROM current_window)
                  AND recorded_at < (SELECT ends_at FROM current_window)
                GROUP BY server, event_key, player_id
            ),
            previous_period AS (
                SELECT server, event_key, player_id,
                       MAX(event_points) AS previous_score
                FROM tracker_event_snapshots
                WHERE server = $1
                  AND event_key = $2
                  AND recorded_at >= (SELECT starts_at FROM previous_window)
                  AND recorded_at < (SELECT ends_at FROM previous_window)
                GROUP BY server, event_key, player_id
            )
            SELECT
                p.player_id,
                p.player_name,
                p.level,
                p.legendary_level,
                p.tracker_updated_at,
                p.synced_at,
                COALESCE(cp.current_score, 0) AS current_score,
                COALESCE(pp.previous_score, 0) AS previous_score,
                cp.current_at,
                (SELECT starts_at FROM current_window) AS event_starts_at,
                (SELECT ends_at FROM current_window) AS event_ends_at,
                (SELECT starts_at FROM previous_window) AS previous_starts_at,
                (SELECT ends_at FROM previous_window) AS previous_ends_at
            FROM tracker_players p
            LEFT JOIN current_period cp
              ON cp.server = p.server AND cp.player_id = p.player_id
            LEFT JOIN previous_period pp
              ON pp.server = p.server AND pp.player_id = p.player_id
            WHERE p.server = $1 AND p.alliance_name = $3
            ORDER BY COALESCE(cp.current_score, 0) DESC, LOWER(p.player_name) ASC
            """,
            TRACKER_SERVER,
            tracker_key,
            ALLIANCE_NAME,
        )
        last_sync = await conn.fetchval(
            """
            SELECT MAX(recorded_at)
            FROM tracker_event_snapshots
            WHERE server = $1 AND event_key = $2
            """,
            TRACKER_SERVER,
            tracker_key,
        )
    players = [dict(row) for row in rows]
    if (last_sync is None or not any(int(row.get("current_score") or 0) for row in players)) and players:
        live_data = await fetch_live_event_fallback(event_slug, players)
        if live_data is not None:
            live_data["sort"] = sort
            live_data["direction"] = direction
            return live_data
    return {
        "players": players,
        "last_sync": last_sync,
        "event_slug": event_slug,
        "event": event_config,
        "sort": sort,
        "direction": direction,
    }


def render_event_tabs(active_slug: str) -> str:
    return "".join(
        f'<a class="tab {"active" if slug == active_slug else ""}" href="/?tab={esc(slug)}">{esc(config["emoji"])} {esc(config["nav"])}</a>'
        for slug, config in EVENT_TYPES.items()
    )


def render_event_page(data: dict[str, Any]) -> str:
    event_slug = data["event_slug"]
    event = data["event"]
    sort = data.get("sort", "score")
    direction = data.get("direction", "desc")
    rows = []
    total_score = 0
    total_previous = 0
    active_players = 0
    prepared_rows = []
    current_start = data["players"][0].get("event_starts_at") if data["players"] else None
    current_end = data["players"][0].get("event_ends_at") if data["players"] else None
    previous_start = data["players"][0].get("previous_starts_at") if data["players"] else None
    previous_end = data["players"][0].get("previous_ends_at") if data["players"] else None
    for row in data["players"]:
        score = int(row["current_score"] or 0)
        previous = int(row["previous_score"] or 0)
        diff = score - previous
        total_score += score
        total_previous += previous
        if score > 0:
            active_players += 1
        prepared_rows.append({**row, "_score": score, "_previous": previous, "_diff": diff})

    sort_field = {
        "meno": "player_name",
        "previous": "_previous",
        "diff": "_diff",
        "level": "legendary_level",
        "update": "current_at",
    }.get(sort, "_score")
    prepared_rows.sort(
        key=lambda row: (
            str(row[sort_field]).lower()
            if sort == "meno"
            else row[sort_field] or row["synced_at"] or datetime.min.replace(tzinfo=timezone.utc)
            if sort == "update"
            else int(row[sort_field] or 0),
            str(row["player_name"]).lower(),
        ),
        reverse=direction == "desc",
    )

    for index, row in enumerate(prepared_rows, start=1):
        score = int(row["_score"])
        previous = int(row["_previous"])
        diff = int(row["_diff"])
        diff_class = "good" if diff > 0 else "bad" if diff < 0 else ""
        updated_at = row["current_at"] or row["tracker_updated_at"] or row["synced_at"]
        updated_label = updated_at.strftime("%d.%m. %H:%M") if hasattr(updated_at, "strftime") else "—"
        rows.append(
            "<tr>"
            f"<td>{index}</td>"
            f"<td>{esc(row['player_name'])}</td>"
            f'<td class="num">{fmt_number(score)}</td>'
            f'<td class="num">{fmt_number(previous)}</td>'
            f'<td class="num {diff_class}">{fmt_signed(diff)}</td>'
            f'<td class="num">{int(row["level"] or 0)}/{int(row["legendary_level"] or 0)}</td>'
            f"<td>{esc(updated_label)}</td>"
            "</tr>"
        )
    body_rows = "\n".join(rows) or '<tr><td colspan="7" class="empty">Zatiaľ nemám eventové dáta z GGE Trackeru. Skús refresh o chvíľu.</td></tr>'
    last_sync = data["last_sync"].strftime("%d.%m. %H:%M") if hasattr(data["last_sync"], "strftime") else "zatiaľ bez syncu"
    total_diff = total_score - total_previous
    cards = [
        ("Aktívni hráči", fmt_number(active_players)),
        ("Body v udalosti", fmt_number(total_score)),
        ("Predošlá udalosť", fmt_number(total_previous)),
        ("Rozdiel", fmt_signed(total_diff)),
    ]
    cards_html = "".join(f'<section class="card"><span>{esc(label)}</span><strong>{esc(value)}</strong></section>' for label, value in cards)
    params = {"tab": event_slug}
    current_window_label = f"{fmt_dt(current_start)} – {fmt_dt(current_end)}"
    previous_window_label = f"{fmt_dt(previous_start)} – {fmt_dt(previous_end)}" if previous_start and previous_end else "—"
    body = f"""
    <section class="hero">
      <div class="hero-card">
        <h1>{esc(event["emoji"])} {esc(event["title"])} · ROYAL SOLDIERS</h1>
        <p class="subtitle">Výsledky hráčov z GGE Trackeru za presné trvanie udalosti, ktoré vracia tracker. Aktuálne zobrazené okno: <strong>{esc(current_window_label)}</strong>. Predošlé okno: <strong>{esc(previous_window_label)}</strong>. Posledný event sync: <strong>{esc(last_sync)}</strong>.</p>
        <div class="periods">{render_event_tabs(event_slug)}</div>
      </div>
      <div class="filters hero-card">
        <h3>Zdroj</h3>
        <p class="subtitle" style="margin:0">Server: <strong>{esc(TRACKER_SERVER)}</strong><br>Aliancia: <strong>{esc(ALLIANCE_NAME)}</strong><br>Tracker kľúč: <strong>{esc(event["tracker_key"])}</strong></p>
      </div>
    </section>
    <section class="cards" style="grid-template-columns:repeat(4,minmax(0,1fr))">{cards_html}</section>
    <section class="panel"><div class="panel-head"><h2>{esc(event["title"])} podľa hráčov</h2><span class="pill">{esc(current_window_label)}</span></div>
      <table><thead><tr><th>#</th>{sort_header("Hráč", base="/", params=params, key="meno", active_sort=sort, direction=direction)}{sort_header("Body", base="/", params=params, key="score", active_sort=sort, direction=direction, numeric=True)}{sort_header("Predošlé", base="/", params=params, key="previous", active_sort=sort, direction=direction, numeric=True)}{sort_header("Rozdiel", base="/", params=params, key="diff", active_sort=sort, direction=direction, numeric=True)}{sort_header("Level", base="/", params=params, key="level", active_sort=sort, direction=direction, numeric=True)}{sort_header("Update", base="/", params=params, key="update", active_sort=sort, direction=direction)}</tr></thead><tbody>{body_rows}</tbody></table>
    </section>
    """
    return layout(f'{event["title"]} · ROYAL SOLDIERS', body, active=event_slug)


def render_leaderboard(rows: list[dict[str, Any]], period: str, date_from: str | None, date_to: str | None) -> str:
    table_rows = []
    qs = query_string(period, date_from, date_to)
    for index, row in enumerate(rows, start=1):
        losses = int(row["total_losses"] or 0)
        kills = int(row["total_kills"] or 0)
        ratio = ratio_score(losses, kills)
        rank_class = "rank top" if index <= 3 else "rank"
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(index, str(index))
        table_rows.append(
            "<tr>"
            f'<td><span class="{rank_class}">{medal}</span></td>'
            f'<td><a class="player-link" href="/player/{int(row["player_id"])}?{qs}">{esc(row["player_name"])}</a></td>'
            f'<td class="num">{fmt_number(kills)}</td>'
            f'<td class="num">{fmt_number(losses)}</td>'
            f'<td class="num {"good" if ratio >= 5 else "bad" if ratio < 2 else ""}">{fmt_ratio(losses, kills)}</td>'
            f'<td class="num">{fmt_number(row["report_count"])}</td>'
            "</tr>"
        )
    body = "\n".join(table_rows) or '<tr><td colspan="6" class="empty">Zatiaľ žiadne reporty.</td></tr>'
    return f"""
    <table>
      <thead><tr><th>#</th><th>Hráč</th><th class="num">Killy</th><th class="num">Straty</th><th class="num">Ratio</th><th class="num">Reporty</th></tr></thead>
      <tbody>{body}</tbody>
    </table>
    """


def render_recent(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<div class="empty">Zatiaľ žiadne reporty.</div>'
    items = []
    for row in rows:
        losses = int(row["own_losses"] or 0)
        kills = int(row["enemy_kills"] or 0)
        created_at = row["created_at"].strftime("%d.%m. %H:%M")
        items.append(
            f"""
            <div class="report-item">
              <div>
                <a class="player-link" href="/player/{int(row["player_id"])}">{esc(row["player_name"])}</a>
                <div class="report-meta">{esc(created_at)} · msg {int(row["message_id"])}</div>
              </div>
              <div class="actions">
                <span class="pill">⚔️ {fmt_number(kills)}</span>
                <span class="pill">🛡️ {fmt_number(losses)}</span>
                <span class="pill">Ratio {fmt_ratio(losses, kills)}</span>
              </div>
            </div>
            """
        )
    return f'<div class="recent-list">{"".join(items)}</div>'


def render_dashboard(data: dict[str, Any], period: str, date_from: str | None, date_to: str | None) -> str:
    label = period_label(period, date_from, date_to)
    body = f"""
    <section class="hero">
      <div class="hero-card">
        <h1>ROYAL SOLDIERS</h1>
        <p class="subtitle">Prehľad killov, strát, ratio, reportov a aktivity hráčov za <strong>{esc(label)}</strong>.</p>
        <div class="periods">{render_period_tabs(period, date_from, date_to)}</div>
      </div>
      {render_filter_form(period, date_from, date_to)}
    </section>
    <section class="cards">{render_cards(data["alliance"], data["blacklist_count"])}</section>
    <section class="grid">
      <div class="panel">
        <div class="panel-head"><h2>Leaderboard</h2><span class="pill">podľa killov</span></div>
        {render_leaderboard(data["leaderboard"], period, date_from, date_to)}
      </div>
      <div class="panel"><div class="panel-head"><h2>Vývoj aktivity</h2></div><div class="panel-body">{render_chart(data["chart"])}</div></div>
    </section>
    <section class="panel" style="margin-top:18px"><div class="panel-head"><h2>Posledné reporty</h2><span class="pill">read-only</span></div><div class="panel-body">{render_recent(data["recent"])}</div></section>
    """
    return layout("GGE Report Dashboard", body)


def render_player_page(player_id: int, data: dict[str, Any], period: str, date_from: str | None, date_to: str | None) -> str:
    summary = data["summary"]
    player_name = summary["player_name"] or f"Hráč {player_id}"
    cards = {
        "report_count": summary["report_count"],
        "player_count": 1 if int(summary["report_count"] or 0) else 0,
        "total_losses": summary["total_losses"],
        "total_kills": summary["total_kills"],
    }
    body = f"""
    <section class="hero">
      <div class="hero-card">
        <a class="tab" href="/?{query_string(period, date_from, date_to)}">← Späť na dashboard</a>
        <h1 style="margin-top:16px">{esc(player_name)}</h1>
        <p class="subtitle">Detail hráča za <strong>{esc(period_label(period, date_from, date_to))}</strong>. Player ID: <code>{int(player_id)}</code></p>
        <div class="periods">{render_period_tabs(period, date_from, date_to, f"/player/{int(player_id)}")}</div>
      </div>
      {render_filter_form(period, date_from, date_to, f"/player/{int(player_id)}")}
    </section>
    <section class="cards">{render_cards(cards, 0)}</section>
    <section class="grid">
      <div class="panel"><div class="panel-head"><h2>Reporty hráča</h2></div><div class="panel-body">{render_recent(data["reports"])}</div></div>
      <div class="panel"><div class="panel-head"><h2>Trend hráča</h2></div><div class="panel-body">{render_chart(data["chart"])}</div></div>
    </section>
    """
    return layout(f"{player_name} · GGE Report Dashboard", body)


def action_form(
    action: str,
    label: str,
    fields: str,
    button_class: str = "btn",
    confirm: str | None = None,
) -> str:
    confirm_attr = f' data-confirm="{esc(confirm)}"' if confirm else ""
    return f'<form class="mini-form" method="post" action="/admin/{esc(action)}"{confirm_attr}>{fields}<button class="{button_class}" type="submit">{esc(label)}</button></form>'


def render_admin_pagination(current_page: int, total_reports: int, per_page: int) -> str:
    total_pages = max(1, (total_reports + per_page - 1) // per_page)
    if total_pages <= 1:
        return ""
    links = []
    for page in range(1, total_pages + 1):
        if total_pages > 9 and page not in {1, total_pages, current_page - 1, current_page, current_page + 1}:
            if not links or links[-1] != '<span class="pill">…</span>':
                links.append('<span class="pill">…</span>')
            continue
        active = " active" if page == current_page else ""
        links.append(f'<a class="page-link{active}" href="/admin?page={page}">{page}</a>')
    return f'<nav class="pagination" aria-label="Report history pages">{"".join(links)}</nav>'


def render_admin(data: dict[str, Any], message: str | None = None) -> str:
    player_options = "".join(
        f'<option value="{int(row["player_id"])}">{esc(row["player_name"])}'
        f' · {fmt_number(row["total_kills"])} killov'
        f'{" · bez reportu" if int(row["report_count"] or 0) == 0 else ""}</option>'
        for row in data["players"]
    )
    assign_options = '<option value="">Vyber Discord meno…</option>' + player_options
    report_rows = []
    for row in data["reports"]:
        losses = int(row["own_losses"] or 0)
        kills = int(row["enemy_kills"] or 0)
        msg_id = int(row["message_id"])
        guild_id = int(row["guild_id"])
        player_id = int(row["player_id"])
        release_fields = f'<input type="hidden" name="message_id" value="{msg_id}">'
        assign_fields = (
            f'<input type="hidden" name="message_id" value="{msg_id}">'
            f'<input type="hidden" name="guild_id" value="{guild_id}">'
            f'<select name="target_player_id" required>{assign_options}</select>'
        )
        report_rows.append(
            "<tr>"
            f"<td>{row['created_at'].strftime('%d.%m. %H:%M')}</td>"
            f"<td>{esc(row['player_name'])}<div class='report-meta'>{player_id}</div></td>"
            f"<td class='num'>{fmt_number(kills)}</td><td class='num'>{fmt_number(losses)}</td><td class='num'>{fmt_ratio(losses, kills)}</td>"
            f"<td class='num'>{msg_id}</td>"
            f"<td>{action_form('release', 'Release', release_fields, confirm=f'Naozaj uvoľniť report {msg_id}? Odpočíta sa hráčovi aj aliancii.')}"
            f"{action_form('blacklist', 'Blacklist', release_fields, 'btn danger', f'Naozaj dať report {msg_id} na blacklist? Vymaže sa zo štatistík a už ho nikto nezapočíta.')}"
            f"{action_form('assign', 'Assign', assign_fields, 'btn primary', f'Naozaj presunúť report {msg_id} na vybraného hráča?')}</td>"
            "</tr>"
        )
    report_table = "\n".join(report_rows) or '<tr><td colspan="7" class="empty">Žiadne reporty.</td></tr>'
    blacklist_rows = []
    for row in data["blacklist"]:
        guild_id = int(row["guild_id"])
        own_losses = int(row["own_losses"] or 0)
        enemy_kills = int(row["enemy_kills"] or 0)
        source_message_id = int(row["source_message_id"] or 0)
        remove_fields = (
            f'<input type="hidden" name="guild_id" value="{guild_id}">'
            f'<input type="hidden" name="own_losses" value="{own_losses}">'
            f'<input type="hidden" name="enemy_kills" value="{enemy_kills}">'
        )
        blacklist_rows.append(
            "<tr>"
            f"<td>{row['created_at'].strftime('%d.%m. %H:%M')}</td>"
            f"<td class='num'>{fmt_number(enemy_kills)}</td><td class='num'>{fmt_number(own_losses)}</td>"
            f"<td>{esc(row['blacklisted_by_name'] or 'admin')}</td><td class='num'>{source_message_id}</td>"
            f"<td>{action_form('blacklist-remove', 'Odstrániť', remove_fields, 'btn danger', f'Naozaj odstrániť blacklist pre {fmt_number(enemy_kills)} killov / {fmt_number(own_losses)} strát? Rovnaký report bude znovu možné započítať.')}</td>"
            "</tr>"
        )
    blacklist_table = "\n".join(blacklist_rows) or '<tr><td colspan="6" class="empty">Blacklist je prázdny.</td></tr>'
    notice = f'<div class="notice">{esc(message)}</div>' if message else ""
    page = int(data.get("page", 1))
    per_page = int(data.get("per_page", ADMIN_REPORTS_PER_PAGE))
    report_count = int(data.get("report_count", len(data["reports"])))
    first_report = ((page - 1) * per_page) + 1 if report_count else 0
    last_report = min(page * per_page, report_count)
    pagination = render_admin_pagination(page, report_count, per_page)
    body = f"""
    <section class="hero-card" style="margin-bottom:18px"><h1>Admin panel</h1><p class="subtitle">Release, blacklist, assign a reset priamo z webu. Toto je chránené dashboard heslom.</p></section>
    {notice}
    <section class="admin-grid">
      <div class="panel"><div class="panel-head"><h2>Rýchle akcie</h2></div><div class="panel-body">
        <h3>Reset hráča</h3>
        <form class="mini-form" method="post" action="/admin/reset" data-confirm="Naozaj resetnúť štatistiky vybraného hráča za zvolené obdobie? Táto akcia vymaže reporty z databázy."><select name="player_id" required>{player_options}</select><select name="period"><option value="all">Celé obdobie</option><option value="30d">30 dní</option><option value="7d">7 dní</option><option value="24h">24h</option></select><button class="btn danger" type="submit">Resetnúť</button></form>
        <h3 style="margin-top:22px">Manuálne podľa message ID</h3>
        {action_form('release', 'Release report', '<input name="message_id" inputmode="numeric" placeholder="message id">', confirm='Naozaj manuálne uvoľniť report podľa message ID?')}
        {action_form('blacklist', 'Blacklist report', '<input name="message_id" inputmode="numeric" placeholder="message id">', 'btn danger', 'Naozaj manuálne pridať report na blacklist podľa message ID?')}
      </div></div>
      <details class="panel"><summary class="panel-head"><h2>Blacklist reportov</h2><span class="pill">Klikni pre rozbalenie</span></summary><table><thead><tr><th>Čas</th><th class="num">Killy</th><th class="num">Straty</th><th>Admin</th><th class="num">Message</th><th>Akcia</th></tr></thead><tbody>{blacklist_table}</tbody></table></details>
    </section>
    <section class="panel" style="margin-top:18px"><div class="panel-head"><h2>Report history</h2><div class="actions"><span class="pill">{first_report}–{last_report} z {report_count}</span></div></div><table><thead><tr><th>Čas</th><th>Hráč</th><th class="num">Killy</th><th class="num">Straty</th><th class="num">Ratio</th><th class="num">Message</th><th>Akcie</th></tr></thead><tbody>{report_table}</tbody></table>{pagination}</section>
    """
    return layout("Admin · GGE Report Dashboard", body, active="admin")


async def release_by_message(message_id: int) -> int:
    db = ensure_pool()
    status = await db.execute("DELETE FROM battle_reports WHERE message_id = $1", message_id)
    return int(status.rsplit(" ", 1)[-1])


async def assign_by_message(guild_id: int, message_id: int, target_player_id: int) -> int:
    db = ensure_pool()
    async with db.acquire() as conn:
        player_name = await conn.fetchval(
            """
            SELECT player_name FROM (
                SELECT COALESCE(NULLIF(display_name, ''), username, user_id::text) AS player_name,
                       updated_at AS sort_time
                FROM discord_members
                WHERE guild_id = $1 AND user_id = $2 AND is_bot = FALSE
                UNION ALL
                SELECT player_name, created_at AS sort_time
                FROM battle_reports
                WHERE guild_id = $1 AND player_id = $2
            ) source
            ORDER BY sort_time DESC
            LIMIT 1
            """,
            guild_id,
            target_player_id,
        )
        if not player_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Selected Discord player was not found in stored members or reports.",
            )
        status_text = await conn.execute(
            "UPDATE battle_reports SET player_id = $3, player_name = $4 WHERE guild_id = $1 AND message_id = $2",
            guild_id,
            message_id,
            target_player_id,
            player_name,
        )
    return int(status_text.rsplit(" ", 1)[-1])


async def blacklist_by_message(message_id: int, admin_name: str) -> tuple[int, int]:
    db = ensure_pool()
    async with db.acquire() as conn:
        async with conn.transaction():
            rows = await conn.fetch("SELECT DISTINCT guild_id, own_losses, enemy_kills FROM battle_reports WHERE message_id = $1", message_id)
            inserted = 0
            for row in rows:
                status = await conn.execute(
                    """
                    INSERT INTO battle_report_blacklist (
                        guild_id, own_losses, enemy_kills,
                        source_message_id, blacklisted_by_name
                    ) VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (guild_id, own_losses, enemy_kills) DO NOTHING
                    """,
                    int(row["guild_id"]),
                    int(row["own_losses"]),
                    int(row["enemy_kills"]),
                    message_id,
                    admin_name,
                )
                if status == "INSERT 0 1":
                    inserted += 1
            status = await conn.execute("DELETE FROM battle_reports WHERE message_id = $1", message_id)
            return int(status.rsplit(" ", 1)[-1]), inserted


async def remove_blacklist_entry(guild_id: int, own_losses: int, enemy_kills: int) -> int:
    db = ensure_pool()
    status = await db.execute(
        """
        DELETE FROM battle_report_blacklist
        WHERE guild_id = $1 AND own_losses = $2 AND enemy_kills = $3
        """,
        guild_id,
        own_losses,
        enemy_kills,
    )
    return int(status.rsplit(" ", 1)[-1])


async def reset_player(player_id: int, period: str) -> int:
    db = ensure_pool()
    clauses = ["player_id = $1"]
    params: list[Any] = [player_id]
    add_period_filter(clauses, params, period=period)
    status = await db.execute(f"DELETE FROM battle_reports {where_sql(clauses)}", *params)
    return int(status.rsplit(" ", 1)[-1])


def admin_redirect(message: str) -> RedirectResponse:
    return RedirectResponse(f"/admin?{urlencode({'message': message})}", status_code=303)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if request.query_params.get("tab") == "rabovanie":
        sort, direction = table_sort(request, {"meno", "this_week", "level", "update"}, "this_week")
        data = await fetch_loot_data(sort, direction)
        return HTMLResponse(render_loot_page(data))

    tab = request.query_params.get("tab")
    if tab in EVENT_TYPES:
        sort, direction = table_sort(request, {"meno", "score", "previous", "diff", "level", "update"}, "score")
        data = await fetch_event_data(tab, sort, direction)
        return HTMLResponse(render_event_page(data))

    if request.query_params.get("tab") == "moc" or request.query_params.get("view") == "moc":
        period = power_period(request)
        sort, direction = power_sort(request)
        power_player_id = request.query_params.get("power_player_id")
        if power_player_id:
            data = await fetch_power_player_data(int(power_player_id), period)
            return HTMLResponse(render_power_player_page(int(power_player_id), data))
        data = await fetch_power_data(period, sort, direction)
        return HTMLResponse(render_power_page(data))

    period, date_from, date_to = get_filters(request)
    data = await fetch_dashboard_data(period=period, date_from=date_from, date_to=date_to)
    return HTMLResponse(render_dashboard(data, period, date_from, date_to))


@app.get("/player/{player_id}", response_class=HTMLResponse)
async def player_detail(player_id: int, request: Request):
    period, date_from, date_to = get_filters(request)
    data = await fetch_player_data(player_id=player_id, period=period, date_from=date_from, date_to=date_to)
    return HTMLResponse(render_player_page(player_id, data, period, date_from, date_to))


@app.get("/might", response_class=HTMLResponse)
async def power_page(request: Request):
    period = power_period(request)
    sort, direction = power_sort(request)
    data = await fetch_power_data(period, sort, direction)
    return HTMLResponse(render_power_page(data))


@app.get("/might/{player_id}", response_class=HTMLResponse)
async def power_player_page(player_id: int, request: Request):
    period = power_period(request)
    data = await fetch_power_player_data(player_id, period)
    return HTMLResponse(render_power_player_page(player_id, data))


@app.get("/loot", response_class=HTMLResponse)
async def loot_page(request: Request):
    sort, direction = table_sort(request, {"meno", "this_week", "level", "update"}, "this_week")
    data = await fetch_loot_data(sort, direction)
    return HTMLResponse(render_loot_page(data))


@app.get("/events/{event_slug}", response_class=HTMLResponse)
async def event_page(event_slug: str, request: Request):
    if event_slug not in EVENT_TYPES:
        raise HTTPException(status_code=404, detail="Event sekcia neexistuje.")
    sort, direction = table_sort(request, {"meno", "score", "previous", "diff", "level", "update"}, "score")
    data = await fetch_event_data(event_slug, sort, direction)
    return HTMLResponse(render_event_page(data))


@app.get("/api/summary")
async def api_summary(request: Request):
    period, date_from, date_to = get_filters(request)
    data = await fetch_dashboard_data(period=period, date_from=date_from, date_to=date_to)
    return JSONResponse(
        {
            "period": {"key": period, "label": period_label(period, date_from, date_to), "from": date_from, "to": date_to},
            "data": {
                "alliance": data["alliance"],
                "leaderboard": data["leaderboard"],
                "recent": [{**row, "created_at": row["created_at"].isoformat()} for row in data["recent"]],
                "blacklist_count": data["blacklist_count"],
            },
        }
    )


@app.get("/admin", response_class=HTMLResponse)
async def admin(page: int = 1, message: str | None = None, user: str = Depends(require_auth)):
    data = await fetch_admin_data(page=page)
    return HTMLResponse(render_admin(data, message))


@app.post("/admin/release")
async def admin_release(request: Request, user: str = Depends(require_auth)):
    form = parse_form_body(await request.body())
    deleted = await release_by_message(int(form["message_id"]))
    return admin_redirect(f"Release hotový. Vymazané reporty: {deleted}.")


@app.post("/admin/assign")
async def admin_assign(request: Request, user: str = Depends(require_auth)):
    form = parse_form_body(await request.body())
    updated = await assign_by_message(
        int(form["guild_id"]), int(form["message_id"]), int(form["target_player_id"])
    )
    return admin_redirect(f"Assign hotový. Presunuté reporty: {updated}.")


@app.post("/admin/blacklist")
async def admin_blacklist(request: Request, user: str = Depends(require_auth)):
    form = parse_form_body(await request.body())
    deleted, inserted = await blacklist_by_message(int(form["message_id"]), user)
    return admin_redirect(f"Blacklist hotový. Vymazané reporty: {deleted}. Nové blacklist záznamy: {inserted}.")


@app.post("/admin/blacklist-remove")
async def admin_blacklist_remove(request: Request, user: str = Depends(require_auth)):
    form = parse_form_body(await request.body())
    deleted = await remove_blacklist_entry(
        int(form["guild_id"]),
        int(form["own_losses"]),
        int(form["enemy_kills"]),
    )
    return admin_redirect(f"Blacklist záznam odstránený. Vymazané blacklist záznamy: {deleted}.")


@app.post("/admin/reset")
async def admin_reset(request: Request, user: str = Depends(require_auth)):
    form = parse_form_body(await request.body())
    deleted = await reset_player(int(form["player_id"]), form.get("period", "all"))
    return admin_redirect(f"Reset hotový. Vymazané reporty: {deleted}.")
