
from __future__ import annotations

import asyncio
import csv
import html
import io
import json
import os
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import parse_qs, quote, urlencode

import asyncpg
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
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
POWER_PERIODS: dict[str, tuple[str, str]] = {
    "24h": ("24h", "24 hours"),
    "7d": ("7 dní", "7 days"),
    "30d": ("30 dní", "30 days"),
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
    """Fetch ROYAL SOLDIERS from GGE Tracker and store current + historic might points."""
    db = ensure_pool()
    alliance_path = f"/alliances/name/{quote(ALLIANCE_NAME)}"
    alliance = await tracker_json_async(alliance_path)
    alliance_id = to_int(alliance.get("alliance_id"))
    if not alliance_id:
        raise RuntimeError(f"GGE Tracker did not return alliance_id for {ALLIANCE_NAME!r}.")

    detail = await tracker_json_async(f"/alliances/id/{alliance_id}")
    players = detail.get("players") or []
    history: list[dict[str, Any]] = []
    try:
        stats = await tracker_json_async(f"/statistics/alliance/{alliance_id}")
        history = (stats.get("points") or {}).get("player_might_history") or []
    except (urlerror.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"[DASHBOARD] Tracker history sync failed: {type(exc).__name__}: {exc}")

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
    return {"players": len(players), "history": len(history)}


async def tracker_sync_loop() -> None:
    while True:
        try:
            result = await sync_tracker_power()
            print(
                f"[DASHBOARD] Synced GGE Tracker power: {result['players']} players, "
                f"{result['history']} history points"
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
        ("might", "/might", "Moc"),
        ("export", "/export.csv", "CSV export"),
        ("admin", "/admin", "Admin"),
    ]
    nav_html = "".join(
        f'<a class="nav-link {"active" if key == active else ""}" href="{href}">{label}</a>'
        for key, href, label in nav
    )
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
    .shell {{ width:min(1320px, calc(100% - 28px)); margin:0 auto; padding:22px 0 48px; }}
    .topbar {{ display:flex; justify-content:space-between; align-items:center; gap:16px; position:sticky; top:0; z-index:10; padding:12px 0 18px; backdrop-filter:blur(18px); }}
    .brand {{ display:flex; align-items:center; gap:12px; text-decoration:none; }}
    .crest {{ width:42px; height:42px; border-radius:12px; display:grid; place-items:center; background:linear-gradient(135deg,#f5c451,#7b4a12); color:#1a1205; box-shadow:var(--shadow); font-size:24px; }}
    .brand strong {{ display:block; font-size:17px; }} .brand span span {{ color:var(--muted); font-size:13px; }}
    .nav,.periods,.actions,.mini-form {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; }}
    .nav-link,.btn,.tab {{ border:1px solid var(--line); background:rgba(255,255,255,.04); color:var(--text); text-decoration:none; border-radius:999px; padding:9px 13px; font-weight:800; font-size:14px; }}
    .nav-link.active,.btn.primary,.tab.active {{ background:linear-gradient(135deg,var(--gold),#e19b31); color:#1d1405; border-color:rgba(245,196,81,.65); }}
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
    @media (max-width:1050px) {{ .hero,.grid,.admin-grid {{ grid-template-columns:1fr; }} .cards {{ grid-template-columns:repeat(3,minmax(0,1fr)); }} }}
    @media (max-width:680px) {{ .shell {{ width:min(100% - 18px,1320px); padding-top:12px; }} .topbar {{ align-items:flex-start; flex-direction:column; }} .cards {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .panel {{ overflow-x:auto; }} th,td {{ padding:10px; }} .report-item {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <main class="shell">
    <header class="topbar">
      <a class="brand" href="/"><span class="crest">♛</span><span><strong>ROYAL SOLDIERS</strong><span>Reporty · ratio · power</span></span></a>
      <nav class="nav">{nav_html}</nav>
    </header>
    {body}
    <footer>Public mód je read-only. Admin akcie sú chránené heslom a zapisujú priamo do rovnakej databázy ako Discord bot.</footer>
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


async def fetch_power_data(period: str) -> dict[str, Any]:
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
    return {"players": [dict(row) for row in rows], "last_sync": last_sync, "period": period}


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


def render_power_tabs(period: str, base: str = "/might") -> str:
    return "".join(
        f'<a class="tab {"active" if key == period else ""}" href="{base}?period={key}">{esc(label)}</a>'
        for key, (label, _) in POWER_PERIODS.items()
    )


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
    period_label_text = POWER_PERIODS[period][0]
    rows = []
    total_might = 0
    total_delta = 0
    for index, row in enumerate(data["players"], start=1):
        current = int(row["might_current"] or 0)
        baseline = int(row["baseline_might"] or current)
        delta = current - baseline
        total_might += current
        total_delta += delta
        delta_class = "good" if delta > 0 else "bad" if delta < 0 else ""
        updated_at = row["tracker_updated_at"] or row["synced_at"]
        updated_label = updated_at.strftime("%d.%m. %H:%M") if hasattr(updated_at, "strftime") else "—"
        rows.append(
            "<tr>"
            f"<td>{index}</td>"
            f'<td><a class="player-link" href="/might/{int(row["player_id"])}?period={period}">{esc(row["player_name"])}</a></td>'
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
        <div class="periods">{render_power_tabs(period)}</div>
      </div>
      <div class="filters hero-card">
        <h3>Zdroj</h3>
        <p class="subtitle" style="margin:0">Server: <strong>{esc(TRACKER_SERVER)}</strong><br>Aliancia: <strong>{esc(ALLIANCE_NAME)}</strong><br>Interval syncu: približne každých {fmt_number(TRACKER_SYNC_INTERVAL_SECONDS // 60)} min.</p>
      </div>
    </section>
    <section class="cards" style="grid-template-columns:repeat(3,minmax(0,1fr))">{cards_html}</section>
    <section class="panel"><div class="panel-head"><h2>Členovia podľa power</h2><span class="pill">{esc(period_label_text)}</span></div>
      <table><thead><tr><th>#</th><th>Hráč</th><th class="num">Power</th><th class="num">Zmena</th><th class="num">Level</th><th>Update</th></tr></thead><tbody>{body_rows}</tbody></table>
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
        <a class="tab" href="/might?period={period}">← Späť na Moc</a>
        <h1 style="margin-top:16px">{esc(player["player_name"])}</h1>
        <p class="subtitle">Vývoj moci hráča podľa dát z GGE Trackeru. Player ID: <code>{int(player_id)}</code></p>
        <div class="periods">{render_power_tabs(period, f"/might/{int(player_id)}")}</div>
      </div>
      <div class="panel"><div class="panel-head"><h2>Trend power</h2></div><div class="panel-body">{render_power_chart(history)}</div></div>
    </section>
    <section class="cards" style="grid-template-columns:repeat(3,minmax(0,1fr))">{cards_html}</section>
    <section class="panel"><div class="panel-head"><h2>Posledné body</h2></div><table><thead><tr><th>Čas</th><th class="num">Power</th></tr></thead><tbody>{history_rows}</tbody></table></section>
    """
    return layout(f"{player['player_name']} · Moc", body, active="might")


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
        <div class="panel-head"><h2>Leaderboard</h2><a class="btn" href="/export.csv?{query_string(period, date_from, date_to)}">Stiahnuť CSV</a></div>
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
    <section class="panel" style="margin-top:18px"><div class="panel-head"><h2>Report history</h2><div class="actions"><span class="pill">{first_report}–{last_report} z {report_count}</span><a class="btn" href="/admin/reports.csv">CSV reporty</a></div></div><table><thead><tr><th>Čas</th><th>Hráč</th><th class="num">Killy</th><th class="num">Straty</th><th class="num">Ratio</th><th class="num">Message</th><th>Akcie</th></tr></thead><tbody>{report_table}</tbody></table>{pagination}</section>
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


def csv_response(filename: str, headers: list[str], rows: list[list[Any]]) -> Response:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(headers)
    writer.writerows(rows)
    return Response(output.getvalue(), media_type="text/csv; charset=utf-8", headers={"Content-Disposition": f'attachment; filename="{filename}"'})


def admin_redirect(message: str) -> RedirectResponse:
    return RedirectResponse(f"/admin?{urlencode({'message': message})}", status_code=303)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
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
    data = await fetch_power_data(period)
    return HTMLResponse(render_power_page(data))


@app.get("/might/{player_id}", response_class=HTMLResponse)
async def power_player_page(player_id: int, request: Request):
    period = power_period(request)
    data = await fetch_power_player_data(player_id, period)
    return HTMLResponse(render_power_player_page(player_id, data))


@app.get("/export.csv")
async def export_csv(request: Request):
    period, date_from, date_to = get_filters(request)
    data = await fetch_dashboard_data(period=period, date_from=date_from, date_to=date_to)
    rows = []
    for index, row in enumerate(data["leaderboard"], start=1):
        losses = int(row["total_losses"] or 0)
        kills = int(row["total_kills"] or 0)
        rows.append([index, int(row["player_id"]), row["player_name"], int(row["report_count"] or 0), kills, losses, fmt_ratio(losses, kills)])
    return csv_response("gge-leaderboard.csv", ["rank", "player_id", "player_name", "reports", "kills", "losses", "ratio"], rows)


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


@app.get("/admin/reports.csv")
async def admin_reports_csv(user: str = Depends(require_auth)):
    data = await fetch_admin_data(report_limit=100)
    rows = [
        [
            row["created_at"].isoformat(),
            int(row["guild_id"]),
            int(row["message_id"]),
            int(row["attachment_id"]),
            int(row["player_id"]),
            row["player_name"],
            int(row["enemy_kills"]),
            int(row["own_losses"]),
            fmt_ratio(int(row["own_losses"]), int(row["enemy_kills"])),
        ]
        for row in data["reports"]
    ]
    return csv_response("gge-reports.csv", ["created_at", "guild_id", "message_id", "attachment_id", "player_id", "player_name", "kills", "losses", "ratio"], rows)


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
