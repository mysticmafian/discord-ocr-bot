
from __future__ import annotations

import html
import os
import secrets
from contextlib import asynccontextmanager
from typing import Any

import asyncpg
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials


DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
DASHBOARD_USERNAME = os.getenv("DASHBOARD_USERNAME", "admin").strip()
DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "").strip()

PERIODS: dict[str, tuple[str, int | None]] = {
    "1d": ("1 deň", 1),
    "7d": ("7 dní", 7),
    "all": ("Celé obdobie", None),
}

security = HTTPBasic()
pool: asyncpg.Pool | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool
    if DATABASE_URL:
        pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    yield
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


def normalize_period(period: str | None) -> tuple[str, str, int | None]:
    key = period if period in PERIODS else "all"
    label, days = PERIODS[key]
    return key, label, days


def fmt_number(value: int | None) -> str:
    return f"{int(value or 0):,}".replace(",", " ")


def fmt_ratio(losses: int | None, kills: int | None) -> str:
    losses = int(losses or 0)
    kills = int(kills or 0)
    if losses == 0:
        return "1:∞" if kills else "1:1.00"
    return f"1:{kills / losses:.2f}"


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


async def fetch_dashboard_data(period_days: int | None) -> dict[str, Any]:
    if pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DATABASE_URL is missing or database connection is not ready.",
        )

    async with pool.acquire() as conn:
        alliance = await conn.fetchrow(
            """
            SELECT COUNT(*) AS report_count,
                   COALESCE(SUM(own_losses), 0) AS total_losses,
                   COALESCE(SUM(enemy_kills), 0) AS total_kills,
                   COUNT(DISTINCT player_id) AS player_count
            FROM battle_reports
            WHERE ($1::integer IS NULL OR created_at >= NOW() - ($1 * INTERVAL '1 day'))
            """,
            period_days,
        )
        leaderboard = await conn.fetch(
            """
            SELECT player_id,
                   MAX(player_name) AS player_name,
                   COUNT(*) AS report_count,
                   COALESCE(SUM(own_losses), 0) AS total_losses,
                   COALESCE(SUM(enemy_kills), 0) AS total_kills
            FROM battle_reports
            WHERE ($1::integer IS NULL OR created_at >= NOW() - ($1 * INTERVAL '1 day'))
            GROUP BY player_id
            ORDER BY total_kills DESC, total_losses ASC, report_count DESC, player_name ASC
            LIMIT 25
            """,
            period_days,
        )
        recent = await conn.fetch(
            """
            SELECT player_name, own_losses, enemy_kills, created_at
            FROM battle_reports
            ORDER BY created_at DESC
            LIMIT 20
            """
        )
        blacklist_count = await conn.fetchval(
            "SELECT COUNT(*) FROM battle_report_blacklist"
        )

    return {
        "alliance": dict(alliance),
        "leaderboard": [dict(row) for row in leaderboard],
        "recent": [dict(row) for row in recent],
        "blacklist_count": int(blacklist_count or 0),
    }


def render_dashboard(data: dict[str, Any], period_key: str, period_label: str) -> str:
    alliance = data["alliance"]
    total_losses = int(alliance["total_losses"] or 0)
    total_kills = int(alliance["total_kills"] or 0)
    cards = [
        ("Reporty", fmt_number(alliance["report_count"])),
        ("Hráči", fmt_number(alliance["player_count"])),
        ("Killy", fmt_number(total_kills)),
        ("Straty", fmt_number(total_losses)),
        ("Ratio", fmt_ratio(total_losses, total_kills)),
        ("Blacklist", fmt_number(data["blacklist_count"])),
    ]

    card_html = "\n".join(
        f'<section class="card"><span>{esc(label)}</span><strong>{esc(value)}</strong></section>'
        for label, value in cards
    )

    tabs = "\n".join(
        f'<a class="tab {"active" if key == period_key else ""}" href="/?period={key}">{esc(label)}</a>'
        for key, (label, _) in PERIODS.items()
    )

    rows = []
    for index, row in enumerate(data["leaderboard"], start=1):
        losses = int(row["total_losses"] or 0)
        kills = int(row["total_kills"] or 0)
        rows.append(
            "<tr>"
            f"<td>{index}</td>"
            f"<td>{esc(row['player_name'])}</td>"
            f"<td>{fmt_number(kills)}</td>"
            f"<td>{fmt_number(losses)}</td>"
            f"<td>{fmt_ratio(losses, kills)}</td>"
            f"<td>{fmt_number(row['report_count'])}</td>"
            "</tr>"
        )
    leaderboard_html = "\n".join(rows) or (
        '<tr><td colspan="6" class="empty">Zatiaľ žiadne reporty.</td></tr>'
    )

    recent_rows = []
    for row in data["recent"]:
        losses = int(row["own_losses"] or 0)
        kills = int(row["enemy_kills"] or 0)
        created_at = row["created_at"].strftime("%d.%m. %H:%M")
        recent_rows.append(
            "<tr>"
            f"<td>{esc(created_at)}</td>"
            f"<td>{esc(row['player_name'])}</td>"
            f"<td>{fmt_number(kills)}</td>"
            f"<td>{fmt_number(losses)}</td>"
            f"<td>{fmt_ratio(losses, kills)}</td>"
            "</tr>"
        )
    recent_html = "\n".join(recent_rows) or (
        '<tr><td colspan="5" class="empty">Zatiaľ žiadne reporty.</td></tr>'
    )

    return f"""<!doctype html>
<html lang="sk">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GGE Report Dashboard</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #11131a;
      --panel: #181b25;
      --panel-2: #202433;
      --text: #f4f6fb;
      --muted: #98a2b3;
      --line: #30384c;
      --accent: #f6c453;
      --good: #7dd87d;
      --bad: #ff7777;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top left, #23283a 0, #11131a 42rem);
      color: var(--text);
    }}
    main {{
      width: min(1180px, calc(100% - 32px));
      margin: 0 auto;
      padding: 32px 0 48px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-end;
      margin-bottom: 22px;
    }}
    h1 {{ margin: 0; font-size: clamp(28px, 5vw, 44px); letter-spacing: 0; }}
    .subtitle {{ color: var(--muted); margin-top: 8px; }}
    .tabs {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    .tab {{
      color: var(--muted);
      text-decoration: none;
      padding: 9px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255,255,255,0.03);
    }}
    .tab.active {{ color: #1b1605; background: var(--accent); border-color: var(--accent); font-weight: 700; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .card {{
      background: linear-gradient(180deg, var(--panel-2), var(--panel));
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      min-height: 92px;
    }}
    .card span {{ display: block; color: var(--muted); font-size: 13px; }}
    .card strong {{ display: block; margin-top: 10px; font-size: 24px; white-space: nowrap; }}
    .grid {{ display: grid; grid-template-columns: 1.4fr 1fr; gap: 18px; align-items: start; }}
    .panel {{
      background: rgba(24, 27, 37, 0.9);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    .panel h2 {{ margin: 0; padding: 18px 18px 0; font-size: 19px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
    th, td {{ padding: 12px 14px; border-top: 1px solid var(--line); text-align: left; white-space: nowrap; }}
    th {{ color: var(--muted); font-weight: 600; font-size: 13px; }}
    td:nth-child(3), td:nth-child(4), td:nth-child(5), td:nth-child(6) {{ text-align: right; }}
    .empty {{ color: var(--muted); text-align: center !important; padding: 28px; }}
    footer {{ color: var(--muted); margin-top: 18px; font-size: 13px; }}
    @media (max-width: 900px) {{
      header {{ align-items: stretch; flex-direction: column; }}
      .cards {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .grid {{ grid-template-columns: 1fr; }}
      .panel {{ overflow-x: auto; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>GGE Report Dashboard</h1>
        <div class="subtitle">Prehľad štatistík aliancie za: {esc(period_label)}</div>
      </div>
      <nav class="tabs">{tabs}</nav>
    </header>
    <div class="cards">{card_html}</div>
    <div class="grid">
      <section class="panel">
        <h2>Leaderboard</h2>
        <table>
          <thead><tr><th>#</th><th>Hráč</th><th>Killy</th><th>Straty</th><th>Ratio</th><th>Reporty</th></tr></thead>
          <tbody>{leaderboard_html}</tbody>
        </table>
      </section>
      <section class="panel">
        <h2>Posledné reporty</h2>
        <table>
          <thead><tr><th>Čas</th><th>Hráč</th><th>Killy</th><th>Straty</th><th>Ratio</th></tr></thead>
          <tbody>{recent_html}</tbody>
        </table>
      </section>
    </div>
    <footer>Dashboard je read-only. Admin úpravy zatiaľ robíš cez Discord príkazy.</footer>
  </main>
</body>
</html>"""


def serialize_dashboard_data(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "alliance": data["alliance"],
        "leaderboard": data["leaderboard"],
        "recent": [
            {
                **row,
                "created_at": row["created_at"].isoformat(),
            }
            for row in data["recent"]
        ],
        "blacklist_count": data["blacklist_count"],
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, period: str = "all", user: str = Depends(require_auth)):
    period_key, period_label, period_days = normalize_period(period)
    data = await fetch_dashboard_data(period_days)
    return HTMLResponse(render_dashboard(data, period_key, period_label))


@app.get("/api/summary")
async def api_summary(period: str = "all", user: str = Depends(require_auth)):
    period_key, period_label, period_days = normalize_period(period)
    data = await fetch_dashboard_data(period_days)
    return JSONResponse(
        {
            "period": {"key": period_key, "label": period_label, "days": period_days},
            "data": serialize_dashboard_data(data),
        }
    )
