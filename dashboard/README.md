# GGE Report Dashboard

Web dashboard pre štatistiky z bota.

Verejná časť je read-only. Admin časť `/admin` je chránená heslom a umožňuje
release, blacklist, assign a reset reportov priamo nad rovnakou PostgreSQL
databázou, ktorú používa Discord bot.

## Railway setup

V Railway vytvor novú service z rovnakého GitHub repozitára.

Nastavenia service:

- Root Directory: `dashboard`
- Start Command: `sh -c 'uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}'`
- Variables:
  - "DATABASE_URL" – rovnaká PostgreSQL URL ako používa bot
  - "DASHBOARD_USERNAME" – napríklad "admin"
  - "DASHBOARD_PASSWORD" – tvoje vlastné heslo

Dashboard používa rovnakú databázu ako bot a nič do nej nezapisuje.
