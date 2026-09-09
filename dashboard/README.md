# GGE Report Dashboard

Read-only web dashboard pre štatistiky z bota.

## Railway setup

V Railway vytvor novú service z rovnakého GitHub repozitára.

Nastavenia service:

- Root Directory: "/"
- Dockerfile Path: "dashboard/Dockerfile"
- Variables:
  - "DATABASE_URL" – rovnaká PostgreSQL URL ako používa bot
  - "DASHBOARD_USERNAME" – napríklad "admin"
  - "DASHBOARD_PASSWORD" – tvoje vlastné heslo

Dashboard používa rovnakú databázu ako bot a nič do nej nezapisuje.
