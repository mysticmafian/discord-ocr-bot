# Discord OCR bot pre Goodgame Empire

Bot analyzuje obrázky bojových reportov, odpovie výsledkom OCR a uloží štatistiky
hráča, ktorý report poslal.

## Štatistiky

Každý úspešne rozpoznaný obrázok sa započíta autorovi správy ako jeden report:

- vlastné straty = straty útočníka,
- zabití nepriatelia = straty obrancu,
- priemerné ratio = celkoví zabití / celkové vlastné straty (vážený priemer).

Hráč si svoje štatistiky zobrazí správou `!stats`. Rovnaká príloha sa v rámci
pôvodnej Discord správy nezapočíta dvakrát.

## Railway

1. Pripoj k službe Railway PostgreSQL databázu.
2. Sprístupni botovi premennú `DATABASE_URL` z PostgreSQL služby.
3. Nastav `DISCORD_BOT_TOKEN`.

Tabuľky sa vytvoria automaticky pri štarte. Ak `DATABASE_URL` nie je nastavená,
bot použije SQLite súbor `battle_stats.sqlite3`. Na Railway je v tom prípade
potrebný Volume a premenná, napríklad `STATS_DB_PATH=/data/battle_stats.sqlite3`,
inak sa dáta pri novom deployi môžu stratiť.

Voliteľné premenné:

- `STATS_COMMAND` – príkaz na výpis štatistík, predvolene `!stats`.
- `GGE_DEBUG=1` – podrobné OCR logy.

