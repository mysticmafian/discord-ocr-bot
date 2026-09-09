# Discord OCR bot pre Goodgame Empire

Bot analyzuje obrázky bojových reportov, odpovie výsledkom OCR a uloží štatistiky
hráča, ktorý report poslal.

## Štatistiky

Každý úspešne rozpoznaný obrázok sa započíta autorovi správy ako jeden report:

- vlastné straty = straty útočníka,
- zabití nepriatelia = straty obrancu,
- priemerné ratio = celkoví zabití / celkové vlastné straty (vážený priemer).

Ak je obranca v reporte sivý, bot ho považuje za rift a odpovie iba
`rift sa nepočíta`. Takýto report nepočíta do štatistík hráča ani aliancie.

Hráč si svoje štatistiky zobrazí slash príkazom `/stats` a vyberie si obdobie
`1 deň`, `7 dní` alebo `Celé obdobie`. Rovnaká príloha sa v rámci pôvodnej
Discord správy nezapočíta dvakrát.

Bot navyše porovnáva vlastné straty a zabitých nepriateľov s predchádzajúcimi
reportmi na Discord serveri. Ak hráč pošle rovnaký report ako novú prílohu, bot
znovu zobrazí ratio a percentá, ale report druhýkrát do štatistík nepridá. Keď
už rovnaký report nahral iný hráč, bot doplní aj jeho meno.

Administrátor môže cez `/stats-reset` vybrať hráča aj obdobie, ktorého uložené
reporty sa majú vymazať. Bot oprávnenie Administrátor kontroluje aj pri vykonaní
príkazu.

Administrátor môže tiež odpovedať na botovu hlášku o započítaní reportu alebo
priamo na správu s reportom textom `!release-report`. Bot tým report vymaže zo
štatistík hráča aj aliancie a rovnaký report sa dá znova započítať správnemu
hráčovi.

Príkaz `/stats-alliance` s rovnakým výberom obdobia zobrazí spoločný súhrn
všetkých hráčov na Discord serveri.

Príkaz `/leaderboard` zobrazí TOP 10 hráčov zoradených podľa počtu zabitých
nepriateľských vojakov. Aj tu si hráč vyberá obdobie `1 deň`, `7 dní` alebo
`Celé obdobie`.

## Railway

1. Pripoj k službe Railway PostgreSQL databázu.
2. Sprístupni botovi premennú `DATABASE_URL` z PostgreSQL služby.
3. Nastav `DISCORD_BOT_TOKEN`.

Tabuľky sa vytvoria automaticky pri štarte. Ak `DATABASE_URL` nie je nastavená,
bot použije SQLite súbor `battle_stats.sqlite3`. Na Railway je v tom prípade
potrebný Volume a premenná, napríklad `STATS_DB_PATH=/data/battle_stats.sqlite3`,
inak sa dáta pri novom deployi môžu stratiť.

Voliteľné premenné:

- `STATS_COMMAND` – voliteľný starší textový príkaz, predvolene `!stats`.
- `RELEASE_REPORT_COMMAND` – admin príkaz na uvoľnenie reportu, predvolene
  `!release-report`.
- `GGE_DEBUG=1` – podrobné OCR logy.
