
# Discord OCR bot pre Goodgame Empire

Bot analyzuje obrázky bojových reportov, odpovie výsledkom OCR a uloží štatistiky
hráča, ktorý report poslal.

## Štatistiky

Každý úspešne rozpoznaný obrázok sa započíta autorovi správy ako jeden report:

- vlastné straty = straty útočníka,
- zabití nepriatelia = straty obrancu,
- priemerné ratio = celkoví zabití / celkové vlastné straty (vážený priemer).

Ak je obranca v reporte sivý, bot ho považuje za rift, pridá reakciu `🤏` a
nič neodpíše. Takýto report nepočíta do štatistík hráča ani aliancie.

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

Príkaz `!assign @hráč` funguje ako admin odpoveď na botovu hlášku alebo priamo
na správu s reportom. Bot priradí započítaný report vybranému hráčovi, takže sa
odobere pôvodnému autorovi a pripíše novému hráčovi.

Príkaz `!blacklist` funguje tiež ako admin odpoveď na botovu hlášku alebo priamo
na správu s reportom. Bot report vymaže zo štatistík a jeho hodnoty natrvalo
zablokuje, takže rovnaký report si už nikto nezapočíta.

Príkaz `/stats-alliance` s rovnakým výberom obdobia zobrazí spoločný súhrn
všetkých hráčov na Discord serveri.

Príkaz `/leaderboard` zobrazí TOP 10 hráčov zoradených podľa počtu zabitých
nepriateľských vojakov. Pri každom hráčovi zobrazí aj celkové straty a ratio.
Aj tu si hráč vyberá obdobie `1 deň`, `7 dní` alebo `Celé obdobie`.

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
- `BLACKLIST_REPORT_COMMAND` – admin príkaz na trvalé zablokovanie reportu,
  predvolene `!blacklist`.
- `ASSIGN_REPORT_COMMAND` – admin príkaz na priradenie reportu inému hráčovi,
  predvolene `!assign`.
- `GGE_DEBUG=1` – podrobné OCR logy.

## Web dashboard

Dashboard je samostatná Railway service v priečinku `dashboard/`. Je read-only,
čiže len číta štatistiky z rovnakej PostgreSQL databázy ako bot.

V Railway vytvor novú service z toho istého GitHub repozitára a nastav:

- Dockerfile Path: `dashboard/Dockerfile`
- `DATABASE_URL` – rovnaká hodnota ako pri botovi
- `DASHBOARD_USERNAME` – napríklad `admin`
- `DASHBOARD_PASSWORD` – tvoje vlastné heslo

Po deployi otvoríš URL novej Railway service a prihlásiš sa týmto menom a
heslom.
