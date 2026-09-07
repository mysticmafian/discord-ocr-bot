"""
Discord bot: automaticky rozpozna screenshot battle reportu (výpis boja)
a odpovie s pomerom strát vojska medzi útočníkom a obrancom.

Princíp:
1. V obrázku nájdeme červené (stratové) čísla pomocou farebnej masky v HSV.
2. Z nájdených "blobov" vyberieme dvojicu, ktorá vyzerá ako dve čísla
   na rovnakej výške, na opačných stranách obrázka (ľavá strana = útočník,
   pravá strana = obranca) - presne ako v UI hry.
3. Každé číslo prečítame cez OCR (Tesseract) a vypočítame pomer.

Toto NEspolieha na presné pixelové súradnice, takže by malo fungovať
aj pri rôznych rozlíšeniach screenshotu (mobil / PC / rôzny zoom),
pokiaľ farba stratových čísel ostáva červená a layout je podobný.
"""

import io
import os
import re

import cv2
import numpy as np
import pytesseract
import discord

# ---------------------------------------------------------------------------
# KONFIGURÁCIA
# ---------------------------------------------------------------------------

# Token bota - najlepšie ako premenná prostredia, aby nebol v kóde.
TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "VLOZ_SI_TU_TOKEN")

# Ak chceš, aby bot reagoval len v konkrétnych kanáloch, vypíš ich ID sem.
# Prázdny zoznam = reaguje vo všetkých kanáloch, kam má prístup.
ALLOWED_CHANNEL_IDS = []  # napr. [123456789012345678]

# Ak sa nepodarí rozpoznať dve čísla, bot môže na správu reagovať emoji,
# aby bolo jasné, že screenshot nevie spracovať. Nastav na None, ak nechceš.
FAILURE_REACTION = ""

# Windows: ak Tesseract nie je v PATH, odkomentuj a nastav cestu, napr.:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ---------------------------------------------------------------------------
# ROZPOZNÁVANIE ČÍSEL V OBRÁZKU
# ---------------------------------------------------------------------------

def _extract_negative_numbers(text: str):
    """
    Z OCR textu vytiahne záporné čísla.
    Funguje aj pri medzerách: "-5 488" -> 5488.
    """
    results = []
    for match in re.finditer(r"-\s*([0-9][0-9\s]{0,14})", text):
        digits = re.sub(r"[^0-9]", "", match.group(1))
        if digits:
            try:
                results.append(int(digits))
            except ValueError:
                pass
    return results


def _ocr_loss_from_roi(img_bgr: np.ndarray, rect):
    """
    Prečíta stratové číslo z relatívneho výrezu obrázka.

    rect = (x0, y0, x1, y1), všetko v rozsahu 0..1.

    Namiesto spoliehania sa iba na presný odtieň červenej skúšame viac
    OCR variantov. To je spoľahlivejšie pri zmenšených Discord obrázkoch.
    """
    h, w = img_bgr.shape[:2]
    x0, y0, x1, y1 = rect

    xa = max(0, min(w, int(x0 * w)))
    ya = max(0, min(h, int(y0 * h)))
    xb = max(0, min(w, int(x1 * w)))
    yb = max(0, min(h, int(y1 * h)))

    crop = img_bgr[ya:yb, xa:xb]
    if crop.size == 0:
        return None

    # Malé Discord náhľady výrazne zväčšíme.
    target_h = 260
    scale = max(2.0, target_h / max(1, crop.shape[0]))
    scale = min(scale, 8.0)

    big = cv2.resize(
        crop,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )

    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)

    otsu = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    # Variant zameraný na výrazne červený text.
    b, g, r = cv2.split(big)
    strongest_other = np.maximum(g, b).astype(np.int16)
    red_dominance = r.astype(np.int16) - strongest_other
    red_only = np.where(
        (r > 135) & (red_dominance > 35),
        0,
        255,
    ).astype(np.uint8)

    candidates = []

    # PSM 6: blok textu; PSM 11: riedky text.
    # Pri jednom screenshote býva lepší 6, pri inom 11.
    for processed in (gray, otsu, red_only):
        for psm in (6, 11):
            text = pytesseract.image_to_string(
                processed,
                config=(
                    f"--psm {psm} "
                    "-c tessedit_char_whitelist=0123456789- "
                ),
            )
            candidates.extend(_extract_negative_numbers(text))

    if not candidates:
        return None

    # Ak OCR ten istý výsledok zachytí viackrát, uprednostníme ho.
    # Pri zhode frekvencie preferujeme väčšie číslo, pretože strata
    # je zvyčajne viacmiestna a tým odfiltrujeme drobné OCR artefakty.
    counts = {}
    for value in candidates:
        counts[value] = counts.get(value, 0) + 1

    return max(counts, key=lambda value: (counts[value], value))


def _analyze_known_layouts(img_bgr: np.ndarray):
    """
    Skúsi známe rozloženia battle reportu.

    Podporuje:
    1. celý screenshot hry,
    2. orezaný spodný battle-report panel.
    """
    h, w = img_bgr.shape[:2]
    aspect = w / max(1, h)

    layouts = []

    # Orezaný report je veľmi široký a nízky.
    if aspect >= 2.5:
        layouts.append((
            (0.04, 0.52, 0.34, 0.98),   # útočník
            (0.69, 0.52, 0.995, 0.98),  # obranca
        ))

        # O niečo širší fallback pre rôzne orezy.
        layouts.append((
            (0.00, 0.42, 0.40, 1.00),
            (0.62, 0.42, 1.00, 1.00),
        ))

    # Celý screenshot – battle report je dole.
    else:
        layouts.append((
            (0.025, 0.765, 0.205, 0.855),  # útočník
            (0.425, 0.765, 0.600, 0.855),  # obranca
        ))

        # Fallback s väčším výrezom.
        layouts.append((
            (0.015, 0.720, 0.230, 0.890),
            (0.405, 0.720, 0.625, 0.890),
        ))

    for attacker_rect, defender_rect in layouts:
        attacker_loss = _ocr_loss_from_roi(img_bgr, attacker_rect)
        defender_loss = _ocr_loss_from_roi(img_bgr, defender_rect)

        if attacker_loss is not None and defender_loss is not None:
            return attacker_loss, defender_loss

    return None


def analyze_battle_report(image_bytes: bytes):
    """
    Hlavná funkcia.

    Vráti:
        (attacker_loss, defender_loss)

    alebo None, ak OCR straty nerozpozná.
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        return None

    return _analyze_known_layouts(img)


def format_ratio(attacker_loss: int, defender_loss: int) -> str:
    """Naformátuje výsledok ako 'X : Y', kde menšia strana = 1."""
    if attacker_loss == 0 or defender_loss == 0:
        return f"{attacker_loss} : {defender_loss}"

    smaller = min(attacker_loss, defender_loss)
    left = attacker_loss / smaller
    right = defender_loss / smaller

    def fmt(v: float) -> str:
        if abs(v - round(v)) < 0.05:
            return str(int(round(v)))
        return str(round(v, 2)).replace(".", ",")

    return f"{fmt(left)} : {fmt(right)}"


# ---------------------------------------------------------------------------
# DISCORD BOT
# ---------------------------------------------------------------------------

intents = discord.Intents.default()
intents.message_content = True  # nutné pre čítanie príloh v obsahu správy

client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print(f"Prihlásený ako {client.user} (ID: {client.user.id})")


@client.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    if ALLOWED_CHANNEL_IDS and message.channel.id not in ALLOWED_CHANNEL_IDS:
        return

    image_attachments = [
        a for a in message.attachments
        if a.content_type and a.content_type.startswith("image/")
    ]
    if not image_attachments:
        return

    for attachment in image_attachments:
        try:
            image_bytes = await attachment.read()
        except discord.HTTPException:
            continue

        result = analyze_battle_report(image_bytes)
        if result is None:
            if FAILURE_REACTION:
                try:
                    await message.add_reaction(FAILURE_REACTION)
                except discord.HTTPException:
                    pass
            continue

        attacker_loss, defender_loss = result
        ratio = format_ratio(attacker_loss, defender_loss)
        await message.reply(
    f"**Battle Report ratio is: {ratio.replace(',', '.')}**",
    mention_author=False,
)


if __name__ == "__main__":
    if TOKEN == "VLOZ_SI_TU_TOKEN":
        raise SystemExit(
            "Nastav token bota - buď premennú prostredia DISCORD_BOT_TOKEN, "
            "alebo priamo v premennej TOKEN v tomto súbore."
        )
    client.run(TOKEN)
