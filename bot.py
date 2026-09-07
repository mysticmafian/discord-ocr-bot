"""
Discord bot: rozpozna battle report a odpovie:
- pomerom strát útočník : obranca
- percentom obrancovej armády, ktorú útočník zabil

Bot rozozná stranu obrancu podľa textu "Obranca" alebo "Defender",
takže obranca môže byť napravo aj naľavo.
"""

import os
import re
from difflib import SequenceMatcher

import cv2
import numpy as np
import pytesseract
import discord


# ---------------------------------------------------------------------------
# KONFIGURÁCIA
# ---------------------------------------------------------------------------

TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "VLOZ_SI_TU_TOKEN")
ALLOWED_CHANNEL_IDS = []
FAILURE_REACTION = ""


# ---------------------------------------------------------------------------
# OCR POMOCNÉ FUNKCIE
# ---------------------------------------------------------------------------

def _crop_relative(img_bgr: np.ndarray, rect):
    """Vyreže relatívny obdĺžnik (x0, y0, x1, y1), hodnoty 0..1."""
    h, w = img_bgr.shape[:2]
    x0, y0, x1, y1 = rect

    xa = max(0, min(w, int(x0 * w)))
    ya = max(0, min(h, int(y0 * h)))
    xb = max(0, min(w, int(x1 * w)))
    yb = max(0, min(h, int(y1 * h)))

    crop = img_bgr[ya:yb, xa:xb]
    return crop if crop.size else None


def _prepare_variants(crop: np.ndarray):
    """Vytvorí viac verzií výrezu, aby mal Tesseract vyššiu šancu."""
    if crop is None or crop.size == 0:
        return []

    target_h = 180
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

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9,
    )

    return [gray, otsu, adaptive]


def _ocr_label(img_bgr: np.ndarray, rect) -> str:
    """OCR textového nadpisu panelu."""
    crop = _crop_relative(img_bgr, rect)
    texts = []

    for processed in _prepare_variants(crop):
        for psm in (7, 6):
            text = pytesseract.image_to_string(
                processed,
                config=f"--psm {psm}",
            )
            text = re.sub(r"[^A-Za-zÀ-ž]", "", text).lower()
            if text:
                texts.append(text)

    return " ".join(texts)


def _defender_score(text: str) -> float:
    """
    Skóre podobnosti k slovám Obranca / Defender.
    Pomáha aj keď OCR spraví malú chybu.
    """
    clean = re.sub(r"[^a-z]", "", text.lower())
    if not clean:
        return 0.0

    targets = ("obranca", "defender")

    if any(target in clean for target in targets):
        return 1.0

    scores = []
    for target in targets:
        # porovná aj menšie kúsky OCR textu
        scores.append(SequenceMatcher(None, clean, target).ratio())

        for i in range(max(1, len(clean) - len(target) + 1)):
            part = clean[i:i + len(target)]
            scores.append(SequenceMatcher(None, part, target).ratio())

    return max(scores, default=0.0)


def _extract_integer(text: str):
    """Z OCR textu vyberie číslice: '138 318' -> 138318."""
    digits = re.sub(r"[^0-9]", "", text)
    if not digits:
        return None

    try:
        return int(digits)
    except ValueError:
        return None


def _ocr_number_from_roi(img_bgr: np.ndarray, rect):
    """
    Prečíta jedno číslo zo známeho riadku.
    Pri strate nepotrebujeme mínus - ROI už presne určuje riadok strát.
    """
    crop = _crop_relative(img_bgr, rect)
    candidates = []

    for processed in _prepare_variants(crop):
        for psm in (7, 8, 13):
            text = pytesseract.image_to_string(
                processed,
                config=(
                    f"--psm {psm} "
                    "-c tessedit_char_whitelist=0123456789 "
                ),
            )
            value = _extract_integer(text)
            if value is not None:
                candidates.append(value)

    if not candidates:
        return None

    counts = {}
    for value in candidates:
        counts[value] = counts.get(value, 0) + 1

    # Preferuj výsledok, ktorý OCR zopakovalo najčastejšie.
    return max(counts, key=lambda value: (counts[value], value))


# ---------------------------------------------------------------------------
# LAYOUTY BATTLE REPORTU
# ---------------------------------------------------------------------------

def _get_layouts(img_bgr: np.ndarray):
    """
    Vráti dvojice panelov. Každý panel má:
    label = Obranca/Defender alebo Útočník/Attacker
    total = počet vojska pred stratou
    loss  = počet strateného vojska
    """
    h, w = img_bgr.shape[:2]
    aspect = w / max(1, h)

    if aspect >= 2.5:
        # OREZANÝ battle report
        return [
            [
                {
                    "label": (0.00, 0.00, 0.31, 0.25),
                    "total": (0.13, 0.52, 0.31, 0.75),
                    "loss":  (0.13, 0.69, 0.31, 0.94),
                },
                {
                    "label": (0.69, 0.00, 1.00, 0.25),
                    "total": (0.82, 0.52, 0.995, 0.75),
                    "loss":  (0.82, 0.69, 0.995, 0.94),
                },
            ],
            # Širší fallback pre trochu inak orezané reporty
            [
                {
                    "label": (0.00, 0.00, 0.40, 0.30),
                    "total": (0.10, 0.46, 0.39, 0.76),
                    "loss":  (0.10, 0.67, 0.39, 1.00),
                },
                {
                    "label": (0.60, 0.00, 1.00, 0.30),
                    "total": (0.70, 0.46, 1.00, 0.76),
                    "loss":  (0.70, 0.67, 1.00, 1.00),
                },
            ],
        ]

    # CELÝ SCREENSHOT
    return [
        [
            {
                "label": (0.015, 0.680, 0.190, 0.718),
                "total": (0.090, 0.775, 0.185, 0.805),
                "loss":  (0.090, 0.800, 0.185, 0.835),
            },
            {
                "label": (0.415, 0.680, 0.595, 0.718),
                "total": (0.500, 0.775, 0.590, 0.805),
                "loss":  (0.500, 0.800, 0.590, 0.835),
            },
        ],
        # Fallback s mierne väčšími výrezmi
        [
            {
                "label": (0.010, 0.665, 0.205, 0.730),
                "total": (0.075, 0.760, 0.200, 0.815),
                "loss":  (0.075, 0.795, 0.200, 0.850),
            },
            {
                "label": (0.405, 0.665, 0.610, 0.730),
                "total": (0.480, 0.760, 0.605, 0.815),
                "loss":  (0.480, 0.795, 0.605, 0.850),
            },
        ],
    ]


# ---------------------------------------------------------------------------
# ANALÝZA REPORTU
# ---------------------------------------------------------------------------

def analyze_battle_report(image_bytes: bytes):
    """
    Vráti:
        (attacker_loss, defender_loss, defender_total)

    defender_total = počet obrancov pred bitkou.
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        return None

    for panels in _get_layouts(img):
        left, right = panels

        left_label = _ocr_label(img, left["label"])
        right_label = _ocr_label(img, right["label"])

        left_score = _defender_score(left_label)
        right_score = _defender_score(right_label)

        # Musíme vedieť, ktorá strana je Obranca / Defender.
        if max(left_score, right_score) < 0.55:
            continue

        if left_score > right_score:
            defender_panel = left
            attacker_panel = right
        else:
            defender_panel = right
            attacker_panel = left

        attacker_loss = _ocr_number_from_roi(img, attacker_panel["loss"])
        defender_loss = _ocr_number_from_roi(img, defender_panel["loss"])
        defender_total = _ocr_number_from_roi(img, defender_panel["total"])

        if (
            attacker_loss is not None
            and defender_loss is not None
            and defender_total is not None
            and defender_total > 0
        ):
            return attacker_loss, defender_loss, defender_total

    return None


def format_ratio(attacker_loss: int, defender_loss: int) -> str:
    """Pomer strát útočník : obranca."""
    if attacker_loss == 0 or defender_loss == 0:
        return f"{attacker_loss} : {defender_loss}"

    smaller = min(attacker_loss, defender_loss)
    left = attacker_loss / smaller
    right = defender_loss / smaller

    def fmt(v: float) -> str:
        if abs(v - round(v)) < 0.05:
            return str(int(round(v)))
        return f"{v:.2f}".rstrip("0").rstrip(".")

    return f"{fmt(left)} : {fmt(right)}"


def format_defender_killed_percent(defender_loss: int, defender_total: int) -> str:
    """
    Vypočíta percento zabitých obrancov:
        strata obrancu / počet obrancov pred bitkou * 100
    """
    if defender_total <= 0:
        return "0"

    percent = (defender_loss / defender_total) * 100
    return f"{percent:.1f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------------------
# DISCORD BOT
# ---------------------------------------------------------------------------

intents = discord.Intents.default()
intents.message_content = True

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

        attacker_loss, defender_loss, defender_total = result

        ratio = format_ratio(attacker_loss, defender_loss)
        killed_percent = format_defender_killed_percent(
            defender_loss,
            defender_total,
        )

        await message.reply(
            f"**Battle Report ratio is:  {ratio}**\n"
            f"**Defenders killed: {killed_percent}%**",
            mention_author=False,
        )


if __name__ == "__main__":
    if TOKEN == "VLOZ_SI_TU_TOKEN":
        raise SystemExit(
            "Nastav token bota cez premennú prostredia DISCORD_BOT_TOKEN."
        )

    client.run(TOKEN)
