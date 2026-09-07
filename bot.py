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
import unicodedata

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


def _normalize_text(text: str) -> str:
    """Odstráni diakritiku a nechá iba písmená a-z."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z]", "", text)


def _role_score(text: str, targets) -> float:
    """
    Skóre podobnosti OCR textu k názvu roly.
    Používame obe roly, aby sa strany nemohli ľahko prehodiť.
    """
    clean = _normalize_text(text)
    if not clean:
        return 0.0

    if any(target in clean for target in targets):
        return 1.0

    scores = []
    for target in targets:
        scores.append(SequenceMatcher(None, clean, target).ratio())

        if len(clean) >= len(target):
            for i in range(len(clean) - len(target) + 1):
                part = clean[i:i + len(target)]
                scores.append(SequenceMatcher(None, part, target).ratio())

    return max(scores, default=0.0)


def _defender_score(text: str) -> float:
    return _role_score(text, ("obranca", "defender"))


def _attacker_score(text: str) -> float:
    return _role_score(text, ("utocnik", "attacker"))


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
# PRESNEJŠIE OCR PRE OREZANÉ / MENEJ KVALITNÉ REPORTY
# ---------------------------------------------------------------------------

def _find_red_loss_bbox_cropped(img_bgr: np.ndarray, side: str):
    """
    Nájde presný bounding box červeného stratového čísla na ľavej/pravej strane.

    Namiesto OCR veľkého výrezu najprv nájdeme samotné červené číslice.
    To výrazne pomáha pri malých a komprimovaných screenshotoch.
    """
    h, w = img_bgr.shape[:2]

    if side == "left":
        xa, xb = int(0.10 * w), int(0.32 * w)
    else:
        xa, xb = int(0.78 * w), int(0.99 * w)

    ya, yb = int(0.48 * h), int(0.95 * h)

    crop = img_bgr[ya:yb, xa:xb]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    mask = (
        cv2.inRange(
            hsv,
            np.array([0, 70, 70]),
            np.array([15, 255, 255]),
        )
        |
        cv2.inRange(
            hsv,
            np.array([165, 70, 70]),
            np.array([180, 255, 255]),
        )
    )

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    components = []

    for i in range(1, count):
        x, y, cw, ch, area = stats[i]

        # Číslice majú určitú minimálnu výšku.
        # Týmto odfiltrujeme tenké červené čiary a malé grafické artefakty.
        min_h = max(4, int(0.04 * h))
        max_h = max(min_h + 1, int(0.16 * h))

        if (
            ch >= min_h
            and ch <= max_h
            and area >= 8
        ):
            components.append(
                (x + xa, y + ya, cw, ch, area)
            )

    if not components:
        return None

    # Zoskupíme jednotlivé číslice, ktoré ležia na rovnakom riadku.
    groups = []

    for comp in sorted(
        components,
        key=lambda c: c[1] + c[3] / 2
    ):
        cy = comp[1] + comp[3] / 2
        found = False

        for group in groups:
            if abs(cy - group["cy"]) <= 0.04 * h:
                group["items"].append(comp)
                group["cy"] = sum(
                    c[1] + c[3] / 2
                    for c in group["items"]
                ) / len(group["items"])
                found = True
                break

        if not found:
            groups.append({
                "cy": cy,
                "items": [comp],
            })

    usable = []

    for group in groups:
        xs = [c[0] for c in group["items"]]
        ys = [c[1] for c in group["items"]]
        x2s = [c[0] + c[2] for c in group["items"]]
        y2s = [c[1] + c[3] for c in group["items"]]

        box = (
            min(xs),
            min(ys),
            max(x2s) - min(xs),
            max(y2s) - min(ys),
        )

        total_area = sum(c[4] for c in group["items"])

        # Straty bývajú v spodnej časti panelu.
        # Kombinujeme veľkosť textu a jeho vertikálnu pozíciu.
        score = total_area + 0.5 * group["cy"]

        usable.append((score, box))

    if not usable:
        return None

    return max(usable, key=lambda item: item[0])[1]


def _ocr_precise_number_box(img_bgr: np.ndarray, box):
    """
    OCR čísla z už presne nájdeného bounding boxu.

    Skúša viac mierok, thresholdov a PSM režimov.
    Výsledok vyberie hlasovaním.
    """
    x, y, w, h = [int(v) for v in box]
    img_h, img_w = img_bgr.shape[:2]

    pad_x = max(5, int(0.25 * w))
    pad_y = max(3, int(0.25 * h))

    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(img_w, x + w + pad_x)
    y1 = min(img_h, y + h + pad_y)

    crop = img_bgr[y0:y1, x0:x1]

    if crop.size == 0:
        return None

    candidates = []

    for scale in (4, 6, 8):
        big = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)

        otsu = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )[1]

        for processed in (gray, otsu):
            for psm in (7, 8, 13):
                text = pytesseract.image_to_string(
                    processed,
                    config=(
                        f"--psm {psm} "
                        "-c tessedit_char_whitelist=0123456789-"
                    ),
                )

                digits = re.sub(r"[^0-9]", "", text)

                if digits:
                    try:
                        candidates.append(int(digits))
                    except ValueError:
                        pass

    if not candidates:
        return None

    counts = {}

    for value in candidates:
        counts[value] = counts.get(value, 0) + 1

    # Najprv počet hlasov. Pri zhode preferujeme kratšie číslo,
    # aby jeden OCR artefakt nepridal náhodnú číslicu na začiatok.
    return max(
        counts,
        key=lambda value: (
            counts[value],
            -len(str(value)),
        ),
    )


def _ocr_total_above_loss(img_bgr: np.ndarray, loss_box, defender_loss=None):
    """
    Prečíta celkový počet vojska z riadku priamo NAD stratami.

    Skúšame niekoľko úzkych vertikálnych posunov. Ak poznáme defender_loss,
    preferujeme hodnotu >= strata a pri viacerých výsledkoch tú, na ktorej
    sa OCR zhodne najčastejšie.
    """
    x, y, w, h = [int(v) for v in loss_box]
    img_h, img_w = img_bgr.shape[:2]

    # Celkový počet je v rovnakom stĺpci ako strata, približne o 1 riadok vyššie.
    rects = [
        (
            max(0, x - int(0.45 * w)),
            max(0, y - int(2.10 * h)),
            min(img_w, x + w + int(0.45 * w)),
            max(0, y - int(0.35 * h)),
        ),
        (
            max(0, x - int(0.35 * w)),
            max(0, y - int(1.90 * h)),
            min(img_w, x + w + int(0.35 * w)),
            max(0, y - int(0.55 * h)),
        ),
        (
            max(0, x - int(0.55 * w)),
            max(0, y - int(2.30 * h)),
            min(img_w, x + w + int(0.55 * w)),
            max(0, y - int(0.45 * h)),
        ),
    ]

    candidates = []

    for x0, y0, x1, y1 in rects:
        if x1 <= x0 or y1 <= y0:
            continue

        crop = img_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            continue

        for scale in (4, 6, 8):
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

            for processed in (gray, otsu):
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

    if defender_loss is not None:
        valid = [v for v in counts if v >= defender_loss]
        if valid:
            # Pri rovnakej zhode preferuj číslo, ktoré nie je presne strata,
            # pretože to často znamená, že OCR omylom čítalo spodný riadok.
            return max(
                valid,
                key=lambda v: (
                    counts[v],
                    1 if v > defender_loss else 0,
                    -len(str(v)),
                ),
            )

    return max(
        counts,
        key=lambda v: (counts[v], -len(str(v))),
    )


def _analyze_cropped_report_precise(img_bgr: np.ndarray):
    """
    Presná analýza širokého orezaného reportu.

    1. OCR nadpisov určí, kde je Obranca / Defender.
    2. Červenou maskou nájdeme presnú polohu oboch stratových čísel.
    3. Každé číslo čítame niekoľkokrát a výsledky hlasujú.
    4. Počet obrancov čítame priamo nad jeho stratovým číslom.
    """
    left_label = _ocr_label(
        img_bgr,
        (0.00, 0.00, 0.35, 0.27),
    )

    right_label = _ocr_label(
        img_bgr,
        (0.65, 0.00, 1.00, 0.27),
    )

    left_def = _defender_score(left_label)
    right_def = _defender_score(right_label)
    left_att = _attacker_score(left_label)
    right_att = _attacker_score(right_label)

    # Vyberieme orientáciu, ktorá najlepšie sedí na OBE hlavičky:
    # Obranca/Defender na jednej strane a Útočník/Attacker na druhej.
    left_is_defender_score = left_def + right_att
    right_is_defender_score = right_def + left_att

    if max(left_is_defender_score, right_is_defender_score) < 0.80:
        return None

    left_loss_box = _find_red_loss_bbox_cropped(
        img_bgr, "left"
    )

    right_loss_box = _find_red_loss_bbox_cropped(
        img_bgr, "right"
    )

    if left_loss_box is None or right_loss_box is None:
        return None

    left_loss = _ocr_precise_number_box(
        img_bgr,
        left_loss_box,
    )

    right_loss = _ocr_precise_number_box(
        img_bgr,
        right_loss_box,
    )

    if left_loss is None or right_loss is None:
        return None

    if left_is_defender_score > right_is_defender_score:
        defender_loss = left_loss
        attacker_loss = right_loss
        defender_total = _ocr_total_above_loss(
            img_bgr,
            left_loss_box,
            defender_loss,
        )
    else:
        defender_loss = right_loss
        attacker_loss = left_loss
        defender_total = _ocr_total_above_loss(
            img_bgr,
            right_loss_box,
            defender_loss,
        )

    if defender_total is None or defender_total <= 0:
        return None

    # Logická kontrola: nemôže zomrieť viac obrancov,
    # než ich bolo pred bitkou.
    if defender_loss > defender_total:
        return None

    return attacker_loss, defender_loss, defender_total



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

    Pri širokých orezaných reportoch používa presnejšiu detekciu
    samotných červených číslic. Pri celých screenshotoch ostáva
    pôvodný layoutový fallback.
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        return None

    h, w = img.shape[:2]
    aspect = w / max(1, h)

    # Presnejší režim pre orezané battle reporty.
    if aspect >= 2.5:
        precise = _analyze_cropped_report_precise(img)

        if precise is not None:
            return precise

    # Fallback pre celé screenshoty alebo atypický orez.
    for panels in _get_layouts(img):
        left, right = panels

        left_label = _ocr_label(img, left["label"])
        right_label = _ocr_label(img, right["label"])

        left_def = _defender_score(left_label)
        right_def = _defender_score(right_label)
        left_att = _attacker_score(left_label)
        right_att = _attacker_score(right_label)

        left_is_defender_score = left_def + right_att
        right_is_defender_score = right_def + left_att

        if max(left_is_defender_score, right_is_defender_score) < 0.80:
            continue

        if left_is_defender_score > right_is_defender_score:
            defender_panel = left
            attacker_panel = right
        else:
            defender_panel = right
            attacker_panel = left

        attacker_loss = _ocr_number_from_roi(
            img,
            attacker_panel["loss"],
        )

        defender_loss = _ocr_number_from_roi(
            img,
            defender_panel["loss"],
        )

        defender_total = _ocr_number_from_roi(
            img,
            defender_panel["total"],
        )

        if (
            attacker_loss is not None
            and defender_loss is not None
            and defender_total is not None
            and defender_total > 0
            and defender_loss <= defender_total
        ):
            return (
                attacker_loss,
                defender_loss,
                defender_total,
            )

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
