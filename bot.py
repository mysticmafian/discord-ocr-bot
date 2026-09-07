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
# OCR V3: presná detekcia dvoch číselných riadkov
# ---------------------------------------------------------------------------

def _find_loss_box_v3(img_bgr: np.ndarray, side: str):
    """
    Nájde spodný červený číselný riadok v ľavom alebo pravom paneli.
    Hľadáme iba v úzkom stĺpci, kde sú čísla, takže ignorujeme erby/ikony.
    """
    h, w = img_bgr.shape[:2]

    if side == "left":
        xa, xb = int(0.14 * w), int(0.31 * w)
    else:
        xa, xb = int(0.80 * w), int(0.99 * w)

    ya, yb = int(0.28 * h), int(0.90 * h)
    crop = img_bgr[ya:yb, xa:xb]

    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    mask = (
        cv2.inRange(hsv, np.array([0, 70, 70]), np.array([15, 255, 255]))
        |
        cv2.inRange(hsv, np.array([165, 70, 70]), np.array([180, 255, 255]))
    )

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    components = []

    for i in range(1, count):
        x, y, cw, ch, area = stats[i]

        min_h = max(3, int(0.03 * h))
        max_h = max(min_h + 1, int(0.14 * h))

        if (
            min_h <= ch <= max_h
            and area >= 5
        ):
            components.append((x + xa, y + ya, cw, ch, area))

    if not components:
        return None

    groups = []

    for comp in sorted(components, key=lambda c: c[1] + c[3] / 2):
        cy = comp[1] + comp[3] / 2

        for group in groups:
            if abs(cy - group["cy"]) <= 0.03 * h:
                group["items"].append(comp)
                group["cy"] = sum(
                    c[1] + c[3] / 2 for c in group["items"]
                ) / len(group["items"])
                break
        else:
            groups.append({"cy": cy, "items": [comp]})

    candidates = []

    for group in groups:
        # Reálne číslo má zvyčajne viac oddelených červených komponentov.
        if len(group["items"]) < 2:
            continue

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

        candidates.append((group["cy"], box))

    if not candidates:
        return None

    # Straty sú spodný z dvoch číselných riadkov.
    return max(candidates, key=lambda item: item[0])[1]


def _ocr_candidates_v3(img_bgr: np.ndarray, box):
    """
    Vráti OCR kandidátov a počet hlasov pre stratové číslo.
    """
    x, y, w, h = [int(v) for v in box]
    img_h, img_w = img_bgr.shape[:2]

    pad_x = max(2, int(0.10 * w))
    pad_y = max(2, int(0.20 * h))

    crop = img_bgr[
        max(0, y - pad_y):min(img_h, y + h + pad_y),
        max(0, x - pad_x):min(img_w, x + w + pad_x),
    ]

    if crop.size == 0:
        return {}

    values = []

    for scale in (4, 6, 8):
        big = cv2.resize(
            crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )

        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
        otsu = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

        b, g, r = cv2.split(big)
        strongest_other = np.maximum(g, b).astype(np.int16)
        red_only = np.where(
            (r.astype(np.int16) > 110)
            & ((r.astype(np.int16) - strongest_other) > 25),
            0,
            255,
        ).astype(np.uint8)

        for processed in (gray, otsu, red_only):
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
                        values.append(int(digits))
                    except ValueError:
                        pass

    counts = {}

    for value in values:
        counts[value] = counts.get(value, 0) + 1

    return counts


def _ocr_total_v3(img_bgr: np.ndarray, loss_box):
    """
    Prečíta horné číslo (počet vojska pred stratami) presne nad loss riadkom.
    """
    x, y, w, h = [int(v) for v in loss_box]
    img_h, img_w = img_bgr.shape[:2]

    values = []

    # Niektoré screenshoty majú pár pixelov navyše dole,
    # preto skúšame viac tesných vertikálnych posunov.
    windows = (
        (2.0, 0.35),
        (2.3, 0.55),
        (1.9, 0.45),
        (2.5, 0.70),
    )

    for top_mul, bottom_mul in windows:
        x0 = max(0, x - int(0.40 * w))
        x1 = min(img_w, x + w + int(0.40 * w))
        y0 = max(0, y - int(top_mul * h))
        y1 = max(0, y - int(bottom_mul * h))

        if x1 <= x0 or y1 <= y0:
            continue

        crop = img_bgr[y0:y1, x0:x1]

        for scale in (4, 6, 8):
            big = cv2.resize(
                crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
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

                    digits = re.sub(r"[^0-9]", "", text)

                    if digits:
                        try:
                            values.append(int(digits))
                        except ValueError:
                            pass

    if not values:
        return None

    counts = {}

    for value in values:
        counts[value] = counts.get(value, 0) + 1

    return max(counts, key=lambda v: (counts[v], -len(str(v))))


def _choose_loss_v3(counts, total):
    """
    Vyberie najpravdepodobnejšiu stratu.
    Kľúčová kontrola: strata nikdy nemôže byť väčšia než počet vojska.
    """
    if not counts:
        return None

    valid = [
        value for value in counts
        if value > 0 and (total is None or value <= total)
    ]

    if not valid:
        return None

    return max(
        valid,
        key=lambda value: (
            counts[value],
            -len(str(value)),
        ),
    )


def _analyze_cropped_report_v3(img_bgr: np.ndarray):
    """
    Robustnejší režim pre široké orezané reporty.
    """
    left_label = _ocr_label(img_bgr, (0.00, 0.00, 0.35, 0.27))
    right_label = _ocr_label(img_bgr, (0.65, 0.00, 1.00, 0.27))

    left_def = _defender_score(left_label)
    right_def = _defender_score(right_label)
    left_att = _attacker_score(left_label)
    right_att = _attacker_score(right_label)

    left_is_defender_score = left_def + right_att
    right_is_defender_score = right_def + left_att

    if max(left_is_defender_score, right_is_defender_score) < 0.80:
        return None

    left_box = _find_loss_box_v3(img_bgr, "left")
    right_box = _find_loss_box_v3(img_bgr, "right")

    if left_box is None or right_box is None:
        return None

    # Najprv čítame CELKOVÉ počty.
    left_total = _ocr_total_v3(img_bgr, left_box)
    right_total = _ocr_total_v3(img_bgr, right_box)

    # Potom loss OCR kandidátov a odfiltrujeme nemožné hodnoty.
    left_loss = _choose_loss_v3(
        _ocr_candidates_v3(img_bgr, left_box),
        left_total,
    )
    right_loss = _choose_loss_v3(
        _ocr_candidates_v3(img_bgr, right_box),
        right_total,
    )

    if (
        left_loss is None
        or right_loss is None
        or left_total is None
        or right_total is None
    ):
        return None

    if left_is_defender_score > right_is_defender_score:
        defender_loss = left_loss
        defender_total = left_total
        attacker_loss = right_loss
    else:
        defender_loss = right_loss
        defender_total = right_total
        attacker_loss = left_loss

    if defender_total <= 0 or defender_loss > defender_total:
        return None

    return attacker_loss, defender_loss, defender_total




# ---------------------------------------------------------------------------
# OCR V4 HARD MODE: segmentácia číslic po znakoch
# ---------------------------------------------------------------------------

def _segment_red_digits(img_bgr: np.ndarray, box):
    """
    Z presného loss boxu vytiahne jednotlivé červené číslice zľava doprava.
    Mínus ignorujeme, pretože strata je už známy typ poľa.
    """
    x, y, w, h = [int(v) for v in box]
    H, W = img_bgr.shape[:2]

    pad_x = max(2, int(0.08 * w))
    pad_y = max(2, int(0.18 * h))

    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(W, x + w + pad_x)
    y1 = min(H, y + h + pad_y)

    crop = img_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return []

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    mask = (
        cv2.inRange(hsv, np.array([0, 65, 65]), np.array([18, 255, 255]))
        |
        cv2.inRange(hsv, np.array([162, 65, 65]), np.array([180, 255, 255]))
    )

    # Jemné spojenie rozbitých častí jednej číslice.
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    comps = []
    crop_h, crop_w = mask.shape[:2]

    for i in range(1, count):
        cx, cy, cw, ch, area = stats[i]

        # Odfiltrujeme mínus, bodky a šum.
        if ch < max(4, int(crop_h * 0.28)):
            continue
        if cw < 1 or area < 5:
            continue
        if cw > crop_w * 0.35:
            continue

        comps.append((cx, cy, cw, ch, area))

    if not comps:
        return []

    comps.sort(key=lambda c: c[0])

    # Zlepíme komponenty, ktoré patria tej istej číslici
    # (napr. rozbitá 8 alebo 4).
    merged = []

    for comp in comps:
        cx, cy, cw, ch, area = comp

        if not merged:
            merged.append([cx, cy, cw, ch, area])
            continue

        px, py, pw, ph, parea = merged[-1]
        gap = cx - (px + pw)

        vertical_overlap = max(
            0,
            min(cy + ch, py + ph) - max(cy, py)
        )

        overlap_ratio = vertical_overlap / max(1, min(ch, ph))

        if gap <= max(1, int(0.10 * max(ch, ph))) and overlap_ratio > 0.45:
            nx0 = min(px, cx)
            ny0 = min(py, cy)
            nx1 = max(px + pw, cx + cw)
            ny1 = max(py + ph, cy + ch)

            merged[-1] = [
                nx0,
                ny0,
                nx1 - nx0,
                ny1 - ny0,
                parea + area,
            ]
        else:
            merged.append([cx, cy, cw, ch, area])

    return [(x0, y0, crop, tuple(m)) for m in merged]


def _ocr_single_digit(digit_img: np.ndarray):
    """
    Prečíta JEDNU číslicu cez Tesseract v single-character režime.
    Vracia (digit, confidence_score_votes) alebo None.
    """
    if digit_img is None or digit_img.size == 0:
        return None

    votes = {}

    for scale in (8, 12, 16):
        big = cv2.resize(
            digit_img,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)

        variants = [
            gray,
            cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1],
            cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )[1],
        ]

        for processed in variants:
            text = pytesseract.image_to_string(
                processed,
                config=(
                    "--psm 10 "
                    "-c tessedit_char_whitelist=0123456789"
                ),
            )

            digits = re.sub(r"[^0-9]", "", text)

            if len(digits) == 1:
                d = digits[0]
                votes[d] = votes.get(d, 0) + 1

    if not votes:
        return None

    digit = max(votes, key=votes.get)
    return digit, votes[digit]


def _ocr_loss_char_by_char(img_bgr: np.ndarray, box):
    """
    Hlavný HARD MODE OCR:
    červené číslo rozseká na jednotlivé číslice a každú číta samostatne.
    """
    segmented = _segment_red_digits(img_bgr, box)

    if not segmented:
        return None

    chars = []
    confidences = []

    for _, _, crop, comp in segmented:
        cx, cy, cw, ch, _ = comp

        # Výrez jednej číslice s malým paddingom.
        px = max(1, int(0.16 * cw))
        py = max(1, int(0.12 * ch))

        x0 = max(0, cx - px)
        y0 = max(0, cy - py)
        x1 = min(crop.shape[1], cx + cw + px)
        y1 = min(crop.shape[0], cy + ch + py)

        digit_crop = crop[y0:y1, x0:x1]

        result = _ocr_single_digit(digit_crop)

        if result is None:
            return None

        digit, conf = result
        chars.append(digit)
        confidences.append(conf)

    if not chars:
        return None

    # Ochrana proti náhodnému OCR šumu.
    if sum(confidences) < len(confidences):
        return None

    try:
        return int("".join(chars))
    except ValueError:
        return None


def _ocr_loss_consensus_v4(img_bgr: np.ndarray, box, total=None):
    """
    Kombinuje:
    1. OCR celého čísla,
    2. OCR po jednotlivých čísliciach.

    HARD MODE dá prednosť znakovej segmentácii, ak dá logický výsledok.
    """
    char_value = _ocr_loss_char_by_char(img_bgr, box)

    if (
        char_value is not None
        and char_value > 0
        and (total is None or char_value <= total)
    ):
        return char_value

    # Fallback na hlasovanie celého čísla.
    counts = _ocr_candidates_v3(img_bgr, box)
    return _choose_loss_v3(counts, total)


def _analyze_cropped_report_v4(img_bgr: np.ndarray):
    """
    Najtvrdší režim pre široké orezané reporty:
    - roly určí z Obranca/Defender a Útočník/Attacker,
    - nájde červený loss riadok,
    - loss číta po JEDNOTLIVÝCH čísliciach,
    - total číta samostatne nad loss riadkom,
    - kontroluje matematickú konzistenciu.
    """
    left_label = _ocr_label(img_bgr, (0.00, 0.00, 0.35, 0.27))
    right_label = _ocr_label(img_bgr, (0.65, 0.00, 1.00, 0.27))

    left_def = _defender_score(left_label)
    right_def = _defender_score(right_label)
    left_att = _attacker_score(left_label)
    right_att = _attacker_score(right_label)

    left_is_defender_score = left_def + right_att
    right_is_defender_score = right_def + left_att

    if max(left_is_defender_score, right_is_defender_score) < 0.80:
        return None

    left_box = _find_loss_box_v3(img_bgr, "left")
    right_box = _find_loss_box_v3(img_bgr, "right")

    if left_box is None or right_box is None:
        return None

    left_total = _ocr_total_v3(img_bgr, left_box)
    right_total = _ocr_total_v3(img_bgr, right_box)

    left_loss = _ocr_loss_consensus_v4(
        img_bgr,
        left_box,
        left_total,
    )

    right_loss = _ocr_loss_consensus_v4(
        img_bgr,
        right_box,
        right_total,
    )

    if (
        left_loss is None
        or right_loss is None
        or left_total is None
        or right_total is None
    ):
        return None

    if left_is_defender_score > right_is_defender_score:
        defender_loss = left_loss
        defender_total = left_total
        attacker_loss = right_loss
    else:
        defender_loss = right_loss
        defender_total = right_total
        attacker_loss = left_loss

    if defender_total <= 0:
        return None

    if defender_loss > defender_total:
        return None

    return attacker_loss, defender_loss, defender_total




# ---------------------------------------------------------------------------
# OCR V5: dynamická detekcia panelov v CELOM / odfotenom screenshote
# ---------------------------------------------------------------------------

def _best_role_anchor_from_data(df, role: str, y_offset=0, scale=1.0):
    """
    Nájde najlepší OCR box pre Obranca/Defender alebo Útočník/Attacker.
    """
    if role == "defender":
        targets = ("obranca", "defender")
    else:
        targets = ("utocnik", "attacker")

    best = None
    best_score = 0.0

    for _, row in df.iterrows():
        raw = str(row.get("text", "") or "").strip()
        if not raw:
            continue

        clean = _normalize_text(raw)
        if not clean:
            continue

        score = _role_score(clean, targets)

        try:
            conf = float(row.get("conf", 0))
        except Exception:
            conf = 0.0

        # OCR môže mať pri moiré fotke nízku confidence, preto je podobnosť
        # názvu dôležitejšia než samotná confidence.
        combined = score + max(0.0, conf) / 500.0

        if score >= 0.52 and combined > best_score:
            x = int(float(row["left"]) / scale)
            y = int(float(row["top"]) / scale) + y_offset
            w = int(float(row["width"]) / scale)
            h = int(float(row["height"]) / scale)

            best = {
                "x": x,
                "y": y,
                "w": max(1, w),
                "h": max(1, h),
                "cx": x + max(1, w) / 2,
                "cy": y + max(1, h) / 2,
                "score": score,
                "text": raw,
            }
            best_score = combined

    return best


def _find_role_anchors_dynamic(img_bgr: np.ndarray):
    """
    Nájde hlavičky Obranca/Defender a Útočník/Attacker bez pevných súradníc.

    Toto je určené najmä pre:
    - celý screenshot,
    - fotografiu monitora/mobilu,
    - screenshot s okrajmi okolo hry.
    """
    H, W = img_bgr.shape[:2]

    # Battle report býva v dolnej polovici obrazovky.
    y0 = int(0.38 * H)
    region = img_bgr[y0:int(0.94 * H), :]

    if region.size == 0:
        return None

    scale = 2.0
    big = cv2.resize(
        region,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )

    variants = [big]

    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )[1]
    variants.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))

    best_def = None
    best_att = None

    for variant in variants:
        for psm in (6, 11, 12):
            try:
                df = pytesseract.image_to_data(
                    variant,
                    config=f"--psm {psm}",
                    output_type=pytesseract.Output.DATAFRAME,
                )
            except Exception:
                continue

            df = df.dropna(subset=["text"])

            defender = _best_role_anchor_from_data(
                df,
                "defender",
                y_offset=y0,
                scale=scale,
            )
            attacker = _best_role_anchor_from_data(
                df,
                "attacker",
                y_offset=y0,
                scale=scale,
            )

            if defender is not None:
                if best_def is None or defender["score"] > best_def["score"]:
                    best_def = defender

            if attacker is not None:
                if best_att is None or attacker["score"] > best_att["score"]:
                    best_att = attacker

    if best_def is None or best_att is None:
        return None

    # Musia byť dva rôzne panely.
    if abs(best_def["cx"] - best_att["cx"]) < 0.12 * W:
        return None

    # Hlavičky by mali byť približne na rovnakej výške.
    if abs(best_def["cy"] - best_att["cy"]) > 0.10 * H:
        return None

    return best_def, best_att


def _find_red_loss_box_near_anchor(img_bgr: np.ndarray, anchor):
    """
    Nájde červené stratové číslo POD konkrétnou hlavičkou panelu.

    Už nehľadáme "ľavú" alebo "pravú" stranu celej fotky.
    Hľadáme lokálne okolo OCR pozície slova Obranca/Defender/Útočník/Attacker.
    """
    H, W = img_bgr.shape[:2]

    cx = anchor["cx"]
    label_y = anchor["y"]
    label_h = max(anchor["h"], int(0.012 * H))

    # Čísla sú pod hlavičkou a viac smerom do stredu panelu.
    xa = max(0, int(cx - 0.11 * W))
    xb = min(W, int(cx + 0.13 * W))

    ya = max(0, int(label_y + 1.3 * label_h))
    yb = min(H, int(label_y + max(9.0 * label_h, 0.13 * H)))

    crop = img_bgr[ya:yb, xa:xb]

    if crop.size == 0:
        return None

    # Pri odfotenej obrazovke je červená často "vyblednutá" kvôli moiré.
    b, g, r = cv2.split(crop)
    r16 = r.astype(np.int16)
    strongest_other = np.maximum(g, b).astype(np.int16)

    red_mask = np.where(
        (r16 > 90)
        & ((r16 - strongest_other) > 18),
        255,
        0,
    ).astype(np.uint8)

    # HSV doplnková maska.
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hsv_mask = (
        cv2.inRange(
            hsv,
            np.array([0, 45, 55]),
            np.array([20, 255, 255]),
        )
        |
        cv2.inRange(
            hsv,
            np.array([160, 45, 55]),
            np.array([180, 255, 255]),
        )
    )

    mask = cv2.bitwise_or(red_mask, hsv_mask)

    # Odstránenie drobného moiré šumu.
    mask = cv2.medianBlur(mask, 3)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    components = []

    for i in range(1, count):
        x, y, cw, ch, area = stats[i]

        min_h = max(3, int(0.012 * H))
        max_h = max(min_h + 1, int(0.055 * H))

        if (
            min_h <= ch <= max_h
            and area >= max(5, int(0.000003 * W * H))
            and cw <= 0.10 * W
        ):
            components.append((x + xa, y + ya, cw, ch, area))

    if not components:
        return None

    # Zoskupenie komponentov do horizontálnych riadkov.
    groups = []

    for comp in sorted(components, key=lambda c: c[1] + c[3] / 2):
        cy = comp[1] + comp[3] / 2

        matched = False
        for group in groups:
            if abs(cy - group["cy"]) <= max(4, 0.012 * H):
                group["items"].append(comp)
                group["cy"] = sum(
                    c[1] + c[3] / 2 for c in group["items"]
                ) / len(group["items"])
                matched = True
                break

        if not matched:
            groups.append({"cy": cy, "items": [comp]})

    candidates = []

    for group in groups:
        items = group["items"]

        # Reálne loss číslo má typicky viac červených komponentov.
        if len(items) < 2:
            continue

        xs = [c[0] for c in items]
        ys = [c[1] for c in items]
        x2s = [c[0] + c[2] for c in items]
        y2s = [c[1] + c[3] for c in items]

        box = (
            min(xs),
            min(ys),
            max(x2s) - min(xs),
            max(y2s) - min(ys),
        )

        bw, bh = box[2], box[3]

        # Číselný riadok je širší než jedna náhodná ikonka.
        if bw < max(12, int(0.018 * W)):
            continue

        # Preferujeme spodnejší riadok pod hlavičkou,
        # pretože strata je pod celkovým počtom vojska.
        score = group["cy"] + 0.15 * bw + 2.0 * len(items)
        candidates.append((score, box))

    if not candidates:
        return None

    return max(candidates, key=lambda item: item[0])[1]


def _ocr_total_dynamic(img_bgr: np.ndarray, loss_box, loss_value=None):
    """
    Prečíta horný počet vojska relatívne k presne nájdenému loss boxu.
    Vhodné aj pre fotku monitora.
    """
    x, y, w, h = [int(v) for v in loss_box]
    H, W = img_bgr.shape[:2]

    candidates = []

    windows = (
        (2.4, 0.30),
        (2.8, 0.45),
        (2.1, 0.25),
        (3.1, 0.60),
    )

    for top_mul, bottom_mul in windows:
        x0 = max(0, x - int(0.45 * w))
        x1 = min(W, x + w + int(0.45 * w))
        y0 = max(0, y - int(top_mul * h))
        y1 = max(0, y - int(bottom_mul * h))

        if x1 <= x0 or y1 <= y0:
            continue

        crop = img_bgr[y0:y1, x0:x1]

        if crop.size == 0:
            continue

        for scale in (3, 4, 6):
            big = cv2.resize(
                crop,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )

            gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)

            # Mierne odšumenie pomáha pri fotografii LCD/monitoru.
            gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

            otsu = cv2.threshold(
                gray_blur,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )[1]

            adaptive = cv2.adaptiveThreshold(
                gray_blur,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                7,
            )

            for processed in (gray, otsu, adaptive):
                for psm in (7, 8, 13):
                    text = pytesseract.image_to_string(
                        processed,
                        config=(
                            f"--psm {psm} "
                            "-c tessedit_char_whitelist=0123456789 "
                        ),
                    )

                    digits = re.sub(r"[^0-9]", "", text)

                    if digits:
                        try:
                            value = int(digits)
                        except ValueError:
                            continue

                        if value > 0:
                            candidates.append(value)

    if not candidates:
        return None

    counts = {}
    for value in candidates:
        counts[value] = counts.get(value, 0) + 1

    valid = list(counts)

    if loss_value is not None:
        logical = [v for v in valid if v >= loss_value]
        if logical:
            valid = logical

    return max(
        valid,
        key=lambda v: (
            counts[v],
            1 if loss_value is None or v > loss_value else 0,
            -len(str(v)),
        ),
    )


def _analyze_dynamic_full_report_v5(img_bgr: np.ndarray):
    """
    V5 režim pre celý screenshot / fotografiu monitora.

    Nepoužíva pevné Y súradnice.
    Najprv cez OCR nájde slová Obranca/Defender a Útočník/Attacker,
    potom podľa nich lokalizuje čísla.
    """
    anchors = _find_role_anchors_dynamic(img_bgr)

    if anchors is None:
        return None

    defender_anchor, attacker_anchor = anchors

    defender_box = _find_red_loss_box_near_anchor(
        img_bgr,
        defender_anchor,
    )

    attacker_box = _find_red_loss_box_near_anchor(
        img_bgr,
        attacker_anchor,
    )

    if defender_box is None or attacker_box is None:
        return None

    # Najprv hard-mode OCR strát.
    defender_candidates = _ocr_candidates_v3(
        img_bgr,
        defender_box,
    )
    attacker_candidates = _ocr_candidates_v3(
        img_bgr,
        attacker_box,
    )

    # Total obrancu čítame najskôr bez obmedzenia,
    # následne použijeme matematickú kontrolu.
    defender_total_rough = _ocr_total_dynamic(
        img_bgr,
        defender_box,
        None,
    )

    defender_loss = _ocr_loss_consensus_v4(
        img_bgr,
        defender_box,
        defender_total_rough,
    )

    attacker_loss = _ocr_loss_consensus_v4(
        img_bgr,
        attacker_box,
        None,
    )

    if defender_loss is None or attacker_loss is None:
        return None

    defender_total = _ocr_total_dynamic(
        img_bgr,
        defender_box,
        defender_loss,
    )

    if defender_total is None or defender_total <= 0:
        return None

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

    # V4 HARD MODE pre široké orezané battle reporty.
    if aspect >= 2.5:
        precise = _analyze_cropped_report_v4(img)

        if precise is not None:
            return precise

    # V5 DYNAMIC MODE:
    # funguje pre celý screenshot aj fotografiu monitora.
    # Hlavičky Obranca/Defender a Útočník/Attacker nájde cez OCR,
    # takže report nemusí byť v pevnej časti obrázka.
    dynamic = _analyze_dynamic_full_report_v5(img)

    if dynamic is not None:
        return dynamic

    # Posledný fallback pre staršie layouty.
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
