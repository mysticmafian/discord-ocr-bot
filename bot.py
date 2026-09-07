"""
Discord OCR bot - V6 FAST + ROBUST

Ciel:
- rychly na beznych orezanych battle reportoch,
- funguje aj na celom screenshote,
- funguje aj na fotografii obrazovky,
- Obranca/Defender moze byt vlavo aj vpravo.

Nova logika:
1. Tesseract spravi JEDEN hlavny OCR prechod a najde "Obranca/Defender".
2. Podla polohy nadpisu urci panel obrancu a opacny panel utocnika.
3. Z OCR dat najde hornu hodnotu vojska a spodnu hodnotu strat.
4. Ak je spodny riadok rozmazany (napr. fotka monitora), OCR sa zopakuje
   iba na malom useku priamo pod hornym cislom.
"""

import os
import re
import unicodedata
from difflib import SequenceMatcher

import cv2
import numpy as np
import pytesseract
import discord
from pytesseract import Output


TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "VLOZ_SI_TU_TOKEN")
ALLOWED_CHANNEL_IDS = []
FAILURE_REACTION = ""


# ---------------------------------------------------------------------------
# TEXT / OCR HELPERS
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text).lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z]", "", text)


def _similarity_to_role(text: str, targets) -> float:
    clean = _normalize_text(text)
    if not clean:
        return 0.0

    best = 0.0

    for target in targets:
        if target in clean:
            return 1.0

        best = max(
            best,
            SequenceMatcher(None, clean, target).ratio()
        )

    return best


def _is_defender(text: str) -> float:
    return _similarity_to_role(
        text,
        ("obranca", "defender"),
    )


def _is_attacker(text: str) -> float:
    return _similarity_to_role(
        text,
        ("utocnik", "attacker"),
    )


def _ocr_rows(img_bgr: np.ndarray):
    """
    Jeden hlavny OCR prechod.

    Pri sirokom orezanom reporte OCRujeme cely obraz.
    Pri celom screenshote/fotke OCRujeme iba spodnych ~55 %,
    kde sa battle panel nachadza.
    """
    H, W = img_bgr.shape[:2]
    aspect = W / max(1, H)

    cropped_mode = aspect >= 2.5

    if cropped_mode:
        y0 = 0
        y1 = H
        target_width = 1500
    else:
        y0 = int(0.43 * H)
        y1 = int(0.91 * H)
        target_width = 2200

    region = img_bgr[y0:y1, :]

    if region.size == 0:
        return [], cropped_mode

    scale = target_width / max(1, W)

    if cropped_mode:
        scale = min(4.0, max(1.8, scale))
    else:
        scale = min(1.7, max(1.0, scale))

    interpolation = (
        cv2.INTER_CUBIC
        if scale >= 1
        else cv2.INTER_AREA
    )

    big = cv2.resize(
        region,
        None,
        fx=scale,
        fy=scale,
        interpolation=interpolation,
    )

    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)

    data = pytesseract.image_to_data(
        gray,
        config="--psm 11",
        output_type=Output.DICT,
    )

    rows = []

    count = len(data.get("text", []))

    for i in range(count):
        text = str(data["text"][i]).strip()

        if not text:
            continue

        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = 0.0

        rows.append({
            "text": text,
            "conf": conf,
            "x": float(data["left"][i]) / scale,
            "y": float(data["top"][i]) / scale + y0,
            "w": float(data["width"][i]) / scale,
            "h": float(data["height"][i]) / scale,
        })

    return rows, cropped_mode


def _find_role_anchor(rows, role: str):
    scorer = _is_defender if role == "defender" else _is_attacker

    candidates = []

    for row in rows:
        score = scorer(row["text"])

        if score >= 0.50:
            # Textova podobnost je dolezitejsia nez OCR confidence.
            combined = score + max(0.0, row["conf"]) / 500.0
            candidates.append((combined, row))

    if not candidates:
        return None

    return max(candidates, key=lambda item: item[0])[1]


def _panel_center(anchor):
    return anchor["x"] + anchor["w"] / 2


def _numeric_groups_near_panel(
    rows,
    panel_cx: float,
    role_y: float,
    img_w: int,
    img_h: int,
    cropped_mode: bool,
):
    """
    Najde ciselne riadky v jednom bočnom paneli.

    Stlpec s cislami je vzdy mierne napravo od stredu nazvu panelu.
    To odfiltruje XP, menu, ikony a ine cisla v strede reportu.
    """
    if cropped_mode:
        min_y = role_y + 0.24 * img_h
        max_y = role_y + 0.72 * img_h
    else:
        min_y = role_y + 0.038 * img_h
        max_y = role_y + 0.145 * img_h

    min_x = panel_cx - 0.025 * img_w
    max_x = panel_cx + 0.105 * img_w

    numeric_tokens = []

    for row in rows:
        cx = row["x"] + row["w"] / 2
        cy = row["y"] + row["h"] / 2

        if not (min_x <= cx <= max_x):
            continue

        if not (min_y <= cy <= max_y):
            continue

        digits = re.sub(r"[^0-9]", "", row["text"])

        if not digits:
            continue

        numeric_tokens.append((row, digits))

    # Zoskupenie tokenov, napr. "680" + "816" -> "680816".
    groups = []

    tolerance = (
        max(3.0, 0.025 * img_h)
        if cropped_mode
        else max(4.0, 0.012 * img_h)
    )

    for row, digits in sorted(
        numeric_tokens,
        key=lambda item: item[0]["y"] + item[0]["h"] / 2,
    ):
        cy = row["y"] + row["h"] / 2

        placed = False

        for group in groups:
            if abs(cy - group["cy"]) <= tolerance:
                group["items"].append((row, digits))
                group["cy"] = sum(
                    r["y"] + r["h"] / 2
                    for r, _ in group["items"]
                ) / len(group["items"])
                placed = True
                break

        if not placed:
            groups.append({
                "cy": cy,
                "items": [(row, digits)],
            })

    result = []

    for group in groups:
        items = sorted(
            group["items"],
            key=lambda item: item[0]["x"],
        )

        digits = "".join(d for _, d in items)

        if not digits:
            continue

        try:
            value = int(digits)
        except ValueError:
            continue

        x0 = min(r["x"] for r, _ in items)
        y0 = min(r["y"] for r, _ in items)
        x1 = max(r["x"] + r["w"] for r, _ in items)
        y1 = max(r["y"] + r["h"] for r, _ in items)

        texts = [r["text"] for r, _ in items]
        confs = [max(0.0, r["conf"]) for r, _ in items]

        result.append({
            "value": value,
            "box": (x0, y0, x1 - x0, y1 - y0),
            "cy": group["cy"],
            "texts": texts,
            "has_minus": any("-" in t for t in texts),
            "avg_conf": (
                sum(confs) / len(confs)
                if confs
                else 0.0
            ),
        })

    return sorted(result, key=lambda g: g["cy"])


def _ocr_loss_below_total(
    img_bgr: np.ndarray,
    total_box,
):
    """
    Fallback iba pre rozmazany loss riadok.

    Pouziva presnu polohu horneho cisla a cita maly riadok priamo pod nim.
    Kandidat s '-' ma prednost, lebo stratovy riadok je zaporny.
    """
    x, y, w, h = total_box
    H, W = img_bgr.shape[:2]

    x0 = max(0, int(x - 0.28 * w - 8))
    x1 = min(W, int(x + w + 0.28 * w + 8))

    # Zacneme az POD hornym riadkom, aby sa total a loss nezlepili do jedneho cisla.
    y0 = max(0, int(y + 1.15 * h))
    y1 = min(H, int(y + 3.80 * h))

    crop = img_bgr[y0:y1, x0:x1]

    if crop.size == 0:
        return None

    candidates = []

    for scale in (2, 3):
        big = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)

        # Jemne rozmazanie potlaci moire z fotografie monitora.
        gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

        for processed in (gray, gray_blur):
            for psm in (7, 11):
                text = pytesseract.image_to_string(
                    processed,
                    config=(
                        f"--psm {psm} "
                        "-c tessedit_char_whitelist=0123456789-"
                    ),
                ).strip()

                digits = re.sub(r"[^0-9]", "", text)

                if not digits:
                    continue

                try:
                    value = int(digits)
                except ValueError:
                    continue

                if value <= 0:
                    continue

                candidates.append({
                    "value": value,
                    "negative": "-" in text,
                })

    if not candidates:
        return None

    # Najprv hlasovanie medzi vysledkami s minusom.
    negatives = [
        c["value"]
        for c in candidates
        if c["negative"]
    ]

    pool = negatives if negatives else [
        c["value"] for c in candidates
    ]

    counts = {}

    for value in pool:
        counts[value] = counts.get(value, 0) + 1

    return max(
        counts,
        key=lambda value: (
            counts[value],
            -len(str(value)),
        ),
    )


def _read_panel_values(
    img_bgr,
    rows,
    panel_cx,
    role_y,
    cropped_mode,
):
    H, W = img_bgr.shape[:2]

    groups = _numeric_groups_near_panel(
        rows,
        panel_cx,
        role_y,
        W,
        H,
        cropped_mode,
    )

    if not groups:
        return None

    # Prvy ciselny riadok = povodny pocet vojska.
    total_group = groups[0]
    total = total_group["value"]

    if total <= 0:
        return None

    loss = None

    # Druhy riadok je loss. Ked ho hlavny OCR precital kvalitne a vidi '-',
    # pouzijeme ho okamzite bez dalsieho Tesseract volania.
    if len(groups) >= 2:
        loss_group = groups[1]

        if (
            loss_group["has_minus"]
            and loss_group["avg_conf"] >= 45
            and 0 < loss_group["value"] <= total
        ):
            loss = loss_group["value"]

    # Nekvalitna fotka: precitaj iba maly riadok pod total.
    if loss is None:
        loss = _ocr_loss_below_total(
            img_bgr,
            total_group["box"],
        )

    if loss is None or loss <= 0:
        return None

    # Matematicka kontrola.
    if loss > total:
        return None

    return total, loss



# ---------------------------------------------------------------------------
# V7 FAST PATH PRE OREZANE REPORTY
# ---------------------------------------------------------------------------

def _verify_red_loss(panel, loss_row, total_row):
    """Read the complete red line independently; reject uncertain OCR."""
    H, W = panel.shape[:2]
    x, y, w, h = loss_row["box"]
    tx, _, tw, _ = total_row["box"]
    # Extend horizontally: the main OCR may have dropped a thousands group.
    x0 = max(int(0.35 * W), int(min(x, tx) - 0.12 * W))
    x1 = min(W, int(max(x + w, tx + tw) + 0.08 * W))
    y0, y1 = max(0, int(y - 2)), min(H, int(y + h + 3))
    crop = panel[y0:y1, x0:x1]
    if not crop.size:
        return None

    blue, green, red = cv2.split(crop.astype(np.int16))
    red_pixels = (red - green > 40) & (red - blue > 40) & (red > 100)
    if np.count_nonzero(red_pixels) < 8:
        return None
    ink = np.clip((red - green - 30) * 2, 0, 255)
    mask = (255 - ink).astype(np.uint8)
    ys, xs = np.nonzero(red_pixels)
    bounds = (slice(max(0, ys.min() - 1), min(crop.shape[0], ys.max() + 2)),
              slice(max(0, xs.min() - 1), min(crop.shape[1], xs.max() + 2)))
    values = []
    for source in (mask, cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)):
        source = source[bounds]
        enlarged = cv2.resize(source, None, fx=5, fy=5,
                              interpolation=cv2.INTER_CUBIC)
        enlarged = cv2.copyMakeBorder(enlarged, 15, 15, 15, 15,
                                     cv2.BORDER_CONSTANT, value=255)
        text = pytesseract.image_to_string(
            enlarged,
            config="--psm 7 -c tessedit_char_whitelist=0123456789-",
        ).strip()
        # Never concatenate separate OCR lines into one large number.
        if not re.fullmatch(r"-?\s*[0-9][0-9 ]*", text):
            return None
        values.append(int(re.sub(r"[^0-9]", "", text)))
    if values[0] != values[1] or not 0 < values[0] <= total_row["value"]:
        return None
    return values[0]


def _ocr_side_panel_cropped(img_bgr: np.ndarray, side: str):
    """
    Pri širokom orezanom reporte OCRujeme každý bočný panel zvlášť.

    Toto je rýchlejšie a hlavne spoľahlivejšie než OCR celého reportu,
    pretože stredné odmeny/ikony už nemôžu pomýliť čísla vojska.
    """
    H, W = img_bgr.shape[:2]

    if side == "left":
        xa, xb = 0, int(0.31 * W)
    else:
        xa, xb = int(0.69 * W), W

    panel = img_bgr[:, xa:xb]

    if panel.size == 0:
        return None

    # Malý panel výrazne zväčšíme. Stále je to rýchle, lebo OCRujeme
    # iba približne tretinu malého obrázka.
    scale = max(2.0, 700.0 / max(1, panel.shape[1]))
    scale = min(scale, 6.0)

    big = cv2.resize(
        panel,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )

    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)

    data = pytesseract.image_to_data(
        gray,
        config="--psm 11",
        output_type=Output.DICT,
    )

    text_tokens = []
    numeric_tokens = []

    count = len(data.get("text", []))

    for i in range(count):
        text = str(data["text"][i]).strip()

        if not text:
            continue

        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = 0.0

        x = float(data["left"][i]) / scale
        y = float(data["top"][i]) / scale
        w = float(data["width"][i]) / scale
        h = float(data["height"][i]) / scale

        text_tokens.append(text)

        digits = re.sub(r"[^0-9]", "", text)

        if digits:
            numeric_tokens.append({
                "text": text,
                "digits": digits,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "conf": conf,
            })

    # Určenie roly panelu. Stačí, keď spoľahlivo nájdeme Obranca/Defender
    # na jednej strane; druhá strana je potom útočník.
    joined_text = " ".join(text_tokens)

    defender_score = _is_defender(joined_text)
    attacker_score = _is_attacker(joined_text)

    # Zoskupíme čísla podľa horizontálneho riadku.
    groups = []

    for item in sorted(
        numeric_tokens,
        key=lambda item: item["y"] + item["h"] / 2,
    ):
        cy = item["y"] + item["h"] / 2

        # Herné total/loss čísla sú približne v strednej/spodnej časti panelu.
        # Horné bonusy alebo dátumy týmto odfiltrujeme.
        if cy < 0.28 * H or cy > 0.86 * H:
            continue

        placed = False

        for group in groups:
            if abs(cy - group["cy"]) <= max(3.0, 0.035 * H):
                group["items"].append(item)
                group["cy"] = sum(
                    i["y"] + i["h"] / 2
                    for i in group["items"]
                ) / len(group["items"])
                placed = True
                break

        if not placed:
            groups.append({
                "cy": cy,
                "items": [item],
            })

    parsed = []

    for group in groups:
        items = sorted(
            group["items"],
            key=lambda item: item["x"],
        )

        digits = "".join(item["digits"] for item in items)

        if not digits:
            continue

        try:
            value = int(digits)
        except ValueError:
            continue

        x0 = min(item["x"] for item in items)
        y0 = min(item["y"] for item in items)
        x1 = max(item["x"] + item["w"] for item in items)
        y1 = max(item["y"] + item["h"] for item in items)
        parsed.append({
            "box": (x0, y0, x1 - x0, y1 - y0),
            "cy": group["cy"],
            "value": value,
            "has_minus": any("-" in item["text"] for item in items),
            "confidence": (
                sum(max(0.0, item["conf"]) for item in items)
                / max(1, len(items))
            ),
        })

    parsed.sort(key=lambda row: row["cy"])

    # Hľadáme dvojicu susedných číselných riadkov:
    # horný = total, spodný = loss.
    best_pair = None
    best_score = None

    for i in range(len(parsed) - 1):
        total_row = parsed[i]
        loss_row = parsed[i + 1]

        gap = loss_row["cy"] - total_row["cy"]

        if gap <= 0:
            continue

        # Pri rôznych výškach cropu môže byť rozostup rôzny,
        # ale stále ide o dva blízke riadky.
        if gap > 0.22 * H:
            continue

        total = total_row["value"]
        loss = loss_row["value"]

        if total <= 0 or loss <= 0 or loss > total:
            continue

        score = 0.0

        # Mínus na spodnom riadku je veľmi silný signál.
        if loss_row["has_minus"]:
            score += 5.0

        score += min(2.0, loss_row["confidence"] / 50.0)
        score += min(2.0, total_row["confidence"] / 50.0)

        # Preferujeme nižšiu dvojicu, pretože total/loss je pod názvom hráča.
        score += loss_row["cy"] / max(1.0, H)

        if best_score is None or score > best_score:
            best_score = score
            best_pair = (total_row, loss_row)

    if best_pair is None:
        return None

    total_row, loss_row = best_pair
    total = total_row["value"]
    loss = _verify_red_loss(panel, loss_row, total_row)
    if loss is None:
        return None

    return {
        "total": total,
        "loss": loss,
        "defender_score": defender_score,
        "attacker_score": attacker_score,
    }


def _analyze_cropped_v7(img_bgr: np.ndarray):
    """
    Rýchla a deterministická analýza širokého orezaného reportu.
    """
    left = _ocr_side_panel_cropped(img_bgr, "left")
    right = _ocr_side_panel_cropped(img_bgr, "right")

    if left is None or right is None:
        return None

    # Ktorá strana je obranca?
    left_role = left["defender_score"] + right["attacker_score"]
    right_role = right["defender_score"] + left["attacker_score"]

    if max(left_role, right_role) < 0.45:
        # Ak OCR nezachytil Útočník, stále stačí samotný Obranca/Defender.
        if left["defender_score"] > right["defender_score"]:
            left_role = 1.0
            right_role = 0.0
        elif right["defender_score"] > left["defender_score"]:
            right_role = 1.0
            left_role = 0.0
        else:
            return None

    if left_role > right_role:
        defender = left
        attacker = right
    else:
        defender = right
        attacker = left

    if defender["loss"] > defender["total"]:
        return None

    return (
        attacker["loss"],
        defender["loss"],
        defender["total"],
    )



# ---------------------------------------------------------------------------
# MAIN ANALYSIS
# ---------------------------------------------------------------------------

def analyze_battle_report(image_bytes: bytes):
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        return None

    H, W = img.shape[:2]
    aspect = W / max(1, H)

    # V7 FAST PATH:
    # široké orezané reporty idú cez dva malé bočné OCR panely.
    # Je to rýchle a nepletú sa do toho stredné bonusy/ikony.
    if aspect >= 2.5:
        cropped_result = _analyze_cropped_v7(img)

        if cropped_result is not None:
            print(
                "OCR V7 cropped:",
                cropped_result,
                flush=True,
            )
            return cropped_result
        # Do not bypass a failed verification via the less strict photo path.
        return None

    # Celý screenshot alebo fotografia obrazovky:
    # použije sa dynamický V6 režim, ktorý lokalizuje Obranca/Defender
    # podľa textu v spodnej časti obrázka.
    rows, cropped_mode = _ocr_rows(img)

    if not rows:
        return None

    defender_anchor = _find_role_anchor(
        rows,
        "defender",
    )

    attacker_anchor = _find_role_anchor(
        rows,
        "attacker",
    )

    # Obranca/Defender je rozhodujuci.
    if defender_anchor is None:
        return None

    defender_cx = _panel_center(defender_anchor)
    role_y = defender_anchor["y"]

    # Ak OCR vidi aj Utocnik/Attacker, pouzijeme jeho skutocnu polohu.
    # Ak nie, je to jednoducho druhy panel na opacnej strane.
    if attacker_anchor is not None:
        attacker_cx = _panel_center(attacker_anchor)
    else:
        attacker_cx = W - defender_cx

    # Ochrana proti tomu, aby OCR omylom vybral dva texty z rovnakeho panela.
    if abs(attacker_cx - defender_cx) < 0.20 * W:
        attacker_cx = W - defender_cx

    defender_values = _read_panel_values(
        img,
        rows,
        defender_cx,
        role_y,
        cropped_mode,
    )

    attacker_values = _read_panel_values(
        img,
        rows,
        attacker_cx,
        role_y,
        cropped_mode,
    )

    if defender_values is None or attacker_values is None:
        return None

    defender_total, defender_loss = defender_values
    _, attacker_loss = attacker_values

    result = (
        attacker_loss,
        defender_loss,
        defender_total,
    )

    print(
        "OCR V7 full/photo:",
        result,
        flush=True,
    )

    return result


def format_ratio(attacker_loss: int, defender_loss: int) -> str:
    if attacker_loss <= 0 or defender_loss <= 0:
        return f"{attacker_loss} : {defender_loss}"

    smaller = min(attacker_loss, defender_loss)

    left = attacker_loss / smaller
    right = defender_loss / smaller

    def fmt(value: float) -> str:
        if abs(value - round(value)) < 0.005:
            return str(int(round(value)))

        return (
            f"{value:.2f}"
            .rstrip("0")
            .rstrip(".")
        )

    return f"{fmt(left)} : {fmt(right)}"


def format_defender_killed_percent(
    defender_loss: int,
    defender_total: int,
) -> str:
    if defender_total <= 0:
        return "0"

    percent = defender_loss / defender_total * 100

    return (
        f"{percent:.1f}"
        .rstrip("0")
        .rstrip(".")
    )


# ---------------------------------------------------------------------------
# DISCORD
# ---------------------------------------------------------------------------

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print(
        f"Prihlásený ako {client.user} "
        f"(ID: {client.user.id})"
    )


@client.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    if (
        ALLOWED_CHANNEL_IDS
        and message.channel.id not in ALLOWED_CHANNEL_IDS
    ):
        return

    image_attachments = [
        attachment
        for attachment in message.attachments
        if (
            attachment.content_type
            and attachment.content_type.startswith("image/")
        )
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
                    await message.add_reaction(
                        FAILURE_REACTION
                    )
                except discord.HTTPException:
                    pass
            continue

        attacker_loss, defender_loss, defender_total = result

        ratio = format_ratio(
            attacker_loss,
            defender_loss,
        )

        killed_percent = format_defender_killed_percent(
            defender_loss,
            defender_total,
        )

        await message.reply(
            f"**Battle Report ratio is: {ratio}**\n"
            f"**Defenders killed: {killed_percent}%**",
            mention_author=False,
        )


if __name__ == "__main__":
    if TOKEN == "VLOZ_SI_TU_TOKEN":
        raise SystemExit(
            "Nastav DISCORD_BOT_TOKEN v Railway Variables."
        )

    client.run(TOKEN)
