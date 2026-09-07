"""
Goodgame Empire battle-report analyzer for Discord.

What it does
------------
- Watches Discord messages for image attachments.
- Finds the *battle result panels* instead of trusting a fixed left/right side.
- Supports Slovak role labels: Útočník / Obranca.
- Supports English role labels: Attacker / Defender.
- Works with full screenshots and cropped battle-result screenshots.
- Extracts attacker losses, defender losses and defender total troops.
- Replies only with:
    Battle ratio: X.XX : 1
    Straty obrancu: XX.XX %

The detector deliberately requires a consistent attacker/defender pair plus the
red loss row and the total row directly above it. If that structure is not
reliably found, it stays silent instead of calculating from unrelated numbers.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import pytesseract

try:
    import discord
except ImportError:  # Lets the OCR functions still be imported/tested without discord.py.
    discord = None


# =============================================================================
# CONFIGURATION
# =============================================================================

# Recommended: set the environment variable DISCORD_BOT_TOKEN.
# Alternatively paste the token below. Do NOT publish a file containing a token.
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "").strip()
# Example fallback if you prefer a token directly in this file:
# DISCORD_TOKEN = DISCORD_TOKEN or "PASTE_YOUR_DISCORD_BOT_TOKEN_HERE"

# Optional. Leave empty if tesseract is available in PATH.
# Windows example:
#   TESSERACT_CMD = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Empty = analyze image attachments in every channel the bot can read.
# Example: {123456789012345678, 987654321098765432}
ALLOWED_CHANNEL_IDS: set[int] = set()

# False = unsupported/unreadable images are ignored silently.
REPLY_ON_FAILURE = False

# Prevent accidental processing of huge files.
MAX_IMAGE_BYTES = 15 * 1024 * 1024

# Common image formats Discord may send without a content_type.
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


# =============================================================================
# DATA TYPES
# =============================================================================


@dataclass(frozen=True)
class LabelCandidate:
    role: str  # "attacker" or "defender"
    score: float
    x: int
    y: int
    w: int
    h: int
    text: str

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def bottom(self) -> int:
        return self.y + self.h


@dataclass(frozen=True)
class NumberCandidate:
    value: int
    x: int
    y: int
    w: int
    h: int
    raw: str
    score: float

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


@dataclass(frozen=True)
class PanelData:
    role: str
    label: LabelCandidate
    total: int
    loss: int
    loss_box: tuple[int, int, int, int]
    structure_score: float


@dataclass(frozen=True)
class BattleResult:
    attacker_loss: int
    defender_loss: int
    defender_total: int
    confidence: float


# =============================================================================
# TEXT / ROLE HELPERS
# =============================================================================


ROLE_TARGETS = {
    "attacker": ("utocnik", "attacker"),
    "defender": ("obranca", "defender"),
}


def _normalize_letters(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text).lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z]", "", text)


def _role_match(text: str) -> tuple[Optional[str], float]:
    clean = _normalize_letters(text)
    if len(clean) < 5:
        return None, 0.0

    best_role: Optional[str] = None
    best_score = 0.0

    for role, targets in ROLE_TARGETS.items():
        for target in targets:
            if target in clean:
                score = 1.0
            else:
                score = SequenceMatcher(None, clean, target).ratio()

            if score > best_score:
                best_role = role
                best_score = score

    # Handles typical OCR variants such as "Utoénik" -> "utoenik".
    if best_score < 0.72:
        return None, 0.0

    return best_role, best_score


def _digits_to_int(text: str) -> Optional[int]:
    digits = re.sub(r"[^0-9]", "", str(text))
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


# =============================================================================
# IMAGE PREPARATION
# =============================================================================


def _decode_image(image_bytes: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return None

    # Cropped GGE report panels can be tiny (e.g. ~478x118). Upscaling before
    # OCR gives Tesseract much more reliable character shapes.
    h, w = img.shape[:2]
    if w < 900:
        scale = min(4.0, max(2.0, 1400.0 / max(1, w)))
        img = cv2.resize(
            img,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    return img


def _ocr_dataframe(img: np.ndarray, psm: int):
    return pytesseract.image_to_data(
        img,
        config=f"--psm {psm}",
        output_type=pytesseract.Output.DICT,
    )


# =============================================================================
# FIND ROLE LABELS IN THE WHOLE SCREENSHOT
# =============================================================================


def _boxes_near(a: LabelCandidate, b: LabelCandidate, img_w: int, img_h: int) -> bool:
    return (
        a.role == b.role
        and abs(a.cx - b.cx) <= max(12, 0.02 * img_w)
        and abs(a.cy - b.cy) <= max(8, 0.02 * img_h)
    )


def _dedupe_labels(
    candidates: Iterable[LabelCandidate], img_w: int, img_h: int
) -> list[LabelCandidate]:
    result: list[LabelCandidate] = []

    for cand in sorted(candidates, key=lambda c: c.score, reverse=True):
        duplicate_index = None
        for i, existing in enumerate(result):
            if _boxes_near(cand, existing, img_w, img_h):
                duplicate_index = i
                break

        if duplicate_index is None:
            result.append(cand)
        elif cand.score > result[duplicate_index].score:
            result[duplicate_index] = cand

    return result


def _find_role_labels(img: np.ndarray) -> list[LabelCandidate]:
    h, w = img.shape[:2]
    found: list[LabelCandidate] = []

    # PSM 11 is strong on sparse UI text. PSM 6 is a useful second opinion.
    for psm in (11, 6):
        data = _ocr_dataframe(img, psm)

        for i, raw_text in enumerate(data.get("text", [])):
            text = str(raw_text).strip()
            if not text:
                continue

            role, score = _role_match(text)
            if role is None:
                continue

            try:
                x = int(data["left"][i])
                y = int(data["top"][i])
                bw = int(data["width"][i])
                bh = int(data["height"][i])
            except (ValueError, TypeError, KeyError, IndexError):
                continue

            if bw <= 0 or bh <= 0:
                continue

            found.append(
                LabelCandidate(
                    role=role,
                    score=score,
                    x=x,
                    y=y,
                    w=bw,
                    h=bh,
                    text=text,
                )
            )

    return _dedupe_labels(found, w, h)


# =============================================================================
# FIND THE RED LOSS NUMBER ASSOCIATED WITH A ROLE LABEL
# =============================================================================


def _red_text_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    Isolate the bright red used by GGE for the loss number.

    Requiring saturation and brightness is important: the GGE panel background
    is brown and can also have a reddish hue, but it is much less saturated.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    is_red_hue = (hue <= 12) | (hue >= 168)
    mask = is_red_hue & (sat >= 90) & (val >= 145)

    # Tesseract expects dark text on a light background.
    out = np.full(mask.shape, 255, dtype=np.uint8)
    out[mask] = 0
    return out


def _group_numeric_tokens(data: dict, origin_x: int, origin_y: int) -> list[NumberCandidate]:
    tokens = []

    for i, raw in enumerate(data.get("text", [])):
        text = str(raw).strip()
        value = _digits_to_int(text)
        if value is None:
            continue

        try:
            x = int(data["left"][i]) + origin_x
            y = int(data["top"][i]) + origin_y
            w = int(data["width"][i])
            h = int(data["height"][i])
        except (ValueError, TypeError, KeyError, IndexError):
            continue

        if w <= 0 or h <= 0:
            continue

        tokens.append(
            {
                "value": value,
                "raw": text,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            }
        )

    if not tokens:
        return []

    # Join numbers Tesseract split because of the thousands separator, e.g.
    # "-138" + "318" on the same line.
    groups: list[list[dict]] = []
    for token in sorted(tokens, key=lambda t: (t["y"] + t["h"] / 2, t["x"])):
        cy = token["y"] + token["h"] / 2
        placed = False

        for group in groups:
            group_cy = sum(t["y"] + t["h"] / 2 for t in group) / len(group)
            tolerance = max(5.0, 0.75 * max(token["h"], max(t["h"] for t in group)))

            if abs(cy - group_cy) <= tolerance:
                group.append(token)
                placed = True
                break

        if not placed:
            groups.append([token])

    results: list[NumberCandidate] = []

    for group in groups:
        group = sorted(group, key=lambda t: t["x"])

        # If tokens on the same row are extremely far apart, keep them as
        # separate candidates rather than accidentally joining unrelated UI.
        clusters: list[list[dict]] = []
        for token in group:
            if not clusters:
                clusters.append([token])
                continue

            prev = clusters[-1][-1]
            gap = token["x"] - (prev["x"] + prev["w"])
            typical_h = max(token["h"], prev["h"])

            if gap <= max(18, 2.0 * typical_h):
                clusters[-1].append(token)
            else:
                clusters.append([token])

        for cluster in clusters:
            raw = "".join(t["raw"] for t in cluster)
            value = _digits_to_int(raw)
            if value is None:
                continue

            x0 = min(t["x"] for t in cluster)
            y0 = min(t["y"] for t in cluster)
            x1 = max(t["x"] + t["w"] for t in cluster)
            y1 = max(t["y"] + t["h"] for t in cluster)
            digit_count = len(re.sub(r"[^0-9]", "", raw))
            minus_bonus = 1.0 if "-" in raw else 0.0

            results.append(
                NumberCandidate(
                    value=value,
                    x=x0,
                    y=y0,
                    w=x1 - x0,
                    h=y1 - y0,
                    raw=raw,
                    score=4.0 * digit_count + minus_bonus,
                )
            )

    return results


def _loss_search_roi(
    img: np.ndarray, label: LabelCandidate
) -> tuple[int, int, int, int]:
    h, w = img.shape[:2]

    # Numbers in the GGE result panel sit slightly to the right of the role
    # heading and several text-heights below it. Using the label's own size
    # makes this work for both full screens and small cropped reports.
    x0 = max(0, int(label.x))
    x1 = min(w, int(label.x + 2.50 * label.w + 0.03 * w))
    y0 = max(0, label.bottom)
    y1 = min(h, int(label.bottom + 10.5 * label.h + 0.04 * h))

    return x0, y0, x1, y1


def _find_loss_number(img: np.ndarray, label: LabelCandidate) -> Optional[NumberCandidate]:
    img_h, img_w = img.shape[:2]
    x0, y0, x1, y1 = _loss_search_roi(img, label)

    if x1 <= x0 or y1 <= y0:
        return None

    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    red_mask = _red_text_mask(crop)
    all_candidates: list[NumberCandidate] = []

    for psm in (11, 6):
        data = pytesseract.image_to_data(
            red_mask,
            config=f"--psm {psm} -c tessedit_char_whitelist=0123456789- ",
            output_type=pytesseract.Output.DICT,
        )
        all_candidates.extend(_group_numeric_tokens(data, x0, y0))

    if not all_candidates:
        return None

    usable: list[NumberCandidate] = []

    for cand in all_candidates:
        # The loss row must actually be below the role heading.
        vertical_offset = (cand.y - label.bottom) / max(1.0, label.h)
        if vertical_offset < 1.8 or vertical_offset > 13.0:
            continue

        # It should be horizontally close to the panel heading, not on the
        # other side of the screenshot.
        if cand.x < label.x - 0.45 * label.w:
            continue
        if cand.x > label.x + 3.2 * label.w + 0.04 * img_w:
            continue

        digit_count = len(str(cand.value))
        if digit_count < 1:
            continue

        # Prefer a real multi-digit red number, a visible minus sign, and the
        # lower part of the local panel. This rejects most red icon artefacts.
        position_bonus = min(3.0, vertical_offset / 3.0)
        minus_bonus = 2.0 if "-" in cand.raw else 0.0
        multi_digit_bonus = min(4.0, digit_count)

        usable.append(
            NumberCandidate(
                value=cand.value,
                x=cand.x,
                y=cand.y,
                w=cand.w,
                h=cand.h,
                raw=cand.raw,
                score=cand.score + position_bonus + minus_bonus + multi_digit_bonus,
            )
        )

    if not usable:
        return None

    # Deduplicate PSM 11/6 readings of the same row by favoring the best score.
    usable.sort(key=lambda c: c.score, reverse=True)
    return usable[0]


# =============================================================================
# READ THE TOTAL DIRECTLY ABOVE THE LOSS ROW
# =============================================================================


def _preprocess_number_crop(crop: np.ndarray) -> list[np.ndarray]:
    if crop is None or crop.size == 0:
        return []

    # Upscale again because the number row itself can be only ~15 px high.
    target_h = 80
    scale = max(2.0, target_h / max(1, crop.shape[0]))
    scale = min(scale, 6.0)

    big = cv2.resize(
        crop,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return [gray, otsu]


def _read_total_above_loss(
    img: np.ndarray, loss: NumberCandidate, minimum_value: int
) -> Optional[int]:
    img_h, img_w = img.shape[:2]
    x, y, w, h = loss.x, loss.y, loss.w, loss.h

    # Slightly different boxes make OCR resilient to font scaling/cropping.
    rects = [
        (
            max(0, int(x - 0.45 * w)),
            max(0, int(y - 2.45 * h)),
            min(img_w, int(x + w + 0.45 * w)),
            max(0, int(y - 0.30 * h)),
        ),
        (
            max(0, int(x - 0.35 * w)),
            max(0, int(y - 2.20 * h)),
            min(img_w, int(x + w + 0.35 * w)),
            max(0, int(y - 0.40 * h)),
        ),
        (
            max(0, int(x - 0.55 * w)),
            max(0, int(y - 2.65 * h)),
            min(img_w, int(x + w + 0.55 * w)),
            max(0, int(y - 0.35 * h)),
        ),
    ]

    votes: dict[int, int] = {}

    for x0, y0, x1, y1 in rects:
        if x1 <= x0 or y1 <= y0:
            continue

        crop = img[y0:y1, x0:x1]
        for processed in _preprocess_number_crop(crop):
            for psm in (7, 8, 13, 6):
                text = pytesseract.image_to_string(
                    processed,
                    config=f"--psm {psm} -c tessedit_char_whitelist=0123456789 ",
                )
                value = _digits_to_int(text)
                if value is None:
                    continue

                votes[value] = votes.get(value, 0) + 1

    if not votes:
        return None

    valid = {value: count for value, count in votes.items() if value >= minimum_value}
    if not valid:
        return None

    # Most OCR votes wins. For a tie, prefer a plausible number that isn't an
    # accidental duplicate of the loss row, then the shorter representation.
    return max(
        valid,
        key=lambda value: (
            valid[value],
            1 if value > minimum_value else 0,
            -len(str(value)),
        ),
    )


def _extract_panel(img: np.ndarray, label: LabelCandidate) -> Optional[PanelData]:
    loss = _find_loss_number(img, label)
    if loss is None:
        return None

    total = _read_total_above_loss(img, loss, minimum_value=loss.value)
    if total is None or total <= 0:
        return None

    if loss.value < 0 or loss.value > total:
        return None

    offset_in_label_heights = (loss.y - label.bottom) / max(1.0, label.h)
    if not 1.8 <= offset_in_label_heights <= 13.0:
        return None

    # Structural score rewards a strong role OCR match and the expected layout.
    offset_quality = max(0.0, 1.0 - abs(offset_in_label_heights - 6.0) / 8.0)
    structure_score = label.score + offset_quality

    return PanelData(
        role=label.role,
        label=label,
        total=total,
        loss=loss.value,
        loss_box=(loss.x, loss.y, loss.w, loss.h),
        structure_score=structure_score,
    )


# =============================================================================
# PAIR ATTACKER + DEFENDER PANELS
# =============================================================================


def _pair_score(attacker: PanelData, defender: PanelData, img: np.ndarray) -> Optional[float]:
    h, w = img.shape[:2]

    label_y_diff = abs(attacker.label.cy - defender.label.cy)
    loss_att_y = attacker.loss_box[1] + attacker.loss_box[3] / 2
    loss_def_y = defender.loss_box[1] + defender.loss_box[3] / 2
    loss_y_diff = abs(loss_att_y - loss_def_y)

    # Battle result panel headings and loss rows are horizontally aligned.
    if label_y_diff > max(20.0, 0.08 * h):
        return None
    if loss_y_diff > max(25.0, 0.08 * h):
        return None

    # A role pair should occupy different horizontal regions.
    horizontal_separation = abs(attacker.label.cx - defender.label.cx)
    if horizontal_separation < max(60.0, 0.12 * w):
        return None

    att_offset = (attacker.loss_box[1] - attacker.label.bottom) / max(1.0, attacker.label.h)
    def_offset = (defender.loss_box[1] - defender.label.bottom) / max(1.0, defender.label.h)
    offset_diff = abs(att_offset - def_offset)

    if offset_diff > 4.5:
        return None

    row_quality = 1.0 - min(1.0, label_y_diff / max(20.0, 0.08 * h))
    loss_row_quality = 1.0 - min(1.0, loss_y_diff / max(25.0, 0.08 * h))
    offset_quality = 1.0 - min(1.0, offset_diff / 4.5)

    return (
        attacker.structure_score
        + defender.structure_score
        + row_quality
        + loss_row_quality
        + offset_quality
    )


def analyze_battle_report(image_bytes: bytes) -> Optional[BattleResult]:
    """Analyze one screenshot. Returns None if the battle panel is uncertain."""
    img = _decode_image(image_bytes)
    if img is None:
        return None

    labels = _find_role_labels(img)
    attackers = [label for label in labels if label.role == "attacker"]
    defenders = [label for label in labels if label.role == "defender"]

    if not attackers or not defenders:
        return None

    # Only run the heavier number OCR around labels that can plausibly pair on
    # the same row. This also prevents random role words elsewhere on the screen
    # from becoming valid battle panels.
    panel_cache: dict[LabelCandidate, Optional[PanelData]] = {}
    scored_pairs: list[tuple[float, PanelData, PanelData]] = []
    img_h, _ = img.shape[:2]

    for att_label in attackers:
        for def_label in defenders:
            if abs(att_label.cy - def_label.cy) > max(25.0, 0.10 * img_h):
                continue

            if att_label not in panel_cache:
                panel_cache[att_label] = _extract_panel(img, att_label)
            if def_label not in panel_cache:
                panel_cache[def_label] = _extract_panel(img, def_label)

            attacker = panel_cache[att_label]
            defender = panel_cache[def_label]

            if attacker is None or defender is None:
                continue

            score = _pair_score(attacker, defender, img)
            if score is None:
                continue

            # Logical checks: losses cannot exceed pre-battle troop totals.
            if attacker.loss > attacker.total:
                continue
            if defender.loss > defender.total:
                continue

            scored_pairs.append((score, attacker, defender))

    if not scored_pairs:
        return None

    scored_pairs.sort(key=lambda item: item[0], reverse=True)
    best_score, attacker, defender = scored_pairs[0]

    # If there are two almost-equally-good but materially different detections,
    # fail safely rather than guess.
    if len(scored_pairs) > 1:
        second_score, second_att, second_def = scored_pairs[1]
        materially_different = (
            second_att.loss != attacker.loss
            or second_def.loss != defender.loss
            or second_def.total != defender.total
        )
        if materially_different and (best_score - second_score) < 0.35:
            return None

    return BattleResult(
        attacker_loss=attacker.loss,
        defender_loss=defender.loss,
        defender_total=defender.total,
        confidence=best_score,
    )


# =============================================================================
# OUTPUT FORMAT
# =============================================================================


def format_battle_ratio(attacker_loss: int, defender_loss: int) -> str:
    """Battle ratio = defender losses / attacker losses, shown as X.XX : 1."""
    if attacker_loss < 0 or defender_loss < 0:
        raise ValueError("Losses cannot be negative")

    if attacker_loss == 0:
        if defender_loss == 0:
            return "0.00 : 0"
        return "∞ : 1"

    ratio = defender_loss / attacker_loss
    return f"{ratio:.2f} : 1"


def format_defender_loss_percent(defender_loss: int, defender_total: int) -> str:
    if defender_total <= 0:
        raise ValueError("defender_total must be positive")

    percent = (defender_loss / defender_total) * 100.0
    return f"{percent:.2f} %"


def format_reply(result: BattleResult) -> str:
    ratio = format_battle_ratio(result.attacker_loss, result.defender_loss)
    percent = format_defender_loss_percent(result.defender_loss, result.defender_total)

    return (
        f"⚔️ **Battle ratio:** `{ratio}`\n"
        f"🛡️ **Straty obrancu:** `{percent}`"
    )


# =============================================================================
# DISCORD BOT
# =============================================================================


def _attachment_looks_like_image(attachment) -> bool:
    content_type = (getattr(attachment, "content_type", None) or "").lower()
    if content_type.startswith("image/"):
        return True

    filename = getattr(attachment, "filename", "") or ""
    return Path(filename).suffix.lower() in IMAGE_EXTENSIONS


async def _analyze_attachment(attachment) -> Optional[BattleResult]:
    size = getattr(attachment, "size", None)
    if size is not None and size > MAX_IMAGE_BYTES:
        return None

    try:
        image_bytes = await attachment.read()
    except Exception as exc:
        print(f"[GGE] Failed to download attachment: {exc}", file=sys.stderr)
        return None

    if len(image_bytes) > MAX_IMAGE_BYTES:
        return None

    try:
        # OCR is CPU-bound; do not freeze the Discord event loop while Tesseract
        # works on a large screenshot.
        return await asyncio.to_thread(analyze_battle_report, image_bytes)
    except pytesseract.TesseractNotFoundError:
        print(
            "[GGE] Tesseract OCR was not found. Install Tesseract or set TESSERACT_CMD.",
            file=sys.stderr,
        )
        return None
    except Exception as exc:
        print(f"[GGE] OCR error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None


def create_discord_client():
    if discord is None:
        raise RuntimeError(
            "discord.py is not installed. Run: pip install -U discord.py"
        )

    intents = discord.Intents.default()
    intents.message_content = True

    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        print(f"[GGE] Logged in as {client.user} (ID: {client.user.id})")
        print("[GGE] Waiting for Goodgame Empire report screenshots...")

    @client.event
    async def on_message(message):
        if message.author.bot:
            return

        if ALLOWED_CHANNEL_IDS and message.channel.id not in ALLOWED_CHANNEL_IDS:
            return

        attachments = [
            attachment
            for attachment in message.attachments
            if _attachment_looks_like_image(attachment)
        ]

        if not attachments:
            return

        any_success = False

        for attachment in attachments:
            result = await _analyze_attachment(attachment)
            if result is None:
                continue

            any_success = True
            try:
                await message.reply(format_reply(result), mention_author=False)
            except discord.HTTPException as exc:
                print(f"[GGE] Failed to send Discord reply: {exc}", file=sys.stderr)

        if REPLY_ON_FAILURE and not any_success:
            try:
                await message.reply(
                    "Nepodarilo sa spoľahlivo rozpoznať battle report.",
                    mention_author=False,
                )
            except discord.HTTPException:
                pass

    return client


def _preflight() -> None:
    if not DISCORD_TOKEN:
        raise SystemExit(
            "Chýba Discord token. Nastav premennú DISCORD_BOT_TOKEN "
            "alebo ho vlož do DISCORD_TOKEN v hornej časti súboru."
        )

    try:
        version = pytesseract.get_tesseract_version()
        print(f"[GGE] Tesseract OCR: {version}")
    except pytesseract.TesseractNotFoundError as exc:
        raise SystemExit(
            "Tesseract OCR nebol nájdený. Nainštaluj Tesseract a prípadne "
            "nastav premennú TESSERACT_CMD na cestu k tesseract.exe."
        ) from exc


def main() -> None:
    _preflight()
    client = create_discord_client()
    client.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
