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
    Battle ratio: 1 : X.XX
    Straty obrancu: XX.XX %

The detector deliberately requires a consistent attacker/defender pair plus the
red loss row and the total row directly above it. If that structure is not
reliably found, it stays silent instead of calculating from unrelated numbers.
"""

from __future__ import annotations

import asyncio
import os
import re
import sqlite3
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

try:
    import asyncpg
except ImportError:  # SQLite remains available for local development/tests.
    asyncpg = None


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

# Set GGE_DEBUG=1 in Railway Variables to see why a screenshot was rejected.
DEBUG = os.getenv("GGE_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}

# Railway injects DATABASE_URL when a PostgreSQL service is connected. Without
# it the bot falls back to SQLite (use a Railway Volume for persistence).
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
STATS_DB_PATH = os.getenv("STATS_DB_PATH", "battle_stats.sqlite3").strip()
STATS_COMMAND = os.getenv("STATS_COMMAND", "!stats").strip().lower()
RELEASE_REPORT_COMMAND = os.getenv("RELEASE_REPORT_COMMAND", "!release-report").strip().lower()
BLACKLIST_REPORT_COMMAND = os.getenv("BLACKLIST_REPORT_COMMAND", "!blacklist").strip().lower()
ASSIGN_REPORT_COMMAND = os.getenv("ASSIGN_REPORT_COMMAND", "!assign").strip().lower()

STATS_PERIODS = {
    "1d": ("za posledný 1 deň", 1),
    "7d": ("za posledných 7 dní", 7),
    "all": ("za celé obdobie", None),
}

def _debug(message: str) -> None:
    if DEBUG:
        print(f"[GGE][DEBUG] {message}")


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
    is_rift: bool = False


@dataclass(frozen=True)
class PlayerStats:
    report_count: int
    total_losses: int
    total_kills: int


@dataclass(frozen=True)
class LeaderboardEntry:
    player_id: int
    player_name: str
    stats: PlayerStats


@dataclass(frozen=True)
class RecordBattleResult:
    counted: bool
    duplicate_player_id: Optional[int] = None
    duplicate_player_name: Optional[str] = None
    blacklisted: bool = False


@dataclass(frozen=True)
class BlacklistReportResult:
    deleted_reports: int
    blacklisted_reports: int


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
                # If OCR glued the role to another word, compare sliding windows too.
                if len(clean) > len(target):
                    for i in range(len(clean) - len(target) + 1):
                        part = clean[i:i + len(target)]
                        score = max(score, SequenceMatcher(None, part, target).ratio())

            if score > best_score:
                best_role = role
                best_score = score

    # Pair geometry and red-loss validation provide additional protection, so a
    # slightly lower threshold improves recall on tiny/blurred screenshots.
    if best_score < 0.66:
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

    # Normalize screenshot size. Tiny/cropped reports are enlarged; very large
    # 4K screenshots are reduced so OCR stays fast on Railway.
    h, w = img.shape[:2]
    if w < 1400:
        scale = min(4.0, 1400.0 / max(1, w))
        img = cv2.resize(
            img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
    elif w > 2200:
        scale = 2200.0 / w
        img = cv2.resize(
            img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
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

    # OCR the raw screenshot plus a contrast-enhanced grayscale version.
    # This costs one extra Tesseract pass but greatly helps small GGE UI text.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    variants = ((img, 11), (img, 6), (clahe, 11))

    for variant, psm in variants:
        data = _ocr_dataframe(variant, psm)

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
                    role=role, score=score, x=x, y=y, w=bw, h=bh, text=text
                )
            )

    labels = _dedupe_labels(found, w, h)
    _debug(
        "whole-screen roles: "
        + ", ".join(f"{x.role}:{x.text!r}@{x.x},{x.y}({x.score:.2f})" for x in labels)
    )
    return labels


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


def _looks_like_gray_name_bar(crop: np.ndarray) -> bool:
    if crop is None or crop.size == 0:
        return False

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    visible = val >= 70
    if not np.any(visible):
        return False

    visible_sat = sat[visible]
    low_saturation_ratio = float(np.mean(visible_sat <= 55))
    colored_ratio = float(np.mean((sat > 70) & (val > 90)))
    median_saturation = float(np.median(visible_sat))

    return (
        median_saturation <= 45
        and low_saturation_ratio >= 0.62
        and colored_ratio <= 0.35
    )


def _is_gray_name_bar_near_label(img: np.ndarray, label: LabelCandidate) -> bool:
    img_h, img_w = img.shape[:2]
    x0 = max(0, int(label.x - 1.1 * label.w))
    x1 = min(img_w, int(label.x + 2.4 * label.w))
    y0 = max(0, int(label.bottom + 0.35 * label.h))
    y1 = min(img_h, int(label.bottom + 3.3 * label.h))
    if x1 <= x0 or y1 <= y0:
        return False
    return _looks_like_gray_name_bar(img[y0:y1, x0:x1])


def _is_gray_name_bar_above_loss(
    img: np.ndarray, loss_box: tuple[int, int, int, int]
) -> bool:
    img_h, img_w = img.shape[:2]
    x, y, bw, bh = loss_box
    x0 = max(0, int(x - 1.6 * bw))
    x1 = min(img_w, int(x + 2.2 * bw))
    y0 = max(0, int(y - 5.6 * bh))
    y1 = min(img_h, int(y - 2.0 * bh))
    if x1 <= x0 or y1 <= y0:
        return False
    return _looks_like_gray_name_bar(img[y0:y1, x0:x1])


def _rift_result() -> BattleResult:
    return BattleResult(0, 0, 1, 1.0, is_rift=True)


def _extract_panel(img: np.ndarray, label: LabelCandidate) -> Optional[PanelData]:
    loss = _find_loss_number(img, label)
    if loss is None:
        return None

    # Whole-mask OCR can occasionally drop one digit on compressed Discord JPGs
    # (e.g. 814 -> 84). Re-read the already-localized box on grayscale/threshold
    # variants and use that consensus when it is strong.
    try:
        precise_value, precise_votes, precise_minus = _ocr_number_box_votes(
            img, (loss.x, loss.y, loss.w, loss.h), signed=True
        )
    except NameError:
        precise_value, precise_votes, precise_minus = None, 0, 0

    if precise_value is not None and precise_votes >= 3 and precise_minus >= 1:
        loss = NumberCandidate(
            value=precise_value, x=loss.x, y=loss.y, w=loss.w, h=loss.h,
            raw=f"-{precise_value}", score=loss.score + 1.0
        )

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
# LOSS-ANCHOR FALLBACK
# =============================================================================


def _redish_component_boxes(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Find small red/reddish text-like rows without assuming a fixed layout."""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    b, g, r = cv2.split(img)

    # Two complementary tests: HSV red and direct red-channel dominance.
    hsv_red = (((hue <= 20) | (hue >= 162)) & (sat >= 55) & (val >= 95))
    ri = r.astype(np.int16)
    gi = g.astype(np.int16)
    bi = b.astype(np.int16)
    dominant_red = ((ri - gi >= 25) & (ri - bi >= 18) & (r >= 90))
    mask = ((hsv_red | dominant_red).astype(np.uint8)) * 255

    # Join characters on one row, but do not join separate panels.
    kernel_w = max(3, int(round(w * 0.010)))
    joined = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1)),
        iterations=1,
    )

    count, _, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    boxes: list[tuple[int, int, int, int]] = []

    for i in range(1, count):
        x, y, bw, bh, area = [int(v) for v in stats[i]]

        if bh < max(3, int(h * 0.004)):
            continue
        if bh > max(45, int(h * 0.060)):
            continue
        if bw < max(8, int(w * 0.010)):
            continue
        if bw > int(w * 0.25):
            continue

        local = mask[y:y + bh, x:x + bw]
        ys, xs = np.where(local > 0)
        if len(xs) < 8:
            continue

        # Tighten back to the actual colored pixels after dilation.
        ax = x + int(xs.min())
        ay = y + int(ys.min())
        aw = int(xs.max() - xs.min() + 1)
        ah = int(ys.max() - ys.min() + 1)

        if aw <= 1 or ah <= 2:
            continue

        boxes.append((ax, ay, aw, ah))

    return boxes


def _box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0


def _candidate_anchor_pairs(
    img: np.ndarray, boxes: list[tuple[int, int, int, int]]
) -> list[tuple[float, tuple[int, int, int, int], tuple[int, int, int, int]]]:
    """Return geometrically plausible two-player rows, best first."""
    h, w = img.shape[:2]
    pairs = []

    # The actual casualty summary is normally in the lower part of the report.
    # This is a score bonus, not a hard layout coordinate.
    usable = [b for b in boxes if _box_center(b)[1] >= 0.42 * h]

    for i, a in enumerate(usable):
        acx, acy = _box_center(a)
        for b in usable[i + 1:]:
            bcx, bcy = _box_center(b)
            if bcx < acx:
                left, right = b, a
                lcx, lcy = bcx, bcy
                rcx, rcy = acx, acy
            else:
                left, right = a, b
                lcx, lcy = acx, acy
                rcx, rcy = bcx, bcy

            lh, rh = left[3], right[3]
            ydiff = abs(lcy - rcy)
            ytol = max(7.0, 1.10 * max(lh, rh), 0.018 * h)
            if ydiff > ytol:
                continue

            separation = rcx - lcx
            if separation < max(70.0, 0.16 * w):
                continue
            if separation > 0.82 * w:
                continue

            height_ratio = min(lh, rh) / max(1.0, max(lh, rh))
            if height_ratio < 0.45:
                continue

            width_ratio = min(left[2], right[2]) / max(1.0, max(left[2], right[2]))
            row_quality = max(0.0, 1.0 - ydiff / ytol)
            bottom_bonus = ((lcy + rcy) / 2.0) / max(1.0, h)
            separation_quality = min(1.0, separation / max(1.0, 0.35 * w))

            score = (
                2.0 * row_quality
                + 0.7 * height_ratio
                + 0.35 * width_ratio
                + 0.65 * bottom_bonus
                + 0.35 * separation_quality
            )
            pairs.append((score, left, right))

    pairs.sort(key=lambda item: item[0], reverse=True)
    return pairs[:10]


def _role_from_region_text(text: str) -> tuple[Optional[str], float]:
    """Find a role word inside a multi-word OCR region."""
    best_role = None
    best_score = 0.0

    parts = re.findall(r"[A-Za-zÀ-ž]+", text)
    parts.extend(line.strip() for line in text.splitlines() if line.strip())

    for part in parts:
        role, score = _role_match(part)
        if role is not None and score > best_score:
            best_role, best_score = role, score

    return best_role, best_score


def _ocr_role_near_anchor(
    img: np.ndarray, box: tuple[int, int, int, int]
) -> tuple[Optional[str], float, str]:
    """Read Attacker/Defender in a generous region above a suspected number row."""
    h, w = img.shape[:2]
    x, y, bw, bh = box

    x0 = max(0, int(x - 3.1 * bw - 0.020 * w))
    x1 = min(w, int(x + 2.0 * bw + 0.015 * w))
    y0 = max(0, int(y - 11.5 * bh - 0.010 * h))
    y1 = max(0, int(y - 1.6 * bh))

    if x1 <= x0 or y1 <= y0:
        return None, 0.0, ""

    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None, 0.0, ""

    target_h = 180
    scale = max(1.5, target_h / max(1, crop.shape[0]))
    scale = min(scale, 5.0)
    big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    best_role: Optional[str] = None
    best_score = 0.0
    best_text = ""

    for processed, psm in ((clahe, 6), (gray, 11)):
        text = pytesseract.image_to_string(processed, config=f"--psm {psm}")
        role, score = _role_from_region_text(text)
        if score > best_score:
            best_role, best_score, best_text = role, score, text
        if best_score >= 0.92:
            break

    return best_role, best_score, best_text


def _ocr_number_box_votes(
    img: np.ndarray,
    box: tuple[int, int, int, int],
    signed: bool,
) -> tuple[Optional[int], int, int]:
    """OCR a tight number box; returns value, winning votes, minus-sign votes."""
    x, y, bw, bh = [int(v) for v in box]
    h, w = img.shape[:2]
    pad_x = max(3, int(0.30 * bw))
    pad_y = max(2, int(0.30 * bh))
    x0, x1 = max(0, x - pad_x), min(w, x + bw + pad_x)
    y0, y1 = max(0, y - pad_y), min(h, y + bh + pad_y)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None, 0, 0

    votes: dict[int, int] = {}
    minus_votes_by_value: dict[int, int] = {}

    for scale in (4, 6):
        big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
        otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        for processed in (gray, otsu):
            for psm in (7, 8):
                whitelist = "0123456789- " if signed else "0123456789 "
                text = pytesseract.image_to_string(
                    processed,
                    config=f"--psm {psm} -c tessedit_char_whitelist={whitelist}",
                )
                digits = re.sub(r"[^0-9]", "", text)
                if not digits:
                    continue
                try:
                    value = int(digits)
                except ValueError:
                    continue
                if value > 20_000_000:
                    continue
                votes[value] = votes.get(value, 0) + 1
                if "-" in text:
                    minus_votes_by_value[value] = minus_votes_by_value.get(value, 0) + 1

    if not votes:
        return None, 0, 0

    value = max(votes, key=lambda v: (votes[v], -len(str(v))))
    return value, votes[value], minus_votes_by_value.get(value, 0)


def _find_total_box_above(
    loss_box: tuple[int, int, int, int],
    boxes: list[tuple[int, int, int, int]],
) -> Optional[tuple[int, int, int, int]]:
    lx, ly, lw, lh = loss_box
    lcx, _ = _box_center(loss_box)
    candidates = []

    for box in boxes:
        if box == loss_box:
            continue
        x, y, bw, bh = box
        cx, cy = _box_center(box)
        vertical_gap = ly - (y + bh)
        if vertical_gap < -0.35 * lh:
            continue
        if vertical_gap > 3.8 * max(lh, bh):
            continue
        if abs(cx - lcx) > max(0.95 * max(lw, bw), 22.0):
            continue
        height_ratio = min(lh, bh) / max(1.0, max(lh, bh))
        if height_ratio < 0.40:
            continue

        score = (
            2.0 * height_ratio
            - 0.035 * abs(cx - lcx)
            - 0.05 * abs(vertical_gap - 0.7 * lh)
        )
        candidates.append((score, box))

    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _analyze_by_loss_anchors(img: np.ndarray) -> Optional[BattleResult]:
    """Fallback: locate the casualty row first, then determine roles around it."""
    boxes = _redish_component_boxes(img)
    pairs = _candidate_anchor_pairs(img, boxes)
    _debug(f"anchor fallback: {len(boxes)} colored boxes, {len(pairs)} row pairs")

    detections: list[tuple[float, BattleResult]] = []

    for geom_score, left_box, right_box in pairs:
        left_role, left_role_score, left_text = _ocr_role_near_anchor(img, left_box)
        right_role, right_role_score, right_text = _ocr_role_near_anchor(img, right_box)

        # At least one side must explicitly tell us the role. If only one is
        # readable, the opposite panel can safely be inferred from the paired UI.
        if left_role is None and right_role is None:
            continue
        if left_role is not None and right_role is not None and left_role == right_role:
            continue

        if left_role is None:
            left_role = "defender" if right_role == "attacker" else "attacker"
            left_role_score = 0.58
        if right_role is None:
            right_role = "defender" if left_role == "attacker" else "attacker"
            right_role_score = 0.58

        left_value, left_votes, left_minus = _ocr_number_box_votes(img, left_box, signed=True)
        right_value, right_votes, right_minus = _ocr_number_box_votes(img, right_box, signed=True)
        if left_value is None or right_value is None:
            continue

        # This specifically rejects the total-troops row just above casualties.
        if left_minus == 0 or right_minus == 0:
            continue

        if left_role == "attacker":
            attacker_loss, attacker_box = left_value, left_box
            defender_loss, defender_box = right_value, right_box
        else:
            attacker_loss, attacker_box = right_value, right_box
            defender_loss, defender_box = left_value, left_box

        if _is_gray_name_bar_above_loss(img, defender_box):
            _debug("anchor fallback detected gray defender name bar; treating as rift")
            return _rift_result()

        # Prefer the paired colored component directly above the defender loss.
        total_box = _find_total_box_above(defender_box, boxes)
        defender_total = None
        total_votes = 0
        if total_box is not None:
            defender_total, total_votes, _ = _ocr_number_box_votes(
                img, total_box, signed=False
            )

        # If component geometry did not expose the total row, use the older
        # OCR-above-loss routine as a final local fallback.
        if defender_total is None or defender_total < defender_loss:
            synthetic_loss = NumberCandidate(
                value=defender_loss,
                x=defender_box[0], y=defender_box[1],
                w=defender_box[2], h=defender_box[3],
                raw=f"-{defender_loss}", score=1.0,
            )
            defender_total = _read_total_above_loss(
                img, synthetic_loss, minimum_value=defender_loss
            )
            total_votes = 1 if defender_total is not None else 0

        if defender_total is None or defender_total <= 0:
            continue
        if defender_loss > defender_total:
            continue

        confidence = (
            geom_score
            + left_role_score
            + right_role_score
            + min(1.0, left_votes / 4.0)
            + min(1.0, right_votes / 4.0)
            + min(1.0, total_votes / 4.0)
        )

        result = BattleResult(
            attacker_loss=attacker_loss,
            defender_loss=defender_loss,
            defender_total=defender_total,
            confidence=confidence,
        )
        detections.append((confidence, result))
        _debug(
            f"anchor candidate roles={left_role}/{right_role}, "
            f"losses={left_value}/{right_value}, defender_total={defender_total}, "
            f"score={confidence:.2f}; text={left_text!r} | {right_text!r}"
        )

    if not detections:
        return None

    detections.sort(key=lambda item: item[0], reverse=True)
    best_score, best = detections[0]

    if len(detections) > 1:
        second_score, second = detections[1]
        materially_different = (
            best.attacker_loss != second.attacker_loss
            or best.defender_loss != second.defender_loss
            or best.defender_total != second.defender_total
        )
        if materially_different and best_score - second_score < 0.40:
            _debug("anchor fallback ambiguous: rejecting close competing detections")
            return None

    return best


# =============================================================================
# SINGLE-ROLE RESCUE
# =============================================================================


def _find_opposite_loss_same_row(
    img: np.ndarray, known_panel: PanelData
) -> Optional[NumberCandidate]:
    """Find the other player's red loss on the same horizontal casualty row."""
    h, w = img.shape[:2]
    kx, ky, kw, kh = known_panel.loss_box
    known_cx = kx + kw / 2.0
    known_cy = ky + kh / 2.0

    if known_cx >= w / 2.0:
        x0 = 0
        x1 = max(1, min(int(w * 0.58), int(kx - 0.035 * w)))
    else:
        x0 = min(w - 1, max(int(w * 0.42), int(kx + kw + 0.035 * w)))
        x1 = w

    y0 = max(0, int(ky - 1.5 * kh))
    y1 = min(h, int(ky + 2.1 * kh))
    if x1 <= x0 or y1 <= y0:
        return None

    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    mask = _red_text_mask(crop)
    candidates: list[NumberCandidate] = []

    for psm in (7, 11, 6):
        data = pytesseract.image_to_data(
            mask,
            config=f"--psm {psm} -c tessedit_char_whitelist=0123456789- ",
            output_type=pytesseract.Output.DICT,
        )
        candidates.extend(_group_numeric_tokens(data, x0, y0))

    if not candidates:
        return None

    # Keep only numbers tightly aligned to the known casualty row.
    aligned = []
    for cand in candidates:
        if abs(cand.cy - known_cy) > max(8.0, 1.8 * kh):
            continue
        if cand.value > 20_000_000:
            continue
        aligned.append(cand)

    if not aligned:
        return None

    # Vote by value across PSM modes, then choose the most row-aligned reading.
    grouped: dict[int, list[NumberCandidate]] = {}
    for cand in aligned:
        grouped.setdefault(cand.value, []).append(cand)

    best_value = max(
        grouped,
        key=lambda value: (
            len(grouped[value]),
            max(c.score for c in grouped[value]),
            len(str(value)),
        ),
    )
    best = min(grouped[best_value], key=lambda c: abs(c.cy - known_cy))

    # Tight-box re-read guards against a dropped digit after compression.
    precise, votes, minus_votes = _ocr_number_box_votes(
        img, (best.x, best.y, best.w, best.h), signed=True
    )
    if precise is not None and votes >= 3:
        best = NumberCandidate(
            value=precise,
            x=best.x, y=best.y, w=best.w, h=best.h,
            raw=f"-{precise}" if minus_votes else str(precise),
            score=best.score + votes,
        )

    return best


def _analyze_from_single_role(
    img: np.ndarray, labels: list[LabelCandidate]
) -> Optional[BattleResult]:
    """If only one role OCRs, infer the opposite side from the aligned loss row."""
    detections: list[tuple[float, BattleResult]] = []

    for label in sorted(labels, key=lambda x: x.score, reverse=True):
        panel = _extract_panel(img, label)
        if panel is None:
            continue

        opposite = _find_opposite_loss_same_row(img, panel)
        if opposite is None:
            continue

        if label.role == "defender":
            if _is_gray_name_bar_near_label(img, label):
                _debug("single-role rescue detected gray defender name bar; treating as rift")
                return _rift_result()
            attacker_loss = opposite.value
            defender_loss = panel.loss
            defender_total = panel.total
        else:
            if _is_gray_name_bar_above_loss(img, (opposite.x, opposite.y, opposite.w, opposite.h)):
                _debug("single-role rescue detected gray inferred defender name bar; treating as rift")
                return _rift_result()
            attacker_loss = panel.loss
            defender_loss = opposite.value
            defender_total = _read_total_above_loss(
                img, opposite, minimum_value=defender_loss
            )
            if defender_total is None:
                continue

        if attacker_loss < 0 or defender_loss < 0:
            continue
        if defender_total <= 0 or defender_loss > defender_total:
            continue

        score = panel.structure_score + label.score + 1.0
        detections.append((
            score,
            BattleResult(
                attacker_loss=attacker_loss,
                defender_loss=defender_loss,
                defender_total=defender_total,
                confidence=score,
            ),
        ))
        _debug(
            f"single-role rescue via {label.role}: "
            f"A={attacker_loss}, D={defender_loss}/{defender_total}"
        )

    if not detections:
        return None
    detections.sort(key=lambda item: item[0], reverse=True)
    best_score, best = detections[0]
    if len(detections) > 1:
        second_score, second = detections[1]
        if (
            (best.attacker_loss, best.defender_loss, best.defender_total)
            != (second.attacker_loss, second.defender_loss, second.defender_total)
            and best_score - second_score < 0.35
        ):
            return None
    return best


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


def _analyze_by_role_labels(img: np.ndarray) -> Optional[BattleResult]:
    """Fast primary detector: explicit role labels -> local loss rows."""
    labels = _find_role_labels(img)
    attackers = [label for label in labels if label.role == "attacker"]
    defenders = [label for label in labels if label.role == "defender"]

    if not attackers or not defenders:
        _debug("role-label detector: only one role type found; trying aligned-row rescue")
        return _analyze_from_single_role(img, labels)

    panel_cache: dict[LabelCandidate, Optional[PanelData]] = {}
    scored_pairs: list[tuple[float, PanelData, PanelData]] = []
    img_h, _ = img.shape[:2]

    for att_label in attackers:
        for def_label in defenders:
            if abs(att_label.cy - def_label.cy) > max(30.0, 0.12 * img_h):
                continue

            if att_label not in panel_cache:
                panel_cache[att_label] = _extract_panel(img, att_label)
            if def_label not in panel_cache:
                panel_cache[def_label] = _extract_panel(img, def_label)

            attacker = panel_cache[att_label]
            defender = panel_cache[def_label]
            if attacker is None or defender is None:
                continue

            if _is_gray_name_bar_near_label(img, def_label):
                _debug("role-label detector detected gray defender name bar; treating as rift")
                return _rift_result()

            score = _pair_score(attacker, defender, img)
            if score is None:
                continue
            if attacker.loss > attacker.total or defender.loss > defender.total:
                continue
            scored_pairs.append((score, attacker, defender))

    if not scored_pairs:
        _debug("role-label detector: labels found, but no valid panel pair; trying single-role rescue")
        return _analyze_from_single_role(img, labels)

    scored_pairs.sort(key=lambda item: item[0], reverse=True)
    best_score, attacker, defender = scored_pairs[0]

    if len(scored_pairs) > 1:
        second_score, second_att, second_def = scored_pairs[1]
        materially_different = (
            second_att.loss != attacker.loss
            or second_def.loss != defender.loss
            or second_def.total != defender.total
        )
        if materially_different and (best_score - second_score) < 0.30:
            _debug("role-label detector ambiguous: trying anchor fallback")
            return None

    return BattleResult(
        attacker_loss=attacker.loss,
        defender_loss=defender.loss,
        defender_total=defender.total,
        confidence=best_score,
    )


def analyze_battle_report(image_bytes: bytes) -> Optional[BattleResult]:
    """Analyze one screenshot using two independent detection strategies."""
    img = _decode_image(image_bytes)
    if img is None:
        _debug("image decode failed")
        return None

    _debug(f"image size after normalization: {img.shape[1]}x{img.shape[0]}")

    # Strategy 1 is quick and very precise when both role labels OCR correctly.
    result = _analyze_by_role_labels(img)
    if result is not None:
        _debug(
            f"accepted by role-label detector: A={result.attacker_loss}, "
            f"D={result.defender_loss}/{result.defender_total}"
        )
        return result

    # Strategy 2 starts from the paired casualty row. This rescues screenshots
    # where whole-screen OCR misses one role label because of scale/compression.
    result = _analyze_by_loss_anchors(img)
    if result is not None:
        _debug(
            f"accepted by anchor detector: A={result.attacker_loss}, "
            f"D={result.defender_loss}/{result.defender_total}"
        )
        return result

    _debug("no reliable battle panel detected")
    return None


# =============================================================================
# OUTPUT FORMAT
# =============================================================================


def format_battle_ratio(attacker_loss: int, defender_loss: int) -> str:
    """
    Normalize the battle ratio so the attacker is always 1.

    Example:
        attacker losses = 6 438
        defender losses = 138 318
        -> 1 : 21.48
    """
    if attacker_loss < 0 or defender_loss < 0:
        raise ValueError("Losses cannot be negative")

    # A flawless attack cannot be normalized by dividing by attacker losses.
    if attacker_loss == 0:
        if defender_loss == 0:
            return "1 : 1.00"
        return "1 : ∞"

    ratio = defender_loss / attacker_loss
    return f"1 : {ratio:.2f}"


def attacker_is_weak(attacker_loss: int, defender_loss: int) -> bool:
    """True when the attacker's battle ratio is worse than 1:1."""
    return attacker_loss > defender_loss


def format_defender_loss_percent(defender_loss: int, defender_total: int) -> str:
    if defender_total <= 0:
        raise ValueError("defender_total must be positive")

    percent = (defender_loss / defender_total) * 100.0
    return f"{percent:.2f} %"


def format_reply(result: BattleResult) -> str:
    ratio = format_battle_ratio(result.attacker_loss, result.defender_loss)
    percent = format_defender_loss_percent(result.defender_loss, result.defender_total)

    reply = (
        f"⚔️ **Battle ratio:** `{ratio}`\n"
        f"🛡️ **Straty obrancu:** `{percent}`"
    )

    if attacker_is_weak(result.attacker_loss, result.defender_loss):
        reply += "\n💀 **Slabý útok!**"

    return reply


# =============================================================================
# PERSISTENT PLAYER STATISTICS
# =============================================================================


class StatsStore:
    """PostgreSQL on Railway, with a lightweight SQLite fallback."""

    def __init__(self, database_url: str = "", sqlite_path: str = STATS_DB_PATH):
        self.database_url = database_url
        self.sqlite_path = Path(sqlite_path)
        self.pool = None

    @property
    def backend_name(self) -> str:
        return "PostgreSQL" if self.database_url else f"SQLite ({self.sqlite_path})"

    async def initialize(self) -> None:
        if self.database_url:
            if asyncpg is None:
                raise RuntimeError("DATABASE_URL is set, but asyncpg is not installed")
            self.pool = await asyncpg.create_pool(self.database_url, min_size=1, max_size=5)
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS battle_reports (
                        id BIGSERIAL PRIMARY KEY,
                        guild_id BIGINT NOT NULL,
                        message_id BIGINT NOT NULL,
                        attachment_id BIGINT NOT NULL,
                        player_id BIGINT NOT NULL,
                        player_name TEXT NOT NULL,
                        own_losses BIGINT NOT NULL CHECK (own_losses >= 0),
                        enemy_kills BIGINT NOT NULL CHECK (enemy_kills >= 0),
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        UNIQUE (guild_id, message_id, attachment_id)
                    )
                    """
                )
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS battle_report_blacklist (
                        id BIGSERIAL PRIMARY KEY,
                        guild_id BIGINT NOT NULL,
                        own_losses BIGINT NOT NULL CHECK (own_losses >= 0),
                        enemy_kills BIGINT NOT NULL CHECK (enemy_kills >= 0),
                        source_message_id BIGINT,
                        blacklisted_by_id BIGINT,
                        blacklisted_by_name TEXT,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        UNIQUE (guild_id, own_losses, enemy_kills)
                    )
                    """
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS battle_reports_player_idx "
                    "ON battle_reports (guild_id, player_id)"
                )
            return

        await asyncio.to_thread(self._initialize_sqlite)

    def _connect_sqlite(self):
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.sqlite_path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize_sqlite(self) -> None:
        with self._connect_sqlite() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS battle_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    guild_id INTEGER NOT NULL,
                    message_id INTEGER NOT NULL,
                    attachment_id INTEGER NOT NULL,
                    player_id INTEGER NOT NULL,
                    player_name TEXT NOT NULL,
                    own_losses INTEGER NOT NULL CHECK (own_losses >= 0),
                    enemy_kills INTEGER NOT NULL CHECK (enemy_kills >= 0),
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (guild_id, message_id, attachment_id)
                );
                CREATE TABLE IF NOT EXISTS battle_report_blacklist (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    guild_id INTEGER NOT NULL,
                    own_losses INTEGER NOT NULL CHECK (own_losses >= 0),
                    enemy_kills INTEGER NOT NULL CHECK (enemy_kills >= 0),
                    source_message_id INTEGER,
                    blacklisted_by_id INTEGER,
                    blacklisted_by_name TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (guild_id, own_losses, enemy_kills)
                );
                CREATE INDEX IF NOT EXISTS battle_reports_player_idx
                    ON battle_reports (guild_id, player_id);
                """
            )

    async def record_battle(
        self,
        *,
        guild_id: int,
        message_id: int,
        attachment_id: int,
        player_id: int,
        player_name: str,
        result: BattleResult,
    ) -> RecordBattleResult:
        """Store a report once per guild and loss pair.

        A different Discord upload with the same own/enemy losses is treated as
        the same battle for the alliance. The PostgreSQL advisory lock makes the
        check atomic even if two images finish OCR at the same time.
        """
        values = (
            guild_id,
            message_id,
            attachment_id,
            player_id,
            player_name,
            result.attacker_loss,
            result.defender_loss,
        )
        if self.pool is not None:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    fingerprint = f"{guild_id}:{result.attacker_loss}:{result.defender_loss}"
                    await conn.execute(
                        "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                        fingerprint,
                    )
                    blacklisted = await conn.fetchrow(
                        """
                        SELECT 1 FROM battle_report_blacklist
                        WHERE guild_id = $1 AND own_losses = $2 AND enemy_kills = $3
                        LIMIT 1
                        """,
                        guild_id,
                        result.attacker_loss,
                        result.defender_loss,
                    )
                    if blacklisted is not None:
                        return RecordBattleResult(False, blacklisted=True)
                    duplicate = await conn.fetchrow(
                        """
                        SELECT player_id, player_name FROM battle_reports
                        WHERE guild_id = $1 AND own_losses = $2 AND enemy_kills = $3
                        ORDER BY created_at ASC, id ASC
                        LIMIT 1
                        """,
                        guild_id,
                        result.attacker_loss,
                        result.defender_loss,
                    )
                    if duplicate is not None:
                        return RecordBattleResult(
                            False,
                            int(duplicate["player_id"]),
                            duplicate["player_name"],
                        )
                    status = await conn.execute(
                        """
                        INSERT INTO battle_reports (
                            guild_id, message_id, attachment_id, player_id,
                            player_name, own_losses, enemy_kills
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT (guild_id, message_id, attachment_id) DO NOTHING
                        """,
                        *values,
                    )
                    if status == "INSERT 0 1":
                        return RecordBattleResult(True)
                    return RecordBattleResult(False, player_id, player_name)
        return await asyncio.to_thread(self._record_sqlite, values)

    def _record_sqlite(self, values: tuple) -> RecordBattleResult:
        with self._connect_sqlite() as conn:
            conn.execute("BEGIN IMMEDIATE")
            blacklisted = conn.execute(
                """
                SELECT 1 FROM battle_report_blacklist
                WHERE guild_id = ? AND own_losses = ? AND enemy_kills = ?
                LIMIT 1
                """,
                (values[0], values[5], values[6]),
            ).fetchone()
            if blacklisted is not None:
                return RecordBattleResult(False, blacklisted=True)
            duplicate = conn.execute(
                """
                SELECT player_id, player_name FROM battle_reports
                WHERE guild_id = ? AND own_losses = ? AND enemy_kills = ?
                ORDER BY created_at ASC, id ASC
                LIMIT 1
                """,
                (values[0], values[5], values[6]),
            ).fetchone()
            if duplicate is not None:
                return RecordBattleResult(False, int(duplicate[0]), duplicate[1])
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO battle_reports (
                    guild_id, message_id, attachment_id, player_id, player_name,
                    own_losses, enemy_kills
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            if cursor.rowcount == 1:
                return RecordBattleResult(True)
            return RecordBattleResult(False, values[3], values[4])

    async def get_player_stats(
        self, guild_id: int, player_id: int, since_days: Optional[int] = None
    ) -> PlayerStats:
        if self.pool is not None:
            row = await self.pool.fetchrow(
                """
                SELECT COUNT(*) AS report_count,
                       COALESCE(SUM(own_losses), 0) AS total_losses,
                       COALESCE(SUM(enemy_kills), 0) AS total_kills
                FROM battle_reports
                WHERE guild_id = $1 AND player_id = $2
                  AND ($3::integer IS NULL OR created_at >= NOW() - ($3 * INTERVAL '1 day'))
                """,
                guild_id,
                player_id,
                since_days,
            )
            return PlayerStats(
                report_count=int(row["report_count"]),
                total_losses=int(row["total_losses"]),
                total_kills=int(row["total_kills"]),
            )
        return await asyncio.to_thread(
            self._get_player_stats_sqlite, guild_id, player_id, since_days
        )

    async def get_alliance_stats(
        self, guild_id: int, since_days: Optional[int] = None
    ) -> PlayerStats:
        """Aggregate every player's reports on one Discord server."""
        if self.pool is not None:
            row = await self.pool.fetchrow(
                """
                SELECT COUNT(*) AS report_count,
                       COALESCE(SUM(own_losses), 0) AS total_losses,
                       COALESCE(SUM(enemy_kills), 0) AS total_kills
                FROM battle_reports
                WHERE guild_id = $1
                  AND ($2::integer IS NULL OR created_at >= NOW() - ($2 * INTERVAL '1 day'))
                """,
                guild_id,
                since_days,
            )
            return PlayerStats(
                report_count=int(row["report_count"]),
                total_losses=int(row["total_losses"]),
                total_kills=int(row["total_kills"]),
            )
        return await asyncio.to_thread(
            self._get_alliance_stats_sqlite, guild_id, since_days
        )

    def _get_alliance_stats_sqlite(
        self, guild_id: int, since_days: Optional[int]
    ) -> PlayerStats:
        with self._connect_sqlite() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(own_losses), 0),
                       COALESCE(SUM(enemy_kills), 0)
                FROM battle_reports
                WHERE guild_id = ?
                  AND (? IS NULL OR created_at >= datetime('now', ?))
                """,
                (
                    guild_id,
                    since_days,
                    f"-{since_days} days" if since_days is not None else None,
                ),
            ).fetchone()
        return PlayerStats(int(row[0]), int(row[1]), int(row[2]))

    def _get_player_stats_sqlite(
        self, guild_id: int, player_id: int, since_days: Optional[int]
    ) -> PlayerStats:
        with self._connect_sqlite() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(own_losses), 0),
                       COALESCE(SUM(enemy_kills), 0)
                FROM battle_reports
                WHERE guild_id = ? AND player_id = ?
                  AND (? IS NULL OR created_at >= datetime('now', ?))
                """,
                (
                    guild_id,
                    player_id,
                    since_days,
                    f"-{since_days} days" if since_days is not None else None,
                ),
            ).fetchone()
        return PlayerStats(int(row[0]), int(row[1]), int(row[2]))

    async def get_leaderboard(
        self,
        guild_id: int,
        since_days: Optional[int] = None,
        limit: int = 10,
    ) -> list[LeaderboardEntry]:
        """Return top players ordered by enemy kills."""
        if self.pool is not None:
            rows = await self.pool.fetch(
                """
                SELECT player_id,
                       MAX(player_name) AS player_name,
                       COUNT(*) AS report_count,
                       COALESCE(SUM(own_losses), 0) AS total_losses,
                       COALESCE(SUM(enemy_kills), 0) AS total_kills
                FROM battle_reports
                WHERE guild_id = $1
                  AND ($2::integer IS NULL OR created_at >= NOW() - ($2 * INTERVAL '1 day'))
                GROUP BY player_id
                ORDER BY total_kills DESC, total_losses ASC, report_count DESC, player_name ASC
                LIMIT $3
                """,
                guild_id,
                since_days,
                limit,
            )
            return [
                LeaderboardEntry(
                    player_id=int(row["player_id"]),
                    player_name=row["player_name"],
                    stats=PlayerStats(
                        report_count=int(row["report_count"]),
                        total_losses=int(row["total_losses"]),
                        total_kills=int(row["total_kills"]),
                    ),
                )
                for row in rows
            ]
        return await asyncio.to_thread(
            self._get_leaderboard_sqlite, guild_id, since_days, limit
        )

    def _get_leaderboard_sqlite(
        self, guild_id: int, since_days: Optional[int], limit: int
    ) -> list[LeaderboardEntry]:
        with self._connect_sqlite() as conn:
            rows = conn.execute(
                """
                SELECT player_id,
                       MAX(player_name) AS player_name,
                       COUNT(*) AS report_count,
                       COALESCE(SUM(own_losses), 0) AS total_losses,
                       COALESCE(SUM(enemy_kills), 0) AS total_kills
                FROM battle_reports
                WHERE guild_id = ?
                  AND (? IS NULL OR created_at >= datetime('now', ?))
                GROUP BY player_id
                ORDER BY total_kills DESC, total_losses ASC, report_count DESC, player_name ASC
                LIMIT ?
                """,
                (
                    guild_id,
                    since_days,
                    f"-{since_days} days" if since_days is not None else None,
                    limit,
                ),
            ).fetchall()
        return [
            LeaderboardEntry(
                player_id=int(row[0]),
                player_name=row[1],
                stats=PlayerStats(
                    report_count=int(row[2]),
                    total_losses=int(row[3]),
                    total_kills=int(row[4]),
                ),
            )
            for row in rows
        ]

    async def release_report(self, guild_id: int, message_id: int) -> int:
        """Delete reports recorded from one Discord message and return the count."""
        if self.pool is not None:
            status = await self.pool.execute(
                """
                DELETE FROM battle_reports
                WHERE guild_id = $1 AND message_id = $2
                """,
                guild_id,
                message_id,
            )
            return int(status.rsplit(" ", 1)[-1])
        return await asyncio.to_thread(self._release_report_sqlite, guild_id, message_id)

    def _release_report_sqlite(self, guild_id: int, message_id: int) -> int:
        with self._connect_sqlite() as conn:
            cursor = conn.execute(
                """
                DELETE FROM battle_reports
                WHERE guild_id = ? AND message_id = ?
                """,
                (guild_id, message_id),
            )
            return cursor.rowcount

    async def assign_report(
        self,
        guild_id: int,
        message_id: int,
        player_id: int,
        player_name: str,
    ) -> int:
        """Move reports recorded from one Discord message to another player."""
        if self.pool is not None:
            status = await self.pool.execute(
                """
                UPDATE battle_reports
                SET player_id = $3, player_name = $4
                WHERE guild_id = $1 AND message_id = $2
                """,
                guild_id,
                message_id,
                player_id,
                player_name,
            )
            return int(status.rsplit(" ", 1)[-1])
        return await asyncio.to_thread(
            self._assign_report_sqlite, guild_id, message_id, player_id, player_name
        )

    def _assign_report_sqlite(
        self,
        guild_id: int,
        message_id: int,
        player_id: int,
        player_name: str,
    ) -> int:
        with self._connect_sqlite() as conn:
            cursor = conn.execute(
                """
                UPDATE battle_reports
                SET player_id = ?, player_name = ?
                WHERE guild_id = ? AND message_id = ?
                """,
                (player_id, player_name, guild_id, message_id),
            )
            return cursor.rowcount

    async def blacklist_report(
        self,
        guild_id: int,
        message_id: int,
        admin_id: int,
        admin_name: str,
    ) -> BlacklistReportResult:
        """Blacklist reports from one message and remove them from stats."""
        if self.pool is not None:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    rows = await conn.fetch(
                        """
                        SELECT DISTINCT own_losses, enemy_kills
                        FROM battle_reports
                        WHERE guild_id = $1 AND message_id = $2
                        """,
                        guild_id,
                        message_id,
                    )
                    blacklisted_reports = 0
                    for row in rows:
                        status = await conn.execute(
                            """
                            INSERT INTO battle_report_blacklist (
                                guild_id, own_losses, enemy_kills,
                                source_message_id, blacklisted_by_id, blacklisted_by_name
                            ) VALUES ($1, $2, $3, $4, $5, $6)
                            ON CONFLICT (guild_id, own_losses, enemy_kills) DO NOTHING
                            """,
                            guild_id,
                            int(row["own_losses"]),
                            int(row["enemy_kills"]),
                            message_id,
                            admin_id,
                            admin_name,
                        )
                        if status == "INSERT 0 1":
                            blacklisted_reports += 1

                    status = await conn.execute(
                        """
                        DELETE FROM battle_reports
                        WHERE guild_id = $1 AND message_id = $2
                        """,
                        guild_id,
                        message_id,
                    )
                    deleted_reports = int(status.rsplit(" ", 1)[-1])
                    return BlacklistReportResult(deleted_reports, blacklisted_reports)
        return await asyncio.to_thread(
            self._blacklist_report_sqlite, guild_id, message_id, admin_id, admin_name
        )

    def _blacklist_report_sqlite(
        self,
        guild_id: int,
        message_id: int,
        admin_id: int,
        admin_name: str,
    ) -> BlacklistReportResult:
        with self._connect_sqlite() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT DISTINCT own_losses, enemy_kills
                FROM battle_reports
                WHERE guild_id = ? AND message_id = ?
                """,
                (guild_id, message_id),
            ).fetchall()

            blacklisted_reports = 0
            for own_losses, enemy_kills in rows:
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO battle_report_blacklist (
                        guild_id, own_losses, enemy_kills,
                        source_message_id, blacklisted_by_id, blacklisted_by_name
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        guild_id,
                        int(own_losses),
                        int(enemy_kills),
                        message_id,
                        admin_id,
                        admin_name,
                    ),
                )
                blacklisted_reports += cursor.rowcount

            cursor = conn.execute(
                """
                DELETE FROM battle_reports
                WHERE guild_id = ? AND message_id = ?
                """,
                (guild_id, message_id),
            )
            return BlacklistReportResult(cursor.rowcount, blacklisted_reports)

    async def reset_player_stats(
        self, guild_id: int, player_id: int, since_days: Optional[int] = None
    ) -> int:
        """Delete a player's reports in the selected period and return the count."""
        if self.pool is not None:
            status = await self.pool.execute(
                """
                DELETE FROM battle_reports
                WHERE guild_id = $1 AND player_id = $2
                  AND ($3::integer IS NULL OR created_at >= NOW() - ($3 * INTERVAL '1 day'))
                """,
                guild_id,
                player_id,
                since_days,
            )
            return int(status.rsplit(" ", 1)[-1])
        return await asyncio.to_thread(
            self._reset_player_stats_sqlite, guild_id, player_id, since_days
        )

    def _reset_player_stats_sqlite(
        self, guild_id: int, player_id: int, since_days: Optional[int]
    ) -> int:
        with self._connect_sqlite() as conn:
            cursor = conn.execute(
                """
                DELETE FROM battle_reports
                WHERE guild_id = ? AND player_id = ?
                  AND (? IS NULL OR created_at >= datetime('now', ?))
                """,
                (
                    guild_id,
                    player_id,
                    since_days,
                    f"-{since_days} days" if since_days is not None else None,
                ),
            )
            return cursor.rowcount


def format_player_stats(
    player_name: str, stats: PlayerStats, period_label: str = "za celé obdobie"
) -> str:
    if stats.report_count == 0:
        return f"📊 **{player_name}** nemá {period_label} uložený žiadny report."

    weighted_ratio = format_battle_ratio(stats.total_losses, stats.total_kills)
    return (
        f"📊 **Štatistiky hráča {player_name}** ({period_label})\n"
        f"🧾 **Reporty:** `{stats.report_count:,}`\n"
        f"💀 **Celkové straty:** `{stats.total_losses:,}`\n"
        f"⚔️ **Zabití nepriatelia:** `{stats.total_kills:,}`\n"
        f"📈 **Priemerné ratio:** `{weighted_ratio}`"
    ).replace(",", " ")


def format_leaderboard(
    entries: list[LeaderboardEntry], period_label: str = "za celé obdobie"
) -> str:
    if not entries:
        return f"🏆 Leaderboard je {period_label} zatiaľ prázdny."

    def compact_name(name: str, max_len: int = 16) -> str:
        clean_name = " ".join(name.split())
        if len(clean_name) <= max_len:
            return clean_name
        return clean_name[: max_len - 1] + "…"

    def pretty_number(value: int) -> str:
        return f"{value:,}".replace(",", " ")

    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    lines = [f"🏆 **Leaderboard podľa zabitých nepriateľov** ({period_label})"]
    for index, entry in enumerate(entries, start=1):
        rank = medals.get(index, f"**{index}.**")
        lines.append(
            f"\n{rank} **{compact_name(entry.player_name, 24)}**\n"
            f"⚔️ **{pretty_number(entry.stats.total_kills)}** killov"
        )
    return "\n".join(lines)


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


async def _fetch_referenced_message(message):
    reference = getattr(message, "reference", None)
    if reference is None or reference.message_id is None:
        return None

    resolved = getattr(reference, "resolved", None)
    if resolved is not None:
        return resolved

    try:
        return await message.channel.fetch_message(reference.message_id)
    except Exception as exc:
        print(f"[GGE] Failed to fetch referenced message: {exc}", file=sys.stderr)
        return None


def create_discord_client():
    if discord is None:
        raise RuntimeError(
            "discord.py is not installed. Run: pip install -U discord.py"
        )

    intents = discord.Intents.default()
    intents.message_content = True

    client = discord.Client(intents=intents)
    command_tree = discord.app_commands.CommandTree(client)
    stats_store = StatsStore(DATABASE_URL, STATS_DB_PATH)
    stats_ready = asyncio.Event()
    commands_synced = False

    period_choices = [
        discord.app_commands.Choice(name="1 deň", value="1d"),
        discord.app_commands.Choice(name="7 dní", value="7d"),
        discord.app_commands.Choice(name="Celé obdobie", value="all"),
    ]

    @command_tree.command(name="stats", description="Zobrazí tvoje bojové štatistiky")
    @discord.app_commands.describe(obdobie="Obdobie, za ktoré chceš štatistiky")
    @discord.app_commands.choices(obdobie=period_choices)
    async def stats_command(
        interaction: discord.Interaction,
        obdobie: discord.app_commands.Choice[str] | None = None,
    ):
        if interaction.guild_id is None:
            await interaction.response.send_message(
                "Tento príkaz je dostupný iba na serveri.", ephemeral=True
            )
            return

        await stats_ready.wait()
        period_key = obdobie.value if obdobie is not None else "all"
        period_label, since_days = STATS_PERIODS[period_key]
        try:
            stats = await stats_store.get_player_stats(
                interaction.guild_id, interaction.user.id, since_days
            )
            display_name = getattr(interaction.user, "display_name", interaction.user.name)
            await interaction.response.send_message(
                format_player_stats(display_name, stats, period_label)
            )
        except Exception as exc:
            print(f"[GGE] Stats read error: {type(exc).__name__}: {exc}", file=sys.stderr)
            await interaction.response.send_message(
                "Štatistiky sa momentálne nepodarilo načítať.", ephemeral=True
            )

    @command_tree.command(
        name="stats-alliance", description="Zobrazí spoločné štatistiky celej aliancie"
    )
    @discord.app_commands.describe(obdobie="Obdobie, za ktoré chceš štatistiky")
    @discord.app_commands.choices(obdobie=period_choices)
    async def stats_alliance_command(
        interaction: discord.Interaction,
        obdobie: discord.app_commands.Choice[str] | None = None,
    ):
        if interaction.guild_id is None:
            await interaction.response.send_message(
                "Tento príkaz je dostupný iba na serveri.", ephemeral=True
            )
            return

        await stats_ready.wait()
        period_key = obdobie.value if obdobie is not None else "all"
        period_label, since_days = STATS_PERIODS[period_key]
        try:
            stats = await stats_store.get_alliance_stats(
                interaction.guild_id, since_days
            )
            await interaction.response.send_message(
                format_player_stats("celej aliancie", stats, period_label)
            )
        except Exception as exc:
            print(
                f"[GGE] Alliance stats read error: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            await interaction.response.send_message(
                "Štatistiky aliancie sa momentálne nepodarilo načítať.",
                ephemeral=True,
            )

    @command_tree.command(
        name="leaderboard",
        description="Zobrazí TOP hráčov podľa zabitých nepriateľov",
    )
    @discord.app_commands.describe(obdobie="Obdobie, za ktoré chceš leaderboard")
    @discord.app_commands.choices(obdobie=period_choices)
    async def leaderboard_command(
        interaction: discord.Interaction,
        obdobie: discord.app_commands.Choice[str] | None = None,
    ):
        if interaction.guild_id is None:
            await interaction.response.send_message(
                "Tento príkaz je dostupný iba na serveri.", ephemeral=True
            )
            return

        await stats_ready.wait()
        period_key = obdobie.value if obdobie is not None else "all"
        period_label, since_days = STATS_PERIODS[period_key]
        try:
            entries = await stats_store.get_leaderboard(
                interaction.guild_id, since_days
            )
            await interaction.response.send_message(
                format_leaderboard(entries, period_label)
            )
        except Exception as exc:
            print(
                f"[GGE] Leaderboard read error: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            await interaction.response.send_message(
                "Leaderboard sa momentálne nepodarilo načítať.", ephemeral=True
            )

    @command_tree.command(
        name="stats-reset", description="Vymaže štatistiky vybraného hráča (iba admin)"
    )
    @discord.app_commands.default_permissions(administrator=True)
    @discord.app_commands.describe(
        hrac="Hráč, ktorému chceš vymazať štatistiky",
        obdobie="Obdobie, ktoré chceš vymazať",
    )
    @discord.app_commands.choices(obdobie=period_choices)
    async def stats_reset_command(
        interaction: discord.Interaction,
        hrac: discord.Member,
        obdobie: discord.app_commands.Choice[str] | None = None,
    ):
        if interaction.guild_id is None:
            await interaction.response.send_message(
                "Tento príkaz je dostupný iba na serveri.", ephemeral=True
            )
            return
        if not interaction.user.guild_permissions.administrator:
            await interaction.response.send_message(
                "Na tento príkaz potrebuješ oprávnenie Administrátor.", ephemeral=True
            )
            return

        await stats_ready.wait()
        period_key = obdobie.value if obdobie is not None else "all"
        period_label, since_days = STATS_PERIODS[period_key]
        try:
            deleted = await stats_store.reset_player_stats(
                interaction.guild_id, hrac.id, since_days
            )
            await interaction.response.send_message(
                f"🗑️ Vymazané reporty hráča **{hrac.display_name}** "
                f"({period_label}): `{deleted}`.",
                ephemeral=True,
            )
        except Exception as exc:
            print(f"[GGE] Stats reset error: {type(exc).__name__}: {exc}", file=sys.stderr)
            await interaction.response.send_message(
                "Štatistiky sa nepodarilo vymazať.", ephemeral=True
            )

    @client.event
    async def on_ready():
        nonlocal commands_synced
        if not stats_ready.is_set():
            await stats_store.initialize()
            stats_ready.set()
        if not commands_synced:
            synced = await command_tree.sync()
            commands_synced = True
            print(f"[GGE] Synced {len(synced)} global slash commands")
        print(f"[GGE] Logged in as {client.user} (ID: {client.user.id})")
        print(f"[GGE] Statistics storage: {stats_store.backend_name}")
        print("[GGE] Robust OCR ready. Waiting for Goodgame Empire report screenshots...")

    @client.event
    async def on_message(message):
        if message.author.bot:
            return

        if ALLOWED_CHANNEL_IDS and message.channel.id not in ALLOWED_CHANNEL_IDS:
            return

        guild_id = message.guild.id if message.guild is not None else 0
        content = (message.content or "").strip().lower()
        if content == ASSIGN_REPORT_COMMAND or content.startswith(f"{ASSIGN_REPORT_COMMAND} "):
            if message.guild is None:
                return
            if not getattr(message.author.guild_permissions, "administrator", False):
                await message.reply(
                    "Na tento príkaz potrebuješ oprávnenie Administrátor.",
                    mention_author=False,
                )
                return

            if not message.mentions:
                await message.reply(
                    f"Použi `{ASSIGN_REPORT_COMMAND} @hráč` ako odpoveď na botovu hlášku "
                    "alebo priamo na správu s reportom.",
                    mention_author=False,
                )
                return

            target_player = message.mentions[0]
            target_message = await _fetch_referenced_message(message)
            if target_message is None:
                await message.reply(
                    f"Použi `{ASSIGN_REPORT_COMMAND} @hráč` ako odpoveď na botovu hlášku "
                    "alebo priamo na správu s reportom.",
                    mention_author=False,
                )
                return

            original_message_id = target_message.id
            if (
                client.user is not None
                and target_message.author.id == client.user.id
                and target_message.reference is not None
                and target_message.reference.message_id is not None
            ):
                original_message_id = target_message.reference.message_id

            await stats_ready.wait()
            try:
                updated = await stats_store.assign_report(
                    guild_id,
                    original_message_id,
                    target_player.id,
                    target_player.display_name,
                )
                if updated:
                    await message.reply(
                        f"✅ Report bol priradený hráčovi **{target_player.display_name}**. "
                        f"Presunuté záznamy: `{updated}`.",
                        mention_author=False,
                    )
                else:
                    await message.reply(
                        "Nenašiel som k tejto správe žiadny započítaný report.",
                        mention_author=False,
                    )
            except Exception as exc:
                print(f"[GGE] Assign report error: {type(exc).__name__}: {exc}", file=sys.stderr)
                await message.reply(
                    "Report sa momentálne nepodarilo priradiť.",
                    mention_author=False,
                )
            return

        if content == BLACKLIST_REPORT_COMMAND:
            if message.guild is None:
                return
            if not getattr(message.author.guild_permissions, "administrator", False):
                await message.reply(
                    "Na tento príkaz potrebuješ oprávnenie Administrátor.",
                    mention_author=False,
                )
                return

            target_message = await _fetch_referenced_message(message)
            if target_message is None:
                await message.reply(
                    f"Použi `{BLACKLIST_REPORT_COMMAND}` ako odpoveď na botovu hlášku "
                    "alebo priamo na správu s reportom.",
                    mention_author=False,
                )
                return

            original_message_id = target_message.id
            if (
                client.user is not None
                and target_message.author.id == client.user.id
                and target_message.reference is not None
                and target_message.reference.message_id is not None
            ):
                original_message_id = target_message.reference.message_id

            await stats_ready.wait()
            try:
                result = await stats_store.blacklist_report(
                    guild_id,
                    original_message_id,
                    message.author.id,
                    message.author.display_name,
                )
                if result.deleted_reports or result.blacklisted_reports:
                    await message.reply(
                        f"⛔ Report je na blackliste. Vymazané záznamy: "
                        f"`{result.deleted_reports}`, nové blokácie: "
                        f"`{result.blacklisted_reports}`.",
                        mention_author=False,
                    )
                else:
                    await message.reply(
                        "Nenašiel som k tejto správe žiadny započítaný report na blacklist.",
                        mention_author=False,
                    )
            except Exception as exc:
                print(f"[GGE] Blacklist report error: {type(exc).__name__}: {exc}", file=sys.stderr)
                await message.reply(
                    "Report sa momentálne nepodarilo pridať na blacklist.",
                    mention_author=False,
                )
            return

        if content == RELEASE_REPORT_COMMAND:
            if message.guild is None:
                return
            if not getattr(message.author.guild_permissions, "administrator", False):
                await message.reply(
                    "Na tento príkaz potrebuješ oprávnenie Administrátor.",
                    mention_author=False,
                )
                return

            target_message = await _fetch_referenced_message(message)
            if target_message is None:
                await message.reply(
                    f"Použi `{RELEASE_REPORT_COMMAND}` ako odpoveď na botovu hlášku "
                    "alebo priamo na správu s reportom.",
                    mention_author=False,
                )
                return

            original_message_id = target_message.id
            if (
                client.user is not None
                and target_message.author.id == client.user.id
                and target_message.reference is not None
                and target_message.reference.message_id is not None
            ):
                original_message_id = target_message.reference.message_id

            await stats_ready.wait()
            try:
                deleted = await stats_store.release_report(guild_id, original_message_id)
                if deleted:
                    await message.reply(
                        f"✅ Report bol uvoľnený. Vymazané záznamy: `{deleted}`. "
                        "Teraz ho môže nahrať správny hráč.",
                        mention_author=False,
                    )
                else:
                    await message.reply(
                        "Nenašiel som k tejto správe žiadny započítaný report.",
                        mention_author=False,
                    )
            except Exception as exc:
                print(f"[GGE] Release report error: {type(exc).__name__}: {exc}", file=sys.stderr)
                await message.reply(
                    "Report sa momentálne nepodarilo uvoľniť.",
                    mention_author=False,
                )
            return

        if content == STATS_COMMAND:
            await stats_ready.wait()
            try:
                stats = await stats_store.get_player_stats(guild_id, message.author.id)
                await message.reply(
                    format_player_stats(message.author.display_name, stats),
                    mention_author=False,
                )
            except Exception as exc:
                print(f"[GGE] Stats read error: {type(exc).__name__}: {exc}", file=sys.stderr)
                await message.reply(
                    "Štatistiky sa momentálne nepodarilo načítať.",
                    mention_author=False,
                )
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
            if result.is_rift:
                try:
                    await message.add_reaction("🤏")
                except discord.HTTPException as exc:
                    print(f"[GGE] Failed to add rift reaction: {exc}", file=sys.stderr)
                continue

            try:
                await stats_ready.wait()
                record_result = await stats_store.record_battle(
                    guild_id=guild_id,
                    message_id=message.id,
                    attachment_id=attachment.id,
                    player_id=message.author.id,
                    player_name=message.author.display_name,
                    result=result,
                )
                reply = format_reply(result)
                if record_result.counted:
                    reply += "\n✅ Report bol započítaný. Svoje súčty zobrazíš cez `/stats`."
                elif record_result.blacklisted:
                    reply += "\n⛔ Tento report je na blackliste a nebude započítaný."
                elif record_result.duplicate_player_id == message.author.id:
                    reply += "\nℹ️ Tento report už máš raz započítaný."
                elif record_result.duplicate_player_name:
                    reply += f"\nℹ️ Tento report už nahral: {record_result.duplicate_player_name}"
                else:
                    reply += "\nℹ️ Tento report už bol započítaný."
                await message.reply(reply, mention_author=False)
            except discord.HTTPException as exc:
                print(f"[GGE] Failed to send Discord reply: {exc}", file=sys.stderr)
            except Exception as exc:
                print(f"[GGE] Stats write error: {type(exc).__name__}: {exc}", file=sys.stderr)
                try:
                    await message.reply(
                        format_reply(result) + "\n⚠️ Report sa nepodarilo uložiť do štatistík.",
                        mention_author=False,
                    )
                except discord.HTTPException:
                    pass

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
        if DEBUG:
            print("[GGE][DEBUG] Debug logging is ON")
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
    
                
