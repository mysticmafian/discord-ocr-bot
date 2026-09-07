"""
Goodgame Empire battle-report analyzer for an existing Discord bot.

What it does
------------
- Accepts a full screenshot OR a cropped battle report.
- Finds the actual Attacker/Defender result-panel pair by layout + validation,
  so unrelated occurrences of the words elsewhere on screen are ignored.
- Supports Slovak and English role labels (with fuzzy OCR matching).
- Attacker/Defender may be on either side.
- Extracts total troops and losses from both result panels.
- Calculates only:
    Battle ratio = defender losses / attacker losses  ->  X.XX : 1
    Straty obrancu = defender losses / defender total -> percent
- Fails safely (returns None) when the result panel cannot be verified.

Dependencies
------------
Python packages:
    pip install opencv-python-headless numpy pytesseract

System package:
    Tesseract OCR must be installed and available on PATH.

This file intentionally DOES NOT create a Discord client and DOES NOT contain
any token/channel settings. Import it into your existing bot and call
`process_discord_message(message)` from your existing on_message listener.
"""

from __future__ import annotations

import asyncio
import os
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable, Optional

import cv2
import numpy as np
import pytesseract
from pytesseract import Output


# Optional runtime configuration. These are analyzer settings, not Discord settings.
TESSERACT_LANG = os.getenv("GGE_TESSERACT_LANG", "eng")
TESSERACT_CMD = os.getenv("GGE_TESSERACT_CMD", "").strip()
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

ROLE_ALIASES = {
    "attacker": ("utocnik", "attacker"),
    "defender": ("obranca", "defender"),
}


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0


@dataclass(frozen=True)
class OCRItem:
    text: str
    conf: float
    box: Box


@dataclass(frozen=True)
class RoleAnchor:
    role: str
    score: float
    item: OCRItem


@dataclass(frozen=True)
class PanelValues:
    total: int
    loss: int
    confidence: float
    total_box: Box


@dataclass(frozen=True)
class BattleAnalysis:
    attacker_total: int
    attacker_loss: int
    defender_total: int
    defender_loss: int
    confidence: float

    @property
    def battle_ratio(self) -> Optional[float]:
        if self.attacker_loss <= 0:
            return None
        return self.defender_loss / self.attacker_loss

    @property
    def defender_loss_percent(self) -> float:
        if self.defender_total <= 0:
            return 0.0
        return self.defender_loss / self.defender_total * 100.0


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def _normalize_text(text: str) -> str:
    """Normalize OCR text so accents / mild OCR damage matter less."""
    s = unicodedata.normalize("NFKD", str(text).lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    # Common OCR substitutions seen in UI fonts.
    s = s.replace("0", "o").replace("1", "i").replace("|", "i")
    return re.sub(r"[^a-z]", "", s)


def _role_similarity(text: str, role: str) -> float:
    clean = _normalize_text(text)
    if not clean:
        return 0.0

    best = 0.0
    for target in ROLE_ALIASES[role]:
        if target in clean:
            return 1.0
        best = max(best, SequenceMatcher(None, clean, target).ratio())
    return best


def _parse_int(text: str) -> Optional[int]:
    """Read 12 205 / 12,205 / -5488 / 5.488 as an integer."""
    digits = re.sub(r"[^0-9]", "", str(text))
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------


def _resize_for_ocr(img: np.ndarray, target_width: int = 1900) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    if w <= 0:
        return img, 1.0

    # Small cropped reports need much more enlargement than full screenshots.
    scale = target_width / float(w)
    scale = min(5.0, max(1.0, scale))
    if abs(scale - 1.0) < 0.03:
        return img, 1.0

    return (
        cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC),
        scale,
    )


def _ocr_items(img: np.ndarray, *, target_width: int = 1900, psm: int = 11) -> list[OCRItem]:
    big, scale = _resize_for_ocr(img, target_width=target_width)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)

    data = pytesseract.image_to_data(
        gray,
        lang=TESSERACT_LANG,
        config=f"--psm {psm}",
        output_type=Output.DICT,
    )

    out: list[OCRItem] = []
    for i, raw in enumerate(data.get("text", [])):
        text = str(raw).strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = 0.0

        out.append(
            OCRItem(
                text=text,
                conf=conf,
                box=Box(
                    float(data["left"][i]) / scale,
                    float(data["top"][i]) / scale,
                    float(data["width"][i]) / scale,
                    float(data["height"][i]) / scale,
                ),
            )
        )
    return out


def _find_role_anchors(items: Iterable[OCRItem]) -> list[RoleAnchor]:
    anchors: list[RoleAnchor] = []
    for item in items:
        for role in ("attacker", "defender"):
            sim = _role_similarity(item.text, role)
            if sim < 0.58:
                continue

            # OCR confidence contributes only a little; text similarity matters more.
            score = sim + max(0.0, min(100.0, item.conf)) / 500.0
            anchors.append(RoleAnchor(role=role, score=score, item=item))
    return anchors


def _recover_role_anchors_from_band(
    img: np.ndarray,
    known: RoleAnchor,
) -> list[RoleAnchor]:
    """
    Fallback: OCR only a thin horizontal band around a known role header.
    This often recovers the opposite role on tiny/blurred screenshots without
    searching the whole screen again.
    """
    h, w = img.shape[:2]
    b = known.item.box
    y0 = max(0, int(b.y - 1.8 * max(8.0, b.h)))
    y1 = min(h, int(b.y + 3.0 * max(8.0, b.h)))
    if y1 <= y0:
        return []

    band = img[y0:y1, :]
    items = _ocr_items(band, target_width=max(1900, int(w * 2.5)), psm=6)

    shifted: list[OCRItem] = []
    for item in items:
        shifted.append(
            OCRItem(
                text=item.text,
                conf=item.conf,
                box=Box(item.box.x, item.box.y + y0, item.box.w, item.box.h),
            )
        )
    return _find_role_anchors(shifted)


# ---------------------------------------------------------------------------
# Panel-value extraction
# ---------------------------------------------------------------------------


def _red_pixel_count(crop: np.ndarray) -> int:
    if crop.size == 0:
        return 0
    b, g, r = cv2.split(crop.astype(np.int16))
    mask = (r >= 100) & ((r - g) >= 28) & ((r - b) >= 22)
    return int(np.count_nonzero(mask))


def _read_loss_below(img: np.ndarray, total_box: Box, total: int) -> tuple[Optional[int], float]:
    """
    Read the red loss number immediately below a candidate total.

    The important safety property is that a total is accepted only when a
    plausible red numeric line exists directly underneath it.
    """
    if total <= 0:
        return None, 0.0

    x, y, bw, bh = total_box.x, total_box.y, total_box.w, total_box.h
    H, W = img.shape[:2]
    bh = max(5.0, bh)
    bw = max(12.0, bw)

    votes: dict[int, float] = {}

    # Slightly different windows help with different UI scales and OCR boxes.
    windows = (
        (0.35, 1.05, 3.60, 1.45),
        (0.20, 0.95, 3.35, 1.35),
        (0.50, 1.15, 3.85, 1.60),
    )

    for left_pad, top_mul, bottom_mul, right_mul in windows:
        x0 = max(0, int(x - left_pad * bw))
        x1 = min(W, int(x + right_mul * bw))
        y0 = max(0, int(y + top_mul * bh))
        y1 = min(H, int(y + bottom_mul * bh))
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            continue

        red_count = _red_pixel_count(crop)
        if red_count < max(4, int(crop.shape[0] * crop.shape[1] * 0.002)):
            continue

        # Green channel suppresses red UI text background surprisingly well;
        # grayscale remains useful for anti-aliased digits. Use both, but only
        # on this tiny crop.
        sources = (
            cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY),
            crop[:, :, 1],
        )

        for source in sources:
            big = cv2.resize(source, None, fx=3.5, fy=3.5, interpolation=cv2.INTER_CUBIC)
            for psm in (7, 13):
                txt = pytesseract.image_to_string(
                    big,
                    lang=TESSERACT_LANG,
                    config=f"--psm {psm} -c tessedit_char_whitelist=0123456789- ,.",
                ).strip()
                val = _parse_int(txt)
                if val is None or val < 0 or val > total:
                    continue
                if val == 0 and total > 0:
                    # A literal 0 loss is possible, but red zero is uncommon and
                    # OCR false-zero is common. Main OCR can still handle it.
                    weight = 0.4
                else:
                    weight = 1.0
                votes[val] = votes.get(val, 0.0) + weight

    if not votes:
        return None, 0.0

    ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
    best_val, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    # Require repeated agreement, and reject ambiguous local OCR.
    if best_score < 2.0:
        return None, 0.0
    if second_score > 0 and (best_score - second_score) < 0.8:
        return None, 0.0

    confidence = min(1.0, 0.45 + 0.12 * best_score)
    return best_val, confidence


def _numeric_groups(items: Iterable[OCRItem], *, panel_h: int) -> list[tuple[int, Box, float, bool]]:
    """Group split OCR tokens such as `12` + `205` into one numeric row."""
    numeric: list[OCRItem] = []
    for item in items:
        if _parse_int(item.text) is None:
            continue
        # Avoid accepting words containing a stray digit.
        if not re.fullmatch(r"[\s\-+0-9.,:]+", item.text):
            continue
        numeric.append(item)

    groups: list[list[OCRItem]] = []
    tolerance = max(4.0, panel_h * 0.025)

    for item in sorted(numeric, key=lambda i: i.box.cy):
        placed = False
        for group in groups:
            group_cy = sum(g.box.cy for g in group) / len(group)
            if abs(item.box.cy - group_cy) <= tolerance:
                group.append(item)
                placed = True
                break
        if not placed:
            groups.append([item])

    parsed: list[tuple[int, Box, float, bool]] = []
    for group in groups:
        group = sorted(group, key=lambda i: i.box.x)
        text = "".join(g.text for g in group)
        val = _parse_int(text)
        if val is None:
            continue

        x0 = min(g.box.x for g in group)
        y0 = min(g.box.y for g in group)
        x1 = max(g.box.x + g.box.w for g in group)
        y1 = max(g.box.y + g.box.h for g in group)
        confs = [max(0.0, min(100.0, g.conf)) for g in group]
        avg_conf = sum(confs) / max(1, len(confs))
        has_minus = any("-" in g.text for g in group)
        parsed.append((val, Box(x0, y0, x1 - x0, y1 - y0), avg_conf, has_minus))

    return sorted(parsed, key=lambda row: row[1].cy)


def _read_panel_values(
    img: np.ndarray,
    role_anchor: RoleAnchor,
    panel_half_width: float,
) -> Optional[PanelValues]:
    H, W = img.shape[:2]
    role = role_anchor.item.box

    # Crop only the validated side-panel neighbourhood. Unlike fixed screen
    # coordinates, this follows the detected role header and therefore works
    # on full screenshots, crops and either left/right role order.
    x0 = max(0, int(role.cx - panel_half_width))
    x1 = min(W, int(role.cx + panel_half_width))
    y0 = max(0, int(role.y - max(3.0, 0.3 * role.h)))
    y1 = H
    panel = img[y0:y1, x0:x1]
    if panel.size == 0:
        return None

    ph, pw = panel.shape[:2]
    items = _ocr_items(panel, target_width=900, psm=11)

    # Numeric total/loss column is on the right half of each player panel in
    # the GGE battle-result UI. This is relative to the detected role header,
    # not to the screenshot edges.
    role_cx_local = role.cx - x0
    role_bottom_local = role.y + role.h - y0
    min_x = role_cx_local - 0.10 * pw
    max_x = role_cx_local + 0.48 * pw
    min_y = role_bottom_local + max(2.0, 0.25 * role.h)

    filtered: list[OCRItem] = []
    for item in items:
        if item.box.cy < min_y:
            continue
        if not (min_x <= item.box.cx <= max_x):
            continue
        filtered.append(item)

    groups = _numeric_groups(filtered, panel_h=ph)
    if not groups:
        return None

    candidates: list[PanelValues] = []

    for total, local_box, avg_conf, has_minus in groups:
        if total <= 0 or has_minus:
            continue
        if avg_conf < 25.0:
            continue

        global_box = Box(
            local_box.x + x0,
            local_box.y + y0,
            local_box.w,
            local_box.h,
        )
        loss, loss_conf = _read_loss_below(img, global_box, total)
        if loss is None:
            continue
        if loss > total:
            continue

        # Better OCR totals + repeated local loss agreement score higher.
        conf = 0.45 * min(1.0, avg_conf / 90.0) + 0.55 * loss_conf
        candidates.append(
            PanelValues(total=total, loss=loss, confidence=conf, total_box=global_box)
        )

    if not candidates:
        return None

    # In a valid result panel there should normally be exactly one total with
    # a red loss row directly below it. If more than one passes, take the most
    # confident and require a useful margin.
    candidates.sort(key=lambda c: c.confidence, reverse=True)
    best = candidates[0]
    if len(candidates) > 1 and best.confidence - candidates[1].confidence < 0.08:
        return None
    return best




def _global_value_candidates(
    img: np.ndarray,
    items: Iterable[OCRItem],
    *,
    min_y: float,
) -> list[PanelValues]:
    """
    Find total+red-loss pairs without using a role header.

    This is intentionally only a fallback for the *opposite* panel after one
    Attacker/Defender panel has already been positively identified. The red
    loss line directly below the total keeps unrelated screen numbers out.
    """
    H, _ = img.shape[:2]
    numeric_items = [i for i in items if i.box.cy >= min_y]
    groups = _numeric_groups(numeric_items, panel_h=H)
    out: list[PanelValues] = []

    for total, box, avg_conf, has_minus in groups:
        if total <= 0 or has_minus or avg_conf < 25.0:
            continue
        loss, loss_conf = _read_loss_below(img, box, total)
        if loss is None or loss > total:
            continue
        conf = 0.45 * min(1.0, avg_conf / 90.0) + 0.55 * loss_conf
        out.append(PanelValues(total=total, loss=loss, confidence=conf, total_box=box))

    return out


def _analyze_from_one_role(
    img: np.ndarray,
    items: list[OCRItem],
    anchors: list[RoleAnchor],
) -> Optional[BattleAnalysis]:
    """
    Safe fallback for cases where OCR sees only one of the two role labels.

    We first validate the labelled panel normally. Only then do we accept an
    unlabelled total/loss pair if it is on the other side and horizontally
    aligned with the labelled panel's total row. Thus a stray 'Defender' word
    elsewhere on a full screenshot is not enough to trigger an analysis.
    """
    H, W = img.shape[:2]
    results: list[BattleAnalysis] = []

    # Stronger role matches first; weak fuzzy matches are considered only if
    # they also validate as a real result panel.
    for known in sorted(anchors, key=lambda a: a.score, reverse=True)[:6]:
        # Without a second role header we do not know panel separation yet.
        # Start with a conservative local panel width derived from the screen.
        # Try two widths because a crop and a full screenshot have different
        # amounts of surrounding UI.
        known_values = None
        for half in (0.13 * W, 0.18 * W, 0.23 * W):
            known_values = _read_panel_values(img, known, half)
            if known_values is not None:
                break
        if known_values is None:
            continue

        other_candidates = _global_value_candidates(
            img,
            items,
            min_y=max(0.0, known.item.box.y + known.item.box.h),
        )

        scored: list[tuple[float, PanelValues]] = []
        kb = known_values.total_box
        for cand in other_candidates:
            cb = cand.total_box

            # Must be a distinct side panel.
            dx = abs(cb.cx - kb.cx)
            if dx < 0.16 * W:
                continue

            # Totals in the two result panels sit on the same row.
            y_tol = max(0.055 * H, 4.0 * max(kb.h, cb.h, 5.0))
            dy = abs(cb.cy - kb.cy)
            if dy > y_tol:
                continue

            y_score = max(0.0, 1.0 - dy / y_tol)
            x_score = min(1.0, dx / max(1.0, 0.35 * W))
            score = 0.55 * cand.confidence + 0.30 * y_score + 0.15 * x_score
            scored.append((score, cand))

        if not scored:
            continue

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, other = scored[0]
        if len(scored) > 1 and best_score - scored[1][0] < 0.08:
            # More than one different opposite panel looks equally plausible.
            continue

        if known.role == "defender":
            defender = known_values
            attacker = other
        else:
            attacker = known_values
            defender = other

        if attacker.loss > attacker.total or defender.loss > defender.total:
            continue

        role_conf = min(1.0, known.score / 1.10)
        confidence = (
            0.24 * role_conf
            + 0.36 * known_values.confidence
            + 0.30 * other.confidence
            + 0.10 * min(1.0, best_score)
        )

        results.append(
            BattleAnalysis(
                attacker_total=attacker.total,
                attacker_loss=attacker.loss,
                defender_total=defender.total,
                defender_loss=defender.loss,
                confidence=confidence,
            )
        )

    if not results:
        return None

    results.sort(key=lambda r: r.confidence, reverse=True)
    best = results[0]
    if best.confidence < 0.62:
        return None
    if len(results) > 1 and best.confidence - results[1].confidence < 0.04:
        r2 = results[1]
        if (
            best.attacker_total != r2.attacker_total
            or best.attacker_loss != r2.attacker_loss
            or best.defender_total != r2.defender_total
            or best.defender_loss != r2.defender_loss
        ):
            return None
    return best


# ---------------------------------------------------------------------------
# Battle-panel pairing
# ---------------------------------------------------------------------------


def _pair_geometry_score(a: RoleAnchor, d: RoleAnchor, W: int, H: int) -> Optional[float]:
    ab = a.item.box
    db = d.item.box
    sep = abs(ab.cx - db.cx)
    if sep < 0.14 * W or sep > 0.82 * W:
        return None

    ydiff = abs(ab.cy - db.cy)
    allowed_y = max(0.07 * H, 4.5 * max(ab.h, db.h, 5.0))
    if ydiff > allowed_y:
        return None

    y_score = max(0.0, 1.0 - ydiff / allowed_y)
    sep_score = min(1.0, sep / max(1.0, 0.35 * W))
    return 0.55 * y_score + 0.45 * sep_score


def _analyze_with_anchors(img: np.ndarray, anchors: list[RoleAnchor]) -> Optional[BattleAnalysis]:
    H, W = img.shape[:2]
    attackers = [a for a in anchors if a.role == "attacker"]
    defenders = [a for a in anchors if a.role == "defender"]

    results: list[BattleAnalysis] = []

    for attacker in attackers:
        for defender in defenders:
            geo = _pair_geometry_score(attacker, defender, W, H)
            if geo is None:
                continue

            sep = abs(attacker.item.box.cx - defender.item.box.cx)
            panel_half = min(0.24 * W, max(0.10 * W, 0.33 * sep))

            av = _read_panel_values(img, attacker, panel_half)
            if av is None:
                continue
            dv = _read_panel_values(img, defender, panel_half)
            if dv is None:
                continue

            # Strong consistency checks: these are losses belonging to the
            # two validated role panels, not arbitrary screen numbers.
            if av.loss > av.total or dv.loss > dv.total:
                continue

            role_score = min(1.0, (attacker.score + defender.score) / 2.2)
            confidence = (
                0.25 * role_score
                + 0.20 * geo
                + 0.275 * av.confidence
                + 0.275 * dv.confidence
            )

            results.append(
                BattleAnalysis(
                    attacker_total=av.total,
                    attacker_loss=av.loss,
                    defender_total=dv.total,
                    defender_loss=dv.loss,
                    confidence=confidence,
                )
            )

    if not results:
        return None

    results.sort(key=lambda r: r.confidence, reverse=True)
    best = results[0]

    # If two unrelated places on screen somehow both look plausible, do not
    # guess unless the best candidate is clearly ahead.
    if len(results) > 1 and best.confidence - results[1].confidence < 0.05:
        same_values = (
            best.attacker_total == results[1].attacker_total
            and best.attacker_loss == results[1].attacker_loss
            and best.defender_total == results[1].defender_total
            and best.defender_loss == results[1].defender_loss
        )
        if not same_values:
            return None

    if best.confidence < 0.58:
        return None
    return best


def analyze_battle_report(image_bytes: bytes) -> Optional[BattleAnalysis]:
    """Analyze one Goodgame Empire battle-report screenshot."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return None

    H, W = img.shape[:2]
    if H < 60 or W < 180:
        return None

    items = _ocr_items(img, target_width=1900, psm=11)
    anchors = _find_role_anchors(items)

    result = _analyze_with_anchors(img, anchors)
    if result is not None:
        return result

    # Fallback only when one side/header was OCRed poorly. Re-OCR a thin band
    # around the strongest known role instead of searching every occurrence
    # on the whole screenshot with increasingly loose rules.
    if anchors:
        strongest = max(anchors, key=lambda a: a.score)
        recovered = _recover_role_anchors_from_band(img, strongest)

        # Deduplicate approximately identical anchors.
        merged = list(anchors)
        for candidate in recovered:
            duplicate = False
            for old in merged:
                if old.role != candidate.role:
                    continue
                if (
                    abs(old.item.box.cx - candidate.item.box.cx) < 0.04 * W
                    and abs(old.item.box.cy - candidate.item.box.cy) < 0.04 * H
                ):
                    duplicate = True
                    break
            if not duplicate:
                merged.append(candidate)

        result = _analyze_with_anchors(img, merged)
        if result is not None:
            return result

        one_role = _analyze_from_one_role(img, items, merged)
        if one_role is not None:
            return one_role

    # Safety first: no verified battle-result panel pair.
    return None


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_battle_ratio(result: BattleAnalysis) -> str:
    if result.attacker_loss == 0:
        if result.defender_loss > 0:
            return "∞ : 1"
        return "0 : 0"
    return f"{result.defender_loss / result.attacker_loss:.2f} : 1"


def format_defender_loss_percent(result: BattleAnalysis) -> str:
    pct = result.defender_loss_percent
    return f"{pct:.2f}".rstrip("0").rstrip(".") + "%"


def format_discord_result(result: BattleAnalysis) -> str:
    return (
        f"**Battle ratio: {format_battle_ratio(result)}**\n"
        f"**Straty obrancu: {format_defender_loss_percent(result)}**"
    )


# ---------------------------------------------------------------------------
# Generic discord.py integration helper
# ---------------------------------------------------------------------------


async def process_discord_message(
    message,
    *,
    failure_reaction: Optional[str] = "❓",
    reply: bool = True,
) -> bool:
    """
    Process image attachments on an existing discord.py Message.

    No discord.py import is required in this module; it intentionally uses the
    normal Message/Attachment interface by duck typing, so it can be dropped
    into an existing Client or commands.Bot project.

    Returns True if at least one battle report was successfully analyzed.
    """
    author = getattr(message, "author", None)
    if getattr(author, "bot", False):
        return False

    attachments = getattr(message, "attachments", None) or []
    handled = False

    for attachment in attachments:
        content_type = (getattr(attachment, "content_type", None) or "").lower()
        filename = (getattr(attachment, "filename", None) or "").lower()
        is_image = content_type.startswith("image/") or filename.endswith(
            (".png", ".jpg", ".jpeg", ".webp", ".bmp")
        )
        if not is_image:
            continue

        try:
            image_bytes = await attachment.read()
        except Exception:
            continue

        result = await asyncio.to_thread(analyze_battle_report, image_bytes)
        if result is None:
            continue

        handled = True
        text = format_discord_result(result)
        try:
            if reply and hasattr(message, "reply"):
                await message.reply(text, mention_author=False)
            else:
                await message.channel.send(text)
        except Exception:
            # Analyzer success should not crash the user's existing bot because
            # of a Discord permission/network error.
            pass

    if not handled and failure_reaction:
        try:
            await message.add_reaction(failure_reaction)
        except Exception:
            pass

    return handled


# ---------------------------------------------------------------------------
# Optional local test: python gge_battle_ratio.py screenshot.png
# ---------------------------------------------------------------------------


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Analyze a GGE battle report screenshot")
    parser.add_argument("image", help="Path to PNG/JPG screenshot")
    args = parser.parse_args()

    with open(args.image, "rb") as f:
        result = analyze_battle_report(f.read())

    if result is None:
        print("Battle report not confidently recognized.")
        return 2

    print(format_discord_result(result).replace("**", ""))
    print(
        f"DEBUG: attacker={result.attacker_loss}/{result.attacker_total}, "
        f"defender={result.defender_loss}/{result.defender_total}, "
        f"confidence={result.confidence:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
