"""Local, LLM-free resource-bar reader.

Reads ``food / wood / gold / stone / population`` (+ ``age``, best-effort) from an
AoE2:DE screenshot **without** calling an LLM — the strategist's source of truth
for the HUD, which replaced Claude vision. The output dict uses the same keys as
``StrategistProvider``'s ``ResourceReadings`` and scores directly against
``gameplay_agent.strategist_eval.evaluate_resource_readings``.

Backends (``read_resource_bar(..., backend=...)``)
--------------------------------------------------
- ``"rapidocr"`` (production): PaddleOCR models on onnxruntime — pip-only, no
  system binary. Used by the live strategist.
- ``"template"``: OpenCV NCC against per-digit glyph crops; needs only ``opencv``
  plus per-resolution templates. Useful with no OCR engine, and as a
  lone-single-digit fallback for the engine backends.
- ``"tesseract"`` (optional): ``pytesseract`` digit-whitelist OCR; needs the
  Tesseract binary.

Field geometry is resolution-specific and supplied two ways, in precedence order:
``autodetect_calibration`` localizes the bar from the live frame at runtime
(resolution-independent, no assets needed), and a hand-tuned
``resource_ocr_assets/calibration.<W>x<H>.yaml`` overrides it when present (see
``resource_ocr_assets/README.md``). Geometry is data, never hardcoded.

On-screen field order is Wood → Food → Gold → Stone → Population
(``prompts/strategist.md``), which differs from the ResourceReadings field order.

Run ``python -m gameplay_agent.resource_ocr --selftest`` to verify the OCR
machinery with synthesized digits (no real screenshots needed).
"""

from __future__ import annotations

import io
import statistics
import threading
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, NotRequired, TypedDict, cast

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from PIL import ImageFont
    from rapidocr_onnxruntime import RapidOCR

    _OcrLine = tuple[Sequence[Sequence[float]], str, float]
    _OcrResult = list[_OcrLine]

# On-screen left-to-right order of the four resource counters + population.
RESOURCE_FIELDS: tuple[str, ...] = ("wood", "food", "gold", "stone")
POP_FIELD = "population"

_IDLE_SAT_THRESHOLD = 25.0
_IDLE_ICON_WIDTH_FRAC = 0.7
_IDLE_ICON_Y_PAD = 6

_IDLE_COUNT_X_LO = 3.5
_IDLE_COUNT_X_HI = 6.8
_IDLE_COUNT_Y_HI = 1.8
_IDLE_WHITE_THR = 185
_IDLE_COUNT_MIN_NCC = 0.45

_IDLE_DIGIT_MIN_H_FRAC = 0.45
_IDLE_DIGIT_MAX_H_FRAC = 1.3
_IDLE_DIGIT_MAX_W_FRAC = 1.2
_IDLE_DIGIT_MIN_AREA = 12
_HUD_DIGITS_DIR = Path(__file__).parent / "resource_ocr_assets" / "templates" / "hud_digits"
_GLYPH_HW: tuple[int, int] = (28, 20)

Backend = Literal["template", "tesseract", "rapidocr"]


class ResourceReadings(TypedDict, total=False):
    """One frame's cleaned HUD readings — a key is present only when read."""

    food: int
    wood: int
    gold: int
    stone: int
    population: str
    age: str
    idle_present: bool
    idle_count: int


_DIGITS = "0123456789"
_DIGITS_SLASH = "0123456789/"
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "


def _filter_charset(text: str, charset: str) -> str:
    return "".join(c for c in text if c in charset)


@dataclass
class FieldBox:
    """Pixel box (inclusive-exclusive) of one reading in the full screenshot."""

    x0: int
    y0: int
    x1: int
    y1: int

    def crop(self, img: np.ndarray) -> np.ndarray:
        return cast("np.ndarray", img[self.y0 : self.y1, self.x0 : self.x1])


class _CalibrationData(TypedDict):
    width: int
    height: int
    fields: dict[str, list[int]]
    template_dir: NotRequired[str]


@dataclass
class Calibration:
    width: int
    height: int
    fields: dict[str, FieldBox]
    template_dir: Path

    def field_rects(self) -> dict[str, tuple[int, int, int, int]]:
        return {name: (box.x0, box.y0, box.x1, box.y1) for name, box in self.fields.items()}

    @classmethod
    def from_yaml(cls, path: str | Path) -> Calibration:
        import yaml

        path = Path(path)
        data = cast("_CalibrationData", yaml.safe_load(path.read_text()))
        fields = {
            name: FieldBox(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            for name, box in data["fields"].items()
        }
        tdir = Path(data.get("template_dir", "templates"))
        if not tdir.is_absolute():
            tdir = path.parent / tdir
        return cls(
            width=int(data["width"]),
            height=int(data["height"]),
            fields=fields,
            template_dir=tdir,
        )


ASSETS_DIR = Path(__file__).parent / "resource_ocr_assets"


def calibration_for(width: int, height: int) -> Calibration | None:
    path = ASSETS_DIR / f"calibration.{width}x{height}.yaml"
    return Calibration.from_yaml(path) if path.exists() else None


def _decode_gray(screenshot_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(screenshot_bytes)).convert("L")
    return np.asarray(img, dtype=np.uint8)


def _decode_rgb(screenshot_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _binarize_digits(gray: np.ndarray) -> np.ndarray:
    import cv2

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if binary.mean() > 127:
        binary = cv2.bitwise_not(binary)
    return binary


def _normalize_glyph(glyph: np.ndarray) -> np.ndarray:
    import cv2

    g = glyph if glyph.dtype == np.uint8 else cast("np.ndarray", glyph.astype(np.uint8))
    ys, xs = np.where(g > 0)
    if ys.size == 0:
        return np.zeros(_GLYPH_HW, dtype=np.uint8)
    ymin, ymax = cast("int", ys.min()), cast("int", ys.max())
    xmin, xmax = cast("int", xs.min()), cast("int", xs.max())
    g = cast("np.ndarray", g[ymin : ymax + 1, xmin : xmax + 1])
    out_h, out_w = _GLYPH_HW
    h, w = cast("tuple[int, int]", g.shape)
    scale = min(out_h / h, out_w / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros(_GLYPH_HW, dtype=np.uint8)
    y0, x0 = (out_h - nh) // 2, (out_w - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    af = cast("np.ndarray", a.astype(np.float32).ravel())
    bf = cast("np.ndarray", b.astype(np.float32).ravel())
    af -= cast("float", af.mean())
    bf -= cast("float", bf.mean())
    denom = cast("float", np.linalg.norm(af)) * cast("float", np.linalg.norm(bf))
    if denom == 0.0:
        return 0.0
    return cast("float", np.dot(af, bf)) / denom


def load_templates(template_dir: Path, *, include_slash: bool) -> dict[str, np.ndarray]:
    chars = [str(d) for d in range(10)]
    paths = {c: template_dir / f"{c}.png" for c in chars}
    if include_slash:
        paths["/"] = template_dir / "slash.png"
    templates: dict[str, np.ndarray] = {}
    for char, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"missing glyph template: {p}")
        gray = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
        templates[char] = _normalize_glyph(np.where(gray > 127, 255, 0).astype(np.uint8))
    return templates


def _segment_components(
    binary: np.ndarray,
    *,
    min_h: float,
    max_h: float | None = None,
    max_w: float | None = None,
    min_area: int,
) -> list[tuple[int, np.ndarray]]:
    import cv2

    n, _labels, stats, _centroids = cast(
        "tuple[int, np.ndarray, np.ndarray, np.ndarray]",
        cv2.connectedComponentsWithStats(binary),
    )
    glyphs: list[tuple[int, np.ndarray]] = []
    for i in range(1, n):
        stat_row = cast("np.ndarray", stats[i])
        x, y, w, h, area = (int(v) for v in cast("list[int]", stat_row.tolist()))
        if h < min_h or area < min_area:
            continue
        if (max_h is not None and h > max_h) or (max_w is not None and w > max_w):
            continue
        glyph = cast("np.ndarray", binary[y : y + h, x : x + w])
        glyphs.append((x, _normalize_glyph(glyph)))
    glyphs.sort(key=lambda g: g[0])
    return glyphs


def _segment_glyphs(field_binary: np.ndarray) -> list[tuple[int, np.ndarray]]:
    field_h = cast("int", field_binary.shape[0])
    return _segment_components(field_binary, min_h=0.3 * field_h, min_area=6)


def _classify_bank(
    glyph: np.ndarray, bank: Mapping[str, Sequence[np.ndarray]]
) -> tuple[str, float]:
    best_char, best_score = "", -2.0
    for char, samples in bank.items():
        for tmpl in samples:
            score = _ncc(glyph, tmpl)
            if score > best_score:
                best_char, best_score = char, score
    return best_char, best_score


_FIELD_MIN_NCC = 0.4


def _read_field(field_img: np.ndarray, templates: dict[str, np.ndarray]) -> str:
    binary = _binarize_digits(field_img)
    bank = {char: (tmpl,) for char, tmpl in templates.items()}
    chars: list[str] = []
    for _x, glyph in _segment_glyphs(binary):
        char, score = _classify_bank(glyph, bank)
        if score < _FIELD_MIN_NCC:
            return ""
        chars.append(char)
    return "".join(chars)


def _preprocess_for_ocr(field_img: np.ndarray, *, pad: int, binarize: bool = True) -> Image.Image:
    import cv2

    if binarize:
        proc = cast("np.ndarray", cv2.bitwise_not(_binarize_digits(field_img)))
        proc = cast(
            "np.ndarray",
            cv2.copyMakeBorder(proc, pad, pad, pad + 10, pad + 10, cv2.BORDER_CONSTANT, value=255),
        )
        scale = 4
    else:
        proc = field_img
        scale = 3
    pil = Image.fromarray(proc)
    return pil.resize((pil.width * scale, pil.height * scale), Image.Resampling.LANCZOS)


def _read_field_tesseract(field_img: np.ndarray, *, whitelist: str, binarize: bool = True) -> str:
    import pytesseract

    pil = _preprocess_for_ocr(field_img, pad=16, binarize=binarize)
    for psm in (7, 10):
        text = pytesseract.image_to_string(
            pil, config=f"--psm {psm} -c tessedit_char_whitelist={whitelist}"
        )
        cleaned = _filter_charset(text, whitelist)
        if cleaned:
            return cleaned
    return ""


_ENGINES: dict[str, RapidOCR] = {}
_ENGINE_LOCK = threading.Lock()

_FIELD_LIMITS: Final = {
    "det_limit_type": "max",
    "det_limit_side_len": 3072,
    "use_cls": False,
}


def _engine(name: str, **settings: object) -> RapidOCR:
    with _ENGINE_LOCK:
        if name not in _ENGINES:
            from rapidocr_onnxruntime import RapidOCR
            _ENGINES[name] = RapidOCR(**settings)
        return _ENGINES[name]


def _field_engine() -> RapidOCR:
    return _engine("field", **_FIELD_LIMITS)


def _band_engine() -> RapidOCR:
    return _engine("band")


def warm_up_ocr() -> None:
    import time
    import structlog

    log = structlog.stdlib.get_logger()
    try:
        started = time.monotonic()
        engine = _field_engine()
        engine(np.zeros((32, 96, 3), dtype=np.uint8))
        log.info("ocr_engine_warmed", seconds=round(time.monotonic() - started, 1))
    except Exception as e:
        log.warning("ocr_warmup_failed", error=str(e))


def _read_field_rapidocr(field_img: np.ndarray, *, whitelist: str, binarize: bool = True) -> str:
    pil = _preprocess_for_ocr(field_img, pad=20, binarize=binarize).convert("RGB")
    raw, _elapse = cast("tuple[_OcrResult | None, object]", _field_engine()(np.asarray(pil)))
    if not raw:
        return ""
    return _filter_charset("".join(line[1] for line in raw), whitelist)


_AGE_KEYWORDS = (
    ("imperial", "Imperial Age"),
    ("castle", "Castle Age"),
    ("feudal", "Feudal Age"),
    ("dark", "Dark Age"),
)


def _map_age(text: str) -> str:
    low = text.lower()
    for keyword, canonical in _AGE_KEYWORDS:
        if keyword in low:
            return canonical
    return ""


class Box(NamedTuple):
    """Axis-aligned pixel box (inclusive-exclusive) from text detection."""

    x0: int
    y0: int
    x1: int
    y1: int


def _bbox(quad: Sequence[Sequence[float]]) -> Box:
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return Box(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


def _detect(engine: RapidOCR, image: np.ndarray, top_frac: float) -> list[tuple[Box, str]]:
    band_h = int(cast("int", image.shape[0]) * top_frac)
    band = cast("np.ndarray", image[:band_h, :])
    raw, _elapse = cast("tuple[_OcrResult | None, object]", engine(band))
    dets: list[tuple[Box, str]] = []
    for quad, text, _score in raw or []:
        dets.append((_bbox(quad), "".join(text.split())))
    return dets


def _extract(dets: list[tuple[Box, str]]) -> tuple[list[Box], Box | None, Box | None]:
    numeric = [b for b, t in dets if t.isdigit()]
    pop_dets = [b for b, t in dets if "/" in t and t.replace("/", "").isdigit()]
    age = [b for b, t in dets if _map_age(t)]
    pop_box = max(pop_dets, key=lambda b: b.y1 - b.y0) if pop_dets else None
    if pop_box is not None:
        yc = (pop_box.y0 + pop_box.y1) / 2
        band = 0.5 * (pop_box.y1 - pop_box.y0)
        main = [b for b in numeric if abs((b.y0 + b.y1) / 2 - yc) <= band]
    else:
        hmax = max((b.y1 - b.y0 for b in numeric), default=0)
        main = [b for b in numeric if (b.y1 - b.y0) >= 0.8 * hmax]
    return main, pop_box, (age[0] if age else None)


def _idle_icon_region(pop: FieldBox) -> tuple[int, int, int, int]:
    pw = pop.x1 - pop.x0
    return (
        pop.x1,
        pop.y0 - _IDLE_ICON_Y_PAD,
        pop.x1 + int(_IDLE_ICON_WIDTH_FRAC * pw),
        pop.y1 + _IDLE_ICON_Y_PAD // 2,
    )


def detect_idle_present(rgb: np.ndarray, pop: FieldBox) -> bool:
    x0, y0, x1, y1 = _idle_icon_region(pop)
    h, w = cast("tuple[int, int]", rgb.shape[:2])
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return False
    patch = rgb[y0:y1, x0:x1].astype(np.float32)
    saturation = float((patch.max(axis=2) - patch.min(axis=2)).mean())
    return saturation > _IDLE_SAT_THRESHOLD


_hud_digit_bank: dict[str, list[np.ndarray]] | None = None


def _load_hud_digit_bank() -> dict[str, list[np.ndarray]]:
    global _hud_digit_bank
    if _hud_digit_bank is None:
        bank: dict[str, list[np.ndarray]] = {}
        for p in sorted(_HUD_DIGITS_DIR.glob("[0-9]_*.png")):
            arr = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
            bank.setdefault(p.stem[0], []).append(np.where(arr > 127, 255, 0).astype(np.uint8))
        _hud_digit_bank = bank
    return _hud_digit_bank


def read_idle_count(rgb: np.ndarray, pop: FieldBox) -> int | None:
    ph = pop.y1 - pop.y0
    frame_h, frame_w = cast("tuple[int, int]", rgb.shape[:2])
    x0 = pop.x0 + int(_IDLE_COUNT_X_LO * ph)
    x1 = min(frame_w, pop.x0 + int(_IDLE_COUNT_X_HI * ph))
    y0 = max(0, pop.y0)
    y1 = min(frame_h, pop.y0 + int(_IDLE_COUNT_Y_HI * ph))
    if x1 <= x0 or y1 <= y0:
        return None
    bank = _load_hud_digit_bank()
    if not bank:
        return None
    patch = rgb[y0:y1, x0:x1]
    mask = cast("np.ndarray", (patch.min(axis=2) >= _IDLE_WHITE_THR).astype(np.uint8) * 255)
    glyphs = _segment_components(
        mask,
        min_h=_IDLE_DIGIT_MIN_H_FRAC * ph,
        max_h=_IDLE_DIGIT_MAX_H_FRAC * ph,
        max_w=_IDLE_DIGIT_MAX_W_FRAC * ph,
        min_area=_IDLE_DIGIT_MIN_AREA,
    )
    if not glyphs:
        return None
    digits = ""
    for _x, glyph in glyphs:
        best_char, best_score = _classify_bank(glyph, bank)
        if best_score < _IDLE_COUNT_MIN_NCC:
            return None
        digits += best_char
    return int(digits) if digits.isdigit() else None


def _column_centers(x0s: list[int], k: int = 4, gap: int = 70) -> list[int]:
    if not x0s:
        return []
    xs = sorted(x0s)
    groups: list[list[int]] = [[xs[0]]]
    for x in xs[1:]:
        (groups.append([x]) if x - groups[-1][-1] > gap else groups[-1].append(x))
    groups.sort(key=len, reverse=True)
    return sorted(int(statistics.median(g)) for g in groups[:k])


def _assign(main: list[Box], centers: list[int], tol: float) -> dict[str, Box]:
    out: dict[str, Box] = {}
    if not centers:
        return out
    for b in main:
        idx = min(range(len(centers)), key=lambda j: abs(b.x0 - centers[j]))
        if idx < len(RESOURCE_FIELDS) and abs(b.x0 - centers[idx]) <= tol:
            name = RESOURCE_FIELDS[idx]
            if name not in out or abs(b.x0 - centers[idx]) < abs(out[name].x0 - centers[idx]):
                out[name] = b
    return out


_Y_TOP_PAD = 2


def _build_fields_single_frame(
    assigned: dict[str, Box],
    pop: Box | None,
    age: Box | None,
    *,
    frame_w: int,
    pad: int,
) -> dict[str, FieldBox]:
    present: dict[str, Box] = {n: assigned[n] for n in RESOURCE_FIELDS if n in assigned}
    if pop is not None:
        present[POP_FIELD] = pop
    order = [n for n in (*RESOURCE_FIELDS, POP_FIELD) if n in present]
    if not order:
        return {}
    row_boxes = [present[n] for n in order]
    y0 = max(0, min(b.y0 for b in row_boxes) - _Y_TOP_PAD)
    y1 = max(b.y1 for b in row_boxes)
    x0s = [present[n].x0 for n in order]
    gaps = [b - a for a, b in pairwise(x0s)]
    pitch = statistics.median(gaps) if gaps else 150
    safety = max(2, round(0.06 * pitch))
    fields: dict[str, FieldBox] = {}
    for i, name in enumerate(order):
        b = present[name]
        x0 = max(0, b.x0)
        x1 = b.x1 + pad
        if i + 1 < len(order):
            x1 = min(x1, present[order[i + 1]].x0 - safety)
        x1 = min(max(x1, b.x1), frame_w)
        fields[name] = FieldBox(x0, y0, x1, y1)
    if age is not None:
        fields["age"] = FieldBox(
            max(0, age.x0), max(0, age.y0 - pad), min(frame_w, age.x1 + pad), age.y1 + pad
        )
    return fields


def autodetect_calibration(
    screenshot_bytes: bytes,
    *,
    top_frac: float = 0.15,
    pad: int = 4,
) -> Calibration | None:
    try:
        rgb = _decode_rgb(screenshot_bytes)
    except Exception:
        return None
    h, w = cast("tuple[int, int]", rgb.shape[:2])
    dets = _detect(_band_engine(), rgb, top_frac)
    main, pop, age = _extract(dets)
    centers = _column_centers([b.x0 for b in main])
    pitch = (centers[-1] - centers[0]) / (len(centers) - 1) if len(centers) >= 2 else 150
    assigned = _assign(main, centers, 0.4 * pitch)
    if pop is None and len(assigned) < 2:
        return None
    fields = _build_fields_single_frame(assigned, pop, age, frame_w=w, pad=pad)
    if not fields:
        return None
    tdir = ASSETS_DIR / "templates" / f"{w}x{h}"
    if not tdir.exists():
        tdir = ASSETS_DIR / "templates" / "__autodetect_none__"
    return Calibration(width=w, height=h, fields=fields, template_dir=tdir)


def read_resource_bar(
    screenshot_bytes: bytes,
    calibration: Calibration,
    *,
    backend: Backend = "template",
) -> dict[str, object]:
    gray = _decode_gray(screenshot_bytes)
    templates_num = templates_pop = None
    try:
        templates_num = load_templates(calibration.template_dir, include_slash=False)
        templates_pop = load_templates(calibration.template_dir, include_slash=True)
    except FileNotFoundError:
        if backend == "template":
            raise

    def ocr(crop: np.ndarray, whitelist: str, *, binarize: bool = True) -> str:
        if backend == "rapidocr":
            return _read_field_rapidocr(crop, whitelist=whitelist, binarize=binarize)
        return _read_field_tesseract(crop, whitelist=whitelist, binarize=binarize)

    def read_num(crop: np.ndarray) -> str:
        if backend == "template":
            assert templates_num is not None
            return _read_field(crop, templates_num)
        digits = ocr(crop, _DIGITS)
        if not digits and templates_num is not None:
            digits = _read_field(crop, templates_num)
        return digits

    def read_pop(crop: np.ndarray) -> str:
        if backend == "template":
            assert templates_pop is not None
            return _read_field(crop, templates_pop)
        raw = ocr(crop, _DIGITS_SLASH)
        if "/" not in raw and templates_pop is not None:
            raw = _read_field(crop, templates_pop)
        return raw

    out: dict[str, object] = {}

    for name in RESOURCE_FIELDS:
        box = calibration.fields.get(name)
        if box is None:
            continue
        digits = read_num(box.crop(gray))
        if digits.isdigit():
            out[name] = int(digits)

    pop_box = calibration.fields.get(POP_FIELD)
    if pop_box is not None:
        raw = read_pop(pop_box.crop(gray))
        if "/" in raw:
            out[POP_FIELD] = raw
        rgb = _decode_rgb(screenshot_bytes)
        idle_present = detect_idle_present(rgb, pop_box)
        out["idle_present"] = idle_present
        if not idle_present:
            out["idle_count"] = 0
        else:
            idle_count = read_idle_count(rgb, pop_box)
            if idle_count is not None:
                out["idle_count"] = idle_count

    age_box = calibration.fields.get("age")
    if age_box is not None and backend != "template":
        out["age"] = _map_age(ocr(age_box.crop(gray), _LETTERS, binarize=False))
    else:
        out["age"] = ""

    return out


def read_age(screenshot_bytes: bytes, calibration: Calibration) -> str:
    age_box = calibration.fields.get("age")
    if age_box is None:
        return ""
    crop = age_box.crop(_decode_gray(screenshot_bytes))
    return _map_age(_read_field_rapidocr(crop, whitelist=_LETTERS, binarize=False))


_SELFTEST_FONTS = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "DejaVuSans.ttf",
)


def _selftest_font(size: int = 32) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    from PIL import ImageFont

    for path in _SELFTEST_FONTS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _render_digit_image(text: str) -> np.ndarray:
    from PIL import ImageDraw

    font = _selftest_font()
    canvas = Image.new("L", (400, 80), color=0)
    ImageDraw.Draw(canvas).text((10, 10), text, fill=255, font=font)
    bbox = canvas.getbbox()
    if bbox is None:
        return np.zeros((40, 40), dtype=np.uint8)
    x0, y0, x1, y1 = bbox
    pad = 3
    crop = canvas.crop((max(0, x0 - pad), max(0, y0 - pad), x1 + pad, y1 + pad))
    return np.asarray(crop, dtype=np.uint8)


def _selftest() -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tdir = Path(tmp)
        for d in range(10):
            Image.fromarray(_render_digit_image(str(d))).save(tdir / f"{d}.png")
        Image.fromarray(_render_digit_image("/")).save(tdir / "slash.png")
        templates_num = load_templates(tdir, include_slash=False)
        templates_pop = load_templates(tdir, include_slash=True)

        cases_num = ["0", "7", "42", "200", "1530"]
        cases_pop = ["8/15", "12/200"]
        ok = True
        for want in cases_num:
            got = _read_field(_render_digit_image(want), templates_num)
            mark = "ok " if got == want else "ERR"
            if got != want:
                ok = False
            print(f"  [{mark}] number  want={want!r:>8}  got={got!r}")
        for want in cases_pop:
            got = _read_field(_render_digit_image(want), templates_pop)
            mark = "ok " if got == want else "ERR"
            if got != want:
                ok = False
            print(f"  [{mark}] pop     want={want!r:>8}  got={got!r}")

    print("\nSELFTEST:", "PASS" if ok else "FAIL")
    print(
        "(This validates the segmentation+classification machinery on a synthetic\n"
        " font. Real-game accuracy still requires calibration + real screenshots —\n"
        " see resource_ocr_assets/README.md.)"
    )
    return 0 if ok else 1


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Local resource-bar OCR.")
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Synthesize digits and verify the template backend (no screenshots needed).",
    )
    args = parser.parse_args()
    if cast("bool", args.selftest):
        return _selftest()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
