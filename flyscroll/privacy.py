"""Privacy filter for the frames FlyScroll *shows* and *logs*.

FlyScroll points a simulated fly brain at a real phone screen. The frame the
retina consumes is raw by necessity -- the model has to see what the screen
showed. But the same frame is also encoded into ``/state`` for the dashboard and
written to disk as ``profile_XXXX.jpg`` next to the Bumble decision log, and a
dating feed is full of other people's faces and names. :class:`PrivacyFilter`
sits on that *display/log* path only: :meth:`PrivacyFilter.apply` returns a
censored **copy** and never mutates its input, so the fly keeps receiving the
original array.

Two censors, both local -- no model files are ever downloaded:

* **Faces** -- OpenCV's **YuNet** CNN detector (``cv2.FaceDetectorYN``), a
  small (~230 KB) ONNX model bundled at ``flyscroll/models/`` that handles
  angled, profile and small faces far more reliably than the old Haar cascades.
  Detected boxes are grown ~30% (to cover hair, chin and ears) and pixelated to
  8x8 blocks -- pixelation reads unambiguously as "this was censored". If the
  model or the YuNet API is unavailable, detection falls back to the bundled
  Haar cascades.
* **Text** -- a *heuristic* region detector, not OCR: horizontal gradient
  energy (Sobel-x) is thresholded with Otsu, closed with a wide horizontal
  kernel, and the resulting connected components are kept when their shape
  looks line-of-text-like (height 8-60 px at the 360x640 working scale, at
  least 2.2x wider than tall, better than 25% filled). Matching regions are
  painted with solid black bars.

  Because it keys on contrast rather than on glyphs, this detector **also
  blacks out some non-text high-contrast horizontal structure** -- UI dividers,
  progress bars, striped clothing, window sills. That over-censoring is the
  deliberate trade for a detector that needs zero external models and no
  network access. It is a privacy convenience, not a guarantee: treat captured
  frames as sensitive regardless.

Detection runs every third call and the boxes are reused in between (frames
arrive around 20x/s and neither faces nor captions move far in 150 ms); the
most recent boxes are readable as :attr:`PrivacyFilter.last_boxes`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_YUNET_PATH = Path(__file__).with_name("models") / "face_detection_yunet_2023mar.onnx"
_DB_TEXT_PATH = Path(__file__).with_name("models") / "text_detection_en_ppocrv3_2023may.onnx"

_CV2 = None
_CV2_WARNED = False


def _cv2():
    """Import cv2 lazily; return ``None`` (warning once) if it is not installed."""
    global _CV2, _CV2_WARNED
    if _CV2 is not None:
        return _CV2 if _CV2 is not False else None
    try:
        import cv2  # type: ignore

        _CV2 = cv2
        return cv2
    except Exception:
        _CV2 = False
        if not _CV2_WARNED:
            _CV2_WARNED = True
            print(
                "[flyscroll.privacy] opencv is not installed; privacy mode is "
                "disabled and frames are shown/logged unfiltered. "
                "Install it with: pip install opencv-python-headless"
            )
        return None


class PrivacyFilter:
    """Pixelate faces and black out text-like regions on a copy of a frame.

    Parameters
    ----------
    enabled:
        When false, :meth:`apply` returns the input array unchanged.
    blur_faces:
        Run the Haar face cascades and pixelate what they find.
    censor_text:
        Run the heuristic text-region detector and draw black bars.
    """

    DETECT_EVERY = 1
    MAX_TEXT_REGIONS = 40
    _FACE_WORK_WIDTH = 180

    def __init__(
        self,
        enabled: bool = True,
        blur_faces: bool = True,
        censor_text: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.blur_faces = bool(blur_faces)
        self.censor_text = bool(censor_text)
        self._calls = 0
        self.last_boxes: dict[str, list[tuple[int, int, int, int]]] = {
            "faces": [],
            "text": [],
        }
        self._cascades = None
        self._yunet = None
        self._yunet_tried = False
        self.faces_backend = "none"
        self._db = None
        self._db_tried = False
        self.text_backend = "none"

    # -- detection ---------------------------------------------------------

    def _load_cascades(self, cv2):
        if self._cascades is None:
            names = (
                "haarcascade_frontalface_default.xml",
                "haarcascade_profileface.xml",
            )
            loaded = []
            for name in names:
                cascade = cv2.CascadeClassifier(cv2.data.haarcascades + name)
                if not cascade.empty():
                    loaded.append(cascade)
            self._cascades = loaded
        return self._cascades

    def _ensure_yunet(self, cv2):
        """Create the YuNet detector once; ``None`` if the model/API is missing."""
        if self._yunet is None and not self._yunet_tried:
            self._yunet_tried = True
            try:
                if hasattr(cv2, "FaceDetectorYN") and _YUNET_PATH.exists():
                    # score 0.6 keeps profile/small faces the 0.9 default drops;
                    # nms 0.3, generous top_k -- a dating grid can be crowded.
                    self._yunet = cv2.FaceDetectorYN.create(
                        str(_YUNET_PATH), "", (320, 320), 0.6, 0.3, 5000
                    )
            except Exception:
                self._yunet = None
        return self._yunet

    def _detect_faces(self, cv2, rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
        detector = self._ensure_yunet(cv2)
        if detector is not None:
            try:
                boxes = self._detect_faces_yunet(cv2, detector, rgb)
                self.faces_backend = "yunet"
                return boxes
            except Exception:
                # A YuNet hiccup on one frame drops to Haar rather than nothing.
                pass
        self.faces_backend = "haar"
        return self._detect_faces_haar(cv2, rgb)

    def _detect_faces_yunet(
        self, cv2, detector, rgb: np.ndarray
    ) -> list[tuple[int, int, int, int]]:
        h, w = rgb.shape[:2]
        detector.setInputSize((w, h))
        # YuNet was trained on BGR (OpenCV convention); our frames are RGB.
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        _n, faces = detector.detect(bgr)
        boxes: list[tuple[int, int, int, int]] = []
        if faces is not None:
            for face in faces:
                x, y, bw, bh = (float(face[0]), float(face[1]),
                                float(face[2]), float(face[3]))
                # Grow ~30% wide and a bit more up/down for hair and chin.
                boxes.append(
                    _clip(
                        int(x - 0.15 * bw), int(y - 0.25 * bh),
                        int(bw * 1.30), int(bh * 1.45), w, h,
                    )
                )
        return [b for b in boxes if b[2] > 1 and b[3] > 1]

    def _detect_faces_haar(self, cv2, rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        scale = 1.0
        if w > self._FACE_WORK_WIDTH:
            scale = self._FACE_WORK_WIDTH / float(w)
            gray = cv2.resize(
                gray,
                (self._FACE_WORK_WIDTH, max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        gray = cv2.equalizeHist(gray)
        boxes: list[tuple[int, int, int, int]] = []
        for cascade in self._load_cascades(cv2):
            found = cascade.detectMultiScale(
                gray, scaleFactor=1.15, minNeighbors=4, minSize=(14, 14)
            )
            for x, y, bw, bh in found:
                fx, fy = x / scale, y / scale
                fw, fh = bw / scale, bh / scale
                gx, gy = fw * 0.35 / 2.0, fh * 0.35 / 2.0
                boxes.append(
                    _clip(
                        int(fx - gx), int(fy - gy), int(fw + 2 * gx), int(fh + 2 * gy), w, h
                    )
                )
        return [b for b in boxes if b[2] > 1 and b[3] > 1]

    def _ensure_db_text(self, cv2):
        """Create the PP-OCRv3 DB text detector once; ``None`` if model/API missing."""
        if self._db is None and not self._db_tried:
            self._db_tried = True
            try:
                dnn = getattr(cv2, "dnn", None)
                if dnn is not None and hasattr(dnn, "TextDetectionModel_DB") and _DB_TEXT_PATH.exists():
                    model = dnn.TextDetectionModel_DB(str(_DB_TEXT_PATH))
                    model.setBinaryThreshold(0.2)
                    model.setPolygonThreshold(0.4)
                    model.setMaxCandidates(200)
                    model.setUnclipRatio(2.2)
                    # Input a bit larger than the 360x640 frame so small captions
                    # survive; DB maps boxes back to the original coordinates.
                    model.setInputParams(
                        1.0 / 255.0, (384, 640), (122.67891434, 116.66876762, 104.00698793)
                    )
                    self._db = model
            except Exception:
                self._db = None
        return self._db

    def _detect_text(self, cv2, rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
        # A real DB text-detection model (reliable, no false positives) is the
        # primary; the model-free heuristic is unioned in so soft/low-contrast
        # text the model misses is still covered. Privacy favors recall.
        boxes: list[tuple[int, int, int, int]] = []
        model = self._ensure_db_text(cv2)
        if model is not None:
            try:
                boxes = self._detect_text_db(cv2, model, rgb)
                self.text_backend = "ppocr_db+heuristic"
            except Exception:
                self.text_backend = "heuristic"
        else:
            self.text_backend = "heuristic"
        boxes = boxes + self._detect_text_heuristic(cv2, rgb)
        boxes = _merge_overlapping(boxes)
        boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        return boxes[: self.MAX_TEXT_REGIONS]

    def _detect_text_db(self, cv2, model, rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = rgb.shape[:2]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        result = model.detect(bgr)
        quads = result[0] if result else None
        boxes: list[tuple[int, int, int, int]] = []
        if quads is not None:
            for quad in quads:
                pts = np.asarray(quad, dtype=np.int32).reshape(-1, 2)
                x, y, bw, bh = cv2.boundingRect(pts)
                # DB boxes are tight; grow a touch so descenders/edges are covered.
                gx, gy = int(bw * 0.06) + 2, int(bh * 0.12) + 2
                boxes.append(_clip(x - gx, y - gy, bw + 2 * gx, bh + 2 * gy, w, h))
        return [b for b in boxes if b[2] > 1 and b[3] > 1]

    def _detect_text_heuristic(self, cv2, rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        # Background-normalized, polarity-independent stroke response. Top-hat
        # keeps bright strokes (white captions over dark video), black-hat keeps
        # dark strokes (dark text on light), and cv2.max unions the two. This is
        # what makes WHITE text on a busy frame detectable: a plain Sobel + Otsu
        # drowns in the background's own edges and finds nothing.
        cell = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        resp = cv2.max(
            cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cell),
            cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cell),
        )
        _, binary = cv2.threshold(resp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Join characters into text lines, then remove isolated speckle.
        line = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, w // 18), 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, line)
        closed = cv2.morphologyEx(
            closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        )
        n, _labels, stats, _cent = cv2.connectedComponentsWithStats(closed, 8)
        # The shape thresholds are quoted at the 360x640 retina frame; scale
        # them so larger source frames behave the same way.
        s = max(h / 640.0, 0.35)
        min_h, max_h = 8.0 * s, 64.0 * s
        boxes: list[tuple[int, int, int, int]] = []
        for i in range(1, n):
            x, y, bw, bh, area = (
                stats[i, cv2.CC_STAT_LEFT],
                stats[i, cv2.CC_STAT_TOP],
                stats[i, cv2.CC_STAT_WIDTH],
                stats[i, cv2.CC_STAT_HEIGHT],
                stats[i, cv2.CC_STAT_AREA],
            )
            if not (min_h <= bh <= max_h):
                continue
            if bw < 2.0 * bh:
                continue
            if area / float(max(bw * bh, 1)) <= 0.30:
                continue
            boxes.append(_clip(int(x), int(y), int(bw), int(bh), w, h))
        return boxes

    # -- application -------------------------------------------------------

    def apply(self, rgb: np.ndarray) -> tuple[np.ndarray, dict]:
        """Return ``(censored_copy, info)``; ``rgb`` is never modified."""
        if not self.enabled:
            return rgb, {"faces": 0, "text_regions": 0, "enabled": False}
        cv2 = _cv2()
        if cv2 is None:
            return rgb, {
                "faces": 0,
                "text_regions": 0,
                "enabled": False,
                "error": "opencv missing",
            }

        frame = np.ascontiguousarray(rgb)
        redetect = self._calls % self.DETECT_EVERY == 0
        self._calls += 1
        if redetect:
            try:
                faces = self._detect_faces(cv2, frame) if self.blur_faces else []
                text = self._detect_text(cv2, frame) if self.censor_text else []
            except Exception as exc:  # a detector hiccup must not stop the fly
                return rgb.copy(), {
                    "faces": 0,
                    "text_regions": 0,
                    "enabled": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            self.last_boxes = {"faces": faces, "text": text}

        out = frame.copy()
        faces = self.last_boxes.get("faces", []) if self.blur_faces else []
        text = self.last_boxes.get("text", []) if self.censor_text else []
        # Faces: a mosaic of ~9 px cells (between the old 12 and the too-fine 7)
        # so the face reads as blurred but isn't obvious. Text: the same mosaic
        # treatment (it used to be a solid black bar), sized to the line height so
        # captions are gone.
        for x, y, bw, bh in faces:
            _pixelate(cv2, out, x, y, bw, bh, min(14, max(9, round(min(bw, bh) / 11))))
        for x, y, bw, bh in text:
            _pixelate(cv2, out, x, y, bw, bh, min(16, max(6, round(bh / 2.2))))
        return out, {
            "faces": len(faces),
            "text_regions": len(text),
            "enabled": True,
            "faces_backend": self.faces_backend,
            "text_backend": self.text_backend,
        }


def _clip(x: int, y: int, w: int, h: int, W: int, H: int) -> tuple[int, int, int, int]:
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def _merge_overlapping(
    boxes: list[tuple[int, int, int, int]]
) -> list[tuple[int, int, int, int]]:
    """Union boxes that overlap, repeating until the set is stable."""
    merged = list(boxes)
    changed = True
    while changed and len(merged) > 1:
        changed = False
        out: list[tuple[int, int, int, int]] = []
        for box in merged:
            x, y, w, h = box
            for i, (ox, oy, ow, oh) in enumerate(out):
                if x < ox + ow and ox < x + w and y < oy + oh and oy < y + h:
                    nx, ny = min(x, ox), min(y, oy)
                    out[i] = (
                        nx,
                        ny,
                        max(x + w, ox + ow) - nx,
                        max(y + h, oy + oh) - ny,
                    )
                    changed = True
                    break
            else:
                out.append(box)
        merged = out
    return merged


def _pixelate(cv2, img: np.ndarray, x: int, y: int, w: int, h: int, block: int):
    """Mosaic the region into ~``block``-px cells: a coarse, obviously-censored blur."""
    if w < 2 or h < 2:
        return
    block = max(2, int(block))
    sw, sh = max(1, w // block), max(1, h // block)
    patch = img[y : y + h, x : x + w]
    small = cv2.resize(patch, (sw, sh), interpolation=cv2.INTER_LINEAR)
    img[y : y + h, x : x + w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
