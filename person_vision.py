"""External detector plugin seam and anonymous person-crop gate.

A detector implementation is installed and configured separately. Confirmed
track IDs and image crops are kept only in memory; identities are not persisted.
"""

from __future__ import annotations

import importlib
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol

LOGGER = logging.getLogger("headroom.person_vision")
DETECTION_INTERVAL_SECONDS = 1.0
TRACK_CONFIRMATION_SAMPLES = 2
MIN_PERSON_CROP_WIDTH = 40
MIN_PERSON_CROP_HEIGHT = 80
MAX_TRACK_IDS = 1024
MAX_PENDING_CROPS = 12
MAX_CROP_AGE_SECONDS = 15.0
PERSON_PRESENCE_GRACE_SECONDS = 5.0
RECURRENT_CROP_INTERVAL_SECONDS = 10.0
MAX_CROPS_PER_TURN = 4
PERSON_DETECTOR_ENV = "HEADROOM_PERSON_DETECTOR"


@dataclass(frozen=True)
class PersonDetection:
    """A person box in frame-pixel xyxy coordinates with a stable track ID."""

    track_id: int
    box: tuple[float, float, float, float]


class PersonDetector(Protocol):
    """Interface implemented by an optional, separately installed detector."""

    def detect(self, frame: Any) -> Iterable[PersonDetection]:
        """Return current person detections; keep IDs stable across nearby frames."""
        ...


def crop_person(frame: Any, xyxy: Any) -> Any | None:
    """Copy a padded, bounded person box from a camera frame, or reject it."""
    if frame is None or getattr(frame, "ndim", 0) < 2:
        return None
    try:
        coordinates = [float(value) for value in xyxy]
    except (TypeError, ValueError, OverflowError):
        return None
    if len(coordinates) != 4 or not all(math.isfinite(value) for value in coordinates):
        return None

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = coordinates
    x1, x2 = max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))
    y1, y2 = max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))
    box_width, box_height = x2 - x1, y2 - y1
    if box_width < MIN_PERSON_CROP_WIDTH or box_height < MIN_PERSON_CROP_HEIGHT:
        return None

    # Leave enough context around clothing and props to make a costume legible.
    pad_x = max(8, round(box_width * 0.10))
    pad_y = max(10, round(box_height * 0.08))
    left, top = max(0, math.floor(x1 - pad_x)), max(0, math.floor(y1 - pad_y))
    right = min(width, math.ceil(x2 + pad_x))
    bottom = min(height, math.ceil(y2 + pad_y))
    if right <= left or bottom <= top:
        return None
    crop = frame[top:bottom, left:right].copy()
    return crop if getattr(crop, "size", 0) else None


class PersonTrackGate:
    """Confirm anonymous tracks and refresh their crops at a bounded interval."""

    def __init__(self) -> None:
        self._confirmed_ids: set[int] = set()
        self._candidate_samples: dict[int, int] = {}
        self._last_crop_at: dict[int, float] = {}
        self._budget_warned = False

    def reset_candidates(self) -> None:
        self._candidate_samples.clear()

    def confirm(
        self,
        frame: Any,
        track_ids: list[int],
        boxes: list[Any],
        on_confirmed: Callable[[], None] | None = None,
    ) -> list[tuple[int, Any]]:
        current_ids = set(track_ids)
        self._candidate_samples = {
            track_id: samples
            for track_id, samples in self._candidate_samples.items()
            if track_id in current_ids and track_id not in self._confirmed_ids
        }
        if len(self._confirmed_ids) >= MAX_TRACK_IDS and not self._budget_warned:
            LOGGER.warning("anonymous track limit reached; only already-confirmed visitors will be refreshed")
            self._budget_warned = True

        crops = []
        now = time.monotonic()
        for index, track_id in enumerate(track_ids):
            if index >= len(boxes):
                continue
            if track_id in self._confirmed_ids:
                self._candidate_samples.pop(track_id, None)
                if now - self._last_crop_at.get(track_id, 0.0) < RECURRENT_CROP_INTERVAL_SECONDS:
                    continue
                crop = crop_person(frame, boxes[index])
                if crop is not None:
                    self._last_crop_at[track_id] = now
                    crops.append((track_id, crop))
                continue
            if len(self._confirmed_ids) >= MAX_TRACK_IDS:
                continue
            samples = self._candidate_samples.get(track_id, 0) + 1
            self._candidate_samples[track_id] = samples
            if samples < TRACK_CONFIRMATION_SAMPLES:
                continue
            self._confirmed_ids.add(track_id)
            self._candidate_samples.pop(track_id, None)
            if on_confirmed is not None:
                try:
                    on_confirmed()
                except Exception:
                    LOGGER.exception("person-presence wake callback failed")
            crop = crop_person(frame, boxes[index])
            if crop is not None:
                self._last_crop_at[track_id] = now
                crops.append((track_id, crop))
        return crops


class LocalPersonVision:
    """Sample frames locally and queue fresh crops for confirmed visitor tracks."""

    def __init__(
        self,
        frame_provider: Callable[[], Any | None],
        on_person_seen: Callable[[], None] | None = None,
    ) -> None:
        self._frame_provider = frame_provider
        self._on_person_seen = on_person_seen
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._detector: PersonDetector | None = None
        self._gate = PersonTrackGate()
        self._pending: dict[int, tuple[Any, float]] = {}
        self._status = "Waiting for external detector plugin"
        self._last_person_count: int | None = None
        self._last_person_seen_at: float | None = None
        self._last_error_log = 0.0

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def pending_count(self) -> int:
        with self._lock:
            self._expire_pending_locked(time.monotonic())
            return len(self._pending)

    @property
    def has_recent_person(self) -> bool:
        with self._lock:
            return (
                self._last_person_seen_at is not None
                and time.monotonic() - self._last_person_seen_at <= PERSON_PRESENCE_GRACE_SECONDS
            )

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._status = "Loading external person detector…"
            self._thread = threading.Thread(
                target=self._run,
                name="headroom-person-vision",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5)
            if thread.is_alive():
                LOGGER.warning("local person detector is still stopping")
        with self._lock:
            self._pending.clear()
            self._status = "Stopped"

    def take_new_crops(self, limit: int = MAX_CROPS_PER_TURN) -> list[Any]:
        now = time.monotonic()
        with self._lock:
            self._expire_pending_locked(now)
            selected = sorted(self._pending.items(), key=lambda item: item[1][1])[:max(0, limit)]
            for track_id, _ in selected:
                self._pending.pop(track_id, None)
            return [crop.copy() for _, (crop, _) in selected]

    def _expire_pending_locked(self, now: float) -> None:
        expired = [
            track_id for track_id, (_, captured_at) in self._pending.items()
            if now - captured_at > MAX_CROP_AGE_SECONDS
        ]
        for track_id in expired:
            self._pending.pop(track_id, None)
        if expired:
            LOGGER.info("expired %d unconsumed anonymous person crop(s)", len(expired))

    def _load_detector(self) -> PersonDetector:
        """Load ``module:factory`` configured by the user; no detector is bundled."""
        configured_plugin = os.getenv(PERSON_DETECTOR_ENV, "").strip()
        if not configured_plugin:
            raise RuntimeError(
                f"no detector plugin configured; set {PERSON_DETECTOR_ENV}=module:factory"
            )
        module_name, separator, factory_name = configured_plugin.partition(":")
        if not separator or not module_name.strip() or not factory_name.strip():
            raise ValueError(f"{PERSON_DETECTOR_ENV} must use the form module:factory")

        module = importlib.import_module(module_name.strip())
        factory = getattr(module, factory_name.strip(), None)
        if not callable(factory):
            raise TypeError(f"detector factory {configured_plugin!r} is not callable")
        detector = factory()
        if not callable(getattr(detector, "detect", None)):
            raise TypeError("detector factory must return an object with detect(frame)")
        return detector

    def _detect(self, frame: Any) -> tuple[int, list[tuple[int, Any]]]:
        if self._detector is None:
            self._gate.reset_candidates()
            return 0, []
        raw_detections = self._detector.detect(frame)
        if raw_detections is None:
            raw_detections = ()
        track_ids: list[int] = []
        boxes: list[Any] = []
        for detection in raw_detections:
            if not isinstance(detection, PersonDetection):
                LOGGER.warning("external person detector returned an unsupported detection; expected PersonDetection")
                continue
            try:
                track_id = int(detection.track_id)
                box = tuple(float(value) for value in detection.box)
            except (TypeError, ValueError, OverflowError):
                continue
            if len(box) != 4 or not all(math.isfinite(value) for value in box):
                continue
            track_ids.append(track_id)
            boxes.append(box)
        if not track_ids:
            self._gate.reset_candidates()
            return 0, []
        return len(track_ids), self._gate.confirm(
            frame,
            track_ids,
            boxes,
            on_confirmed=self._on_person_seen,
        )

    def _queue_crops(self, crops: list[tuple[int, Any]]) -> None:
        now = time.monotonic()
        queued = 0
        refreshed = 0
        with self._lock:
            self._expire_pending_locked(now)
            for track_id, crop in crops:
                if track_id in self._pending:
                    refreshed += 1
                else:
                    queued += 1
                self._pending[track_id] = (crop.copy(), now)
                while len(self._pending) > MAX_PENDING_CROPS:
                    oldest = min(self._pending, key=lambda key: self._pending[key][1])
                    self._pending.pop(oldest, None)
                    LOGGER.warning("pending person-crop queue full; dropped its oldest crop")
        if queued:
            LOGGER.info("local detector queued %d newly confirmed or refreshed person crop(s)", queued)
        if refreshed:
            LOGGER.debug("refreshed %d unconsumed person crop(s) with recent frames", refreshed)

    def _run(self) -> None:
        if os.getenv("HEADROOM_LOCAL_PERSON_DETECTION", "1").strip().lower() in {"0", "false", "no"}:
            with self._lock:
                self._status = "Disabled by HEADROOM_LOCAL_PERSON_DETECTION"
            LOGGER.info("local person detection disabled by environment")
            return
        try:
            self._detector = self._load_detector()
        except Exception as error:
            configured_plugin = os.getenv(PERSON_DETECTOR_ENV, "").strip()
            if not configured_plugin:
                status = "No detector plugin · motion wake only"
                LOGGER.info("no external person detector configured; motion wake remains available")
            else:
                status = f"Unavailable · {type(error).__name__} loading external detector"
                LOGGER.warning(
                    "external person detector unavailable (%s); visitor replies remain text-only",
                    type(error).__name__,
                )
            with self._lock:
                self._status = status
            return
        if self._stop_event.is_set():
            return
        with self._lock:
            self._status = "Ready · external local detector"
        LOGGER.info("external local person detector ready; sampling at %.1f Hz", 1 / DETECTION_INTERVAL_SECONDS)

        while not self._stop_event.is_set():
            sample_started = time.monotonic()
            frame = self._frame_provider()
            if frame is not None:
                try:
                    person_count, crops = self._detect(frame)
                    with self._lock:
                        self._status = f"Ready · {person_count} person box(es), local CPU"
                        if person_count > 0:
                            self._last_person_seen_at = time.monotonic()
                    if person_count != self._last_person_count:
                        LOGGER.info("local detector estimates %d visible person box(es)", person_count)
                        self._last_person_count = person_count
                    if crops:
                        self._queue_crops(crops)
                except Exception as error:
                    self._gate.reset_candidates()
                    with self._lock:
                        self._status = f"Detection error · {type(error).__name__}"
                    now = time.monotonic()
                    if now - self._last_error_log >= 30:
                        LOGGER.warning("local person detection failed (%s)", type(error).__name__)
                        self._last_error_log = now
            remaining = DETECTION_INTERVAL_SECONDS - (time.monotonic() - sample_started)
            self._stop_event.wait(max(0.0, remaining))
