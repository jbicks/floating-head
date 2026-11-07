"""WSL-side visitor capture and conversation loop for the Pepper's Ghost display.

Raw camera frames and microphone audio stay in memory. Visual-context JPEGs and
recognized visitor transcripts are saved locally for diagnostics; raw audio is
not saved.
"""

from __future__ import annotations

import base64
import contextlib
import datetime
import glob
import json
import logging
import os
import queue
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from person_vision import LocalPersonVision, MAX_CROPS_PER_TURN

try:
    import cv2  # type: ignore
except ImportError:  # The speech-only demo remains usable without camera extras.
    cv2 = None

try:
    import speech_recognition as sr  # type: ignore
except ImportError:
    sr = None


LOGGER = logging.getLogger("headroom.conversation")
PROJECT_ROOT = Path(__file__).resolve().parent
VISION_FRAME_DIRECTORY = PROJECT_ROOT / "logs" / "vision_frames"
SAVE_VISION_FRAMES = os.getenv("HEADROOM_SAVE_VISION_FRAMES", "1").strip().lower() not in {"0", "false", "no"}
TRANSCRIPT_LOG_PATH = PROJECT_ROOT / "logs" / "visitor_transcripts.jsonl"
SAVE_VISITOR_TRANSCRIPTS = os.getenv("HEADROOM_SAVE_TRANSCRIPTS", "1").strip().lower() not in {"0", "false", "no"}
MAX_TRANSCRIPT_LOG_BYTES = 3 * 1024 * 1024
TRANSCRIPT_LOG_BACKUPS = 3
_TRANSCRIPT_LOG_LOCK = threading.Lock()
MAX_SAVED_VISION_FRAMES = 100
VISION_MODEL = os.getenv("HEADROOM_VISION_MODEL", "gpt-4o-mini")
TRANSCRIPTION_MODEL = os.getenv("HEADROOM_TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe")
MAX_REPLY_TOKENS = 64
MAX_INTERACTION_SECONDS = 300
MAX_SILENT_LISTENS = 2
SILENCE_RECOVERY_REPLY = "Oops, my ghost ears got a little fuzzy. Could you say that once more?"
GREETING_GENERATION_LISTEN_POLL_SECONDS = 0.25
LISTEN_TIMEOUT_SECONDS = 8
PHRASE_LIMIT_SECONDS = 12
LISTEN_PAUSE_THRESHOLD_SECONDS = 0.9
MINIMUM_SPEECH_DURATION_SECONDS = 0.18
CAPTURE_POLL_TIMEOUT_SECONDS = 0.2
AMBIENT_RECALIBRATION_INTERVAL_SECONDS = 5 * 60
AMBIENT_QUIET_WINDOW_SECONDS = 10
AMBIENT_CALIBRATION_DURATION_SECONDS = 0.45
MOTION_THRESHOLD_PIXELS = 420
BUILTIN_CAMERA_HINTS = (
    "integrated", "internal", "built-in", "built in", "builtin", "laptop",
    "front camera", "user-facing", "user facing", "webcam", "wide vision",
)
BUILTIN_MIC_HINTS = (
    "built-in microphone", "built in microphone", "internal microphone",
    "internal mic", "integrated microphone", "integrated mic",
    "microphone array", "mic array", "laptop microphone",
)
VIRTUAL_CAMERA_HINTS = ("virtual", "loopback", "obs", "dummy", "metadata")

SYSTEM_INSTRUCTIONS = (
    "You are Bobby, a cheerful, spooky-but-gentle talking Halloween head at a "
    "family event. For this setup, you are displayed by a home's front door on Halloween. "
    "Children and families walk up the path to trick-or-treat; they may be chatting excitedly "
    "or calling out as they approach. A candy bowl is set out by the display for visitors; "
    "frame the candy as a cheerful treat for visitors: warmly invite each visitor to enjoy a "
    "couple of pieces and leave some for the next children. Keep this generous and friendly, "
    "not scolding or focused on a limit. You cannot see or count the bowl, so never claim to "
    "be watching it or know what remains. "
    "The hosts may be away from the door at times; never reveal their whereabouts or say "
    "the home is unattended. You are the friendly display they meet. Your backstory is local "
    "to the Calgary area: you lived here "
    "about a century ago, when Calgary was much smaller than it is today, and "
    "remember being one of a few spooky neighbors. If asked your name, say Bobby. "
    "You may introduce yourself as Bobby in a greeting, but do not bring up your "
    "origin, age, or old Calgary stories unless directly asked where you are from, "
    "about your past, or the visitor clearly invites that story. If asked where "
    "you are from, say the Calgary area and briefly mention that you lived here "
    "about a century ago, when the city was much smaller. Never imply the area "
    "was uninhabited or had few human residents: Indigenous peoples have lived "
    "here since time immemorial. Do not invent or speak for local Indigenous "
    "histories or traditions. Use natural Canadian English, with occasional "
    "Western Canadian phrasing but no caricature. Be especially welcoming to children "
    "without talking down to them: sound genuinely cheerful, encouraging, and delighted "
    "to meet them. In a greeting, add one brief, fresh bit of ghostly whimsy, a silly "
    "observation, or a gentle pun when it fits. In later replies, sprinkle in original "
    "light-hearted sayings now and then, not in every line; vary them rather than leaning "
    "on a repeated catchphrase. A friendly boo-tiful compliment or a joke about floaty "
    "ghost hands are examples for inspiration, not lines to repeat. If someone seems "
    "sad, worried, or scared, respond kindly and reassure them before trying to cheer "
    "them up. Never tease, mock, startle, pressure, or embarrass a visitor. Keep humor "
    "simple, kind, and family-safe: no violence, gore, threats, profanity, or frightening "
    "personal questions. Prioritize the visitor's latest words: answer a direct question "
    "or respond to their statement before adding whimsy or a follow-up. Never replace "
    "an answer with another question or a costume comment. If a short answer is "
    "ambiguous, gently clarify instead of guessing; if asked about yourself, answer "
    "briefly as Bobby. Prefer one short sentence of no more than 16 words. Use a second "
    "brief sentence only when it is important to answer a direct question; keep the total "
    "under 22 words. Keep language "
    "easy to follow, and ask at most one short, playful follow-up, such as about a "
    "costume or favorite spooky thing; never ask for personal information. Stay in "
    "character, but never claim to be a real person. "
    "Each attached image, if any, is a recent crop of one anonymously tracked "
    "visitor; it may be a refreshed view of someone seen earlier, not a full scene. "
    "When a costume, clothing item, or prop is clearly "
    "visible, normally make one brief, warm, playful comment about that detail; "
    "use only broad costume descriptions when well supported. The pictured people "
    "may not be the speakers, so address the group naturally rather than identifying "
    "or attributing speech to anyone. Never identify people, infer private or "
    "sensitive traits, or guess when unsure. If no crop is attached or the outfit "
    "is unclear and someone asks what they are wearing, say you cannot quite make "
    "it out and invite them to tell you about it. Never mention cameras, "
    "microphones, AI, or technical image input. Never ask for personal information."
)


def choose_microphone_index(devices: list[dict[str, Any]], default_index: int | None) -> int | None:
    """Prefer a Logitech BRIO input, then a built-in input, then the system default."""
    inputs = [item for item in devices if int(item.get("maxInputChannels", 0) or 0) > 0]
    for matcher in (("logitech", "brio"), *tuple((hint,) for hint in BUILTIN_MIC_HINTS)):
        for item in inputs:
            name = str(item.get("name", "")).lower()
            if all(part in name for part in matcher):
                return int(item["index"])
    if default_index is not None and any(int(item["index"]) == default_index for item in inputs):
        return default_index
    return None


@contextlib.contextmanager
def _quiet_native_audio_stderr():
    """Hide ALSA/JACK probing noise from PortAudio in WSL when PulseAudio works."""
    saved_fd = null_fd = None
    try:
        saved_fd = os.dup(2)
        null_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null_fd, 2)
        yield
    finally:
        if saved_fd is not None:
            os.dup2(saved_fd, 2)
            os.close(saved_fd)
        if null_fd is not None:
            os.close(null_fd)


def available_microphones() -> tuple[list[dict[str, Any]], int | None]:
    if sr is None:
        return [], None
    with _quiet_native_audio_stderr():
        audio = sr.Microphone.get_pyaudio().PyAudio()
        try:
            try:
                default_index = int(audio.get_default_input_device_info()["index"])
            except Exception:
                default_index = None
            devices = []
            for index in range(audio.get_device_count()):
                info = audio.get_device_info_by_index(index)
                if int(info.get("maxInputChannels", 0) or 0) > 0:
                    devices.append({**info, "index": index})
            return devices, default_index
        finally:
            audio.terminate()


def response_text(payload: dict[str, Any]) -> str:
    """Extract plain assistant text from a Responses API result."""
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    parts: list[str] = []
    for item in payload.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text":
                text = content.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
    return " ".join(parts).strip()


def camera_candidates() -> list[tuple[str, str]]:
    """Return existing V4L2 nodes in Logitech, built-in, then fallback order."""
    entries: dict[str, str] = {}
    for path in glob.glob("/dev/v4l/by-id/*"):
        if "video-index0" not in path:
            continue
        try:
            resolved = str(Path(path).resolve(strict=True))
        except OSError:
            continue
        entries[resolved] = Path(path).name
    for path in sorted(glob.glob("/dev/video*")):
        name_path = Path("/sys/class/video4linux") / Path(path).name / "name"
        try:
            name = name_path.read_text(encoding="utf-8").strip()
        except OSError:
            name = Path(path).name
        entries.setdefault(path, name)

    def rank(entry: tuple[str, str]) -> tuple[int, str]:
        path, name = entry
        # usb:// aliases use underscores; normalize them so built-in names rank
        # ahead of alternate low-resolution V4L2 nodes for the same camera.
        label = f"{path} {name}".lower().replace("_", " ").replace("-", " ")
        if any(part in label for part in VIRTUAL_CAMERA_HINTS):
            score = 3
        elif "logitech" in label or "brio" in label:
            score = 0
        elif any(part in label for part in BUILTIN_CAMERA_HINTS):
            score = 1
        else:
            score = 2
        return score, path

    return [entry for entry in sorted(entries.items(), key=rank) if rank(entry)[0] < 3]


def save_visitor_transcript(text: str, turn: int) -> None:
    """Keep recognized visitor text in a private, bounded local JSONL log."""
    if not SAVE_VISITOR_TRANSCRIPTS or not text:
        return
    path = TRANSCRIPT_LOG_PATH
    line = json.dumps(
        {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds"),
            "turn": int(turn),
            "text": text,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ) + "\n"
    encoded_line = line.encode("utf-8")
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(path.parent, 0o700)
        with _TRANSCRIPT_LOG_LOCK:
            try:
                current_size = path.stat().st_size
            except FileNotFoundError:
                current_size = 0
            if current_size + len(encoded_line) > MAX_TRANSCRIPT_LOG_BYTES:
                for index in range(TRANSCRIPT_LOG_BACKUPS, 0, -1):
                    source = path if index == 1 else path.with_name(f"{path.name}.{index - 1}")
                    destination = path.with_name(f"{path.name}.{index}")
                    if destination.exists():
                        destination.unlink()
                    if source.exists():
                        os.replace(source, destination)
                        os.chmod(destination, 0o600)
            if path.exists():
                os.chmod(path, 0o600)
            descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            with os.fdopen(descriptor, "a", encoding="utf-8") as transcript_file:
                transcript_file.write(line)
    except OSError as error:
        LOGGER.warning("could not save visitor transcript (%s)", type(error).__name__)


class ConversationManager:
    """Owns WSL capture, motion/person-presence wake, and transient dialogue."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._frame_lock = threading.Lock()
        self._active = False
        self._mode = "stopped"
        self._message = "Visitor mode is off."
        self._camera_status = "Not started"
        self._microphone_status = "Not started"
        self._last_heard = ""
        self._motion_count = 0
        self._last_motion_at: float | None = None
        self._manual_greeting_count = 0
        self._heard_count = 0
        self._last_heard_at: float | None = None
        self._saved_vision_frame_count = 0
        self._last_vision_frame_name: str | None = None
        self._snapshot_sequence = 0
        self._pending_reply: dict[str, Any] | None = None
        self._playback_claimed_reply_id: int | None = None
        self._reply_counter = 0
        self._history: list[dict[str, str]] = []
        self._stop_event = threading.Event()
        self._motion_event = threading.Event()
        self._playback_started = threading.Event()
        self._playback_finished = threading.Event()
        self._camera_thread: threading.Thread | None = None
        self._worker_thread: threading.Thread | None = None
        self._recognizer: Any | None = None
        self._recognizer_calibrated = False
        self._last_recognizer_calibration_at: float | None = None
        self._capture: Any = None
        self._latest_frame: Any = None
        self._person_vision: LocalPersonVision | None = None
        self._camera_label = ""
        self._microphone_index: int | None = None
        self._triggered_at = 0.0

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active": self._active,
                "mode": self._mode,
                "message": self._message,
                "camera": self._camera_status,
                "microphone": self._microphone_status,
                "personDetection": self._person_vision.status if self._person_vision else "No camera detector",
                "pendingPersonCrops": self._person_vision.pending_count if self._person_vision else 0,
                "motionCount": self._motion_count,
                "lastMotionAt": self._last_motion_at,
                "manualGreetingCount": self._manual_greeting_count,
                "heardCount": self._heard_count,
                "lastHeardAt": self._last_heard_at,
                "savedVisionFrameCount": self._saved_vision_frame_count,
                "lastVisionFrame": self._last_vision_frame_name,
                "reply": dict(self._pending_reply) if self._pending_reply else None,
            }

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self._active:
                return self.status()
            if cv2 is None or sr is None:
                missing = []
                if cv2 is None:
                    missing.append("OpenCV")
                if sr is None:
                    missing.append("SpeechRecognition/PyAudio")
                LOGGER.error("cannot enable visitor mode; missing %s", ", ".join(missing))
                return {
                    **self.status(),
                    "error": (
                        f"Visitor capture needs {', '.join(missing)}. "
                        "Run ./scripts/install_conversation_deps.sh, then restart Headroom."
                    ),
                }

            self._stop_event.clear()
            self._motion_event.clear()
            self._playback_started.clear()
            self._playback_finished.clear()
            self._latest_frame = None
            self._pending_reply = None
            self._playback_claimed_reply_id = None
            self._history = []
            self._last_heard = ""
            self._motion_count = 0
            self._last_motion_at = None
            self._manual_greeting_count = 0
            self._heard_count = 0
            self._last_heard_at = None
            self._saved_vision_frame_count = 0
            self._last_vision_frame_name = None
            self._triggered_at = 0.0
            self._active = True
            self._mode = "watching"
            self._message = "Visitor mode is on. Waiting for a visitor to be seen."
            self._open_camera()
            self._select_microphone()
            self._prepare_recognizer()
            self._camera_thread = threading.Thread(target=self._camera_loop, name="headroom-camera", daemon=True)
            self._worker_thread = threading.Thread(target=self._conversation_loop, name="headroom-conversation", daemon=True)
            self._camera_thread.start()
            if self._capture is not None:
                self._person_vision = LocalPersonVision(
                    self._latest_frame_copy,
                    on_person_seen=self._person_presence_detected,
                )
                self._person_vision.start()
            else:
                self._person_vision = None
            self._worker_thread.start()
            LOGGER.info("visitor mode enabled; camera=%s; microphone=%s", self._camera_status, self._microphone_status)
            return self.status()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if not self._active:
                return self.status()
            self._active = False
            self._mode = "stopped"
            self._message = "Visitor mode is off."
            self._pending_reply = None
            self._playback_claimed_reply_id = None
            self._stop_event.set()
            self._motion_event.set()
            self._playback_started.set()
            self._playback_finished.set()
            camera_thread = self._camera_thread
            worker_thread = self._worker_thread
            person_vision = self._person_vision
            self._person_vision = None
        LOGGER.info("visitor mode shutdown requested")
        if person_vision is not None:
            person_vision.stop()
        if worker_thread and worker_thread is not threading.current_thread():
            worker_thread.join(timeout=22)
        if camera_thread and camera_thread is not threading.current_thread():
            camera_thread.join(timeout=3)
        with self._lock:
            if self._capture is not None:
                self._capture.release()
                self._capture = None
            self._camera_thread = None
            self._worker_thread = None
            self._latest_frame = None
            LOGGER.info("visitor mode stopped; camera released")
            return self.status()

    def greet_now(self) -> dict[str, Any]:
        with self._lock:
            if not self._active:
                return {**self.status(), "error": "Enable visitor mode first."}
            self._triggered_at = time.monotonic()
            self._manual_greeting_count += 1
            self._motion_event.set()
            LOGGER.info("operator requested manual greeting #%d", self._manual_greeting_count)
            return self.status()

    def _person_presence_detected(self) -> None:
        """Wake an idle visitor session when local vision confirms a person."""
        with self._lock:
            now = time.monotonic()
            if (
                not self._active
                or self._mode != "watching"
                or self._motion_event.is_set()
                or now - self._triggered_at <= 7
            ):
                return
            self._triggered_at = now
            self._motion_count += 1
            self._last_motion_at = time.time()
            trigger_count = self._motion_count
            self._message = "A visitor is here. Preparing a greeting…"
            self._motion_event.set()
        LOGGER.info("confirmed local person presence triggered visitor interaction #%d", trigger_count)

    def claim_reply_playback(self, reply_id: int) -> bool:
        """Allow only the server speech dispatcher to start a queued reply."""
        with self._lock:
            pending = self._pending_reply
            if not self._active or pending is None or pending["id"] != reply_id:
                return False
            if self._playback_claimed_reply_id == reply_id:
                LOGGER.warning("duplicate speech playback request rejected for reply id=%d", reply_id)
                return False
            self._playback_claimed_reply_id = reply_id
            LOGGER.info("server speech dispatcher claimed reply id=%d", reply_id)
            return True

    def set_playback(self, active: bool, reply_id: int | None) -> dict[str, Any]:
        with self._lock:
            pending = self._pending_reply
            if pending is None or (reply_id is not None and pending["id"] != reply_id):
                return self.status()
            if active:
                if self._playback_started.is_set():
                    return self.status()
                LOGGER.info("reply audio started id=%s", pending["id"])
                self._playback_started.set()
                self._playback_finished.clear()
                self._mode = "speaking"
                self._message = "The visitor reply is playing on the laptop."
            else:
                if self._playback_finished.is_set():
                    return self.status()
                LOGGER.info("reply audio finished id=%s", pending["id"])
                self._playback_started.set()
                self._playback_finished.set()
            return self.status()

    def _open_camera(self) -> None:
        self._capture = None
        self._camera_label = ""
        for path, label in camera_candidates():
            LOGGER.info("trying camera candidate %s (%s)", path, label or "V4L2 device")
            capture = cv2.VideoCapture(path, cv2.CAP_V4L2)
            if not capture.isOpened():
                LOGGER.warning("camera candidate could not be opened: %s", path)
                capture.release()
                continue
            requested_fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            if not capture.set(cv2.CAP_PROP_FOURCC, requested_fourcc):
                LOGGER.warning("camera did not accept MJPG request: %s", path)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            capture.set(cv2.CAP_PROP_FPS, 10)
            actual_width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if actual_width < 640 or actual_height < 360:
                LOGGER.warning(
                    "camera candidate negotiated only %dx%d (wanted at least 640x360); trying next node: %s",
                    actual_width, actual_height, path,
                )
                capture.release()
                continue
            fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
            actual_fourcc = "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4)).strip() or "unknown"
            frame = None
            for _ in range(4):
                ok, candidate_frame = capture.read()
                if ok and candidate_frame is not None:
                    frame = candidate_frame
                time.sleep(0.03)
            if frame is None:
                LOGGER.warning("camera candidate opened but returned no warm-up frames: %s", path)
                capture.release()
                continue
            LOGGER.info(
                "camera negotiated %s %dx%d at %.1f fps: %s",
                actual_fourcc, actual_width, actual_height, capture.get(cv2.CAP_PROP_FPS), path,
            )
            self._capture = capture
            name_path = Path("/sys/class/video4linux") / Path(path).name / "name"
            try:
                self._camera_label = name_path.read_text(encoding="utf-8").strip() or label or path
            except OSError:
                self._camera_label = label or path
            self._latest_frame = frame
            self._camera_status = f"Ready · {self._camera_label} · {actual_width}×{actual_height} {actual_fourcc}"
            LOGGER.info("camera opened: %s", self._camera_status)
            return
        self._camera_status = "No WSL camera found; motion wake and visual context are unavailable."
        LOGGER.warning("%s", self._camera_status)

    def _select_microphone(self) -> None:
        self._microphone_index = None
        try:
            devices, default_index = available_microphones()
            chosen = choose_microphone_index(devices, default_index)
            if chosen is not None:
                self._microphone_index = chosen
                self._microphone_status = f"Ready · {next(item.get('name', 'selected input') for item in devices if item['index'] == chosen)}"
                LOGGER.info("microphone selected: %s", self._microphone_status)
            else:
                self._microphone_status = "No WSL microphone input found."
                LOGGER.warning("%s", self._microphone_status)
        except Exception as error:
            self._microphone_status = f"Microphone unavailable · {error}"
            LOGGER.exception("microphone enumeration failed")

    def _camera_loop(self) -> None:
        previous_small = None
        consecutive_motion = 0
        last_debug_sample = 0.0
        while not self._stop_event.is_set():
            capture = self._capture
            if capture is None:
                time.sleep(0.2)
                continue
            ok, frame = capture.read()
            if not ok or frame is None:
                time.sleep(0.1)
                continue
            with self._frame_lock:
                self._latest_frame = frame.copy()
            small = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if previous_small is not None:
                delta = cv2.absdiff(gray, previous_small)
                _, changed = cv2.threshold(delta, 24, 255, cv2.THRESH_BINARY)
                count = int(cv2.countNonZero(changed))
                with self._lock:
                    can_wake = self._active and self._mode == "watching" and time.monotonic() - self._triggered_at > 7
                if can_wake and count >= MOTION_THRESHOLD_PIXELS:
                    consecutive_motion += 1
                else:
                    consecutive_motion = 0
                if consecutive_motion >= 2:
                    with self._lock:
                        self._triggered_at = time.monotonic()
                        self._motion_count += 1
                        self._last_motion_at = time.time()
                        motion_count = self._motion_count
                    LOGGER.info("motion trigger #%d detected (changed pixels=%d)", motion_count, count)
                    self._motion_event.set()
                    consecutive_motion = 0
                if LOGGER.isEnabledFor(logging.DEBUG) and time.monotonic() - last_debug_sample >= 5.0:
                    with self._lock:
                        current_mode = self._mode
                    LOGGER.debug(
                        "camera frames healthy: %dx%d; changed_pixels=%d; mode=%s",
                        frame.shape[1], frame.shape[0], count, current_mode,
                    )
                    last_debug_sample = time.monotonic()
            previous_small = gray
            time.sleep(0.08)

    def _conversation_loop(self) -> None:
        while not self._stop_event.is_set():
            if not self._motion_event.wait(timeout=0.2):
                self._maybe_refresh_recognizer_calibration()
                continue
            self._motion_event.clear()
            if self._stop_event.is_set():
                break
            self._run_interaction()

    def _answer_transcribed_visitor(
        self,
        heard: str,
        image_data_urls: list[str] | None = None,
    ) -> bool:
        with self._lock:
            self._last_heard = heard
            self._heard_count += 1
            self._last_heard_at = time.time()
            heard_count = self._heard_count
            self._mode = "thinking"
            self._message = "Considering the visitor's reply…"
        LOGGER.info("visitor speech transcribed (turn=%d, characters=%d)", heard_count, len(heard))
        save_visitor_transcript(heard, heard_count)
        images = self._take_person_images() if image_data_urls is None else image_data_urls
        answer = self._ask_assistant(heard, images)
        if not self._queue_reply(answer):
            return False
        if not self._wait_for_playback():
            return False
        self._history.extend((
            {"role": "user", "content": heard},
            {"role": "assistant", "content": answer},
        ))
        return True

    def _run_interaction(self) -> None:
        LOGGER.info("visitor interaction started")
        self._history = []
        interaction_started_at = time.monotonic()
        missed = 0
        offered_speech_repair = False
        opening_listen_cancel: threading.Event | None = None
        opening_listen_thread: threading.Thread | None = None
        with self._lock:
            self._mode = "thinking"
            self._message = "A visitor is here. Preparing a welcome…"
            self._last_heard = ""
        try:
            LOGGER.info("preparing Bobby's greeting immediately while listening for visitor speech")
            with self._lock:
                self._mode = "thinking"
                self._message = "A visitor is here. Thinking of a greeting…"
            greeting_images = self._take_person_images()
            opening_listen_cancel = threading.Event()
            opening_phrase_started = threading.Event()
            opening_phrase_lock = threading.Lock()
            opening_listen_done = threading.Event()
            opening_listen_result: dict[str, str | None] = {"transcript": None}
            greeting_done = threading.Event()
            greeting_result: dict[str, Any] = {"text": None, "error": None}
            greeting_prompt = (
                "Give the visitor or group a bright, delighted, child-friendly welcome and introduce "
                "yourself as Bobby. Include one short, fresh, playful ghostly saying or gentle joke; "
                "if a clear costume detail is visible, warmly notice it. Keep it very brief, aiming for "
                "no more than 12 words, and do not share your backstory unless the visitor asks."
            )

            def listen_during_greeting_generation() -> None:
                try:
                    opening_listen_result["transcript"] = self._listen_for_visitor(
                        first_phrase_timeout_seconds=GREETING_GENERATION_LISTEN_POLL_SECONDS,
                        listen_cancel_event=opening_listen_cancel,
                        phrase_started_event=opening_phrase_started,
                        phrase_start_lock=opening_phrase_lock,
                    )
                except Exception:
                    LOGGER.exception("background opening listen failed")
                    opening_listen_result["transcript"] = ""
                finally:
                    opening_listen_done.set()

            def start_opening_listener() -> None:
                nonlocal opening_listen_thread
                opening_listen_result["transcript"] = None
                opening_listen_done.clear()
                with opening_phrase_lock:
                    opening_phrase_started.clear()
                opening_listen_thread = threading.Thread(
                    target=listen_during_greeting_generation,
                    name="headroom-opening-listen",
                    daemon=True,
                )
                opening_listen_thread.start()

            def prepare_greeting() -> None:
                try:
                    greeting_result["text"] = self._ask_assistant(greeting_prompt, greeting_images)
                except Exception as error:
                    greeting_result["error"] = error
                finally:
                    greeting_done.set()

            greeting_thread = threading.Thread(
                target=prepare_greeting,
                name="headroom-greeting-generation",
                daemon=True,
            )
            greeting_thread.start()
            start_opening_listener()

            heard_during_generation: str | None = None
            while not self._stop_event.is_set():
                if opening_listen_done.is_set():
                    with opening_phrase_lock:
                        phrase_was_started = opening_phrase_started.is_set()
                    heard = opening_listen_result["transcript"]
                    if heard:
                        heard_during_generation = heard
                        break
                    if phrase_was_started and not greeting_done.is_set():
                        LOGGER.info("opening audio yielded no speech; continuing to listen while the greeting generates")
                        start_opening_listener()
                        continue
                    if greeting_done.is_set():
                        break
                    # The microphone failed or is unavailable; wait for the greeting result instead of spinning.
                    opening_listen_cancel.set()

                if greeting_done.is_set():
                    with opening_phrase_lock:
                        visitor_started_speaking = opening_phrase_started.is_set()
                        if not visitor_started_speaking:
                            opening_listen_cancel.set()
                    if visitor_started_speaking:
                        continue
                    if not opening_listen_done.wait(timeout=1.5):
                        raise RuntimeError("Could not close the opening microphone before Bobby's speech.")
                    heard_during_generation = opening_listen_result["transcript"]
                    break

                self._stop_event.wait(timeout=0.03)

            if self._stop_event.is_set():
                return
            if heard_during_generation:
                opening_listen_cancel.set()
                if not greeting_done.is_set():
                    LOGGER.info("visitor speech preempted Bobby's greeting; responding without waiting for the greeting request")
                if not self._answer_transcribed_visitor(heard_during_generation, greeting_images):
                    return
            else:
                if greeting_result["error"] is not None:
                    raise greeting_result["error"]
                greeting = greeting_result["text"]
                if not greeting:
                    raise RuntimeError("The greeting request returned no spoken text.")
                self._history.append({"role": "assistant", "content": greeting})
                if not self._queue_reply(greeting):
                    return
                if not self._wait_for_playback():
                    return

            while not self._stop_event.is_set():
                if time.monotonic() - interaction_started_at >= MAX_INTERACTION_SECONDS:
                    LOGGER.info("visitor interaction reached its %.0f-second safety limit", MAX_INTERACTION_SECONDS)
                    break
                with self._lock:
                    self._mode = "thinking"
                    self._message = "Preparing the microphone…"
                heard = self._listen_for_visitor()
                if self._stop_event.is_set():
                    return
                if not heard:
                    missed += 1
                    if missed >= MAX_SILENT_LISTENS:
                        vision = self._person_vision
                        if vision is not None and vision.has_recent_person:
                            LOGGER.info(
                                "no speech in %d listen windows, but a visitor is still visible; continuing to listen",
                                missed,
                            )
                            missed = 0
                            if not offered_speech_repair:
                                offered_speech_repair = True
                                LOGGER.info("offering one gentle repeat prompt after missed speech")
                                if not self._queue_reply(SILENCE_RECOVERY_REPLY):
                                    return
                                if not self._wait_for_playback():
                                    return
                                self._history.append({"role": "assistant", "content": SILENCE_RECOVERY_REPLY})
                            continue
                        LOGGER.info("ending visitor interaction after %d consecutive quiet listen windows", missed)
                        break
                    continue
                missed = 0
                if not self._answer_transcribed_visitor(heard):
                    return
        except Exception as error:
            LOGGER.exception("visitor conversation turn failed")
            with self._lock:
                self._mode = "error"
                self._message = f"Conversation turn failed: {error}"
        finally:
            if opening_listen_cancel is not None:
                opening_listen_cancel.set()
            if opening_listen_thread is not None and opening_listen_thread is not threading.current_thread():
                opening_listen_thread.join(timeout=1.5)
            with self._lock:
                if self._active and not self._stop_event.is_set():
                    self._mode = "watching"
                    self._message = "Waiting for another visitor."
                    self._pending_reply = None
                    self._playback_claimed_reply_id = None
                    self._triggered_at = time.monotonic()
                    LOGGER.info("visitor interaction ended; returning to motion watch")

    def _create_recognizer(self) -> Any | None:
        if sr is None:
            return None
        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = LISTEN_PAUSE_THRESHOLD_SECONDS
        recognizer.phrase_threshold = MINIMUM_SPEECH_DURATION_SECONDS
        recognizer.non_speaking_duration = 0.35
        return recognizer

    def _prepare_recognizer(self) -> None:
        self._recognizer = self._create_recognizer()
        self._recognizer_calibrated = False
        self._last_recognizer_calibration_at = None
        if self._recognizer is None:
            return
        try:
            with sr.Microphone(device_index=self._microphone_index, sample_rate=16000) as source:
                self._recognizer.adjust_for_ambient_noise(
                    source,
                    duration=AMBIENT_CALIBRATION_DURATION_SECONDS,
                )
            self._recognizer_calibrated = True
            self._last_recognizer_calibration_at = time.monotonic()
            LOGGER.info(
                "microphone calibrated at visitor-mode startup; short-phrase threshold=%.2fs",
                MINIMUM_SPEECH_DURATION_SECONDS,
            )
        except Exception as error:
            self._last_recognizer_calibration_at = time.monotonic()
            LOGGER.warning(
                "microphone warm-up calibration failed (%s); will retry before first listen",
                type(error).__name__,
            )

    def _maybe_refresh_recognizer_calibration(self) -> None:
        """Refresh a stale noise baseline only in a quiet, visitor-free watch period."""
        recognizer = self._recognizer
        last_calibrated = self._last_recognizer_calibration_at
        if (
            sr is None
            or recognizer is None
            or last_calibrated is None
            or time.monotonic() - last_calibrated < AMBIENT_RECALIBRATION_INTERVAL_SECONDS
        ):
            return
        with self._lock:
            if (
                not self._active
                or self._mode != "watching"
                or self._stop_event.is_set()
                or self._motion_event.is_set()
            ):
                return
            person_vision = self._person_vision
            last_motion_at = self._last_motion_at
            last_triggered_at = self._triggered_at
        if person_vision is not None and person_vision.has_recent_person:
            return
        if time.monotonic() - last_triggered_at < AMBIENT_QUIET_WINDOW_SECONDS:
            return
        if (
            last_motion_at is not None
            and time.time() - last_motion_at < AMBIENT_QUIET_WINDOW_SECONDS
        ):
            return

        previous_threshold = recognizer.energy_threshold
        try:
            with sr.Microphone(device_index=self._microphone_index, sample_rate=16000) as source:
                recognizer.adjust_for_ambient_noise(
                    source,
                    duration=AMBIENT_CALIBRATION_DURATION_SECONDS,
                )
            with self._lock:
                visitor_arrived = (
                    self._stop_event.is_set()
                    or self._motion_event.is_set()
                    or (person_vision is not None and person_vision.has_recent_person)
                )
            if visitor_arrived:
                recognizer.energy_threshold = previous_threshold
                LOGGER.info("discarded ambient recalibration because visitor activity appeared during the quiet window")
                return
            self._recognizer_calibrated = True
            self._last_recognizer_calibration_at = time.monotonic()
            LOGGER.info("microphone ambient threshold refreshed during an idle quiet window")
        except Exception as error:
            recognizer.energy_threshold = previous_threshold
            self._last_recognizer_calibration_at = time.monotonic()
            LOGGER.warning("quiet-window microphone recalibration failed (%s)", type(error).__name__)

    def _listen_for_visitor(
        self,
        first_phrase_timeout_seconds: float = LISTEN_TIMEOUT_SECONDS,
        listen_cancel_event: threading.Event | None = None,
        phrase_started_event: threading.Event | None = None,
        phrase_start_lock: threading.Lock | None = None,
    ) -> str | None:
        if listen_cancel_event is not None and listen_cancel_event.is_set():
            return None
        if phrase_start_lock is None:
            phrase_start_lock = threading.Lock()
        if sr is None:
            return ""

        if self._recognizer is None:
            self._recognizer = self._create_recognizer()
            self._recognizer_calibrated = False
        recognizer = self._recognizer
        if recognizer is None:
            return ""

        capture_events: queue.Queue[tuple[str, Any]] = queue.Queue()
        capture_stop = threading.Event()
        capture_state_lock = threading.Lock()
        capture_state = {
            "accepting": True,
            "phrase_candidate": False,
            "transcription_cancel_event": None,
        }
        capture_thread: threading.Thread | None = None
        current_transcription_id = 0
        pending_result: tuple[int, str] | None = None
        pending_transcription_error: tuple[int, Exception] | None = None
        audio_frames = bytearray()
        audio_sample_rate = 0
        audio_sample_width = 0

        try:
            LOGGER.info("opening microphone for visitor speech")

            def capture_audio() -> None:
                try:
                    with sr.Microphone(device_index=self._microphone_index, sample_rate=16000) as source:
                        if not self._recognizer_calibrated:
                            recognizer.adjust_for_ambient_noise(
                                source,
                                duration=AMBIENT_CALIBRATION_DURATION_SECONDS,
                            )
                            self._recognizer_calibrated = True
                            self._last_recognizer_calibration_at = time.monotonic()
                        with self._lock:
                            if self._active and not self._stop_event.is_set():
                                self._mode = "listening"
                                self._message = "Listening for the visitor…"
                        LOGGER.info("microphone capture ready; listening for visitor speech")
                        first_phrase = True
                        while (
                            not capture_stop.is_set()
                            and not self._stop_event.is_set()
                            and not (listen_cancel_event is not None and listen_cancel_event.is_set())
                        ):
                            phrase_chunks: list[bytes] = []
                            phrase_active = False
                            timeout = first_phrase_timeout_seconds if first_phrase else CAPTURE_POLL_TIMEOUT_SECONDS
                            try:
                                audio_stream = recognizer.listen(
                                    source,
                                    timeout=timeout,
                                    phrase_time_limit=PHRASE_LIMIT_SECONDS,
                                    stream=True,
                                )
                                for chunk in audio_stream:
                                    if not phrase_active and first_phrase and phrase_started_event is not None:
                                        with phrase_start_lock:
                                            if listen_cancel_event is not None and listen_cancel_event.is_set():
                                                return
                                            phrase_started_event.set()
                                    with capture_state_lock:
                                        if (
                                            not capture_state["accepting"]
                                            or capture_stop.is_set()
                                            or self._stop_event.is_set()
                                            or (listen_cancel_event is not None and listen_cancel_event.is_set())
                                        ):
                                            return
                                        if not phrase_active:
                                            capture_state["phrase_candidate"] = True
                                            phrase_active = True
                                    phrase_chunks.append(chunk.frame_data)
                            except sr.WaitTimeoutError:
                                with capture_state_lock:
                                    capture_state["phrase_candidate"] = False
                                if first_phrase:
                                    if listen_cancel_event is not None and listen_cancel_event.is_set():
                                        return
                                    if listen_cancel_event is not None:
                                        continue
                                    capture_events.put(("timeout", None))
                                    return
                                continue

                            if not phrase_chunks:
                                if first_phrase:
                                    capture_events.put(("timeout", None))
                                    return
                                continue

                            audio = sr.AudioData(
                                b"".join(phrase_chunks),
                                source.SAMPLE_RATE,
                                source.SAMPLE_WIDTH,
                            )
                            with capture_state_lock:
                                capture_state["phrase_candidate"] = False
                                if not capture_state["accepting"] or capture_stop.is_set():
                                    return
                                capture_events.put(("audio", audio))
                            first_phrase = False
                except Exception as error:
                    with capture_state_lock:
                        capture_state["phrase_candidate"] = False
                    capture_events.put(("error", error))

            capture_thread = threading.Thread(
                target=capture_audio,
                name="headroom-visitor-audio-capture",
                daemon=True,
            )
            capture_thread.start()

            first_audio = None
            while first_audio is None:
                if listen_cancel_event is not None and listen_cancel_event.is_set():
                    return None
                if self._stop_event.is_set():
                    return ""
                try:
                    event, payload = capture_events.get(timeout=0.1)
                except queue.Empty:
                    continue
                if event == "timeout":
                    LOGGER.info("no visitor speech detected before listen timeout")
                    return ""
                if event == "error":
                    raise payload
                first_audio = payload

            audio_sample_rate = first_audio.sample_rate
            audio_sample_width = first_audio.sample_width
            audio_frames.extend(first_audio.frame_data)

            def start_transcription() -> None:
                nonlocal current_transcription_id, pending_result, pending_transcription_error
                with capture_state_lock:
                    previous_cancel_event = capture_state["transcription_cancel_event"]
                    if previous_cancel_event is not None:
                        previous_cancel_event.set()
                    cancel_event = threading.Event()
                    capture_state["transcription_cancel_event"] = cancel_event
                current_transcription_id += 1
                request_id = current_transcription_id
                wav_data = sr.AudioData(
                    bytes(audio_frames), audio_sample_rate, audio_sample_width
                ).get_wav_data(convert_rate=24000, convert_width=2)
                audio_duration = len(audio_frames) / (audio_sample_rate * audio_sample_width)
                LOGGER.info(
                    "visitor audio captured (%.2f seconds, %d bytes); starting speculative transcription (attempt=%d)",
                    audio_duration,
                    len(wav_data),
                    request_id,
                )
                pending_result = None
                pending_transcription_error = None

                def transcribe_snapshot() -> None:
                    try:
                        if cancel_event.is_set():
                            capture_events.put(("transcription_cancelled", request_id))
                            return
                        result = self._transcribe(wav_data)
                        capture_events.put(("transcription", (request_id, result, None)))
                    except Exception as error:
                        capture_events.put(("transcription", (request_id, None, error)))

                threading.Thread(
                    target=transcribe_snapshot,
                    name=f"headroom-transcription-{request_id}",
                    daemon=True,
                ).start()

            start_transcription()

            while (
                not self._stop_event.is_set()
                and not (listen_cancel_event is not None and listen_cancel_event.is_set())
            ):
                audio_changed = False
                try:
                    event, payload = capture_events.get(timeout=0.05)
                    pending_events = [(event, payload)]
                except queue.Empty:
                    pending_events = []
                while True:
                    try:
                        pending_events.append(capture_events.get_nowait())
                    except queue.Empty:
                        break

                for event, payload in pending_events:
                    if event == "error":
                        if audio_frames:
                            LOGGER.warning(
                                "microphone monitoring failed after visitor audio capture; falling back to captured audio (%s)",
                                type(payload).__name__,
                            )
                            with capture_state_lock:
                                capture_state["accepting"] = False
                                capture_state["phrase_candidate"] = False
                            capture_stop.set()
                            continue
                        raise payload
                    if event == "timeout" or event == "transcription_cancelled":
                        continue
                    if event == "transcription":
                        request_id, transcript, error = payload
                        with capture_state_lock:
                            cancel_event = capture_state["transcription_cancel_event"]
                            request_was_cancelled = cancel_event is not None and cancel_event.is_set()
                        if request_id != current_transcription_id or request_was_cancelled:
                            LOGGER.info("discarding stale transcription attempt=%d", request_id)
                            continue
                        if error is not None:
                            with capture_state_lock:
                                phrase_candidate = (
                                    capture_state["phrase_candidate"]
                                    or not capture_events.empty()
                                )
                            if phrase_candidate:
                                pending_transcription_error = (request_id, error)
                                continue
                            raise error
                        pending_result = (request_id, transcript)
                        if transcript:
                            LOGGER.info(
                                "transcription attempt=%d returned (%d characters)",
                                request_id,
                                len(transcript),
                            )
                        else:
                            LOGGER.info("transcription attempt=%d returned no speech", request_id)
                        continue
                    audio = payload
                    if audio.sample_rate != audio_sample_rate or audio.sample_width != audio_sample_width:
                        raise RuntimeError("Microphone audio format changed during visitor speech.")
                    audio_frames.extend(audio.frame_data)
                    audio_changed = True

                if audio_changed:
                    LOGGER.info(
                        "visitor resumed while transcription was pending; invalidating attempt=%d and transcribing the combined utterance",
                        current_transcription_id,
                    )
                    start_transcription()

                if pending_result is not None or pending_transcription_error is not None:
                    with capture_state_lock:
                        if not capture_state["phrase_candidate"] and capture_events.empty():
                            capture_state["accepting"] = False
                            capture_stop.set()
                            accepted_transcript = pending_result[1] if pending_result is not None else None
                            accepted_error = pending_transcription_error[1] if pending_transcription_error is not None else None
                        else:
                            accepted_transcript = None
                            accepted_error = None
                    if accepted_error is not None:
                        raise accepted_error
                    if pending_result is not None and accepted_transcript is not None:
                        if accepted_transcript:
                            LOGGER.info("transcription returned (%d characters)", len(accepted_transcript))
                        return accepted_transcript

            return None if listen_cancel_event is not None and listen_cancel_event.is_set() else ""
        except Exception as error:
            with self._lock:
                self._microphone_status = f"Input error · {error}"
            LOGGER.exception("visitor audio capture or transcription failed")
            time.sleep(0.4)
            return ""
        finally:
            capture_stop.set()
            with capture_state_lock:
                capture_state["accepting"] = False
                cancel_event = capture_state["transcription_cancel_event"]
                if cancel_event is not None:
                    cancel_event.set()
            if capture_thread is not None and capture_thread is not threading.current_thread():
                capture_thread.join(timeout=1.0)

    def _latest_frame_copy(self) -> Any | None:
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def _take_person_images(self) -> list[str]:
        vision = self._person_vision
        if vision is None:
            return []
        crops = vision.take_new_crops(MAX_CROPS_PER_TURN)
        images: list[str] = []
        for crop in crops:
            if cv2 is None:
                continue
            height, width = crop.shape[:2]
            scale = min(1.0, 1024.0 / max(height, width))
            if scale < 1.0:
                crop = cv2.resize(
                    crop,
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                continue
            jpeg_data = encoded.tobytes()
            self._save_vision_frame(jpeg_data)
            images.append(f"data:image/jpeg;base64,{base64.b64encode(jpeg_data).decode('ascii')}")
        if images:
            LOGGER.info("using %d recent person crop(s) for visual context", len(images))
        return images

    def _save_vision_frame(self, jpeg_data: bytes) -> None:
        if not SAVE_VISION_FRAMES:
            return
        try:
            VISION_FRAME_DIRECTORY.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(VISION_FRAME_DIRECTORY, 0o700)
            with self._lock:
                self._snapshot_sequence += 1
                sequence = self._snapshot_sequence
            timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            filename = f"headroom-visitor-{timestamp}-{sequence:04d}.jpg"
            path = VISION_FRAME_DIRECTORY / filename
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as image_file:
                image_file.write(jpeg_data)
            os.chmod(path, 0o600)
            with self._lock:
                self._saved_vision_frame_count += 1
                self._last_vision_frame_name = filename
            LOGGER.info("saved vision snapshot logs/vision_frames/%s (%d bytes)", filename, len(jpeg_data))
        except OSError as error:
            LOGGER.warning("could not save vision snapshot (%s)", type(error).__name__)
            return

        try:
            frames = sorted(VISION_FRAME_DIRECTORY.glob("headroom-visitor-*.jpg"))
            for stale_frame in frames[:-MAX_SAVED_VISION_FRAMES]:
                stale_frame.unlink(missing_ok=True)
        except OSError as error:
            LOGGER.warning("could not prune old vision snapshots (%s)", type(error).__name__)

    def _transcribe(self, wav_data: bytes) -> str:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("No OpenAI key is available for conversation transcription.")
        boundary = f"----headroom-{os.urandom(12).hex()}"
        body = bytearray()
        body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{TRANSCRIPTION_MODEL}\r\n".encode())
        body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\njson\r\n".encode())
        body.extend(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"visitor.wav\"\r\n"
            "Content-Type: audio/wav\r\n\r\n".encode()
        )
        body.extend(wav_data)
        body.extend(f"\r\n--{boundary}--\r\n".encode())
        request = urllib.request.Request(
            "https://api.openai.com/v1/audio/transcriptions",
            data=bytes(body),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "User-Agent": "headroom-local-demo/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=35) as response:
            result = json.loads(response.read().decode("utf-8"))
        return str(result.get("text", "")).strip()

    def _ask_assistant(self, visitor_text: str, image_data_urls: list[str] | None = None) -> str:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("No OpenAI key is available for visitor conversation.")
        if self._history:
            conversation_input: list[dict[str, Any]] = [
                {"role": turn["role"], "content": turn["content"]} for turn in self._history
            ]
        else:
            conversation_input = []
        content: list[dict[str, Any]] = [{"type": "input_text", "text": visitor_text}]
        image_data_urls = image_data_urls or []
        for index, image_data_url in enumerate(image_data_urls, start=1):
            content.append({"type": "input_text", "text": f"Anonymous visitor crop {index}; newly detected or recently refreshed."})
            content.append({"type": "input_image", "image_url": image_data_url, "detail": "high"})
        conversation_input.append({"role": "user", "content": content})
        payload = {
            "model": VISION_MODEL,
            "instructions": SYSTEM_INSTRUCTIONS,
            "input": conversation_input,
            "max_output_tokens": MAX_REPLY_TOKENS,
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "headroom-local-demo/1.0",
            },
            method="POST",
        )
        LOGGER.info("requesting dialogue response (person_crops=%d)", len(image_data_urls))
        with urllib.request.urlopen(request, timeout=45) as response:
            result = json.loads(response.read().decode("utf-8"))
        answer = response_text(result)
        if not answer:
            raise RuntimeError("The conversation model returned no spoken text.")
        LOGGER.info("dialogue response generated (%d words, %d characters)", len(answer.split()), len(answer))
        return answer[:900]

    def _queue_reply(self, text: str) -> bool:
        with self._lock:
            if not self._active or self._stop_event.is_set():
                return False
            self._reply_counter += 1
            self._pending_reply = {"id": self._reply_counter, "text": text}
            self._playback_claimed_reply_id = None
            self._playback_started.clear()
            self._playback_finished.clear()
            self._mode = "reply_ready"
            self._message = "A visitor reply is queued for laptop playback."
            LOGGER.info("reply ready id=%d; %d characters", self._reply_counter, len(text))
            return True

    def _wait_for_playback(self) -> bool:
        with self._lock:
            pending_id = self._pending_reply["id"] if self._pending_reply else None
        LOGGER.info("waiting for laptop speech playback to start reply id=%s", pending_id)
        if not self._playback_started.wait(timeout=90):
            LOGGER.warning("timed out waiting for server speech playback to start")
            return False
        if self._stop_event.is_set():
            return False
        finished = self._playback_finished.wait(timeout=180)
        if not finished:
            LOGGER.warning("timed out waiting for reply playback to finish")
        return finished and not self._stop_event.is_set()
