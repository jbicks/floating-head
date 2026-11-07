#!/usr/bin/env python3
"""Local Headroom server: laptop speech playback + Rhubarb mouth-cue analysis."""

from __future__ import annotations

import json
import logging
import os
import posixpath
import re
import shutil
import signal
import struct
import subprocess
import sys
import concurrent.futures
import queue
import collections
import tracemalloc
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from conversation import ConversationManager, SAVE_VISION_FRAMES, SAVE_VISITOR_TRANSCRIPTS

ROOT = Path(__file__).resolve().parent
MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
Rhubarb = os.getenv("RHUBARB_PATH")
VOICE_NAMES = (
    "alloy", "ash", "ballad", "cedar", "coral", "echo", "fable",
    "marin", "nova", "onyx", "sage", "shimmer", "verse",
)
DEFAULT_VOICE = os.getenv("OPENAI_TTS_VOICE", "nova")
DEFAULT_RATE = 0.94
PAPLAY_PATH = shutil.which("paplay")
TTS_INSTRUCTIONS = (
    "Use a bright, cheerful, smiling adult female voice with inviting warmth and a "
    "little sparkle. Sound genuinely delighted to meet children, with lively but gentle "
    "pitch variation and playful emphasis on a whimsical word or light joke. Keep the "
    "pace conversational and the words clear. The ghostly flavor should feel friendly "
    "and charming, never spooky or frightening. Avoid sounding flat, robotic, breathy, "
    "raspy, overexcited, sing-song, or theatrical."
)
RHUBARB_RECOGNIZER = os.getenv("RHUBARB_RECOGNIZER", "phonetic")
if RHUBARB_RECOGNIZER not in {"pocketSphinx", "phonetic"}:
    raise ValueError("RHUBARB_RECOGNIZER must be 'pocketSphinx' or 'phonetic'.")
MAX_TEXT_LENGTH = 600
TTS_TIMEOUT_SECONDS = 75
LIPSYNC_TIMEOUT_SECONDS = 90
PCM_SAMPLE_RATE = 24_000
PCM_PACKET_BYTES = 9_600  # 200 ms of mono, 16-bit PCM
PCM_START_BUFFER_PACKETS = 5  # Absorb brief OpenAI/WSLg scheduling jitter before audible playback.
PCM_PLAYBACK_QUEUE_PACKETS = 40  # Keep up to 8 seconds of streamed audio ahead of PulseAudio.
PULSE_OUTPUT_LATENCY_MS = 200
LOG_PATH = ROOT / "logs" / "headroom.log"
LOG_MAX_BYTES = 3 * 1024 * 1024
LOG_BACKUPS = 3


class SpeechEventHub:
    """Retain a small, bounded animation-event history independent of display clients."""

    def __init__(self, max_events_per_reply: int = 512, max_replies: int = 8) -> None:
        self._condition = threading.Condition()
        self._events: collections.OrderedDict[int, collections.deque] = collections.OrderedDict()
        self._completed: set[int] = set()
        self._failed: set[int] = set()
        self._sequence: dict[int, int] = {}
        self._max_events = max_events_per_reply
        self._max_replies = max_replies

    def publish(self, reply_id: int, event: dict) -> None:
        with self._condition:
            events = self._events.setdefault(reply_id, collections.deque(maxlen=self._max_events))
            sequence = self._sequence.get(reply_id, 0) + 1
            self._sequence[reply_id] = sequence
            events.append({**event, "replyId": reply_id, "sequence": sequence})
            self._events.move_to_end(reply_id)
            while len(self._events) > self._max_replies:
                expired_id, _ = self._events.popitem(last=False)
                self._completed.discard(expired_id)
                self._failed.discard(expired_id)
                self._sequence.pop(expired_id, None)
            if event.get("type") in {"done", "error"}:
                self._completed.add(reply_id)
            if event.get("type") == "error":
                self._failed.add(reply_id)
            self._condition.notify_all()

    def wait(self, reply_id: int, after_sequence: int, timeout: float = 10.0) -> tuple[list[dict], bool]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                events = self._events.get(reply_id, ())
                fresh = [event for event in events if event["sequence"] > after_sequence]
                if fresh or reply_id in self._completed:
                    return fresh, reply_id in self._completed
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return [], False
                self._condition.wait(remaining)

    def completed(self, reply_id: int) -> bool:
        with self._condition:
            return reply_id in self._completed

    def failed(self, reply_id: int) -> bool:
        with self._condition:
            return reply_id in self._failed

    def knows(self, reply_id: int) -> bool:
        with self._condition:
            return reply_id in self._events


class DisplayHeartbeat:
    """Low-volume diagnostics for the display page's normal status polling."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_monotonic: float | None = None
        self._last_wall: float | None = None
        self._poll_count = 0

    def record(self) -> dict:
        now = time.monotonic()
        with self._lock:
            gap = None if self._last_monotonic is None else now - self._last_monotonic
            self._last_monotonic = now
            self._last_wall = time.time()
            self._poll_count += 1
            count = self._poll_count
            last_wall = self._last_wall
        if gap is None or gap > 5 or count % 120 == 0:
            logging.info("display heartbeat: poll=%d gap_seconds=%s", count, "first" if gap is None else f"{gap:.2f}")
        return {"lastPollAt": last_wall, "ageSeconds": 0.0, "pollCount": count, "gapSeconds": round(gap, 2) if gap is not None else None}

    def snapshot(self) -> dict:
        with self._lock:
            last_wall = self._last_wall
            count = self._poll_count
            last_mono = self._last_monotonic
        return {
            "lastPollAt": last_wall,
            "ageSeconds": round(time.monotonic() - last_mono, 1) if last_mono is not None else None,
            "pollCount": count,
        }


class RuntimeDiagnostics:
    """Periodic low-overhead process resource sampling, with optional tracemalloc."""

    def __init__(self) -> None:
        raw_interval = os.getenv("HEADROOM_DIAGNOSTICS_INTERVAL_SECONDS", "60")
        try:
            self.interval = max(5, min(300, int(raw_interval)))
        except ValueError:
            self.interval = 60
        self.profile = os.getenv("HEADROOM_PROFILE", "0") == "1"
        if self.profile and not tracemalloc.is_tracing():
            tracemalloc.start(10)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="headroom-diagnostics", daemon=True)
        self._last_cpu = self._cpu_seconds()
        self._thread.start()
        logging.info("runtime diagnostics enabled: interval_seconds=%d tracemalloc=%s", self.interval, self.profile)

    @staticmethod
    def _cpu_seconds() -> float:
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            return usage.ru_utime + usage.ru_stime
        except (ImportError, OSError):
            return 0.0

    @staticmethod
    def _proc_memory() -> tuple[int | None, int | None]:
        try:
            values = {}
            with open("/proc/self/status", encoding="ascii") as status:
                for line in status:
                    if line.startswith(("VmRSS:", "VmHWM:")):
                        name, value, *_ = line.split()
                        values[name.rstrip(":")] = int(value) * 1024
            return values.get("VmRSS"), values.get("VmHWM")
        except (OSError, ValueError):
            return None, None

    @staticmethod
    def _child_process_count() -> int | None:
        try:
            children: set[str] = set()
            for thread_id in os.listdir("/proc/self/task"):
                try:
                    with open(f"/proc/self/task/{thread_id}/children", encoding="ascii") as child_file:
                        children.update(child_file.read().split())
                except OSError:
                    continue
            return len(children)
        except OSError:
            return None

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            rss, peak = self._proc_memory()
            cpu = self._cpu_seconds()
            cpu_percent = max(0.0, (cpu - self._last_cpu) * 100 / self.interval)
            self._last_cpu = cpu
            try:
                fd_count = len(os.listdir("/proc/self/fd"))
            except OSError:
                fd_count = None
            children = self._child_process_count()
            detail = (
                f"runtime resources: rss_mb={rss / 1048576:.1f}" if rss is not None else "runtime resources: rss_mb=unavailable"
            )
            detail += f" peak_rss_mb={peak / 1048576:.1f}" if peak is not None else " peak_rss_mb=unavailable"
            detail += f" threads={threading.active_count()} children={children} fds={fd_count} cpu_percent={cpu_percent:.1f}"
            if self.profile and tracemalloc.is_tracing():
                current, peak_allocated = tracemalloc.get_traced_memory()
                detail += f" traced_mb={current / 1048576:.1f} traced_peak_mb={peak_allocated / 1048576:.1f}"
            logging.info(detail)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=2)


class ReplySpeechDispatcher:
    """Claims each conversation reply on the server and streams animation to any display."""

    def __init__(self, server: ThreadingHTTPServer, hub: SpeechEventHub) -> None:
        self.server = server
        self.hub = hub
        self._stop = threading.Event()
        self._last_reply_id = 0
        self._workers: list[threading.Thread] = []
        self._thread = threading.Thread(target=self._run, name="headroom-speech-dispatch", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _dispatch(self, reply: dict) -> None:
        reply_id = int(reply["id"])
        body = {
            "text": str(reply.get("text", "")),
            "voice": getattr(self.server, "conversation_voice", DEFAULT_VOICE),
            "rate": getattr(self.server, "conversation_rate", DEFAULT_RATE),
            "replyId": reply_id,
        }
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.server.server_port}/api/speak-stream",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "Headroom-Speech-Dispatcher/1.0"},
            method="POST",
        )
        logging.info("server speech dispatcher started reply id=%d", reply_id)
        try:
            with urllib.request.urlopen(request, timeout=TTS_TIMEOUT_SECONDS + LIPSYNC_TIMEOUT_SECONDS) as response:
                for raw_line in response:
                    if self._stop.is_set():
                        break
                    try:
                        event = json.loads(raw_line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    self.hub.publish(reply_id, event)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            logging.error("server speech dispatcher failed reply id=%d (%s)", reply_id, type(error).__name__)
            manager = getattr(self.server, "conversation_manager", None)
            if manager is not None:
                manager.set_playback(False, reply_id)
            self.hub.publish(reply_id, {"type": "error", "error": "Laptop speech playback could not start."})

    def _run(self) -> None:
        while not self._stop.is_set():
            manager = getattr(self.server, "conversation_manager", None)
            status = manager.status() if manager else {}
            reply = status.get("reply")
            reply_id = reply.get("id") if isinstance(reply, dict) else None
            if isinstance(reply_id, int) and reply_id > self._last_reply_id:
                self._last_reply_id = reply_id
                self._workers = [worker for worker in self._workers if worker.is_alive()]
                worker = threading.Thread(
                    target=self._dispatch, args=(reply,), name=f"headroom-speech-{reply_id}", daemon=True
                )
                self._workers.append(worker)
                worker.start()
            else:
                self._stop.wait(0.1)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=3)
        for worker in self._workers:
            if worker is not threading.current_thread():
                worker.join(timeout=3)


class _TeeStream:
    """Mirror process output to the original terminal and the private log file."""

    def __init__(self, terminal, logfile, lock: threading.Lock) -> None:
        self._terminal = terminal
        self._logfile = logfile
        self._lock = lock

    def write(self, text: str) -> int:
        if not text:
            return 0
        with self._lock:
            try:
                self._terminal.write(text)
                self._terminal.flush()
            except (BrokenPipeError, OSError):
                pass
            try:
                self._logfile.write(text)
                if "\n" in text:
                    self._logfile.flush()
            except OSError:
                pass
        return len(text)

    def flush(self) -> None:
        with self._lock:
            for stream in (self._terminal, self._logfile):
                try:
                    stream.flush()
                except (BrokenPipeError, OSError):
                    pass

    def isatty(self) -> bool:
        return bool(getattr(self._terminal, "isatty", lambda: False)())

    @property
    def encoding(self):
        return getattr(self._terminal, "encoding", "utf-8")

    def fileno(self) -> int:
        return self._terminal.fileno()


def configure_diagnostics_log() -> Path:
    LOG_PATH.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(LOG_PATH.parent, 0o700)
    try:
        if LOG_PATH.exists() and LOG_PATH.stat().st_size >= LOG_MAX_BYTES:
            for index in range(LOG_BACKUPS, 0, -1):
                source = LOG_PATH if index == 1 else Path(f"{LOG_PATH}.{index - 1}")
                destination = Path(f"{LOG_PATH}.{index}")
                if destination.exists():
                    destination.unlink()
                if source.exists():
                    os.replace(source, destination)
    except OSError as error:
        print(f"Could not rotate diagnostics log: {error}", file=sys.stderr)
    old_umask = os.umask(0o077)
    try:
        logfile = LOG_PATH.open("a", encoding="utf-8", buffering=1)
    finally:
        os.umask(old_umask)
    os.chmod(LOG_PATH, 0o600)
    lock = threading.Lock()
    sys.stdout = _TeeStream(sys.stdout, logfile, lock)
    sys.stderr = _TeeStream(sys.stderr, logfile, lock)
    logging.basicConfig(
        level=logging.DEBUG if os.getenv("HEADROOM_DEBUG") == "1" else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )
    print(f"Diagnostics log: {LOG_PATH}", flush=True)
    print("Raw visitor audio is not saved.", flush=True)
    if SAVE_VISITOR_TRANSCRIPTS:
        print("Recognized visitor transcripts are saved under logs/visitor_transcripts.jsonl (private, rotated).", flush=True)
    else:
        print("Visitor transcript saving is disabled (HEADROOM_SAVE_TRANSCRIPTS=0).", flush=True)
    if SAVE_VISION_FRAMES:
        print("Visual-context JPEG snapshots are saved under logs/vision_frames/ (latest 100 retained).", flush=True)
    else:
        print("Visual-context JPEG snapshot saving is disabled (HEADROOM_SAVE_VISION_FRAMES=0).", flush=True)
    return LOG_PATH


def pcm_to_wav(pcm: bytes) -> bytes:
    if not pcm or len(pcm) % 2:
        raise ValueError("OpenAI returned an empty or misaligned PCM stream.")
    return (
        b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, PCM_SAMPLE_RATE, PCM_SAMPLE_RATE * 2, 2, 16)
        + b"data" + struct.pack("<I", len(pcm)) + pcm
    )


def jaw_envelope(pcm: bytes, window_ms: int = 10) -> list[float]:
    """Reduce raw PCM to small, per-window jaw values for the silent display client."""
    window_samples = max(1, PCM_SAMPLE_RATE * window_ms // 1000)
    envelope = []
    for offset in range(0, len(pcm) - (len(pcm) % 2), window_samples * 2):
        end = min(len(pcm) - (len(pcm) % 2), offset + window_samples * 2)
        total = 0.0
        count = 0
        for (sample,) in struct.iter_unpack("<h", pcm[offset:end]):
            normalized = sample / 32768.0
            total += normalized * normalized
            count += 1
        rms = (total / count) ** 0.5 if count else 0.0
        envelope.append(round(max(0.0, min(0.42, (rms - 0.018) * 2.8)), 3))
    return envelope


class LaptopAudioPlayer:
    """Stream the exact TTS PCM to the laptop's WSL PulseAudio default sink."""

    def __init__(self) -> None:
        if not PAPLAY_PATH:
            raise RuntimeError("paplay is unavailable; install PulseAudio utilities for laptop audio output.")
        self.process = subprocess.Popen(
            [
                PAPLAY_PATH, "--raw", "--format=s16le", f"--rate={PCM_SAMPLE_RATE}",
                "--channels=1", f"--latency-msec={PULSE_OUTPUT_LATENCY_MS}", "--client-name=Headroom",
                "--stream-name=Bobby TTS",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self._lock = threading.Lock()
        self._next_packet_at: float | None = None
        self._late_packet_count = 0
        self._total_late_ms = 0.0
        self._max_late_ms = 0.0
        self._finished = False
        time.sleep(0.03)
        if self.process.poll() is not None:
            raise RuntimeError("paplay could not connect to the laptop's PulseAudio output.")

    def write(self, pcm: bytes) -> None:
        if not pcm:
            return
        with self._lock:
            if self._finished or self.process.poll() is not None:
                raise RuntimeError("Laptop audio output stopped unexpectedly.")
            now = time.monotonic()
            if self._next_packet_at is not None:
                late_ms = max(0.0, (now - self._next_packet_at) * 1000)
                if late_ms > PULSE_OUTPUT_LATENCY_MS:
                    self._late_packet_count += 1
                    self._total_late_ms += late_ms
                    self._max_late_ms = max(self._max_late_ms, late_ms)
                if now < self._next_packet_at:
                    time.sleep(self._next_packet_at - now)
            if self.process.stdin is None:
                raise RuntimeError("Laptop audio output has no input pipe.")
            try:
                self.process.stdin.write(pcm)
                self.process.stdin.flush()
            except (BrokenPipeError, OSError) as error:
                raise RuntimeError("Could not stream audio to the laptop output.") from error
            self._next_packet_at = max(now, self._next_packet_at or now) + len(pcm) / (PCM_SAMPLE_RATE * 2)

    def finish(self) -> None:
        with self._lock:
            if self._finished:
                return
            self._finished = True
            if self.process.stdin and not self.process.stdin.closed:
                self.process.stdin.close()
        try:
            return_code = self.process.wait(timeout=180)
        except subprocess.TimeoutExpired as error:
            self.stop()
            raise RuntimeError("Laptop audio playback did not finish before the timeout.") from error
        if return_code != 0:
            raise RuntimeError(f"paplay exited with status {return_code}.")

    def timing_metrics(self) -> dict[str, int]:
        with self._lock:
            return {
                "latePackets": self._late_packet_count,
                "totalLateMs": round(self._total_late_ms),
                "maxLateMs": round(self._max_late_ms),
            }

    def stop(self) -> None:
        with self._lock:
            self._finished = True
            if self.process.poll() is not None:
                return
            try:
                self.process.terminate()
            except OSError:
                return
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=2)


def play_laptop_wav(wav_data: bytes) -> None:
    if not PAPLAY_PATH:
        raise RuntimeError("paplay is unavailable; install PulseAudio utilities for laptop audio output.")
    process = subprocess.Popen(
        [PAPLAY_PATH, f"--latency-msec={PULSE_OUTPUT_LATENCY_MS}", "--client-name=Headroom", "--stream-name=Bobby TTS"],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        process.communicate(input=wav_data, timeout=180)
    except subprocess.TimeoutExpired as error:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
        raise RuntimeError("Laptop audio playback did not finish before the timeout.") from error
    if process.returncode != 0:
        raise RuntimeError(f"paplay exited with status {process.returncode}.")


def normalize_streamed_wav(audio: bytes) -> bytes:
    """Repair the streaming RIFF size sentinels used by OpenAI's WAV response."""
    if len(audio) < 44 or audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
        raise ValueError("OpenAI did not return a valid RIFF/WAVE file.")
    data = bytearray(audio)
    offset = 12
    found_data = False
    while offset + 8 <= len(data):
        chunk_id = bytes(data[offset : offset + 4])
        chunk_size = struct.unpack_from("<I", data, offset + 4)[0]
        if chunk_id == b"data":
            actual_size = len(data) - offset - 8
            if chunk_size == 0xFFFFFFFF or offset + 8 + chunk_size > len(data):
                struct.pack_into("<I", data, offset + 4, actual_size)
            found_data = True
            break
        if chunk_size == 0xFFFFFFFF or offset + 8 + chunk_size > len(data):
            raise ValueError("WAV contains an invalid chunk length.")
        offset += 8 + chunk_size + (chunk_size & 1)
    if not found_data:
        raise ValueError("WAV has no data chunk.")
    struct.pack_into("<I", data, 4, len(data) - 8)
    return bytes(data)


def laptop_audio_status() -> dict[str, str | bool]:
    if not PAPLAY_PATH:
        return {"available": False, "sink": "paplay not installed"}
    pactl = shutil.which("pactl")
    if not pactl:
        return {"available": True, "sink": "PulseAudio default sink (pactl unavailable)"}
    try:
        result = subprocess.run(
            [pactl, "info"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, timeout=2, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"available": False, "sink": "PulseAudio server unavailable"}
    if result.returncode != 0:
        return {"available": False, "sink": "PulseAudio server unavailable"}
    sink = next((line.partition(":")[2].strip() for line in result.stdout.splitlines()
                 if line.startswith("Default Sink:")), "PulseAudio default sink")
    return {"available": True, "sink": sink}


def find_rhubarb() -> Path | None:
    candidates = []
    if Rhubarb:
        candidates.append(Path(Rhubarb).expanduser())
    candidates.extend(
        [
            Path.home() / ".local/share/headroom/rhubarb-1.14.0/rhubarb",
            ROOT / ".tools/rhubarb/rhubarb",
        ]
    )
    found = shutil.which("rhubarb")
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    return None


def split_speech_segments(text: str, max_chars: int = 220) -> list[str]:
    """Keep nearby clauses and sentences together to preserve natural prosody."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    units: list[str] = []
    for sentence in sentences:
        units.extend(re.split(r"(?<=[,;:])\s+", sentence.strip()))

    segments: list[str] = []
    current = ""
    for unit in units:
        pieces: list[str] = []
        if len(unit) > max_chars:
            piece = ""
            for word in unit.split():
                candidate = f"{piece} {word}".strip()
                if piece and len(candidate) > max_chars:
                    pieces.append(piece)
                    piece = word
                else:
                    piece = candidate
            if piece:
                pieces.append(piece)
        else:
            pieces.append(unit)

        for piece in pieces:
            candidate = f"{current} {piece}".strip()
            if current and len(candidate) > max_chars:
                segments.append(current)
                current = piece
            else:
                current = candidate
    if current:
        segments.append(current)
    return segments or [text.strip()]


def request_tts_chunk(text: str, voice: str, rate: float, api_key: str) -> tuple[bytes, float]:
    request_body = {
        "model": MODEL,
        "voice": voice,
        "input": text,
        "instructions": TTS_INSTRUCTIONS,
        "response_format": "wav",
        "speed": rate,
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/audio/speech",
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "audio/wav",
            "User-Agent": "headroom-local-demo/1.0",
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=TTS_TIMEOUT_SECONDS) as response:
            audio = response.read()
    except urllib.error.HTTPError as error:
        try:
            body = json.loads(error.read().decode("utf-8", "replace"))
            message = body.get("error", {}).get("message", "")
        except (json.JSONDecodeError, AttributeError):
            message = ""
        raise RuntimeError(message or f"OpenAI returned HTTP {error.code}.") from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise RuntimeError(f"Could not reach OpenAI TTS: {error}") from error
    return audio, (time.perf_counter() - started) * 1000


def analyze_rhubarb_chunk(audio: bytes, text: str, rhubarb: Path) -> tuple[list[dict], float, float]:
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="headroom-speech-") as temp_dir:
            directory = Path(temp_dir)
            wav_path = directory / "speech.wav"
            dialog_path = directory / "dialog.txt"
            wav_path.write_bytes(audio)
            dialog_path.write_text(text, encoding="utf-8")
            command = [
                str(rhubarb), "--quiet", "--exportFormat", "json",
                "--recognizer", RHUBARB_RECOGNIZER, "--dialogFile", str(dialog_path),
                str(wav_path),
            ]
            result = subprocess.run(
                command,
                cwd=str(rhubarb.parent),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=LIPSYNC_TIMEOUT_SECONDS,
                check=False,
            )
            if result.returncode != 0:
                detail = result.stderr.strip()[-1200:] or "Rhubarb could not analyze the speech audio."
                raise RuntimeError(detail)
            analysis = json.loads(result.stdout)
            cues = analysis.get("mouthCues", [])
            duration = float(analysis.get("metadata", {}).get("duration", 0))
            if duration <= 0 or not cues:
                raise RuntimeError("Rhubarb returned invalid or empty mouth timing for this audio.")
            if not any(cue.get("value") != "X" for cue in cues):
                raise RuntimeError("Rhubarb could not detect spoken mouth shapes in the audio.")
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("Lip-sync analysis timed out. Try a shorter line.") from error
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise RuntimeError(f"Could not analyze the generated audio: {error}") from error
    return cues, duration, (time.perf_counter() - started) * 1000


class Handler(SimpleHTTPRequestHandler):
    server_version = "Headroom/1.0"

    @staticmethod
    def _client_version() -> int:
        try:
            return int((ROOT / "index.html").stat().st_mtime)
        except OSError:
            return 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        # Avoid noisy health polling and never log client IPs, query strings,
        # request bodies, audio, prompts, or credentials.
        path = urlparse(self.path).path
        if path in {"/api/status", "/api/conversation/status", "/api/conversation/reply-stream"}:
            return
        print(f"[http] {self.command} {path}", file=sys.stderr, flush=True)

    def end_headers(self) -> None:
        # The app's JavaScript is inline in index.html. Avoid a stale cached
        # page keeping old visitor-polling or playback logic after updates.
        if urlparse(self.path).path in {"/", "/index.html"}:
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
        super().end_headers()

    def _json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    @staticmethod
    def _is_public_static_path(path: str) -> bool:
        if path in {"/index.html", "/ATTRIBUTION.md", "/LICENSE"}:
            return True
        if not path.startswith("/assets/") or not path.lower().endswith(".glb"):
            return False
        return not any(part.startswith(".") for part in path.split("/") if part)

    def _normalized_request_path(self) -> str:
        decoded_path = urlparse(self.path).path
        for _ in range(8):
            unquoted_path = unquote(decoded_path)
            if unquoted_path == decoded_path:
                break
            decoded_path = unquoted_path
        else:
            return ""
        return posixpath.normpath("/" + decoded_path.lstrip("/"))

    def do_GET(self) -> None:
        requested_path = urlparse(self.path).path
        normalized_path = self._normalized_request_path()
        path = requested_path
        if path == "/api/conversation/status":
            manager = getattr(self.server, "conversation_manager", None)
            status = manager.status() if manager else {"active": False, "mode": "unavailable"}
            heartbeat = getattr(self.server, "display_heartbeat", None)
            if heartbeat is not None:
                heartbeat.record()
                status["display"] = heartbeat.snapshot()
            self._json(status, 200)
            return
        if path == "/api/conversation/reply-stream":
            query = parse_qs(urlparse(self.path).query)
            try:
                reply_id = int(query.get("replyId", [""])[0])
            except (TypeError, ValueError):
                self._json({"error": "Invalid reply id."}, 400)
                return
            self._stream_reply_events(reply_id)
            return
        if path == "/api/status":
            rhubarb = find_rhubarb()
            laptop_audio = laptop_audio_status()
            self._json(
                {
                    "openaiAvailable": bool(os.getenv("OPENAI_API_KEY")),
                    "clientVersion": self._client_version(),
                    "rhubarbAvailable": bool(rhubarb),
                    "laptopAudioAvailable": laptop_audio["available"],
                    "laptopAudioSink": laptop_audio["sink"],
                    "rhubarbVersion": "1.14.0" if rhubarb else None,
                    "rhubarbRecognizer": RHUBARB_RECOGNIZER,
                    "model": MODEL,
                    "voices": list(VOICE_NAMES),
                    "defaultVoice": DEFAULT_VOICE if DEFAULT_VOICE in VOICE_NAMES else "nova",
                }
            )
            return
        static_path = "/index.html" if path == "/" else normalized_path
        if not self._is_public_static_path(static_path):
            self.send_error(404)
            return
        query = urlparse(self.path).query
        self.path = static_path + (f"?{query}" if query else "")
        super().do_GET()

    def do_HEAD(self) -> None:
        normalized_path = self._normalized_request_path()
        static_path = "/index.html" if urlparse(self.path).path == "/" else normalized_path
        if not self._is_public_static_path(static_path):
            self.send_error(404)
            return
        query = urlparse(self.path).query
        self.path = static_path + (f"?{query}" if query else "")
        super().do_HEAD()

    def _stream_reply_events(self, reply_id: int) -> None:
        hub = getattr(self.server, "speech_event_hub", None)
        if reply_id < 1 or hub is None:
            self._json({"error": "Speech event stream is unavailable."}, 404 if reply_id < 1 else 503)
            return
        manager = getattr(self.server, "conversation_manager", None)
        pending = manager.status().get("reply") if manager is not None else None
        if not hub.knows(reply_id) and not (isinstance(pending, dict) and pending.get("id") == reply_id):
            self._json({"error": "No animation stream is available for that reply."}, 404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        # Do not replay stale mouth motion after a display reconnects once audio
        # has already finished; laptop playback is independent of this stream.
        if hub.completed(reply_id):
            events = (
                ({"type": "error", "error": "Laptop speech playback failed; check the server log.", "replyId": reply_id},)
                if hub.failed(reply_id) else (
                    {"type": "start", "segments": 0, "replyId": reply_id},
                    {"type": "audio-complete", "replyId": reply_id},
                    {"type": "done", "segments": 0, "replyId": reply_id},
                )
            )
            for event in events:
                try:
                    self._stream_event(event)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
            return
        cursor = 0
        while True:
            events, completed = hub.wait(reply_id, cursor, timeout=10)
            try:
                if events:
                    for event in events:
                        self._stream_event(event)
                        cursor = event["sequence"]
                        if event.get("type") in {"done", "error"}:
                            return
                elif completed:
                    return
                else:
                    self._stream_event({"type": "heartbeat", "replyId": reply_id})
            except (BrokenPipeError, ConnectionResetError, OSError):
                logging.info("display disconnected from lip-sync events for reply id=%d; laptop playback continues", reply_id)
                return

    def _stream_event(self, event: dict) -> None:
        data = (json.dumps(event, separators=(",", ":")) + "\n").encode("utf-8")
        self.wfile.write(data)
        self.wfile.flush()

    def _stream_tts_pcm(self, text: str, voice: str, rate: float, api_key: str, index: int, count: int, player: LaptopAudioPlayer, emit) -> tuple[bytes, float, float, float, int, dict[str, int]]:
        request_body = {
            "model": MODEL,
            "voice": voice,
            "input": text,
            "instructions": TTS_INSTRUCTIONS,
            "response_format": "pcm",
            "stream_format": "audio",
            "speed": rate,
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/audio/speech",
            data=json.dumps(request_body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "audio/pcm",
                "User-Agent": "headroom-local-demo/1.0",
            },
            method="POST",
        )
        started = time.perf_counter()
        first_byte_ms = None
        pcm = bytearray()
        pending = bytearray()
        chunk_index = 0
        envelope_total_ms = 0.0
        start_buffer: list[bytes] = []
        start_buffer_bytes = 0
        playback_started = False
        playback_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=PCM_PLAYBACK_QUEUE_PACKETS)
        playback_errors: list[Exception] = []
        playback_thread: threading.Thread | None = None
        queue_high_water = 0
        producer_blocked_ms = 0.0

        def send_envelope(packet: bytes) -> None:
            nonlocal chunk_index, envelope_total_ms
            player.write(packet)
            envelope_started = time.perf_counter()
            envelope = jaw_envelope(packet)
            envelope_ms = (time.perf_counter() - envelope_started) * 1000
            envelope_total_ms += envelope_ms
            emit({
                "type": "envelope",
                "index": index,
                "chunkIndex": chunk_index,
                "segments": count,
                "voice": voice,
                "recognizer": RHUBARB_RECOGNIZER,
                "jawEnvelope": envelope,
                "envelopeStep": 0.01,
                "duration": len(packet) / (PCM_SAMPLE_RATE * 2),
                "playbackLeadMs": PULSE_OUTPUT_LATENCY_MS if chunk_index == 0 else None,
                "openaiTtfbMs": round(first_byte_ms or 0) if chunk_index == 0 else None,
                "envelopeMs": round(envelope_ms),
            })
            chunk_index += 1

        def enqueue_for_playback(packet: bytes | None) -> None:
            nonlocal queue_high_water, producer_blocked_ms
            started = time.perf_counter()
            while True:
                if playback_errors:
                    raise playback_errors[0]
                try:
                    playback_queue.put(packet, timeout=0.1)
                    producer_blocked_ms += max(0.0, (time.perf_counter() - started) * 1000 - 1)
                    queue_high_water = max(queue_high_water, playback_queue.qsize())
                    return
                except queue.Full:
                    continue

        def play_queued_audio() -> None:
            try:
                while True:
                    packet = playback_queue.get()
                    if packet is None:
                        return
                    send_envelope(packet)
            except Exception as error:
                playback_errors.append(error)
                player.stop()

        def queue_for_playback(packet: bytes) -> None:
            nonlocal playback_started, start_buffer_bytes
            if playback_started:
                enqueue_for_playback(packet)
                return
            start_buffer.append(packet)
            start_buffer_bytes += len(packet)
            if start_buffer_bytes >= PCM_PACKET_BYTES * PCM_START_BUFFER_PACKETS:
                playback_started = True
                buffered_packets = start_buffer[:]
                start_buffer.clear()
                start_buffer_bytes = 0
                for buffered_packet in buffered_packets:
                    enqueue_for_playback(buffered_packet)

        try:
            response = urllib.request.urlopen(request, timeout=TTS_TIMEOUT_SECONDS)
        except urllib.error.HTTPError as error:
            logging.warning("OpenAI streamed TTS returned HTTP %d", error.code)
            try:
                body = json.loads(error.read().decode("utf-8", "replace"))
                message = body.get("error", {}).get("message", "")
            except (json.JSONDecodeError, AttributeError):
                message = ""
            raise RuntimeError(message or f"OpenAI returned HTTP {error.code}.") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            logging.warning("OpenAI streamed TTS connection failed (%s)", type(error).__name__)
            raise RuntimeError(f"Could not reach OpenAI TTS: {error}") from error

        playback_thread = threading.Thread(
            target=play_queued_audio,
            name=f"headroom-pcm-playback-{index}",
            daemon=True,
        )
        playback_thread.start()
        try:
            with response:
                reader = response.read1 if hasattr(response, "read1") else response.read
                while True:
                    data = reader(8192)
                    if not data:
                        break
                    if first_byte_ms is None:
                        first_byte_ms = (time.perf_counter() - started) * 1000
                    pcm.extend(data)
                    pending.extend(data)
                    while len(pending) >= PCM_PACKET_BYTES:
                        packet = bytes(pending[:PCM_PACKET_BYTES])
                        del pending[:PCM_PACKET_BYTES]
                        queue_for_playback(packet)
                if len(pending) % 2:
                    raise RuntimeError("OpenAI returned a misaligned PCM stream.")
                if pending:
                    queue_for_playback(bytes(pending))
                if start_buffer:
                    buffered_packets = start_buffer[:]
                    start_buffer.clear()
                    start_buffer_bytes = 0
                    playback_started = True
                    for buffered_packet in buffered_packets:
                        enqueue_for_playback(buffered_packet)
            if not pcm or first_byte_ms is None:
                raise RuntimeError("OpenAI returned an empty audio stream.")
            if len(pcm) % 2:
                raise RuntimeError("OpenAI returned a misaligned PCM stream.")
            stream_duration_ms = (time.perf_counter() - started) * 1000
            enqueue_for_playback(None)
            playback_thread.join(timeout=180)
            if playback_thread.is_alive():
                raise RuntimeError("Buffered laptop audio did not finish before the timeout.")
            if playback_errors:
                raise playback_errors[0]
        except urllib.error.URLError as error:
            player.stop()
            if playback_thread.is_alive():
                try:
                    playback_queue.put(None, timeout=0.5)
                except queue.Full:
                    pass
                playback_thread.join(timeout=2)
            raise RuntimeError(f"OpenAI audio stream failed: {error}") from error
        except Exception:
            player.stop()
            if playback_thread.is_alive():
                try:
                    playback_queue.put(None, timeout=0.5)
                except queue.Full:
                    pass
                playback_thread.join(timeout=2)
            raise
        queue_metrics = {
            "highWaterPackets": queue_high_water,
            "producerBlockedMs": round(producer_blocked_ms),
        }
        return bytes(pcm), first_byte_ms, stream_duration_ms, envelope_total_ms, chunk_index, queue_metrics

    def _stream_speech(
        self,
        text: str,
        voice: str,
        rate: float,
        api_key: str,
        rhubarb: Path,
        reply_id: int | None = None,
        manager: ConversationManager | None = None,
    ) -> None:
        segments = split_speech_segments(text)
        try:
            player = LaptopAudioPlayer()
        except RuntimeError as error:
            logging.error("laptop audio playback unavailable (%s)", error)
            self._json({"error": str(error)}, 503)
            return
        logging.info("streamed speech started on laptop: voice=%s rate=%.2f segments=%d", voice, rate, len(segments))
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store, no-transform")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()
        totals = {"openaiTts": 0.0, "openaiTtfb": 0.0, "rhubarb": 0.0, "jawEnvelope": 0.0}
        pcm_queue_high_water = 0
        pcm_producer_blocked_ms = 0
        event_queue: queue.SimpleQueue[tuple[dict, float] | None] = queue.SimpleQueue()
        writer_failed = threading.Event()
        playback_finished = False
        playback_started = False
        playback_lock = threading.Lock()

        try:
            self.connection.settimeout(2.0)
        except OSError:
            pass

        def write_events() -> None:
            while True:
                item = event_queue.get()
                if item is None:
                    return
                event, queued_at = item
                if writer_failed.is_set():
                    continue
                write_started = time.perf_counter()
                try:
                    self._stream_event(event)
                except (BrokenPipeError, ConnectionResetError, OSError) as error:
                    writer_failed.set()
                    player.stop()
                    logging.info("speech event consumer stopped (%s); stopping laptop playback", type(error).__name__)
                    return
                if event.get("type") in {"audio-complete", "done"}:
                    logging.info(
                        "speech stream event delivered: type=%s queue_wait_ms=%d write_ms=%d",
                        event["type"],
                        round((write_started - queued_at) * 1000),
                        round((time.perf_counter() - write_started) * 1000),
                    )

        writer_thread = threading.Thread(target=write_events, name="headroom-speech-event-writer", daemon=True)
        writer_thread.start()

        def emit(event: dict) -> None:
            nonlocal playback_started
            if event.get("type") == "envelope" and reply_id is not None and manager is not None:
                with playback_lock:
                    if not playback_started:
                        manager.set_playback(True, reply_id)
                        playback_started = True
            if not writer_failed.is_set():
                event_queue.put((event, time.perf_counter()))

        def analyze_and_emit(index: int, wav: bytes, segment_text: str, ttfb_ms: float, tts_ms: float, envelope_ms: float, chunks: int):
            try:
                cues, duration, rhubarb_ms = analyze_rhubarb_chunk(wav, segment_text, rhubarb)
                timings = {
                    "openaiTtfb": round(ttfb_ms),
                    "openaiTts": round(tts_ms),
                    "rhubarb": round(rhubarb_ms),
                    "jawEnvelope": round(envelope_ms),
                }
                emit({
                    "type": "cues",
                    "index": index,
                    "recognizer": RHUBARB_RECOGNIZER,
                    "cues": cues,
                    "duration": duration,
                    "timingsMs": timings,
                })
                print(
                    f"[speech timing] segment={index + 1}/{len(segments)} recognizer={RHUBARB_RECOGNIZER} "
                    + " ".join(f"{name}={value}ms" for name, value in timings.items())
                    + f" audioChunks={chunks}",
                    flush=True,
                )
                return timings
            except Exception as error:
                logging.error("Rhubarb analysis failed for speech segment %d (%s)", index + 1, type(error).__name__)
                try:
                    emit({"type": "error", "error": str(error) or "Speech analysis failed."})
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                return {"error": str(error)}

        try:
            emit({"type": "start", "segments": len(segments), "voice": voice})
            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as analysis_pool:
                for index, segment in enumerate(segments):
                    pcm, ttfb_ms, tts_ms, envelope_ms, chunk_count, queue_metrics = self._stream_tts_pcm(
                        segment, voice, rate, api_key, index, len(segments), player, emit
                    )
                    pcm_queue_high_water = max(pcm_queue_high_water, queue_metrics["highWaterPackets"])
                    pcm_producer_blocked_ms += queue_metrics["producerBlockedMs"]
                    wav = pcm_to_wav(pcm)
                    futures.append(analysis_pool.submit(
                        analyze_and_emit, index, wav, segment, ttfb_ms, tts_ms, envelope_ms, chunk_count
                    ))
                player.finish()
                playback_finished = True
                audio_output = player.timing_metrics()
                audio_output.update({
                    "pcmQueueHighWaterPackets": pcm_queue_high_water,
                    "pcmProducerBlockedMs": pcm_producer_blocked_ms,
                })
                if reply_id is not None and manager is not None:
                    manager.set_playback(False, reply_id)
                emit({"type": "audio-complete", "audioOutput": audio_output})
                logging.info(
                    "laptop speech playback completed: voice=%s segments=%d late_packets=%d total_late_ms=%d max_late_ms=%d pcm_queue_high_water=%d producer_blocked_ms=%d",
                    voice, len(segments), audio_output["latePackets"], audio_output["totalLateMs"], audio_output["maxLateMs"],
                    audio_output["pcmQueueHighWaterPackets"], audio_output["pcmProducerBlockedMs"],
                )
                results = [future.result() for future in futures]
            if any("error" in result for result in results):
                return
            for result in results:
                totals["openaiTts"] += result["openaiTts"]
                totals["openaiTtfb"] += result["openaiTtfb"]
                totals["rhubarb"] += result["rhubarb"]
                totals["jawEnvelope"] += result["jawEnvelope"]
            emit({
                "type": "done",
                "segments": len(segments),
                "totalTimingsMs": {name: round(value) for name, value in totals.items()},
                "audioOutput": audio_output,
            })
        except (BrokenPipeError, ConnectionResetError):
            logging.info("speech event consumer disconnected before completion")
            return
        except Exception as error:
            logging.error("streamed laptop speech failed (%s)", type(error).__name__)
            try:
                emit({"type": "error", "error": str(error) or "Speech stream failed."})
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
        finally:
            if not playback_finished:
                player.stop()
            if reply_id is not None and manager is not None:
                manager.set_playback(False, reply_id)
            event_queue.put(None)
            writer_thread.join(timeout=2)

    def _conversation_post(self, path: str) -> None:
        manager = getattr(self.server, "conversation_manager", None)
        if manager is None:
            self._json({"error": "Conversation service is unavailable."}, 503)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length > 4096:
            self._json({"error": "Conversation request is too large."}, 413)
            return
        payload: dict = {}
        if length:
            try:
                parsed = json.loads(self.rfile.read(length))
                if not isinstance(parsed, dict):
                    raise ValueError("Expected a JSON object.")
                payload = parsed
            except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
                self._json({"error": "Invalid conversation request."}, 400)
                return
        if path == "/api/conversation/start":
            voice = str(payload.get("voice", DEFAULT_VOICE))
            try:
                rate = float(payload.get("rate", DEFAULT_RATE))
            except (TypeError, ValueError):
                self._json({"error": "Invalid speech speed."}, 400)
                return
            if voice not in VOICE_NAMES or not 0.65 <= rate <= 1.25:
                self._json({"error": "Choose a supported voice and speech speed."}, 400)
                return
            self.server.conversation_voice = voice
            self.server.conversation_rate = rate
            result = manager.start()
        elif path == "/api/conversation/stop":
            result = manager.stop()
        elif path == "/api/conversation/greet":
            result = manager.greet_now()
        elif path == "/api/conversation/playback":
            reply_id = payload.get("replyId")
            if reply_id is not None and not isinstance(reply_id, int):
                self._json({"error": "Invalid reply id."}, 400)
                return
            result = manager.set_playback(bool(payload.get("active")), reply_id)
        else:
            self._json({"error": "Not found."}, 404)
            return
        self._json(result, 503 if result.get("error") else 200)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path.startswith("/api/conversation/"):
            self._conversation_post(path)
            return
        if path not in {"/api/speak", "/api/speak-stream"}:
            self._json({"error": "Not found."}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length < 1 or length > 12_000:
            self._json({"error": "Request body is empty or too large."}, 413)
            return
        try:
            payload = json.loads(self.rfile.read(length))
            text = str(payload.get("text", "")).strip()
            voice = str(payload.get("voice", DEFAULT_VOICE))
            rate = float(payload.get("rate", DEFAULT_RATE))
            reply_id = payload.get("replyId")
        except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
            self._json({"error": "Invalid speech request."}, 400)
            return
        if reply_id is not None and (isinstance(reply_id, bool) or not isinstance(reply_id, int) or reply_id < 1):
            self._json({"error": "Invalid conversation reply id."}, 400)
            return
        if reply_id is not None and path != "/api/speak-stream":
            self._json({"error": "Conversation replies require the streamed speech endpoint."}, 400)
            return
        if not text:
            self._json({"error": "Enter something for her to say."}, 400)
            return
        if len(text) > MAX_TEXT_LENGTH:
            self._json({"error": f"Please keep the line under {MAX_TEXT_LENGTH} characters."}, 400)
            return
        if voice not in VOICE_NAMES:
            self._json({"error": "Choose a supported OpenAI voice."}, 400)
            return
        if not 0.65 <= rate <= 1.25:
            self._json({"error": "Speech speed is outside the allowed range."}, 400)
            return
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            self._json({"error": "No OpenAI key is available. Start with ./run.sh or set OPENAI_API_KEY."}, 503)
            return
        rhubarb = find_rhubarb()
        if not rhubarb:
            self._json({"error": "Rhubarb Lip Sync is missing. Run ./scripts/install_rhubarb.sh and restart."}, 503)
            return
        manager = getattr(self.server, "conversation_manager", None) if reply_id is not None else None
        if reply_id is not None and (manager is None or not manager.claim_reply_playback(reply_id)):
            self._json({"error": "This reply is already playing on another display or is no longer pending."}, 409)
            return
        if path == "/api/speak-stream":
            self._stream_speech(text, voice, rate, api_key, rhubarb, reply_id=reply_id, manager=manager)
            return

        request_body = {
            "model": MODEL,
            "voice": voice,
            "input": text,
            "instructions": TTS_INSTRUCTIONS,
            "response_format": "wav",
            "speed": rate,
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/audio/speech",
            data=json.dumps(request_body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "audio/wav",
                "User-Agent": "headroom-local-demo/1.0",
            },
            method="POST",
        )
        tts_started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=TTS_TIMEOUT_SECONDS) as response:
                audio = response.read()
            tts_ms = (time.perf_counter() - tts_started) * 1000
        except urllib.error.HTTPError as error:
            logging.warning("OpenAI speech request returned HTTP %d", error.code)
            try:
                body = json.loads(error.read().decode("utf-8", "replace"))
                message = body.get("error", {}).get("message", "")
            except (json.JSONDecodeError, AttributeError):
                message = ""
            message = message or f"OpenAI returned HTTP {error.code}."
            self._json({"error": message}, 502)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            logging.warning("OpenAI speech request failed (%s)", type(error).__name__)
            self._json({"error": f"Could not reach OpenAI TTS: {error}"}, 502)
            return
        normalization_started = time.perf_counter()
        try:
            audio = normalize_streamed_wav(audio)
            normalization_ms = (time.perf_counter() - normalization_started) * 1000
        except ValueError as error:
            logging.warning("OpenAI returned an invalid WAV response (%s)", type(error).__name__)
            self._json({"error": str(error)}, 502)
            return

        rhubarb_started = time.perf_counter()
        try:
            with tempfile.TemporaryDirectory(prefix="headroom-speech-") as temp_dir:
                directory = Path(temp_dir)
                wav_path = directory / "speech.wav"
                dialog_path = directory / "dialog.txt"
                wav_path.write_bytes(audio)
                dialog_path.write_text(text, encoding="utf-8")
                command = [
                    str(rhubarb), "--quiet", "--exportFormat", "json",
                    "--recognizer", RHUBARB_RECOGNIZER, "--dialogFile", str(dialog_path),
                    str(wav_path),
                ]
                result = subprocess.run(
                    command,
                    cwd=str(rhubarb.parent),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=LIPSYNC_TIMEOUT_SECONDS,
                    check=False,
                )
                if result.returncode != 0:
                    logging.warning("Rhubarb returned exit code %d", result.returncode)
                    detail = result.stderr.strip()[-1200:] or "Rhubarb could not analyze the speech audio."
                    self._json({"error": detail}, 502)
                    return
                analysis = json.loads(result.stdout)
                cues = analysis.get("mouthCues", [])
                duration = float(analysis.get("metadata", {}).get("duration", 0))
                if duration <= 0 or not cues:
                    self._json({"error": "Rhubarb returned invalid or empty mouth timing for this audio."}, 502)
                    return
                if not any(cue.get("value") != "X" for cue in cues):
                    self._json({"error": "Rhubarb could not detect spoken mouth shapes in the audio."}, 502)
                    return
        except subprocess.TimeoutExpired:
            logging.warning("Rhubarb lip-sync analysis timed out")
            self._json({"error": "Lip-sync analysis timed out. Try a shorter line."}, 504)
            return
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            logging.error("Rhubarb response could not be processed (%s)", type(error).__name__)
            self._json({"error": f"Could not analyze the generated audio: {error}"}, 502)
            return

        rhubarb_ms = (time.perf_counter() - rhubarb_started) * 1000
        playback_started = time.perf_counter()
        try:
            play_laptop_wav(audio)
        except RuntimeError as error:
            logging.error("laptop audio playback failed (%s)", type(error).__name__)
            self._json({"error": str(error)}, 503)
            return
        playback_ms = (time.perf_counter() - playback_started) * 1000
        timings_ms = {
            "openaiTts": round(tts_ms),
            "wavNormalize": round(normalization_ms),
            "rhubarb": round(rhubarb_ms),
            "laptopPlayback": round(playback_ms),
        }
        print(
            f"[speech timing] recognizer={RHUBARB_RECOGNIZER} "
            + " ".join(f"{name}={value}ms" for name, value in timings_ms.items()),
            flush=True,
        )
        self._json(
            {
                "cues": cues,
                "duration": duration,
                "voice": voice,
                "timingsMs": timings_ms,
                "recognizer": RHUBARB_RECOGNIZER,
            }
        )


def _handle_termination_signal(signum: int, _frame) -> None:
    raise KeyboardInterrupt(f"Received signal {signum}")


def main() -> None:
    configure_diagnostics_log()
    # Ensure launcher restarts and service managers both get a clean shutdown,
    # including when Python was started as a background process.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, _handle_termination_signal)
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8766"))
    server = ThreadingHTTPServer((host, port), Handler)
    # Let request threads finish their TTS/Rhubarb child work before process
    # exit so Ctrl+C/restarts don't strand subprocesses or partial streams.
    server.daemon_threads = False
    server.block_on_close = True
    server.conversation_manager = ConversationManager()
    server.speech_event_hub = SpeechEventHub()
    server.display_heartbeat = DisplayHeartbeat()
    server.conversation_voice = DEFAULT_VOICE if DEFAULT_VOICE in VOICE_NAMES else "nova"
    server.conversation_rate = DEFAULT_RATE
    speech_dispatcher = ReplySpeechDispatcher(server, server.speech_event_hub)
    diagnostics = RuntimeDiagnostics()
    speech_dispatcher.start()
    rhubarb_path = find_rhubarb()
    laptop_audio = laptop_audio_status()
    logging.info(
        "server listening: host=%s port=%d; OpenAI credential=%s; Rhubarb=%s; laptop audio=%s (%s)",
        host, port, "available" if os.getenv("OPENAI_API_KEY") else "missing",
        str(rhubarb_path) if rhubarb_path else "missing",
        "available" if laptop_audio["available"] else "unavailable", laptop_audio["sink"],
    )
    print(f"Headroom local URL: http://127.0.0.1:{port} (bind {host}; Ctrl+C to stop)", flush=True)
    print(f"Rhubarb recognizer: {RHUBARB_RECOGNIZER}", flush=True)
    if not os.getenv("OPENAI_API_KEY"):
        print("No OpenAI credential found. Start with ./run.sh or set OPENAI_API_KEY.", flush=True)
    if not find_rhubarb():
        print("Rhubarb is missing. Install it with ./scripts/install_rhubarb.sh.", flush=True)
    if not laptop_audio["available"]:
        print(f"Laptop audio output unavailable: {laptop_audio['sink']}.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("server shutdown requested")
        print("\nShutting down Headroom.", flush=True)
    finally:
        speech_dispatcher.stop()
        server.conversation_manager.stop()
        diagnostics.stop()
        server.server_close()
        logging.info("server stopped cleanly")


if __name__ == "__main__":
    main()
