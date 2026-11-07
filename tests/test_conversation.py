import json
import io
import os
import pathlib
import struct
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest.mock import patch

import numpy as np

import conversation
from conversation import ConversationManager, choose_microphone_index, response_text, save_visitor_transcript
from person_vision import (
    LocalPersonVision,
    PERSON_DETECTOR_ENV,
    PERSON_PRESENCE_GRACE_SECONDS,
    RECURRENT_CROP_INTERVAL_SECONDS,
    PersonDetection,
    PersonTrackGate,
    crop_person,
)
from server import (
    DisplayHeartbeat, Handler, PCM_PACKET_BYTES, PULSE_OUTPUT_LATENCY_MS,
    ReplySpeechDispatcher, SpeechEventHub, jaw_envelope, split_speech_segments,
)


class SpeechEventHubTests(unittest.TestCase):
    def test_server_dispatcher_relays_laptop_events_without_a_display_claim(self):
        events = b'{"type":"start","segments":1}\n{"type":"done","segments":1}\n'
        fake_server = type("ServerStub", (), {"server_port": 8766, "conversation_voice": "nova", "conversation_rate": 0.94})()
        hub = SpeechEventHub()
        dispatcher = ReplySpeechDispatcher(fake_server, hub)
        with patch("urllib.request.urlopen", return_value=io.BytesIO(events)):
            dispatcher._dispatch({"id": 4, "text": "Hello there."})
        collected, completed = hub.wait(4, 0, timeout=0.01)
        self.assertTrue(completed)
        self.assertEqual([event["type"] for event in collected], ["start", "done"])
        self.assertEqual([event["replyId"] for event in collected], [4, 4])

    def test_events_are_ordered_and_completion_is_retained(self):
        hub = SpeechEventHub()
        hub.publish(17, {"type": "start"})
        hub.publish(17, {"type": "envelope", "jawEnvelope": [0.2]})
        events, completed = hub.wait(17, 0, timeout=0.01)
        self.assertFalse(completed)
        self.assertEqual([event["sequence"] for event in events], [1, 2])
        self.assertEqual(events[-1]["replyId"], 17)

        hub.publish(17, {"type": "done"})
        events, completed = hub.wait(17, 2, timeout=0.01)
        self.assertTrue(completed)
        self.assertEqual([event["type"] for event in events], ["done"])
        self.assertTrue(hub.completed(17))

    def test_event_history_is_bounded(self):
        hub = SpeechEventHub(max_events_per_reply=2)
        for index in range(4):
            hub.publish(1, {"type": "envelope", "index": index})
        events, _ = hub.wait(1, 0, timeout=0.01)
        self.assertEqual([event["index"] for event in events], [2, 3])


class DisplayHeartbeatTests(unittest.TestCase):
    def test_heartbeat_reports_poll_count_and_age(self):
        heartbeat = DisplayHeartbeat()
        first = heartbeat.record()
        second = heartbeat.record()
        self.assertEqual(first["pollCount"], 1)
        self.assertEqual(second["pollCount"], 2)
        self.assertIsNotNone(second["gapSeconds"])
        self.assertEqual(heartbeat.snapshot()["pollCount"], 2)
        self.assertLessEqual(heartbeat.snapshot()["ageSeconds"], 1)


class StaticPageCacheTests(unittest.TestCase):
    def test_conversation_status_exposes_display_poll_heartbeat(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.display_heartbeat = DisplayHeartbeat()
        server.conversation_manager = type("StatusStub", (), {"status": lambda _self: {"active": True, "mode": "watching"}})()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/api/conversation/status", timeout=5) as response:
                status = json.loads(response.read())
            self.assertEqual(status["display"]["pollCount"], 1)
            self.assertEqual(status["mode"], "watching")
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_reconnected_display_does_not_replay_completed_mouth_motion(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.speech_event_hub = SpeechEventHub()
        server.speech_event_hub.publish(7, {"type": "start", "segments": 1})
        server.speech_event_hub.publish(7, {"type": "envelope", "jawEnvelope": [0.4]})
        server.speech_event_hub.publish(7, {"type": "done", "segments": 1})
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{server.server_port}/api/conversation/reply-stream?replyId=7", timeout=5
            ) as response:
                events = [json.loads(line) for line in response.read().splitlines()]
            self.assertEqual([event["type"] for event in events], ["start", "audio-complete", "done"])
            self.assertFalse(any("jawEnvelope" in event for event in events))
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_static_server_only_exposes_public_app_files(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urllib.request.urlopen(f"{base}/ATTRIBUTION.md", timeout=5) as response:
                self.assertIn(b"Model attribution", response.read(256))
            with urllib.request.urlopen(f"{base}/assets/mpfb.glb", timeout=5) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(response.read(4), b"glTF")
                response.read()
            with urllib.request.urlopen(f"{base}/peppers-ghost-demo.html", timeout=5) as response:
                demo_page = response.read().decode("utf-8")
                self.assertIn("Bobby in a Pepper's Ghost display", demo_page)
                self.assertIn("media/bobby-peppers-ghost-display.mp4", demo_page)
            video_head = urllib.request.Request(f"{base}/media/bobby-peppers-ghost-display.mp4", method="HEAD")
            with urllib.request.urlopen(video_head, timeout=5) as response:
                self.assertEqual(response.status, 200)
                self.assertTrue(response.headers.get("Content-Type", "").startswith("video/mp4"))

            private_paths = (
                "/.git/config", "/.env", "/server.py", "/README.md",
                "/scripts/start-headroom.ps1", "/logs/visitor_transcripts.jsonl",
                "/assets/%2e%2e/server.py",
            )
            for path in private_paths:
                with self.subTest(path=path), self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(f"{base}{path}", timeout=5)
                self.assertEqual(error.exception.code, 404)

            request = urllib.request.Request(f"{base}/.git/config", method="HEAD")
            with self.assertRaises(urllib.error.HTTPError) as head_error:
                urllib.request.urlopen(request, timeout=5)
            self.assertEqual(head_error.exception.code, 404)
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_client_version_matches_the_served_page_mtime(self):
        page_path = pathlib.Path(__file__).resolve().parents[1] / "index.html"
        self.assertEqual(Handler._client_version(), int(page_path.stat().st_mtime))

    def test_index_page_is_never_cached(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/", timeout=5) as response:
                response.read()
                self.assertIn("no-store", response.headers.get("Cache-Control", ""))
                self.assertEqual(response.headers.get("Pragma"), "no-cache")
                self.assertEqual(response.headers.get("Expires"), "0")
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()


class MicrophoneSelectionTests(unittest.TestCase):
    def test_prefers_logitech_brio_over_builtin_and_default(self):
        devices = [
            {"index": 0, "name": "PulseAudio Default", "maxInputChannels": 1},
            {"index": 1, "name": "Built-in Microphone", "maxInputChannels": 2},
            {"index": 2, "name": "Logitech BRIO USB Audio", "maxInputChannels": 1},
            {"index": 3, "name": "Logitech BRIO Speaker", "maxInputChannels": 0},
        ]
        self.assertEqual(choose_microphone_index(devices, 0), 2)

    def test_falls_back_to_integrated_input(self):
        devices = [
            {"index": 0, "name": "PulseAudio Default", "maxInputChannels": 1},
            {"index": 1, "name": "HP Wide Vision Integrated Mic", "maxInputChannels": 1},
        ]
        self.assertEqual(choose_microphone_index(devices, 0), 1)

    def test_falls_back_to_system_default(self):
        devices = [{"index": 0, "name": "pulse", "maxInputChannels": 1}]
        self.assertEqual(choose_microphone_index(devices, 0), 0)
        self.assertIsNone(choose_microphone_index([], None))


class ConversationDiagnosticsTests(unittest.TestCase):
    def test_confirmed_person_presence_wakes_an_idle_watch_once(self):
        manager = ConversationManager()
        manager._active = True
        manager._mode = "watching"

        manager._person_presence_detected()
        self.assertTrue(manager._motion_event.is_set())
        self.assertEqual(manager.status()["motionCount"], 1)
        manager._person_presence_detected()
        self.assertEqual(manager.status()["motionCount"], 1)

    def test_visible_visitor_gets_only_one_gentle_repeat_prompt_after_missed_speech(self):
        class FakeVision:
            has_recent_person = True

            @staticmethod
            def take_new_crops(_limit=4):
                return []

        manager = ConversationManager()
        manager._active = True
        manager._person_vision = FakeVision()
        listen_count = 0

        def listen_until_stopped(**_kwargs):
            nonlocal listen_count
            listen_count += 1
            if listen_count == 7:
                manager._stop_event.set()
            return ""

        with (
            patch.object(manager, "_ask_assistant", return_value="Welcome, visitor!"),
            patch.object(manager, "_listen_for_visitor", side_effect=listen_until_stopped),
            patch.object(manager, "_wait_for_playback", return_value=True),
        ):
            manager._run_interaction()

        self.assertEqual(manager._reply_counter, 2)  # Greeting plus one recovery line.
        self.assertEqual(manager._pending_reply["text"], conversation.SILENCE_RECOVERY_REPLY)
        self.assertEqual(manager._history[-1], {"role": "assistant", "content": conversation.SILENCE_RECOVERY_REPLY})

    def test_does_not_repeat_to_an_empty_space_after_missed_speech(self):
        class FakeVision:
            has_recent_person = False

            @staticmethod
            def take_new_crops(_limit=4):
                return []

        manager = ConversationManager()
        manager._active = True
        manager._person_vision = FakeVision()
        with (
            patch.object(manager, "_ask_assistant", return_value="Welcome, visitor!"),
            patch.object(manager, "_listen_for_visitor", side_effect=["", "", "", ""]),
            patch.object(manager, "_wait_for_playback", return_value=True),
        ):
            manager._run_interaction()

        self.assertEqual(manager._reply_counter, 1)  # Only the initial greeting.
        self.assertEqual(manager._history, [{"role": "assistant", "content": "Welcome, visitor!"}])

    def test_greeting_generation_runs_while_opening_listener_is_active(self):
        class FakeVision:
            @staticmethod
            def take_new_crops(_limit=4):
                return []

        manager = ConversationManager()
        manager._active = True
        manager._person_vision = FakeVision()
        listener_started = threading.Event()
        greeting_requested = threading.Event()

        def listen_until_cancelled(**kwargs):
            listen_cancel = kwargs.get("listen_cancel_event")
            if listen_cancel is not None:
                listener_started.set()
                self.assertTrue(greeting_requested.wait(timeout=1))
                self.assertTrue(listen_cancel.wait(timeout=1))
                return None
            manager._stop_event.set()
            return ""

        def request_greeting(text, _images):
            self.assertIn("no more than 12 words", text)
            greeting_requested.set()
            self.assertTrue(listener_started.wait(timeout=1))
            return "Happy Halloween!"

        with (
            patch.object(manager, "_ask_assistant", side_effect=request_greeting) as ask,
            patch.object(manager, "_listen_for_visitor", side_effect=listen_until_cancelled),
            patch.object(manager, "_wait_for_playback", return_value=True),
        ):
            manager._run_interaction()

        ask.assert_called_once()
        self.assertEqual(manager._reply_counter, 1)
        self.assertEqual(manager._pending_reply["text"], "Happy Halloween!")
        self.assertEqual(manager._history, [{"role": "assistant", "content": "Happy Halloween!"}])

    def test_speech_during_greeting_generation_preempts_the_unheard_greeting(self):
        class FakeVision:
            @staticmethod
            def take_new_crops(_limit=4):
                return []

        manager = ConversationManager()
        manager._active = True
        manager._person_vision = FakeVision()
        listener_started = threading.Event()
        visitor_response_started = threading.Event()
        greeting_request_finished = threading.Event()
        listen_count = 0
        assistant_inputs = []

        def listen_for_arrival(**kwargs):
            nonlocal listen_count
            listen_count += 1
            if kwargs.get("listen_cancel_event") is not None:
                listener_started.set()
                with kwargs["phrase_start_lock"]:
                    kwargs["phrase_started_event"].set()
                return "Trick or treat!"
            if listen_count == 1:
                return ""
            manager._stop_event.set()
            return ""

        def generate_response(text, _images):
            assistant_inputs.append(text)
            if "Give the visitor or group" in text:
                self.assertTrue(listener_started.wait(timeout=1))
                self.assertTrue(visitor_response_started.wait(timeout=1))
                greeting_request_finished.set()
                return "Unused generated greeting."
            self.assertFalse(greeting_request_finished.is_set())
            visitor_response_started.set()
            return "Happy Halloween!"

        with (
            patch.object(manager, "_ask_assistant", side_effect=generate_response),
            patch.object(manager, "_listen_for_visitor", side_effect=listen_for_arrival),
            patch.object(manager, "_wait_for_playback", return_value=True),
            patch.object(conversation, "save_visitor_transcript"),
        ):
            manager._run_interaction()

        self.assertTrue(greeting_request_finished.wait(timeout=1))
        self.assertEqual(manager._reply_counter, 1)
        self.assertEqual(manager._pending_reply["text"], "Happy Halloween!")
        self.assertIn("Trick or treat!", assistant_inputs)
        self.assertEqual(manager._history, [
            {"role": "user", "content": "Trick or treat!"},
            {"role": "assistant", "content": "Happy Halloween!"},
        ])

    def test_no_speech_during_greeting_generation_cancels_listener_before_playback(self):
        class FakeVision:
            @staticmethod
            def take_new_crops(_limit=4):
                return []

        manager = ConversationManager()
        manager._active = True
        manager._person_vision = FakeVision()
        cancel_observed = threading.Event()
        listen_count = 0

        def listen_until_cancelled(**kwargs):
            nonlocal listen_count
            listen_count += 1
            listen_cancel = kwargs.get("listen_cancel_event")
            if listen_cancel is None:
                if listen_count > 1:
                    manager._stop_event.set()
                return ""
            self.assertTrue(listen_cancel.wait(timeout=1))
            cancel_observed.set()
            return None

        with (
            patch.object(manager, "_ask_assistant", return_value="Welcome, visitor!"),
            patch.object(manager, "_listen_for_visitor", side_effect=listen_until_cancelled),
            patch.object(manager, "_wait_for_playback", return_value=True),
        ):
            manager._run_interaction()

        self.assertTrue(cancel_observed.is_set())
        self.assertEqual(manager._reply_counter, 1)
        self.assertEqual(manager._pending_reply["text"], "Welcome, visitor!")

    def test_only_one_display_can_claim_each_reply(self):
        manager = ConversationManager()
        manager._active = True
        self.assertTrue(manager._queue_reply("First reply"))
        first_reply_id = manager.status()["reply"]["id"]
        self.assertTrue(manager.claim_reply_playback(first_reply_id))
        with self.assertLogs("headroom.conversation", level="WARNING"):
            self.assertFalse(manager.claim_reply_playback(first_reply_id))

        self.assertTrue(manager._queue_reply("Next reply"))
        next_reply_id = manager.status()["reply"]["id"]
        self.assertGreater(next_reply_id, first_reply_id)
        self.assertTrue(manager.claim_reply_playback(next_reply_id))

    def test_server_playback_completion_cannot_be_reopened_by_late_display_events(self):
        manager = ConversationManager()
        manager._active = True
        self.assertTrue(manager._queue_reply("Reply"))
        reply_id = manager.status()["reply"]["id"]

        manager.set_playback(True, reply_id)
        self.assertTrue(manager._playback_started.is_set())
        self.assertFalse(manager._playback_finished.is_set())
        manager.set_playback(False, reply_id)
        self.assertTrue(manager._playback_finished.is_set())

        manager.set_playback(True, reply_id)
        manager.set_playback(False, reply_id)
        self.assertTrue(manager._playback_finished.is_set())

    def test_status_reports_speech_events_without_exposing_transcript(self):
        manager = ConversationManager()
        manager._last_heard = "private visitor words"

        status = manager.status()

        self.assertNotIn("lastHeard", status)
        self.assertNotIn("private visitor words", repr(status))
        self.assertEqual(status["motionCount"], 0)
        self.assertEqual(status["manualGreetingCount"], 0)
        self.assertEqual(status["heardCount"], 0)

    def test_vision_snapshots_are_saved_privately_by_default(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = pathlib.Path(temporary_directory) / "vision_frames"
            manager = ConversationManager()
            with patch.object(conversation, "VISION_FRAME_DIRECTORY", output), patch.object(conversation, "SAVE_VISION_FRAMES", True):
                manager._save_vision_frame(b"fake-jpeg")

            status = manager.status()
            saved_image = output / status["lastVisionFrame"]
            self.assertTrue(saved_image.is_file())
            self.assertEqual(saved_image.read_bytes(), b"fake-jpeg")
            self.assertEqual(os.stat(output).st_mode & 0o777, 0o700)
            self.assertEqual(os.stat(saved_image).st_mode & 0o777, 0o600)
            self.assertEqual(status["savedVisionFrameCount"], 1)

    def test_visitor_transcripts_are_saved_privately_as_jsonl(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = pathlib.Path(temporary_directory) / "logs" / "visitor_transcripts.jsonl"
            with patch.object(conversation, "TRANSCRIPT_LOG_PATH", output), patch.object(conversation, "SAVE_VISITOR_TRANSCRIPTS", True):
                save_visitor_transcript("Yes, I can hear you — Bobby!", 2)

            record = json.loads(output.read_text(encoding="utf-8").strip())
            self.assertEqual(record["turn"], 2)
            self.assertEqual(record["text"], "Yes, I can hear you — Bobby!")
            self.assertIn("timestamp", record)
            self.assertEqual(os.stat(output.parent).st_mode & 0o777, 0o700)
            self.assertEqual(os.stat(output).st_mode & 0o777, 0o600)

    def test_visitor_transcript_saving_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = pathlib.Path(temporary_directory) / "visitor_transcripts.jsonl"
            with patch.object(conversation, "TRANSCRIPT_LOG_PATH", output), patch.object(conversation, "SAVE_VISITOR_TRANSCRIPTS", False):
                save_visitor_transcript("private visitor words", 1)
            self.assertFalse(output.exists())

    def test_visitor_transcript_log_rotates_with_private_permissions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = pathlib.Path(temporary_directory) / "visitor_transcripts.jsonl"
            output.write_text("previous transcript\n", encoding="utf-8")
            backup = output.with_name(output.name + ".1")
            with patch.object(conversation, "TRANSCRIPT_LOG_PATH", output), patch.object(conversation, "SAVE_VISITOR_TRANSCRIPTS", True), patch.object(conversation, "MAX_TRANSCRIPT_LOG_BYTES", 1):
                save_visitor_transcript("new transcript", 1)
            self.assertTrue(backup.is_file())
            self.assertEqual(backup.read_text(encoding="utf-8"), "previous transcript\n")
            self.assertEqual(os.stat(output).st_mode & 0o777, 0o600)
            self.assertEqual(os.stat(backup).st_mode & 0o777, 0o600)

    def test_vision_frame_saving_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = pathlib.Path(temporary_directory) / "vision_frames"
            manager = ConversationManager()
            with patch.object(conversation, "VISION_FRAME_DIRECTORY", output), patch.object(conversation, "SAVE_VISION_FRAMES", False):
                manager._save_vision_frame(b"fake-jpeg")
            self.assertFalse(output.exists())

    def test_vision_frame_retention_prunes_oldest_images(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = pathlib.Path(temporary_directory) / "vision_frames"
            output.mkdir()
            oldest = output / "headroom-visitor-20000101T000000.000000Z-0000.jpg"
            newer = output / "headroom-visitor-20010101T000000.000000Z-0000.jpg"
            oldest.write_bytes(b"old")
            newer.write_bytes(b"newer")
            manager = ConversationManager()
            with patch.object(conversation, "VISION_FRAME_DIRECTORY", output), patch.object(conversation, "SAVE_VISION_FRAMES", True), patch.object(conversation, "MAX_SAVED_VISION_FRAMES", 2):
                manager._save_vision_frame(b"current")
            remaining = sorted(path.name for path in output.glob("headroom-visitor-*.jpg"))
            self.assertEqual(len(remaining), 2)
            self.assertNotIn(oldest.name, remaining)
            self.assertIn(newer.name, remaining)


class SpeculativeTranscriptionTests(unittest.TestCase):
    def test_greeting_listener_uses_a_short_poll_timeout(self):
        listen_timeouts = []
        listen_cancel = threading.Event()

        class FakeWaitTimeoutError(Exception):
            pass

        class FakeRecognizer:
            def __init__(self):
                self.dynamic_energy_threshold = True
                self.pause_threshold = conversation.LISTEN_PAUSE_THRESHOLD_SECONDS
                self.phrase_threshold = conversation.MINIMUM_SPEECH_DURATION_SECONDS
                self.non_speaking_duration = 0.35

            def adjust_for_ambient_noise(self, *_args, **_kwargs):
                pass

            def listen(self, *_args, **kwargs):
                listen_timeouts.append(kwargs.get("timeout"))
                listen_cancel.set()

                def timed_out():
                    raise FakeWaitTimeoutError()
                    yield
                return timed_out()

        class FakeSource:
            SAMPLE_RATE = 16000
            SAMPLE_WIDTH = 2

        class FakeMicrophone:
            def __init__(self, **_kwargs):
                pass

            def __enter__(self):
                return FakeSource()

            def __exit__(self, *_args):
                return False

        class FakeSpeechRecognition:
            Recognizer = FakeRecognizer
            Microphone = FakeMicrophone
            WaitTimeoutError = FakeWaitTimeoutError

        manager = ConversationManager()
        manager._recognizer = FakeRecognizer()
        manager._recognizer_calibrated = True
        with patch.object(conversation, "sr", FakeSpeechRecognition):
            self.assertIsNone(
                manager._listen_for_visitor(
                    first_phrase_timeout_seconds=conversation.GREETING_GENERATION_LISTEN_POLL_SECONDS,
                    listen_cancel_event=listen_cancel,
                )
            )

        self.assertEqual(listen_timeouts, [conversation.GREETING_GENERATION_LISTEN_POLL_SECONDS])

    def test_cancelled_opening_listen_closes_the_microphone_without_transcription(self):
        capture_started = threading.Event()
        listen_cancel = threading.Event()

        class FakeWaitTimeoutError(Exception):
            pass

        class FakeRecognizer:
            def listen(self, *_args, **_kwargs):
                def wait_for_cancellation():
                    capture_started.set()
                    listen_cancel.wait(timeout=1)
                    raise FakeWaitTimeoutError()
                    yield
                return wait_for_cancellation()

        class FakeSource:
            SAMPLE_RATE = 16000
            SAMPLE_WIDTH = 2

        class FakeMicrophone:
            def __init__(self, **_kwargs):
                pass

            def __enter__(self):
                return FakeSource()

            def __exit__(self, *_args):
                return False

        class FakeSpeechRecognition:
            Microphone = FakeMicrophone
            WaitTimeoutError = FakeWaitTimeoutError

        manager = ConversationManager()
        manager._recognizer = FakeRecognizer()
        manager._recognizer_calibrated = True
        canceller = threading.Thread(target=lambda: (capture_started.wait(timeout=1), listen_cancel.set()))
        canceller.start()
        with patch.object(conversation, "sr", FakeSpeechRecognition):
            result = manager._listen_for_visitor(
                first_phrase_timeout_seconds=0.25,
                listen_cancel_event=listen_cancel,
            )
        canceller.join(timeout=1)

        self.assertTrue(capture_started.is_set())
        self.assertTrue(listen_cancel.is_set())
        self.assertIsNone(result)
        self.assertFalse(canceller.is_alive())

    def test_resumed_speech_invalidates_partial_transcript_and_retries_combined_audio(self):
        first_request_started = threading.Event()
        continuation_yielded = threading.Event()
        transcription_inputs = []
        listen_call_lock = threading.Lock()
        listen_call_count = 0

        class FakeWaitTimeoutError(Exception):
            pass

        class FakeAudioData:
            def __init__(self, frame_data, sample_rate, sample_width):
                self.frame_data = frame_data
                self.sample_rate = sample_rate
                self.sample_width = sample_width

            def get_wav_data(self, **_kwargs):
                return self.frame_data

        class FakeSource:
            SAMPLE_RATE = 16000
            SAMPLE_WIDTH = 2

        class FakeMicrophone:
            def __init__(self, **_kwargs):
                self.source = FakeSource()

            def __enter__(self):
                return self.source

            def __exit__(self, *_args):
                return False

        class FakeRecognizer:
            def __init__(self):
                self.dynamic_energy_threshold = False
                self.pause_threshold = 0
                self.non_speaking_duration = 0

            def adjust_for_ambient_noise(self, *_args, **_kwargs):
                pass

            def listen(self, *_args, **_kwargs):
                nonlocal listen_call_count
                with listen_call_lock:
                    listen_call_count += 1
                    call_number = listen_call_count

                if call_number == 1:
                    return iter([FakeAudioData(b"first ", 16000, 2)])
                if call_number == 2:
                    def continuation():
                        if not first_request_started.wait(timeout=2):
                            raise AssertionError("first transcription never started")
                        yield FakeAudioData(b"second", 16000, 2)
                        continuation_yielded.set()
                    return continuation()

                def idle():
                    raise FakeWaitTimeoutError()
                    yield  # Keep this a generator so the timeout occurs during iteration.
                return idle()

        class FakeSpeechRecognition:
            Recognizer = FakeRecognizer
            Microphone = FakeMicrophone
            AudioData = FakeAudioData
            WaitTimeoutError = FakeWaitTimeoutError

        manager = ConversationManager()

        def transcribe(wav_data):
            transcription_inputs.append(wav_data)
            if len(transcription_inputs) == 1:
                first_request_started.set()
                self.assertTrue(continuation_yielded.wait(timeout=2))
                return "partial transcript"
            return "complete transcript"

        with patch.object(conversation, "sr", FakeSpeechRecognition), patch.object(
            manager, "_transcribe", side_effect=transcribe
        ):
            transcript = manager._listen_for_visitor()

        self.assertEqual(transcript, "complete transcript")
        self.assertEqual(transcription_inputs, [b"first ", b"first second"])

    def test_short_noise_during_transcription_does_not_discard_valid_result(self):
        first_request_started = threading.Event()
        noise_candidate_seen = threading.Event()
        transcription_inputs = []
        listen_call_count = 0

        class FakeWaitTimeoutError(Exception):
            pass

        class FakeAudioData:
            def __init__(self, frame_data, sample_rate, sample_width):
                self.frame_data = frame_data
                self.sample_rate = sample_rate
                self.sample_width = sample_width

            def get_wav_data(self, **_kwargs):
                return self.frame_data

        class FakeSource:
            SAMPLE_RATE = 16000
            SAMPLE_WIDTH = 2

        class FakeMicrophone:
            def __init__(self, **_kwargs):
                pass

            def __enter__(self):
                return FakeSource()

            def __exit__(self, *_args):
                return False

        class FakeRecognizer:
            def __init__(self):
                pass

            def adjust_for_ambient_noise(self, *_args, **_kwargs):
                pass

            def listen(self, *_args, **_kwargs):
                nonlocal listen_call_count
                listen_call_count += 1
                if listen_call_count == 1:
                    return iter([FakeAudioData(b"visitor", 16000, 2)])
                if listen_call_count == 2:
                    def brief_noise():
                        if not first_request_started.wait(timeout=2):
                            raise AssertionError("first transcription never started")
                        yield FakeAudioData(b"click", 16000, 2)
                        noise_candidate_seen.set()
                        raise FakeWaitTimeoutError()
                    return brief_noise()

                def idle():
                    raise FakeWaitTimeoutError()
                    yield
                return idle()

        class FakeSpeechRecognition:
            Recognizer = FakeRecognizer
            Microphone = FakeMicrophone
            AudioData = FakeAudioData
            WaitTimeoutError = FakeWaitTimeoutError

        manager = ConversationManager()

        def transcribe(wav_data):
            transcription_inputs.append(wav_data)
            first_request_started.set()
            self.assertTrue(noise_candidate_seen.wait(timeout=2))
            return "valid transcript"

        with patch.object(conversation, "sr", FakeSpeechRecognition), patch.object(
            manager, "_transcribe", side_effect=transcribe
        ):
            transcript = manager._listen_for_visitor()

        self.assertEqual(transcript, "valid transcript")
        self.assertEqual(transcription_inputs, [b"visitor"])

    def test_ambient_calibration_is_reused_for_the_first_listen(self):
        class FakeWaitTimeoutError(Exception):
            pass

        class FakeAudioData:
            def __init__(self, frame_data, sample_rate, sample_width):
                self.frame_data = frame_data
                self.sample_rate = sample_rate
                self.sample_width = sample_width

            def get_wav_data(self, **_kwargs):
                return self.frame_data

        class FakeSource:
            SAMPLE_RATE = 16000
            SAMPLE_WIDTH = 2

        class FakeMicrophone:
            def __init__(self, **_kwargs):
                pass

            def __enter__(self):
                return FakeSource()

            def __exit__(self, *_args):
                return False

        recognizer_instances = []

        class FakeRecognizer:
            def __init__(self):
                self.adjustment_count = 0
                self.listen_count = 0
                recognizer_instances.append(self)

            def adjust_for_ambient_noise(self, *_args, **_kwargs):
                self.adjustment_count += 1

            def listen(self, *_args, **_kwargs):
                self.listen_count += 1
                if self.listen_count == 1:
                    return iter([FakeAudioData(b"yes!", 16000, 2)])

                def idle():
                    raise FakeWaitTimeoutError()
                    yield
                return idle()

        class FakeSpeechRecognition:
            Recognizer = FakeRecognizer
            Microphone = FakeMicrophone
            AudioData = FakeAudioData
            WaitTimeoutError = FakeWaitTimeoutError

        manager = ConversationManager()
        with patch.object(conversation, "sr", FakeSpeechRecognition), patch.object(
            manager, "_transcribe", return_value="yes"
        ):
            manager._prepare_recognizer()
            self.assertEqual(manager._listen_for_visitor(), "yes")

        self.assertEqual(len(recognizer_instances), 1)
        self.assertEqual(recognizer_instances[0].adjustment_count, 1)

    def test_no_speech_timeout_does_not_call_transcription(self):
        recognizer_instances = []

        class FakeWaitTimeoutError(Exception):
            pass

        class FakeRecognizer:
            def __init__(self):
                self.dynamic_energy_threshold = False
                self.pause_threshold = 0
                self.phrase_threshold = 0
                self.non_speaking_duration = 0
                recognizer_instances.append(self)

            def adjust_for_ambient_noise(self, *_args, **_kwargs):
                pass

            def listen(self, *_args, **_kwargs):
                def timed_out():
                    raise FakeWaitTimeoutError()
                    yield
                return timed_out()

        class FakeAudioData:
            def __init__(self, *_args, **_kwargs):
                raise AssertionError("AudioData should not be built without speech")

        class FakeSource:
            SAMPLE_RATE = 16000
            SAMPLE_WIDTH = 2

        class FakeMicrophone:
            def __init__(self, **_kwargs):
                pass

            def __enter__(self):
                return FakeSource()

            def __exit__(self, *_args):
                return False

        class FakeSpeechRecognition:
            Recognizer = FakeRecognizer
            Microphone = FakeMicrophone
            AudioData = FakeAudioData
            WaitTimeoutError = FakeWaitTimeoutError

        manager = ConversationManager()
        with patch.object(conversation, "sr", FakeSpeechRecognition), patch.object(
            manager, "_transcribe", side_effect=AssertionError("unexpected transcription")
        ):
            self.assertEqual(manager._listen_for_visitor(), "")

        self.assertEqual(len(recognizer_instances), 1)
        self.assertEqual(
            recognizer_instances[0].phrase_threshold,
            conversation.MINIMUM_SPEECH_DURATION_SECONDS,
        )
        self.assertEqual(
            recognizer_instances[0].pause_threshold,
            conversation.LISTEN_PAUSE_THRESHOLD_SECONDS,
        )


class AmbientCalibrationTests(unittest.TestCase):
    def _fake_speech_recognition(self):
        class FakeSource:
            pass

        class FakeMicrophone:
            def __init__(self, **_kwargs):
                pass

            def __enter__(self):
                return FakeSource()

            def __exit__(self, *_args):
                return False

        class FakeSpeechRecognition:
            Microphone = FakeMicrophone

        return FakeSpeechRecognition

    def test_due_calibration_refreshes_only_during_an_idle_quiet_window(self):
        adjustment_calls = []
        manager = ConversationManager()
        manager._active = True
        manager._mode = "watching"
        manager._recognizer = type(
            "FakeRecognizer",
            (),
            {
                "energy_threshold": 300.0,
                "adjust_for_ambient_noise": lambda _self, *_args, **_kwargs: (
                    adjustment_calls.append(True), setattr(_self, "energy_threshold", 450.0)
                ),
            },
        )()
        manager._recognizer_calibrated = True
        manager._last_recognizer_calibration_at = (
            time.monotonic() - conversation.AMBIENT_RECALIBRATION_INTERVAL_SECONDS - 1
        )
        manager._triggered_at = 0

        with patch.object(conversation, "sr", self._fake_speech_recognition()):
            manager._maybe_refresh_recognizer_calibration()

        self.assertEqual(adjustment_calls, [True])
        self.assertEqual(manager._recognizer.energy_threshold, 450.0)
        self.assertGreater(manager._last_recognizer_calibration_at, 0)

    def test_skips_periodic_calibration_when_a_person_is_present(self):
        adjustment_calls = []
        manager = ConversationManager()
        manager._active = True
        manager._mode = "watching"
        manager._recognizer = type(
            "FakeRecognizer",
            (),
            {
                "energy_threshold": 300.0,
                "adjust_for_ambient_noise": lambda _self, *_args, **_kwargs: adjustment_calls.append(True),
            },
        )()
        manager._recognizer_calibrated = True
        manager._last_recognizer_calibration_at = (
            time.monotonic() - conversation.AMBIENT_RECALIBRATION_INTERVAL_SECONDS - 1
        )
        manager._triggered_at = 0

        class VisiblePerson:
            has_recent_person = True

        manager._person_vision = VisiblePerson()
        with patch.object(conversation, "sr", self._fake_speech_recognition()):
            manager._maybe_refresh_recognizer_calibration()

        self.assertEqual(adjustment_calls, [])
        self.assertEqual(manager._recognizer.energy_threshold, 300.0)

    def test_restores_the_old_threshold_if_activity_starts_during_refresh(self):
        manager = ConversationManager()
        manager._active = True
        manager._mode = "watching"
        manager._triggered_at = 0
        manager._last_recognizer_calibration_at = (
            time.monotonic() - conversation.AMBIENT_RECALIBRATION_INTERVAL_SECONDS - 1
        )

        class FakeRecognizer:
            energy_threshold = 300.0

            def adjust_for_ambient_noise(self, *_args, **_kwargs):
                self.energy_threshold = 900.0
                manager._motion_event.set()

        manager._recognizer = FakeRecognizer()
        manager._recognizer_calibrated = True
        with patch.object(conversation, "sr", self._fake_speech_recognition()):
            manager._maybe_refresh_recognizer_calibration()

        self.assertEqual(manager._recognizer.energy_threshold, 300.0)


class PersonVisionRequestTests(unittest.TestCase):
    def test_only_supplied_person_crops_become_openai_images(self):
        requests = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"output_text":"Hello there."}'

        def capture_request(request, timeout):
            requests.append(json.loads(request.data.decode("utf-8")))
            return FakeResponse()

        manager = ConversationManager()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), patch(
            "conversation.urllib.request.urlopen", side_effect=capture_request
        ):
            manager._ask_assistant("A visitor has arrived.")
            manager._ask_assistant("Nice to meet you.", ["data:image/jpeg;base64,ZmFrZQ=="])

        first_content = requests[0]["input"][0]["content"]
        second_content = requests[1]["input"][0]["content"]
        self.assertFalse(any(item["type"] == "input_image" for item in first_content))
        images = [item for item in second_content if item["type"] == "input_image"]
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0]["image_url"], "data:image/jpeg;base64,ZmFrZQ==")
        self.assertIn("recent crop", requests[1]["instructions"])


class PersonVisionGateTests(unittest.TestCase):
    def test_external_detector_contract_requires_stable_tracked_boxes(self):
        class FakeDetector:
            def detect(self, _frame):
                return [PersonDetection(track_id=17, box=(12, 8, 145, 230))]

        frame = np.zeros((240, 160, 3), dtype=np.uint8)
        vision = LocalPersonVision(lambda: None)
        vision._detector = FakeDetector()
        wakeups = []
        vision._on_person_seen = lambda: wakeups.append(True)

        person_count, first_crops = vision._detect(frame)
        self.assertEqual(person_count, 1)
        self.assertEqual(first_crops, [])
        person_count, confirmed_crops = vision._detect(frame)
        self.assertEqual(person_count, 1)
        self.assertEqual(len(confirmed_crops), 1)
        self.assertEqual(confirmed_crops[0][0], 17)
        self.assertEqual(wakeups, [True])

    def test_external_detector_plugin_factory_is_loaded_from_configured_module(self):
        detector = type("FakeDetector", (), {"detect": lambda self, _frame: []})()
        module = type("PluginModule", (), {"create_detector": staticmethod(lambda: detector)})()
        vision = LocalPersonVision(lambda: None)

        with patch.dict(os.environ, {PERSON_DETECTOR_ENV: "external_adapter:create_detector"}):
            with patch("person_vision.importlib.import_module", return_value=module) as importer:
                self.assertIs(vision._load_detector(), detector)
        importer.assert_called_once_with("external_adapter")

    def test_missing_plugin_leaves_motion_wake_available(self):
        vision = LocalPersonVision(lambda: None)
        with patch.dict(os.environ, {PERSON_DETECTOR_ENV: "", "HEADROOM_LOCAL_PERSON_DETECTION": "1"}):
            vision._run()
        self.assertEqual(vision.status, "No detector plugin · motion wake only")

    def test_presence_wake_does_not_depend_on_crop_size(self):
        frame = np.zeros((240, 160, 3), dtype=np.uint8)
        gate = PersonTrackGate()
        wakeups = []
        small_box = (1, 1, 30, 50)

        self.assertEqual(gate.confirm(frame, [17], [small_box], on_confirmed=lambda: wakeups.append(1)), [])
        self.assertEqual(gate.confirm(frame, [17], [small_box], on_confirmed=lambda: wakeups.append(1)), [])
        self.assertEqual(wakeups, [1])
        gate.confirm(frame, [17], [small_box], on_confirmed=lambda: wakeups.append(1))
        self.assertEqual(wakeups, [1])

    def test_requires_two_samples_and_refreshes_confirmed_tracks_at_a_bounded_rate(self):
        frame = np.zeros((240, 160, 3), dtype=np.uint8)
        gate = PersonTrackGate()
        person_box = (12, 8, 145, 230)

        self.assertEqual(gate.confirm(frame, [17], [person_box]), [])
        confirmed = gate.confirm(frame, [17], [person_box])
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0][0], 17)
        self.assertEqual(gate.confirm(frame, [17], [person_box]), [])
        gate._last_crop_at[17] -= RECURRENT_CROP_INTERVAL_SECONDS
        refreshed = gate.confirm(frame, [17], [person_box])
        self.assertEqual(len(refreshed), 1)
        self.assertEqual(refreshed[0][0], 17)

    def test_resets_confirmation_when_track_is_absent(self):
        frame = np.zeros((240, 160, 3), dtype=np.uint8)
        gate = PersonTrackGate()
        person_box = (12, 8, 145, 230)

        self.assertEqual(gate.confirm(frame, [17], [person_box]), [])
        self.assertEqual(gate.confirm(frame, [], []), [])
        self.assertEqual(gate.confirm(frame, [17], [person_box]), [])
        self.assertEqual(len(gate.confirm(frame, [17], [person_box])), 1)

    def test_rejects_too_small_or_invalid_person_boxes(self):
        frame = np.zeros((240, 160, 3), dtype=np.uint8)
        self.assertIsNone(crop_person(frame, (1, 1, 30, 50)))
        self.assertIsNone(crop_person(frame, (float("nan"), 1, 100, 200)))
        crop = crop_person(frame, (-10, 5, 150, 235))
        self.assertIsNotNone(crop)
        self.assertLessEqual(crop.shape[1], frame.shape[1])
        self.assertLessEqual(crop.shape[0], frame.shape[0])
        context_crop = crop_person(frame, (40, 40, 80, 180))
        self.assertEqual(context_crop.shape[:2], (162, 56))

    def test_recent_person_presence_has_a_short_detection_grace_period(self):
        vision = LocalPersonVision(lambda: None)
        vision._last_person_seen_at = time.monotonic()
        self.assertTrue(vision.has_recent_person)
        vision._last_person_seen_at -= PERSON_PRESENCE_GRACE_SECONDS + 1
        self.assertFalse(vision.has_recent_person)

    def test_unconsumed_crop_is_replaced_by_the_newer_frame(self):
        vision = LocalPersonVision(lambda: None)
        old_crop = np.zeros((100, 100, 3), dtype=np.uint8)
        new_crop = np.ones((100, 100, 3), dtype=np.uint8)
        vision._queue_crops([(17, old_crop)])
        vision._queue_crops([(17, new_crop)])
        np.testing.assert_array_equal(vision.take_new_crops()[0], new_crop)

    def test_pending_crops_expire_and_are_consumed_in_bounded_batches(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        vision = LocalPersonVision(lambda: None)
        now = time.monotonic()
        vision._pending = {
            index: (frame.copy(), now)
            for index in range(5)
        }
        vision._pending[99] = (frame.copy(), now - 16)

        self.assertEqual(len(vision.take_new_crops(limit=3)), 3)
        self.assertEqual(vision.pending_count, 2)
        self.assertEqual(len(vision.take_new_crops()), 2)
        self.assertEqual(vision.pending_count, 0)


class SpeechSegmentationTests(unittest.TestCase):
    def test_keeps_short_clauses_and_sentences_in_one_phrase(self):
        line = "Hello there, I'm Bobby. It's lovely to meet you!"
        self.assertEqual(split_speech_segments(line), [line])

    def test_splits_long_phrases_at_words_without_losing_punctuation(self):
        line = "This is a longer sentence made of several words, which should be split only when needed."
        segments = split_speech_segments(line, max_chars=35)
        self.assertGreater(len(segments), 1)
        self.assertTrue(all(len(segment) <= 35 for segment in segments))
        self.assertEqual(" ".join(segments), line)


class StreamingTtsBufferTests(unittest.TestCase):
    def test_stream_waits_for_start_buffer_then_preserves_all_pcm_packets(self):
        class FakeResponse:
            def __init__(self, payload):
                self.payload = payload
                self.offset = 0
                self.bytes_read = 0
                self.read_complete = threading.Event()

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read1(self, size):
                packet = self.payload[self.offset:self.offset + size]
                self.offset += len(packet)
                self.bytes_read += len(packet)
                if self.offset == len(self.payload):
                    self.read_complete.set()
                return packet

        class FakePlayer:
            def __init__(self, response):
                self.response = response
                self.writes = []
                self.first_write_after = None
                self.network_reader_kept_running = False

            def write(self, packet):
                if self.first_write_after is None:
                    self.first_write_after = self.response.bytes_read
                    self.network_reader_kept_running = self.response.read_complete.wait(timeout=1)
                self.writes.append(packet)

        payload = bytes(PCM_PACKET_BYTES * 6)
        response = FakeResponse(payload)
        player = FakePlayer(response)
        events = []
        with patch("server.urllib.request.urlopen", return_value=response):
            result = Handler._stream_tts_pcm(
                object(), "Hello there.", "nova", 0.94, "not-a-real-key", 0, 1, player, events.append,
            )

        self.assertEqual(result[0], payload)
        self.assertEqual(result[4], 6)
        self.assertIn("highWaterPackets", result[5])
        self.assertIn("producerBlockedMs", result[5])
        self.assertLessEqual(result[5]["highWaterPackets"], 40)
        self.assertEqual(len(player.writes), 6)
        self.assertGreaterEqual(player.first_write_after, PCM_PACKET_BYTES * 5)
        self.assertTrue(player.network_reader_kept_running)
        self.assertEqual(len(events), 6)
        self.assertEqual(events[0]["playbackLeadMs"], PULSE_OUTPUT_LATENCY_MS)
        self.assertTrue(all(event["type"] == "envelope" for event in events))


class JawEnvelopeTests(unittest.TestCase):
    def test_sends_only_silence_or_jaw_values_from_pcm(self):
        silence = jaw_envelope(bytes(480))
        voiced = jaw_envelope(struct.pack("<h", 12000) * 240)
        self.assertEqual(silence, [0.0])
        self.assertEqual(voiced, [0.42])


class ResponsesParsingTests(unittest.TestCase):
    def test_reads_output_text_shortcut(self):
        self.assertEqual(response_text({"output_text": " Hello "}), "Hello")

    def test_extracts_text_from_output_items(self):
        payload = {
            "output": [
                {"type": "message", "content": [
                    {"type": "output_text", "text": "Hello"},
                    {"type": "output_text", "text": "there."},
                ]}
            ]
        }
        self.assertEqual(response_text(payload), "Hello there.")


if __name__ == "__main__":
    unittest.main()
