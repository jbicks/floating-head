# Visitor mode and operation

## Setup and launch

In WSL, install the required tools and capture dependencies:

```bash
./scripts/install_rhubarb.sh
./scripts/install_conversation_deps.sh
./run.sh
```

Open `http://localhost:8766`. The clean page starts visitor mode when the avatar and backend are ready. Use `?debug=1` for diagnostics and manual controls; debug mode starts with visitor mode off. Stop with **Ctrl+C**. The server drains active speech work and releases capture devices on shutdown.

For an iPad display, run `run.ps1` from **Administrator PowerShell** in the WSL checkout. It attaches a USB camera, preflights camera/microphone/laptop-audio bridges, and configures the port proxy for the active Private Wi-Fi subnet. Open the printed URL on the iPad. The laptop still owns all input and sound; the iPad displays the avatar and mouth animation only. Do not expose the unauthenticated server outside a trusted private network.

Use `./scripts/check_capture_devices.py` to inspect devices, or `./scripts/check_capture_devices.py --strict` to actively check camera, microphone, and laptop output readiness.

## Capture and conversation behavior

- Headroom prefers the Logitech BRIO, then a recognized built-in camera/microphone, then the WSL default microphone. Visitor mode calibrates the microphone once at startup; it refreshes the ambient threshold only during a quiet, visitor-free interval.
- Motion wake remains available without person detection. If an external detector is installed, a person track must be present in two samples before the person-presence wake fires or a crop is queued.
- A short pause (about 0.9 seconds) tentatively ends an utterance while capture continues. Very short phrases (about 0.18 seconds) are accepted. If speech resumes before transcription is accepted, the partial transcript is discarded and the combined utterance is retried.
- Bobby starts an arrival greeting while the microphone listens. If the visitor speaks first, their speech takes priority. The microphone closes during Bobby's playback to avoid recording speaker echo; playback without verified echo cancellation must not overlap listening.
- Conversations can continue through quiet pauses while a person remains visible, up to a five-minute safety limit. After two quiet listen windows with no visible visitor, the interaction ends. If a visible visitor is not understood, Bobby offers at most one gentle repeat prompt, then keeps listening.

## Speech and display pipeline

Visitor speech is transcribed with OpenAI; Bobby's response is sent to OpenAI TTS. The laptop plays the streamed audio through WSLg/PulseAudio. Rhubarb analyzes the same generated WAV and supplies approximate timed mouth cues. The browser receives compact animation data and cues, not speech audio. A one-second PCM prebuffer and bounded playback queue help absorb network/audio scheduling jitter. A disconnected display does not stop laptop playback.

Useful status endpoints are `/api/status` and `/api/conversation/status`; `/api/conversation/reply-stream` provides display animation events. Static files are deliberately limited to the public page, license/attribution files, and GLBs.

## Diagnostics

- Runtime log: `logs/headroom.log` (private; rotated with three backups at about 3 MiB).
- Debug camera details: set `HEADROOM_DEBUG=1` before `./run.sh`.
- Resource sampling: `HEADROOM_DIAGNOSTICS_INTERVAL_SECONDS` defaults to 60 seconds and accepts 5–300. `HEADROOM_PROFILE=1` enables allocation tracing with additional overhead.
- For visitor transcript and image retention settings, see [PRIVACY.md](PRIVACY.md).
