# Headroom — Bobby's floating-head display

A browser-based Pepper's Ghost avatar featuring Bobby, a cheerful, gently spooky ghost. Her mint complexion, luminous eyes, stitched details, animated cyan mist, and separate Otohiko wig are applied at runtime; the original MPFB avatar and its facial morph targets remain unchanged. OpenAI provides speech and Rhubarb creates mouth cues from the same audio.

## Live visual preview

[See Bobby on GitHub Pages](https://jbicks.github.io/floating-head/). Drag to orbit and scroll to zoom. This static preview does not use the Python backend, camera, microphone, speech, lip-sync service, or visitor mode. It automatically activates on `*.github.io`; add `?preview=1` to try it locally.

## Run locally

Requirements: Python 3, `curl`, and `unzip`. Install Rhubarb and, for visitor capture, the camera/audio dependencies:

```bash
./scripts/install_rhubarb.sh
./scripts/install_conversation_deps.sh
./run.sh
```

Open <http://localhost:8766>. The clean display starts visitor mode when services are ready; add `?debug=1` for controls and diagnostics, with visitor mode off by default. OpenAI credentials are read server-side from Pi auth or `OPENAI_API_KEY`; they are never sent to the browser. Laptop audio plays through WSLg/PulseAudio. Stop with **Ctrl+C**.

## iPad visitor display (WSL)

The laptop owns the camera, microphone, conversation, and speech; the iPad is a silent visual display. From **Administrator PowerShell**, run the helper from the WSL checkout:

```powershell
Set-Location '\\wsl.localhost\Ubuntu\home\<wsl-user>\3d_head_playground'
.\run.ps1
```

Replace `<wsl-user>` with your WSL username. The helper attaches a camera, checks audio/camera, and configures iPad access on the active Private Wi-Fi network; then open its printed URL on the iPad. Keep this unauthenticated demo on a trusted private network—do not expose it to the internet or a Public network.

## Optional person-detection plugin

No detector or model is bundled or installed by default. Without a plugin, motion wake and conversation work, but person-presence wake and person crops are unavailable. Install an adapter in `.venv`, then start with `HEADROOM_PERSON_DETECTOR=module:factory ./run.sh`. The factory returns an object whose `detect(frame)` method yields `PersonDetection(track_id, (left, top, right, bottom))` values; track IDs must be stable across nearby frames. Headroom confirms tracks across two samples and bounds crops before using them as conversation context. Adapters must perform local inference and must not upload frames themselves.

Headroom is compatible with YOLO-style adapters, but does not include or install one. Review the adapter, detector, and model licenses independently; Ultralytics software is AGPL-3.0. See [ATTRIBUTION.md](ATTRIBUTION.md).

## Privacy and licensing

- Raw microphone audio is sent to OpenAI for transcription but is not saved. Recognized transcripts and person-crop images are saved locally by default under `logs/`; disable them with `HEADROOM_SAVE_TRANSCRIPTS=0` and `HEADROOM_SAVE_VISION_FRAMES=0`. Crops are sent to OpenAI only with conversation turns. No face recognition or persistent identity tracking is used. Follow event consent requirements when vision is active.
- Logs, local environments, credentials, and generated media are excluded from Git. The web server serves only the intended app page and GLB assets.
- The repository’s [MIT license](LICENSE) covers Headroom’s original code. Bundled assets and runtime dependencies have separate terms; see [ATTRIBUTION.md](ATTRIBUTION.md).
