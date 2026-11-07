# Privacy, data flow, and network safety

This page distinguishes local processing, information sent to OpenAI, and files saved on the laptop. Tell visitors when vision is active and follow the event's consent requirements, especially when children are present.

## Data flow

| Data | Processing and retention |
|---|---|
| Camera frames | Captured and held in memory on the laptop. An optional local detector returns anonymous person boxes/tracks; Headroom does not perform face recognition or retain identity histories. Full-scene frames are not sent to OpenAI. |
| Person crops | After a track is confirmed, bounded crops may be saved locally and sent to OpenAI with a conversation turn (up to four per reply). They are not sent continuously. |
| Microphone audio | Captured in memory and sent to OpenAI for transcription. Raw audio is not written to disk. The microphone is closed during Bobby's speech to avoid speaker echo. |
| Transcripts | Recognized visitor text is saved locally by default in `logs/visitor_transcripts.jsonl`; it is rotated at about 3 MiB with three backups. |
| TTS text/audio | Reply text is sent to OpenAI TTS. Streamed audio plays on the laptop; it is not sent to the browser or saved in the project. Rhubarb analyzes the same audio locally and sends only mouth-animation cues to the display. |
| Runtime logs | `logs/headroom.log` records device/status/timing diagnostics, not transcript text, audio payloads, prompts, request bodies, or credential values. It is private and rotated at about 3 MiB with three backups. |

The most recent 100 person-crop images are retained under `logs/vision_frames/`. Track IDs and unconsumed crops remain in memory only; unused crops expire. Without an external person detector, no person crops are created and motion wake remains available.

## Opt-outs and cleanup

Set these variables before `./run.sh` to disable persistent visitor data:

```bash
export HEADROOM_SAVE_TRANSCRIPTS=0
export HEADROOM_SAVE_VISION_FRAMES=0
```

Remove existing local data with:

```bash
rm -f logs/visitor_transcripts.jsonl*
rm -rf logs/vision_frames
```

Logs, crops, transcripts, credentials, local environments, caches, and generated media are excluded by `.gitignore`. Review `git status` and the staged diff before any public push; ignore rules do not prevent an explicit force-add.

## Credentials and network exposure

`run.sh` reads an OpenAI key from Pi auth by default, then falls back to `OPENAI_API_KEY`. Set `OPENAI_USE_PI_AUTH=0` to opt out of Pi auth. The key is passed only to the local server process and is never embedded in the page. Never commit `.env` files or private SSH/API keys.

In WSL, the server listens on an address reachable by Windows for the iPad port proxy. The PowerShell helper restricts forwarding to the active Private Wi-Fi interface and `LocalSubnet`. The demo has no authentication: do not expose it on a Public Wi-Fi network or the internet. The GitHub Pages site is a separate static visual preview; it does not access this local backend, camera, or microphone.

The preview loads Three.js from jsDelivr and fonts from Google Fonts. GitHub Pages and those services receive ordinary browser requests; the preview does not send them visitor audio or images.
