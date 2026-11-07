# Project intent

Headroom is a local, browser-based Pepper's Ghost display for Bobby, a friendly Halloween character. This note records the product boundaries that should guide future changes.

## Character and appearance

- Bobby should be cheerful, whimsical, gentle, and kind to children. Replies should be natural and brief, answer the visitor's latest words, and avoid frightening, teasing, or pressuring them.
- She may share her Calgary-area, roughly century-old backstory when asked; greetings should not become a fixed script.
- Preserve the current avatar, runtime styling, facial morph targets, and Otohiko wig attribution. Visual edits should remain reversible; do not overwrite the original MPFB GLB.

## System boundaries

- The laptop owns the camera, microphone, conversation, and speech playback. The iPad is a silent visual display and receives animation cues, not speech audio.
- Do not listen to the microphone during Bobby's speech unless echo cancellation is added and verified.
- The normal local display starts visitor mode automatically. `?debug=1` is for testing and leaves visitor mode off. The GitHub Pages view is static and must not attempt backend requests.
- Keep OpenAI credentials on the server. Keep the unauthenticated visitor server restricted to a trusted private network; it is not an internet-facing service.

## Privacy and extensibility

- Use anonymous, short-lived person tracks only. No face recognition or identity history; send bounded person crops only with conversation turns, not continuous scene video.
- Raw microphone audio is not retained. Transcripts and crops are private local diagnostics by default, with documented opt-outs. Follow applicable event consent requirements before using vision with visitors.
- The MIT-licensed core has no bundled person detector. Detector integrations are external plugins with their own dependencies and licenses; preserve motion wake as a fallback.

## Non-goals

- Do not copy code or visuals from the separate Halloween Guy project.
- Do not replace the current capture/transcription/response/TTS pipeline with a realtime system without a separate design and evaluation.
- Do not expose the local unauthenticated app publicly or put secrets, visitor data, model caches, or local environments in Git.
