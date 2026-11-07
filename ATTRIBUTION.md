# Model attribution

`assets/mpfb.glb` is derived from the example avatar `avatars/mpfb.glb` in [met4citizen/TalkingHead](https://github.com/met4citizen/TalkingHead), a JavaScript talking-avatar project. Its README credits this example as created with [Blender](https://www.blender.org/) and the [MPFB](https://static.makehumancommunity.org/mpfb.html) Blender extension, and states that the avatar is licensed **CC0** (public domain dedication). The upstream project also notes that MPFB uses assets from the MakeHuman ecosystem, with CC0/CC-BY licensing.

The model file here was optimized for browser download using glTF Transform: meshopt geometry compression and WebP textures. Geometry was not simplified; facial morph targets were retained. The original source model is 36.8 MB; the optimized GLB is about 3.5 MB. The demo applies a local floating-head treatment and zombie look (clean skin-toned neck cutoff, sickly complexion, luminous milky eyes, animated cyan ghost haze and drifting lights, stitched details, and a slack expression). The default ponytail is hidden; the MPFB source GLB and its facial morph targets are not altered.

- Upstream model/repository: [met4citizen/TalkingHead — avatars/mpfb.glb](https://github.com/met4citizen/TalkingHead/tree/main/avatars)
- MPFB: [MakeHuman Community](https://static.makehumancommunity.org/mpfb.html)
- MPFB source project: [makehumancommunity/mpfb](https://github.com/makehumancommunity/mpfb)
- MPFB example avatar license noted upstream: [CC0](https://creativecommons.org/publicdomain/zero/1.0/)

## Otohiko custom wig

`assets/otohikos_custom_hair_source.glb` is the user-provided original; `assets/otohikos_custom_hair.glb` is the browser-ready derivative used by the demo. The derivative keeps the source wig mesh and material/texture, removes unrelated scene objects, and centers the mesh for fitting to the avatar's head bone.

- Source model: [Otohiko's custom hair](https://skfb.ly/pCUrt)
- Creator: Da cat (Sketchfab display name; the source GLB metadata also identifies the author as SLEEPKITTY / `dekomenodekotovis`).
- License: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) — attribution is required; adaptations and commercial use are allowed under the license terms.
- Changes made: removed non-wig scene objects, centered the mesh, and recessed a small section of the viewer-left cheek curl about 1 cm behind the face to prevent interpenetration; original material/texture preserved.

## Optional curly-hair asset (currently unused)

`assets/clementine_curly_hair.glb` is retained as an optional reference asset but is not loaded by the current demo. It is an extracted and modified derivative of **“Nurse Clementine, a stylized retro pin-up medic”** by **XRProfXR (@pinckneyb)**. The source model is published on Sketchfab under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license. Only the curly hair geometry is included here; it was isolated from the full character, recolored purple, and packaged as a separate, non-rigged mesh. The MPFB head and its facial morph targets remain unchanged.

- Source model: [Nurse Clementine](https://sketchfab.com/3d-models/nurse-clementine-a-stylized-retro-pin-up-medic-3ca7ab257d234228bf7fabfec6bdfc8f)
- Creator: [XRProfXR (@pinckneyb)](https://sketchfab.com/pinckneyb)
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — attribution is required; adaptations and commercial use are allowed under the license terms.
- Changes made: isolated the hair mesh from the full character and recolored it purple.

## Lip-sync tool

[Rhubarb Lip Sync](https://github.com/DanielSWolf/rhubarb-lip-sync) v1.14.0 analyzes the generated WAV and outputs time-coded mouth-shape cues. It is licensed under the [MIT License](https://github.com/DanielSWolf/rhubarb-lip-sync/blob/master/LICENSE.md). The setup script installs the official Linux release under `~/.local/share/headroom/`; it is not copied into the project. Rhubarb's English recognizer/model resources are included in that release and retain their upstream license notices.

## Optional external person detector

This repository defines an adapter seam for optional local person detection in `person_vision.py`. The default setup includes no detector implementation or model weights and does not import one; the runtime imports a plugin only when the user configures it. Users may install a compatible adapter separately. For example, an independently supplied adapter can connect the Ultralytics YOLO software, which is licensed under **AGPL-3.0**. Review the [Ultralytics license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) and the separate terms for the specific model weights when choosing or redistributing an adapter or detector. Mentioning compatibility does not include or relicense that external software. When a user-configured detector is active, Headroom confirms local tracks and sends bounded person crops to OpenAI only with conversation turns.

## Runtime software and fonts

The repository's [MIT license](LICENSE) applies to Headroom's original code only. It does not replace the separate terms for the assets above or third-party software installed/fetched when running the app. These dependencies are not vendored in this repository (except the separately attributed GLB assets).

| Dependency | Use | License / distribution note |
|---|---|---|
| [Three.js r170](https://github.com/mrdoob/three.js) and its `GLTFLoader`, `OrbitControls`, and `MeshoptDecoder` modules | Browser rendering, model loading, controls, and meshopt decoding | MIT; loaded from jsDelivr at runtime, not vendored. `MeshoptDecoder` is based on [meshoptimizer](https://github.com/zeux/meshoptimizer), also MIT. |
| [DM Sans](https://fonts.google.com/specimen/DM+Sans), [DM Mono](https://fonts.google.com/specimen/DM+Mono), and [Manrope](https://fonts.google.com/specimen/Manrope) | Browser typography | SIL Open Font License 1.1; requested from Google Fonts at runtime, not redistributed here. |
| [OpenCV-Python](https://github.com/opencv/opencv-python) | Camera capture and image processing | Apache License 2.0. |
| [SpeechRecognition](https://github.com/Uberi/speech_recognition) | Microphone capture and audio utilities | BSD 3-Clause. |
| [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/) | Microphone device access | MIT. |
| [NumPy](https://github.com/numpy/numpy) | Test arrays and transitive numerical support | BSD 3-Clause; installed as a package dependency. |
| External person-detection adapter/model selected by the user | Optional local person-box detection via the documented plugin seam | Not included or installed by this repository; the user must review the adapter, detector package, and model licenses. For example, Ultralytics software is AGPL-3.0; check the specific model's terms separately. |

The setup scripts install Python packages from their upstream package indexes; versions are constrained in `requirements-conversation.txt` but are not vendored. This is an attribution and license inventory, not legal advice; re-check the upstream license terms for the exact versions and use case before redistribution or commercial deployment.
