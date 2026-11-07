# External person-detector plugin

Headroom's MIT-licensed core does not bundle or install a person detector. Without a plugin, visitor conversation still works with motion wake, but person-presence wake and person crops are unavailable.

## Contract

Install the adapter and detector dependencies in the same `.venv` used by `./run.sh`. Configure an importable `module:factory` before startup:

```bash
HEADROOM_PERSON_DETECTOR=my_adapter:create_detector ./run.sh
```

The zero-argument factory must return an object with `detect(frame)`. `frame` is the latest OpenCV camera image. Each call returns an iterable of `PersonDetection` objects from `person_vision`, using frame-pixel `(left, top, right, bottom)` coordinates and integer track IDs. IDs must remain stable for the same track across nearby samples, but must not be used as persistent identities.

Adapter skeleton:

```python
from person_vision import PersonDetection

class Adapter:
    def __init__(self):
        # Load the chosen detector and its local tracker here.
        ...

    def detect(self, frame):
        # Run local inference/tracking; replace with the chosen library's API.
        tracked_boxes = []
        return [
            PersonDetection(int(track_id), tuple(float(v) for v in xyxy))
            for track_id, xyxy in tracked_boxes
        ]

def create_detector():
    return Adapter()
```

The example is only a scaffold; it does not install or invoke YOLO. A plugin can be installed as a package in `.venv`, or exposed on Python's import path using the environment's normal packaging tools.

## Headroom's handling of detections

The detector worker samples at about 1 Hz. Headroom confirms a track in two samples before waking the conversation or queuing its crop. A confirmed person can wake the visitor interaction even if the box is too small to crop. Crops smaller than 40×80 pixels are rejected; confirmed tracks are refreshed no more than once every 10 seconds. At most 12 unconsumed crops are held, each expires after 15 seconds, and at most four crops are attached to a reply. Crop creation, saving, and OpenAI submission remain in Headroom—not in the plugin. The adapter must not upload frames or perform face recognition.

Set `HEADROOM_LOCAL_PERSON_DETECTION=0` to disable the plugin. If it is unset, missing, or fails to load, Headroom reports detector status and falls back to motion wake without crops.

## Licensing

YOLO compatibility means an external adapter can implement this interface; it does not make YOLO part of Headroom or change its license. Ultralytics software is AGPL-3.0, and individual model weights may have separate terms. Check all adapter, package, and model licenses for the intended use. Do not add YOLO packages, adapter code, or weights to the core requirements or claim a YOLO-enabled combined deployment is MIT-only. See [ATTRIBUTION.md](../ATTRIBUTION.md).
