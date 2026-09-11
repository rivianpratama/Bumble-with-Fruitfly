# Third-party code and data

## MaleCNS v1.0 connectome

The neuron annotations, neurotransmitter predictions and synaptic weights in
`connectome_data/malecns_v1/` are the MaleCNS v1.0 release, distributed by the
MaleCNS collaboration under **CC BY 4.0**. See [NOTICE](NOTICE) for the required
attribution. The data is downloaded separately and is not part of this repository.

## fly-wirehead — 3D observation chamber

The following files in `flyscroll/ui/` come from
[mattyhempstead/fly-wirehead](https://github.com/mattyhempstead/fly-wirehead) at
commit `fcefe9441f80e25aab713411ebced53f5e5ea172` (2026-09-11):

- `motion.js`, `swipe.js`, `dopamine-plot.js`, `favicon.svg` — copied
  **unmodified**
- `scene.js` — the Three.js chamber (platform, low-poly Drosophila, cranial
  implant and tether, phone), **modified in place** by FlyScroll: the upstream
  sunlit garden was replaced by an "infinite white" studio (white clear color and
  fog, a large white ground plane with a fading grid, neutral lighting). The
  upstream `garden.js` is no longer used and is not vendored.

`flyscroll/ui/chamber.js` is FlyScroll's own spectator adapter. It replaces that
project's `main.js`, `backend.js` and `video-feed.js`: FlyScroll's Python session
already drives the feed and the simulation, so the chamber only *displays*
telemetry it reads from `/state`. The measured rates that animate the fly
(PAM11, MN9 + DNp09, DNa02 right-minus-left) are computed in
`flyscroll/session.py` from FlyScroll's own spike counts, following the readout
choices fly-wirehead documents in its `docs/model.md`.

**License status.** At the commit above, the fly-wirehead repository ships no
top-level `LICENSE` file. Its `README.md` and `THIRD_PARTY.md` state that its
*neural backend* is adapted from `nftechie/stonkfly` under MIT (with that notice
preserved), that the 3D scene is original to fly-wirehead, and that Three.js is
distributed under its own license. The scene files vendored here therefore carry
no explicit license grant from their author. Confirm terms with the fly-wirehead
author before redistributing FlyScroll with these files included; they can be
removed by deleting the files listed above and `chamber.js`, which restores the
previous spectator layout.

## Three.js 0.180.0

`flyscroll/ui/vendor/three.module.js` and `three.core.min.js` are the Three.js
library, **MIT** — see `flyscroll/ui/vendor/THREE-LICENSE.txt`.

## BumbleClaw — adb phone-automation approach

`flyscroll/bumble.py` drives an Android phone with three adb primitives
(`get-state`, `exec-out screencap -p`, `shell input swipe`), an injectable
`runner` so those primitives can be tested without hardware, and the
human-like pacing of `--bumble-247` (a short randomized delay per decision plus
a 2–3 minute pause once an hour). That approach is taken from
**BumbleClaw** (`bumble_phone_auto.py`), a separate project by the same author.

The code in `flyscroll/bumble.py` is an independent re-implementation written
for FlyScroll's feed contract: **no code is copied and nothing is imported from
BumbleClaw**, which is not a dependency of this repository. The decision policy
(rolling right-rate over per-profile mean interest from the fly) is FlyScroll's
own.

## Short-video platforms

With `flyscroll serve --shorts`, Playwright browses public YouTube Shorts, TikTok
or Instagram Reels without signing in. FlyScroll never enters credentials or
works around CAPTCHAs and login walls. Videos are not downloaded or stored; the
fly sees only the transient screen capture. All content and rights remain with
the platforms and their creators.

With `flyscroll serve --bumble`, FlyScroll swipes a phone that the operator has
already unlocked and signed in; it never enters credentials. Automating swipes
is very likely against Bumble's terms of service. The per-decision log under
`outputs/flyscroll/bumble/` contains other people's profile photos and stays
local.

## PP-OCRv3 text detector (privacy mode)

`flyscroll/models/text_detection_en_ppocrv3_2023may.onnx` (~2.4 MB) is the English
**PP-OCRv3** text-detection (DB) model, distributed via the
[OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/text_detection_ppocr)
and originally from PaddlePaddle/PaddleOCR. FlyScroll runs it locally, through
`cv2.dnn.TextDetectionModel_DB`, to black out text on the frames it shows and logs;
it is never sent anywhere. See the OpenCV Zoo / PaddleOCR repositories for the
model's license and citation.

## YuNet face detector (privacy mode)

`flyscroll/models/face_detection_yunet_2023mar.onnx` (~230 KB) is the **YuNet**
face-detection model by Shiqi Yu and Yuantao Feng, distributed via the
[OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).
FlyScroll uses it locally, through `cv2.FaceDetectorYN`, to pixelate faces on the
frames it shows and logs (privacy mode); it is never sent anywhere. See the OpenCV
Zoo repository for the model's license and citation.
