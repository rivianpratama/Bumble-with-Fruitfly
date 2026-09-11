# Bumble with Fruitfly

This app makes a fruit fly brain swipe on Bumble for you. The fly watches each
profile on your phone screen. When the fly likes what it sees, the app swipes
right. When the fly is bored, the app swipes left. The fly does not know it is
on a dating app. It only sees pictures and reacts to them.

This is a fork of [Fruitfly-Doomscroller](https://github.com/ranagwho/Fruitfly-Doomscroller)
by ranagwho. The base app makes a fly watch short videos. This fork adds the
Bumble mode and some other features. See the Credits part at the end.

![The fly watching a feed](docs/flyscroll-demo.gif)

Above: the fly watching a video feed. Bumble mode uses the same brain to swipe a
real phone instead.

## How the fly decides

The brain is a real fly connectome. It has about 166,700 neurons. The app sends
a phone photo to the fly eyes. The brain then makes neurons fire. The app reads
how much interest the fly has. Interest is a number in Hz. A high number means
the fly likes the photo. A low number means the fly is bored.

For each profile, the app watches the photos for a short time. It reads the
interest over the last 500 ms. If the mean interest is high, the app swipes
right to like. If the interest is low, the app swipes left to pass. On the
screen you see a green tick for a like and a red cross for a pass.

## What you need

- An Android phone.
- A USB cable, with USB debugging turned on.
- `adb` on your computer (the Android platform tools).
- Bumble open on the swipe screen, and the phone unlocked.
- Python 3.11 or newer.

Check that your computer can see the phone:

```sh
adb devices
```

It should show your phone as `device`.

## Install

```sh
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS or Linux:
source .venv/bin/activate

pip install -e ".[video,dev]"
```

You also need the fly brain data. It is about 1.1 GB. Put the three MaleCNS
files in `connectome_data/malecns_v1/`, then run:

```sh
flyscroll import
flyscroll prepare --scale full
```

## Run Bumble mode

Please start with a dry run. A dry run means the app decides and writes a log,
but it does not touch the phone. This is safe. Watch it for a while and check
the numbers first.

```sh
flyscroll serve --scale full --bumble --bumble-dry-run --bumble-max-swipes 10
```

Open http://127.0.0.1:8767/ in a browser. You will see these things:

- the phone screen that the fly looks at (faces and text are blurred for privacy),
- the fly in a small 3D room,
- a green tick or a red cross when it swipes,
- a list of profiles, with the average interest in Hz for each one.

When you are happy with it, run real swipes. Start with a small number.

```sh
flyscroll serve --scale full --bumble --bumble-max-swipes 5
```

## Main options

| Option | What it does |
| --- | --- |
| `--bumble-dry-run` | Decide and log, but do not swipe. |
| `--bumble-max-swipes N` | Stop after N swipes. |
| `--bumble-target-right-rate 0.25` | Try to like about 25% of profiles. |
| `--bumble-like-threshold 20` | A fixed interest level to like, in Hz. |
| `--bumble-max-profile-scrolls 3` | How many photos to scroll before deciding. |
| `--bumble-decision-window 0.5` | Seconds of interest that the swipe uses. |
| `--bumble-fps 3` | Screen photos per second. |
| `--adb` / `--serial` | The adb path and the phone id. |

The app can look at more than one photo in a profile. It swipes up to see the
next photo, and then it decides.

## How the like level is set

By default the level moves on its own. It looks at the last 40 profiles and
picks a level so about 25% of profiles get a like. You can change the rate with
`--bumble-target-right-rate`. For the first 8 profiles there is no history yet,
so it uses the fixed `--bumble-like-threshold` value. Run the dry run first and
watch the Hz numbers before you trust it.

## Privacy

The app blurs faces and text on the screen you see, and in the saved images.
Faces use a small model called YuNet. Text uses a model called PP-OCR. The fly
brain still gets the real photo, because it needs the real pixels to work. Only
the screen and the saved files are blurred.

The app saves one photo per profile in `outputs/flyscroll/bumble/`. These are
other people's photos. Keep this folder private. It stays on your computer. It
is listed in `.gitignore`, so it is not sent to GitHub.

## Please be careful

The app swipes on your own Bumble account. You are already logged in on your
phone. The app never asks for your password, and it never makes an account.

Auto swiping is very likely against the Bumble rules. Your account can get a
warning or a ban. This is your choice and your risk. Please start with a dry run
and a small `--bumble-max-swipes` value.

## Other feeds

The fly can also watch videos. This part comes from the base project.

- Local video files: put MP4 or WebM files in `reels/`, and use `--reels reels`.
- Made-up patterns: the default when there are no video files.
- Live YouTube Shorts, TikTok, or Instagram Reels:
  `--shorts --platform youtube|tiktok|instagram`. This opens a browser. It does
  not log in to anything.

## Credits

- This is a fork of [Fruitfly-Doomscroller](https://github.com/ranagwho/Fruitfly-Doomscroller)
  by ranagwho. The base app and the fly simulation come from there.
- The 3D fly room is from [fly-wirehead](https://github.com/mattyhempstead/fly-wirehead).
- The brain data is MaleCNS v1.0, shared under CC BY 4.0. See [NOTICE](NOTICE).
- The face and text models are from the OpenCV Zoo.
- The phone swipe method with adb follows the same author's BumbleClaw project.

See [THIRD_PARTY.md](THIRD_PARTY.md) for the full list and the licenses.

## License

The code is under the MIT License. See [LICENSE](LICENSE). The brain data is
under CC BY 4.0.
