# android – open-holter recorder

Android app that turns a **Polar H10** chest strap and a dedicated phone
(tested target: Google Pixel 8a, Android 13+) into a long-term ECG recorder.
It records continuously for weeks, has a symptom-marker button, and keeps
all raw data on the phone as open CSV files that `analysis/ecgpipe` reads
directly.

> **Not a medical device.** For personal research and education only. It does
> not diagnose anything, it is not certified, and it can miss events (strap
> off, Bluetooth gaps, flat battery). If you feel unwell, seek medical help.

## Features

- Scan for a Polar H10, remember it, and **reconnect automatically** after
  drop-outs or when the strap is put back on.
- ECG at **130 Hz**, the H10's heart rate and RR intervals with skin contact,
  and the **accelerometer** (25 or 50 Hz). The accelerometer and contact data
  are recorded so the analysis can tell movement artefacts from real beats.
- Recording runs in a **foreground service** (type `connectedDevice`) with a
  persistent notification and a partial wake lock. It keeps running with the
  screen off, after the app is swiped away and in Doze. It restarts after a
  **reboot** or an **app update** if it was recording before.
- **Hourly CSV files** in the ECGLogger layout. Data is handed to the OS every
  10 s and fsync'ed to flash every 60 s, so a crash loses at most a few
  seconds.
- A big **MARK SYMPTOM** button in the app and a **Mark symptom** action in
  the notification. The phone time of the press goes into the `marker` column.
- **Pause for shower / Resume**: stops the ECG and ACC streams cleanly and then
  disconnects. Also available from the notification.
- A status screen with connection state, H10 battery, a live ECG trace (last
  10 s), heart rate, hours recorded today, storage used and free, and the last
  marker.
- **Export**: share one day's files (share sheet: Drive, Quick Share,
  e-mail …), or copy all new files into any folder through the system file
  picker (USB stick, Drive folder, …).
- Backups are off (`allowBackup=false`), there are no analytics, and the app
  never connects to the network.

### H10 battery protection

The Polar SDK documents a known issue: if the phone does not stop ECG/ACC
streaming, the H10 stays on until its battery is empty, even after it is taken
off the strap. The app handles this as follows:

- Every disconnect the app starts (pause, stop, reconnect) stops the
  streams first and waits 1 s. The state machine is unit-tested for this rule
  (`RecorderMachineTest`, including a randomised test).
- If the strap reports **no skin contact for 3 minutes** (strap taken off
  without pressing Pause), ECG and ACC are stopped. Heart rate keeps running,
  and ECG restarts as soon as there is contact again.
- **Use "Pause for shower" before you take the strap off.** It is the clean
  way to do it.

## Install on the Pixel

The APK is built by GitHub Actions. You do not need Android Studio.

1. **Download the APK** on the phone:
   - Releases: open the repository's **Releases** page and download
     `open-holter-<version>-debug.apk` from the latest release (created when a
     `v*` tag is pushed); or
   - Latest build: **Actions → Android →** the newest green run →
     **Artifacts → open-holter-apk** (you need to be logged in to GitHub; the
     artifact is a zip that contains the APK).
2. **Allow the install**: open the APK. Android asks you to allow
   "Install unknown apps" for your browser or Files app. Allow it, go back and
   tap **Install**. Play Protect may warn about an unknown developer. Choose
   *More details → Install anyway*.
3. Open **open-holter** and tap **Grant permissions**: allow *Nearby devices*
   (Bluetooth) and *Notifications*.
4. Tap **Allow unrestricted battery** (or go to *Settings → Apps → open-holter
   → App battery usage → Unrestricted*).
5. **Charge limit**: *Settings → Battery → Charging optimisation → Limit to
   80%* (on older Pixel builds: *Adaptive charging* / *Battery → Charge
   limit*). This protects the battery when the phone charges every night.
6. Put the strap on (wet the electrodes), go to **Device → Scan**, and tap your
   H10. Recording starts and the app remembers the strap.

**Updating:** install the new APK over the old one. Recordings stay, and
recording restarts by itself after the update. Debug APKs from different
build machines are signed with different debug keys. If Android says "App not
installed" when you update, the signing key changed. You must then uninstall
first, **which deletes the recordings, so export them first**. A release
keystore (below) avoids this.

Recommended Pixel settings for a dedicated recorder phone: Bluetooth always
on, *Adaptive Battery* can stay on (the app is exempt), turn off automatic
system-update restarts at night if you can, and keep at least 20 GB free.

## Daily use

- **Morning:** put on the strap. If recording was paused, tap **Resume**.
- **Symptom:** tap **MARK SYMPTOM** (or *Mark symptom* in the notification) as
  soon as you can. The phone vibrates briefly.
- **Shower or taking the strap off:** tap **Pause for shower** first, and
  **Resume** afterwards.
- **Night:** charge the phone. Whether you keep wearing the strap is up to
  you.
- **Every few days:** *Files → Export new files* to a folder outside the app.

## Files and format

### Contract between app and analysis

`analysis/ecgpipe` reads these files with loaders that expect exactly these
layouts. Each file starts with one header line, then comma-separated rows
in this column order and with these units:

| file (hourly) | columns | units / values |
|---|---|---|
| `ecg_*.csv` | `timestamp_ns,ecg_mV,pulse_bpm,rr_ms,marker` | ECGLogger layout: Unix ns, mV, bpm, ms, `1` or empty |
| `acc_*.csv` | `timestamp_ns,x_mG,y_mG,z_mG` | Unix ns, milli-g (integers) |
| `hr_*.csv` | `timestamp_ns,hr_bpm,rr_ms,contact,contact_supported` | Unix ns, bpm, ms (may be empty), `0`/`1`, `0`/`1` |

The headers are defined once, in `core/.../RecordingFiles.kt`, and the unit
tests check them. Changing a column means changing the analysis loaders in
the same PR.

All files are in app-specific storage:

```
/sdcard/Android/data/org.openholter.recorder/files/recordings/
  2026-10-21/
    ecg_2026-10-21_14.csv      ECG + HR/RR + markers (ECGLogger layout)
    acc_2026-10-21_14.csv      accelerometer
    hr_2026-10-21_14.csv       H10 heart rate, RR, skin contact
    events_2026-10-21.log      connection / pause / gap log
```

Files are split by the **local hour** of the data they contain. If the app
restarts within the same hour, it appends to the existing file and does not
write the header again. The folder can also be reached over USB (MTP) from a
computer. **Uninstalling the app deletes it.**

All timestamps are **Unix epoch nanoseconds** (UTC). ECG and ACC sample times
come from the H10's own sample clock, which gives exact 1/130 s spacing. They
are mapped to phone time with a minimum-latency estimate that follows clock
drift slowly (`ClockMapper`). HR rows and markers use the phone time at
arrival or at the button press.

### `ecg_*.csv`: ECGLogger layout, read by `analysis/ecgpipe`

```
timestamp_ns,ecg_mV,pulse_bpm,rr_ms,marker
1792576800000000000,0.123,,,
1792576800007692307,0.131,62,968,
1792576800015384615,0.140,,,1
```

| column | content |
|---|---|
| `timestamp_ns` | sample time, Unix ns |
| `ecg_mV` | ECG in millivolts (H10 µV ÷ 1000, exact to 3 decimals) |
| `pulse_bpm` | H10 heart rate; only on the first row at or after each new value |
| `rr_ms` | H10 RR interval; one per row, only when new |
| `marker` | `1` on the first ECG row at or after a button press |

There is one row per ECG sample, and the sparse columns are empty when
nothing is new. If no ECG arrives within 3 s (e.g. while disconnected), the
heart rate, RR and markers are still written as rows with an empty
`ecg_mV`. `ecgpipe` accepts these rows.

### `acc_*.csv`: accelerometer

```
timestamp_ns,x_mG,y_mG,z_mG
1792576800012000000,-12,980,31
```

Values are in milli-g, measured at 25 or 50 Hz (set under *Device*) with a
range of ±8 g. Same hourly rotation and flush rules as the ECG.

### `hr_*.csv`: H10 heart-rate stream

```
timestamp_ns,hr_bpm,rr_ms,contact,contact_supported
1792576800512000000,62,968,1,1
1792576800512000000,62,975,1,1
1792576801498000000,63,,0,1
```

There is one row per RR interval of each H10 heart-rate notification (about
1 per second). All rows of one notification share its phone arrival time.
A notification without RR gives one row with an empty `rr_ms`. `contact` and
`contact_supported` are the H10's sensor-contact flags (`1`/`0`).

### `events_*.log`: event log (tab-separated, one file per day)

```
time_local	timestamp_ns	event	detail
2026-10-21 12:00:03.120	1792576803120000000	start	A1B2C3D4
2026-10-21 12:00:05.410	1792576805410000000	connected	A1B2C3D4
2026-10-21 12:00:06.002	1792576806002000000	streams_start
2026-10-21 13:12:40.877	1792581160877000000	disconnected	waiting for reconnect
2026-10-21 13:13:02.310	1792581182310000000	ecg_gap	21.6 s without ECG since 1792581160801000000
2026-10-21 16:29:59.000	1792592999000000000	pause
2026-10-21 16:29:59.950	1792592999950000000	marker	symptom marker
```

Main events: `start`, `stop`, `pause`, `resume`, `connected`, `disconnected`,
`disconnect_reason`, `reconnect`, `streams_start`, `no_data`, `no_contact`,
`contact_back`, `ecg_gap`, `marker`, `h10_battery`, `bluetooth_off`/`_on`,
`service_restart`, and `*_error`.

Storage needed: about 15 MB per hour of ECG, roughly 0.4–0.5 GB per day with
all files. The status screen shows the free space.

## Manual test checklist

BLE cannot be tested in CI. Run through this list on the phone after each
change. Use `adb logcat -s RecorderService` if a computer is at hand. The
events log on the phone shows the same information.

1. [ ] Fresh install: permission prompts appear and the battery exemption
       opens the system dialog.
2. [ ] Scan finds the H10 (strap worn). Tapping it starts recording, and
       after a few seconds the status shows *Recording*, heart rate, H10
       battery and a moving ECG trace.
3. [ ] The notification is present and cannot be swiped away. *Mark symptom*
       in the notification vibrates, and *Last marker* updates.
4. [ ] The in-app MARK button: the row with `marker=1` appears in the current
       `ecg_*.csv` within about 10 s at the right time.
5. [ ] Screen off for 30 min in the pocket: the ECG file keeps growing with no
       gap in `events_*.log`.
6. [ ] Swipe the app away in Recents: recording continues (notification
       still there).
7. [ ] Walk out of Bluetooth range for 2 min, then come back:
       `disconnected` → `connected` → `streams_start`, and one `ecg_gap`.
8. [ ] *Pause for shower*: status *Paused*. Take the H10 off the strap for
       10 min, put it back, tap *Resume*: recording resumes. (The H10's LED
       should not keep blinking while it is off the strap.)
9. [ ] Take the strap off **without** pausing: after about 3 min the status
       shows *No skin contact* (`no_contact` in the log). Put it back on:
       ECG resumes (`contact_back`, or a reconnect).
10. [ ] Turn Bluetooth off for 1 min and on again: recording resumes.
11. [ ] Reboot the phone without opening the app: recording restarts
        (`service_restart` in the log).
12. [ ] Install a newer APK over the old one: recording restarts and the old
        files are still there.
13. [ ] Hour change: a new `ecg_…_HH.csv` starts at the full hour with a
        single header.
14. [ ] *Files → Share* a day to another app, and *Export to folder*. Run
        export twice: the second run copies only the changed files.
15. [ ] Copy the files to a computer and run
        `python -m ecgpipe ecg_2026-10-21_*.csv -o report`: the markers
        appear in the report.
16. [ ] 24 h soak test: check the hours recorded today, the phone battery
        drain and the H10 battery.

## Build

CI (`.github/workflows/android.yml`) runs on every push or PR that touches
`android/`:

- `./gradlew :core:test :app:testDebugUnitTest` runs the unit tests (CSV
  format, µV→mV, sparse columns, markers, hourly rotation, clock mapping, and
  the reconnect/pause state machine).
- `./gradlew :app:assembleDebug` builds the APK, which is uploaded as the
  artifact `open-holter-apk`.
- On a tag `v*` (e.g. `git tag v0.1.0 && git push origin v0.1.0`), CI also
  creates a GitHub Release and attaches the APK.

Locally (needs JDK 21 and the Android SDK): `cd android && ./gradlew test assembleDebug`.

Code layout:

| module | content |
|---|---|
| `core/` | plain Kotlin/JVM: CSV writers, hourly rotation, `ClockMapper`, `RecorderMachine` (connect / reconnect / pause / no-contact logic). Unit-tested. |
| `app/` | Android: `RecorderService` (foreground service, Polar SDK), `BootReceiver`, Compose UI (`ui/`), export (`Storage`). |

### Release signing (later)

For now the APK uses debug signing, and no signing material is in the
repository. To get stable, properly signed releases (so updates install over
each other from any build machine):

1. Create a keystore once, on your own computer, and keep it out of git
   (`*.jks` and `*.keystore` are git-ignored):
   `keytool -genkeypair -v -keystore release.jks -alias release -keyalg RSA -keysize 4096 -validity 10000`
2. In GitHub, go to *Settings → Secrets and variables → Actions* and add:
   - `RELEASE_KEYSTORE_BASE64`: output of `base64 -i release.jks`
   - `RELEASE_KEYSTORE_PASSWORD`, `RELEASE_KEY_ALIAS` (`release`), `RELEASE_KEY_PASSWORD`
3. Push a `v*` tag. CI then also builds `open-holter-<version>.apk`, signed
   with your key, and attaches it to the release.

Switching from the debug APK to the release APK changes the signing key once,
so export the recordings and uninstall the debug app first.

## Licences

- open-holter (this app): MIT, see [`../LICENSE`](../LICENSE).
- **Polar BLE SDK** (`com.github.polarofficial:polar-ble-sdk`, via JitPack),
  Copyright © Polar Electro Oy, used under the *Polar SDK Limited License
  Agreement*. The licence text and the SDK's third-party software list ship
  inside the app (`app/src/main/assets/licenses/`) and can be read on the
  *About* screen. The SDK licence also states that Polar devices and the SDK
  are not intended for medical purposes.
- This project is not affiliated with or endorsed by Polar. "Polar" and
  "H10" are trademarks of Polar Electro Oy and are used only to name the
  supported hardware.
- AndroidX, Jetpack Compose and Kotlin coroutines: Apache 2.0.
- No code was taken from GPL-licensed projects.
