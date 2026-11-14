# Live motion training

From the Mac checkout, with the existing Vexo capture running:

```sh
./omi motion
```

The display shows accelerometer (g), gyroscope (degrees/second), roll/pitch,
a neutral-reference tilt preview, proposed pointer/scroll directions and recent
built-in gestures (tap, double-tap, tilt, orientation and motion start/stop).
A three-reversal gyro heuristic displays a wave candidate. It has not been trained
or validated as an intentional gesture classifier.

This version only displays motion; it contains no Mac event-posting implementation.

- Hold your wrist comfortably and press `c` to calibrate its neutral orientation.
- Use `+`/`-` to adjust preview sensitivity (0.1–3x).
- `p` toggles a precision scale at one fifth of normal sensitivity.
- `d` cycles the neutral dead zone through 3, 6 and 10 degrees.
- `m` switches the proposal label between pointer and scroll. These remain proposals,
  not OS pointer movement or real scroll events.
- `q` or Ctrl+C closes the view. Capture/captions continue; motion/events turn off
  when the monitor's three-second lease expires.

`./omi motion --json --seconds 10` streams local telemetry/proposals for inspection.
It does not record or expose microphone audio. Do not run a second Bluetooth client.
The existing owner enables 25 Hz motion and events only while the view is active.
It updates a private atomic snapshot at up to 10 Hz, with a 30-event ring. The view
refreshes at up to 10 Hz. These are monitor rates, not sensor-to-display latency claims.
Closing/replacing the source invalidates its generation and calibration; stale samples
are marked unavailable. Optional telemetry disk errors do not prevent microphone cleanup.
The viewer and capture each retain a process lock to avoid duplicate owners.

## Expand control deliberately

Next steps are hypotheses to test with normal wearing position and repeated motions:

1. Neutral calibration, axis direction, dead zone, sensitivity and precision mode.
2. Stable tilt-based pointer velocity and scroll previews, with smoothing measured
   against response delay. Absolute point-at-screen mapping is not available from
   gravity/gyro alone; yaw integration drifts and acceleration is not hand position.
3. Explicitly engaged pointer/scroll modes with a clutch/recenter gesture, bounded
   velocity, stale/disconnect cancellation and conventional keyboard equivalents.
4. Discrete tap/select, swipe/previous-next, and wave/Escape candidates, with cooldown,
   hysteresis and false-activation tests during ordinary conversation/walking.
5. Measure repeatability, target acquisition, fine corrections and command-to-screen
   delay alongside unchanged audio health before enabling real Mac controls.

The accelerometer/gyro cannot establish finger-pinching or hand tracking accuracy.
Learn intentional wrist movements from this device's data, rather than importing accuracy
claims from other wearable sensor types. The separate memory-timeline research remains
a proposal; this monitor does not implement that application.

## Guided left-wrist reference

```sh
./omi motion --calibrate --hand left
```

From the regular view, `r` opens the guided reference too. Each trial waits for Space
and then counts down. The 17 trials take about 2.5 minutes of motion plus rests: neutral,
left/right/up/down poses repeated twice, slow/fast sweeps, scrolling movements, waves,
taps, double taps, ordinary non-command activity and return to neutral.

The reference records unique timestamped accel/gyro samples and detected events, using
25 Hz sensor data from a bounded 50-sample ring. A final private JSON report under
`logs/motion-calibration-*.json` includes labeled raw motion, neutral noise, gyro bias,
observed sampling rate, missing-motion counts, host telemetry age, directional ranges,
rotation speeds, repeated-pose differences and a suggested dead zone. It contains no
audio and does not apply Mac controls or train a validated classifier.

Connection interruption discards the current trial. Completed trials remain, with their
source generations; after a reconnect, reconfirm the same wrist/strap position before
continuing. No trial joins samples from disconnected sources. If you move the band or
change your control posture, start a new neutral reference. `r` repeats the current
trial; `q` quits. References are saved only after the full session completes.

The reported speed/range comes from the IMU. Gravity-based tilt is approximate during
fast acceleration. Host telemetry age does not establish physical response latency.
Repeated validation and ordinary-activity tests are still needed before claiming precise
selection, stable pointing or low false-activation rates.

When the guided reference completes, the viewer prints its summary once and exits,
leaving captions running. The raw personal reference and derived analyses stay private
under ignored `logs/`; the completion flag means all labeled trials were collected,
not that the gestures are accurate enough to control the Mac. Slow/fast separation,
pose repeatability, neutral return, single/double-tap event counts and ordinary-activity
responses must be assessed before selecting mappings. Absolute angular speed alone
can confuse deliberate waves with scroll motions.

The Bluetooth owner also requests the band's battery stream every 30 seconds.
`./omi status --json` includes fresh battery percent, charging state and voltage;
stale/disconnected readings are omitted. This does not open a competing USB or BLE
connection. With the motion view closed, the private status snapshot is updated at
most once per second, and motion/event streams remain off.
