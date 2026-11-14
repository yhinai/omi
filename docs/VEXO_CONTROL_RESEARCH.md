# Vexo 43: useful motion controls and the evidence behind them

Researched and checked on the Mac, 14 November 2026. Hardware limits below are the kit's documented limits, not a measured performance guarantee for this radio, battery, posture or firmware session.

## Recommendation

Use the band first as a tilt joystick for scrolling and coarse pointer movement, with a separate precision scale. Add a small, personally trained command vocabulary after measuring false activations. Keep the motion-to-input path local and deterministic. Speech recognition and an LLM need not be in that path.

The present implementation is `./omi motion --control`. `./omi motion` remains a display-only preview. Actual Mac input is opt-in and starts paused. Training accuracy and wearer usability remain to be measured; the 17-trial reference was a baseline, not a trained recognizer.

## What the device provides

Vexo's API exposes accelerometer readings in g, gyroscope readings in degrees/second, sensor timestamps/sample numbers, battery, LED feedback and built-in motion events. Its documented six-axis Bluetooth ceiling is 416 Hz and USB ceiling is 833 Hz; one-sensor modes can go faster. Those ceilings are unnecessary starting points for wrist control. We currently request **25 Hz** in both control and preview modes. An attempted 50 Hz live run showed repeated audio packet gaps; that observation does not isolate the radio or firmware as the cause, so the higher rate was rolled back. [Vexo API](https://github.com/VexoAI-Org/vexo-open-kit/blob/main/docs/api.md)

The contract says audio and motion share a send queue. It recommends staying at or below 208 Hz with both sensors alongside audio unless loss-free operation is observed. Its built-in tilt event requires a large direction change held for two seconds, so that event is unsuitable for responsive continuous cursor movement. We read raw motion instead. [Vexo contract](https://github.com/VexoAI-Org/vexo-open-kit/blob/main/contract/contract.md)

| Intended action | Practical approach | Main limitation |
|---|---|---|
| Read a page or captions | Tilt up/down for scroll velocity; neutral stops | Wrist posture and dead zone need tuning |
| Move the pointer | Calibrated right/down axes, bounded velocity, precision toggle | Target selection and fatigue are unmeasured |
| Click | Explicit click shortcut; optional delayed single-band-tap click | Band taps can be missed or triggered by bumps |
| Pause control | Escape globally, explicit toggle, or double-tap pause | Double-tap alone is not the only stop path |
| Slides/media/bookmarks | Personally trained discrete gestures | Requires action mapping and held-out negative testing; not implemented yet |
| Point at an exact screen location | External reference or a constrained calibrated model | A single wrist IMU does not directly measure hand position |
| Pinch/type with fingers | Additional sensing or a separately validated specialized model | No validated finger model or EMG hardware in this setup |

An IMU measures motion and orientation, not position directly. Integrating acceleration twice accumulates error; a general free-space hand tracker needs more constraints or references. That is why cursor velocity is our starting recommendation. [x-io position FAQ](https://x-io.co.uk/contact/), [Analog Devices integration error example](https://www.analog.com/media/en/technical-documentation/application-notes/513772624AN602.pdf)

## The complete local pipeline

Band accelerometer/gyro → existing Bluetooth owner in `vexo_source.py` → `MotionTelemetry` → atomic local snapshot → `ControlSession` calibration and motion checks → `MacInput` → Quartz pointer/scroll/click events.

The controller shares the existing owner and never opens another Bluetooth connection. In control mode, snapshots publish at up to 25 Hz instead of the preview's 10 Hz; the controller polls at about 50 Hz. Sampling and polling rates are not physical movement-to-screen latency measurements. Capture continues independently to the chosen speech recognizer.

The controller requires fresh samples and snapshots, the correct band number, valid finite motion values, an explicit neutral/right/down setup and explicit engagement. A reconnect discards the session mapping. Stale data, unusually large acceleration, a control-loop stall, a mode change or Escape pauses input. Returning inside the dead zone stops movement immediately. Single taps do nothing until tap clicking is explicitly enabled; a single click waits 550 ms so a following double-tap can cancel it. These interaction choices are provisional, not a certified classifier.

Mac input uses Apple's Quartz APIs. The terminal running control needs Accessibility and global keyboard access. Escape/global shortcuts are checked without saving typed text. [Apple Quartz Event Services](https://developer.apple.com/documentation/coregraphics/quartz-event-services), [Apple accessibility trust](https://developer.apple.com/documentation/applicationservices/1459186-axisprocesstrustedwithoptions)

## What our reference actually established

Private source: `logs/motion-calibration-20261114T215308431230Z.json`, left wrist, snug strap. The neutral noise p95 was **1.51 degrees**, suggesting a **5.52-degree** dead zone. Repeated directions differed by approximately 12.6 degrees left, 23.0 right, 8.2 up and 10.8 down. Returning to neutral differed by 8.9 degrees. The reference spans reconnects; each trial used one source generation.

These observations justify learning the current neutral and comfortable right/down axes before control. They do not establish fine pointing accuracy, universal gestures, session-to-session stability or a false-command rate. The new controller uses the saved noise recommendation, then asks for a short posture-specific mapping.

## How to get used to it

1. Rest the elbow and wear the band in the same snug left-wrist position. Keep the physical mouse available.
2. Press C at neutral, X while tilted comfortably right, then Y while tilted comfortably down. Use distinct wrist movements. Return to neutral and press Space. Each reference tilt should be 10–75 degrees; closer-to-neutral comfortable movements are preferable to extreme poses.
3. Move between large targets before trying small ones. Return to neutral after each move. P switches to one-fifth precision sensitivity; +/- adjusts speed.
4. Press M to select scroll; the mode change pauses. Return to neutral and press Space again. Try following a paragraph slowly, then quickly.
5. Enter clicks while the control window has focus. In another app use Control+Option+Command+Enter. The same chord with Space toggles control, M changes mode, and P changes precision. Escape pauses globally. Q quits from the control window.
6. Only after pointer/scroll feels predictable, press T to try band tap clicking in a harmless target area. Double-tap pauses. The wearer tap/double-tap reliability still needs testing.

The control window reports ACTIVE/PAUSED and the current mode. It never auto-engages from motion or after a reconnect.

## Improvements supported by the research

**Combine gyro and gravity for better orientation.** Apple distinguishes gravity, user acceleration and attitude. x-io Fusion provides a maintained complementary-filter implementation supporting accelerometer+gyro operation. A future fusion pass should use sensor timestamps, bias estimates and acceleration rejection; it still needs recentering for heading drift. The current controller uses gravity-derived roll/pitch and a dynamic-acceleration pause, not a full AHRS implementation. [Apple device motion concepts](https://developer.apple.com/documentation/coremotion/getting-processed-device-motion-data), [x-io Fusion](https://github.com/xioTechnologies/Fusion)

**Tune filtering against measured lag.** The One Euro filter adapts its cutoff to movement speed, aiming to reduce slow-motion jitter without excessive fast-motion delay. Compare it with the current short exponential velocity smoothing using recorded motion replay and target tasks before choosing parameters. [Authors' implementation and tuning guide](https://gery.casiez.net/1euro/)

**Train a few commands, including a rejection class.** Vexo already ships a sample recognizer: motion segmentation, resampling to 24 points, and nearest-example comparison across six channels. Its authors explicitly label wrist thresholds unvalidated. Adapt that logic to the existing capture owner instead of connecting its browser demo to the live band. DTW is a next comparison for gesture speed variation, not a reason to put a large model in every control tick. [Vexo gesture trainer](https://github.com/VexoAI-Org/vexo-open-kit/tree/main/projects/gesture-trainer), [MathWorks quaternion-DTW example](https://www.mathworks.com/help/fusion/ug/gesture-recognition-using-inertial-measurement-units.html), [IMU gesture-learning research](https://pmc.ncbi.nlm.nih.gov/articles/PMC6767360/)

**Collect useful training data, rather than every possible motion.** Start with 3–5 distinct commands and roughly 10–15 deliberate repetitions each at comfortable slow/normal/fast speeds, then repeat on another session. Record ordinary typing, conversation, walking and adjusting the band as negative examples. Keep whole sessions out of training for evaluation. Count wrong commands during ordinary use, unknown/rejected gestures, per-command misses and confusion pairs; add longer background tests before making per-hour reliability claims. These are proposed experiments, not measured results.

**Evaluate actual utility.** Measure target acquisition at several sizes/distances, overshoot, corrections, unintended clicks, scrolling accuracy, response time and fatigue. Compare against the physical trackpad and the same task before tuning. Preserve caption packet-loss, reconnect and recognition metrics while testing motion rates. More hertz alone is not a success criterion.

**Use a React UI as feedback, if desired.** A browser front end can show live axes, a wrist model, targets, neutral zones, gesture confidence, training examples and a confusion matrix. It should receive local telemetry from our backend, leaving the existing single Bluetooth owner in charge. Direct Web Bluetooth additionally requires a secure context and a user-triggered chooser; it is not our Mac input backend. A React dashboard is not implemented by this change. [Chrome Web Bluetooth documentation](https://developer.chrome.com/docs/capabilities/bluetooth)

**Add sensing only for a specific missing capability.** A camera can provide hand pose; sEMG can provide muscular signals for subtle finger interaction. These are separate hardware/data paths, not capabilities obtained by downloading a model for Vexo. [Apple hand-pose example](https://developer.apple.com/documentation/vision/detecting-hand-poses-with-vision), [Meta sEMG datasets and interfaces](https://ai.meta.com/blog/open-sourcing-surface-electromyography-datasets-neurips-2024/)

**Avoid a discontinued dependency.** Doublepoint's official notice says WowMouse and its Touch SDK are being discontinued, with no planned open sourcing. They are useful historical examples, not the foundation recommended for this project now. [Doublepoint notice](https://docs.doublepoint.com/)

## Verification and limits

Offline motion/control tests cover explicit engagement, learned axes, overlapping-axis rejection, neutral stop, precision, bounded movement, stale/invalid/disconnected data, wrong-band rejection, reconnect invalidation, mode changes, click cooldown, delayed taps, double-tap cancellation and loop stalls. A native temporary test window received actual pointer movement, scroll and click events from the Quartz adapter; it was then closed and pointer position restored.

That verifies the software output path. Comfortable operation on the wearer's wrist, real tap reliability, physical latency and precision are separate live tests. No accuracy percentage, endurance figure or mouse-replacement claim follows from the adapter test.
