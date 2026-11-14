# Raw Vexo motion → Gemini Decision live preview

This is the current live experiment. It does **not** use the earlier calibration as a reference. That recorded reference remains on disk for historical comparisons, but the live preview never reads it.

Run the launcher prepared on this Mac:

```sh
/Users/alhinai/vexo-captions/logs/Gemini-Live-Motion.command
```

The earlier `logs/Gemini-Motion-Test.command` launcher now opens this same raw live preview. Reproducing the historical reference benchmark requires explicitly running its separate benchmark script.

Or run the script directly with your existing credential file:

```sh
.venv/bin/python scripts/live_gemini_motion.py --env-file /absolute/path/to/private.env
```

The window starts in manual mode. **Space** sends one recent motion window. **A** toggles automatic classification at no more than one request every two seconds. **Q** or Escape closes it. The process ends after 120 seconds or can make at most 24 requests; reaching the request cap pauses automatic requests until the window closes. The cap and timeout are explicit command arguments, bounded to 24 requests and five minutes.

Try holding still, tilting slowly, turning, moving back and forth, then a gentle shake. Do each movement for 2–3 seconds. A held pose with no continuing movement can correctly be classified as still; this experiment identifies physical motion, not an intended screen command.

The input contains every received sample in the selected roughly 0.8–1.2 second window, with no resampling or numeric rounding: the band's millisecond timestamp and accelerometer X/Y/Z in g plus gyroscope X/Y/Z in degrees/second. These are the SDK's decoded sensor readings in physical units, not electrical sensor-register values. Units and column names accompany them. Acceleration includes gravity. No saved neutral, direction mapping, labeled examples, calculated roll/pitch, motion summaries, microphone audio or screenshots are included.

Output choices are `still`, `tilting`, `rotating`, `back_and_forth`, `shaking`, `impact`, and `unknown`. A turn around gravity and a tilt that changes gravity direction have separate descriptions, but these are hypotheses to validate on real movements. An impact is not proof of a deliberate band tap. The confidence is the model's score, not an established accuracy for this person.

The existing Bluetooth owner publishes motion; this viewer opens no second band connection. It uses the same single-viewer lock as the motion/control tools, so close another motion viewer first while leaving captions open. The lease enables 25 Hz motion and faster local snapshot publication. One cloud request can be in flight at a time; no unbounded queue is created. Responses from an old source generation, a paused request epoch, stale data, or a request window older than two seconds are rejected. Detected telemetry holes are not compressed into a contiguous gesture.

There is **no Mac input path** in this preview. Classification labels do not click, scroll or move the pointer. Screen-direction mapping and command intent are separate future tests; a raw six-axis stream provides no absolute screen coordinate reference.

Private status is `logs/gemini-live-status.json`. Small per-run JSON reports contain labels, scores, token counts, request latency, sample count and window duration; they do not retain raw sensor payloads or credentials. API failures pause/end the experiment without printing provider bodies or automatically retrying. A request already sent may finish after quitting, but its result cannot perform an action.

A live, reference-free smoke request returned `still` at about 83% confidence with a 532 ms API round trip. This establishes delivery and parsing only, not correctness across movements. The offline tests cover fresh-window construction, duplicate/invalid sample handling, discontinuities, wrong-band/stale/disconnected input, source-generation changes, raw sample preservation, and rejection of stale responses.
