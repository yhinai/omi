# Gemini Decision 1 and Vexo motion

Checked on this Mac on 14 November 2026. This is an account-access check and a small recorded-data experiment, not a production gesture-accuracy claim.

**Current live experiment:** the user requested reference-free classification after this benchmark. [The raw live preview](GEMINI_RAW_LIVE.md) sends all received six-axis samples in each window directly to Gemini, without reading this calibration. The reference-backed results below are historical comparisons, not a required input or an endorsement of that calibration's correctness.

## Model discovery and access

The credential supplied by the user was read from its existing private environment file, using only the Google/Gemini key variable. It was sent in the official API header, never printed or copied into project configuration. The paginated `v1beta/models` endpoint returned **64 models**. The complete private inventory is `logs/gemini-models-20261114.md`, with JSON metadata beside it.

Only **`gemini-decision-1`** advertised `createDecision`. An actual call to `POST https://generativelanguage.googleapis.com/v1beta/decisions` succeeded with this key. Listing a model alone would not establish inference access; the successful request does.

Google announced the model on **11 November 2026**. It scores a predefined set of outcomes instead of generating a chat response. Supported decision types are yes/no, a choice from fixed options, and an ordered rubric score. Google's launch reports 150–250 ms end-to-end on its workloads; our measurements below are separate. [Official launch](https://blog.google/innovation-and-ai/technology/developers-tools/gemini-decisions-api/)

The model has no function-calling, text-generation or Live API support. Your application receives a decision and implements any action itself. The separate Robotics ER and general thinking models are not substitutes for this endpoint. [Model documentation](https://aistudio.google.com/docs/models/gemini-decision-1)

## Calibration: three different meanings

1. **Physical calibration:** Which wrist, band mounting, neutral pose, gyro bias, comfortable movement directions and range? The current control session learns neutral/right/down. The saved longer reference also measures noise and repeatability.
2. **Classifier calibration:** Do confidence numbers predict correctness on this person's movements? A model's general calibration does not establish this for an unfamiliar sensor task.
3. **Examples in a request:** A few labeled reference poses can explain what left/right/up/down mean for this wearer. This is inference context, not training new model weights.

Google explicitly recommends choosing thresholds with labeled examples from the application's own traffic. Choice scores distribute probability across the options even when none fits, so an explicit unknown class and rejection policy matter. [Decisions API confidence guidance](https://aistudio.google.com/docs/decisions-api#confidence)

## What we measured

The existing left-wrist session supplied all motion; no microphone recording or screen content was sent. The first neutral and first left/right/up/down trials supplied the personal references. The four repeated directions, final neutral and ordinary activity were six held-out trials. Each condition used the same input values; XML versus JSON changed serialization.

| Input condition | Correct top choices | Median API round trip | Input tokens, six calls |
|---|---:|---:|---:|
| Schema and motion, JSON | 1/6 | 459 ms | 4,761 |
| Schema and motion, XML | 1/6 | 307 ms | 7,643 |
| Personal reference and motion, JSON | 6/6 | 406 ms | 6,651 |
| Personal reference and motion, XML | 6/6 | 397 ms | 10,559 |

The schema-only JSON condition confused several held poses with neutral or abstained; schema-only XML abstained on all six, correctly rejecting only ordinary activity. With references both encodings selected all six intended labels. Correct reference-backed confidence ranged approximately **0.60–0.88 for XML** and **0.62–0.88 for JSON**. These are top-choice observations, not a validated automatic-action threshold. A strict 0.90 threshold would accept none of these results.

The reference XML encoding consumed about **59% more input tokens** than reference JSON, with no accuracy improvement in this tiny sample. Timing was not randomized and has network variation; the 397 versus 406 ms difference does not establish that XML is faster. XML can be written more compactly than this serializer, so the result applies to these encodings rather than every XML document.

Limitations: six held-out trials from one session, one call per condition, no independent-day or strap-position test, and whole 6–20 second trials compressed to 12 time points. Measured API time excludes the time needed to collect that observation. This does not establish recognition of short live gestures or reliable clicks. A prior reference-cautious prompt was checked separately, then replaced by a neutral prompt to reduce evaluation bias; the table uses the neutral-prompt run.

Private evidence: `logs/gemini-motion-benchmark-20261114.json`, `logs/gemini-decision-smoke.json`. The benchmark sends a fixed 24 requests and has no Mac input backend.

## XML is packaging, not a physical reference

A schema can tell the API what acceleration, angular velocity, axes and timestamps mean. It does not tell it this wearer's neutral posture or what an intentional command looks like. In this experiment providing a personal reference mattered much more than markup format. Use compact JSON for the next prototype; include units, time deltas, wrist/mounting description, a few reference features and the observed motion window.

If “model information” means an XML hardware or body model, it may add useful context but does not supply sensor observations or personalize intent by itself. If it means trained classifier weights, the Decisions endpoint does not load those as a custom gesture classifier.

The official structured-data guidance serializes JSON as text in the shared input. Requests still use the documented JSON API envelope and typed questions. Each choice should have a clear description. [Structured-data input](https://aistudio.google.com/docs/decisions-api#structured-data)

## Recommended pipeline

```text
Vexo → existing Bluetooth owner → local motion processing
                                  ├─ calibrated tilt → smooth pointer/scroll
                                  └─ completed gesture/context → Gemini Decision
                                                                ↓
                                             allowed action or unknown/no action
```

Keep gravity/gyro processing, cursor velocity, neutral stop, emergency pause and disconnect handling local. Use a bounded cloud decision for discrete intents such as pointer versus scroll, next versus previous, pause, or dismiss. A roughly 400 ms API call can fit some discrete interactions; putting it in every cursor update adds latency and variable network delay to continuous control.

Use one in-flight decision at most, reject obsolete responses after a mode/source change, and send requests only on candidate events. Keep the input limited to motion and explicitly chosen context. A probability threshold, a margin between the best two choices, cooldown and an unknown/no-action route are candidate policies that need wearer evaluation. Greater apparent decisiveness obtained by simply lowering a threshold can increase accidental commands.

Persist the personal profile locally and reuse it. A short neutral/mounting check after putting the band on can be much less work than repeating all 17 trials. Recalibrate directions after a material fit or posture change. This is the proposed next calibration UX; the running controller still uses its explicit C/X/Y setup and does not automatically trust a profile across reconnects.

Decision 1 does not support context caching, so include only compact necessary reference features per request. Keep large reference files and ordinary sensor math local. [Model capability table](https://aistudio.google.com/docs/models/gemini-decision-1)

For “natural control,” also consider clear mode feedback and a clutch/recenter action before adding more gesture classes. Measuring confusion during ordinary typing and conversation is more useful than a large untested vocabulary. A local template classifier remains a comparison baseline; cloud reasoning should earn its place by improving held-out intent recognition or contextual mapping.

## Cost and quotas

Google's current paid list price is **$0.10 per million input tokens**, with zero output tokens billed. The latest 24-call run used 29,614 input tokens, equivalent to about **$0.003 at that list price**. This is arithmetic, not a bill or confirmation of this account's billing tier. The initial run and smoke are additional small calls. [Official pricing](https://aistudio.google.com/docs/pricing#gemini-decision)

At roughly 1,109 input tokens/request in our reference JSON condition, five event-triggered requests/minute would be about $0.033/hour at the paid list price. Sending 25 requests/second would instead be about $10/hour and would require 1,500 RPM and roughly 1.66M TPM. Input length and actual request frequency determine cost.

The documented free limits are 200 RPM / 50,000 TPM / 10,000 RPD; Tier 1 is 1,500 RPM / 1M TPM / 50,000 RPD. Thus a frame-by-frame 25 Hz design exceeds even Tier 1's documented token budget at this input size. Account quotas may differ; inspect AI Studio for the project. [Official rate limits](https://aistudio.google.com/docs/rate-limits#gemini-decision)

## Reproduce

Offline preparation sends nothing:

```sh
.venv/bin/python scripts/benchmark_gemini_motion.py \
  --reference logs/motion-calibration-20261114T215308431230Z.json \
  --output logs/gemini-motion-dry-run.json
```

Add `--run --env-file /absolute/path/to/private.env` to send the bounded experiment to Google's Decisions API. The file is parsed for the Google/Gemini key only, never sourced as shell code. HTTP errors stop the run without printing provider bodies or retrying. No API key, raw motion payload, audio, screen content or Mac actions are written by the report.

The tested API adapter is advisory and separate from the running controller. Connecting Gemini to automatic actions is a later integration step requiring shorter live windows, held-out wearer tests, rejection thresholds and stale-result handling.
