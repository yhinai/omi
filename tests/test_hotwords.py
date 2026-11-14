"""Decoder prompt biasing toward names and domain vocabulary."""
import os
import types
import unittest
from unittest import mock

import whisper
from common import RATE


class HotwordPromptTests(unittest.TestCase):
    def test_default_prompt_names_the_owner_and_the_domain(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OMI_HOTWORDS", None)
            prompt = whisper.hotword_prompt()
        self.assertIn("Yahya Alhinai", prompt)
        self.assertIn("Omi", prompt)
        self.assertIn("KV cache", prompt)

    def test_environment_override_wins(self):
        with mock.patch.dict(os.environ, {"OMI_HOTWORDS": "SGLang Triton"}):
            self.assertEqual(whisper.hotword_prompt(), "SGLang Triton")

    def test_whitespace_is_collapsed(self):
        self.assertEqual(whisper.hotword_prompt("  a\n\tb   c "), "a b c")

    def test_prompt_is_capped_because_whisper_keeps_only_the_tail(self):
        self.assertEqual(len(whisper.hotword_prompt("x" * 5000)), whisper.HOTWORDS_MAX_CHARS)

    def test_empty_prompt_disables_biasing(self):
        self.assertEqual(whisper.hotword_prompt("   "), "")
        with mock.patch.dict(os.environ, {"OMI_HOTWORDS": ""}):
            self.assertEqual(whisper.hotword_prompt(), "")


class DegenerateTests(unittest.TestCase):
    """Guards against prompt echo and decoder loops."""

    POISON = "Names and terms, " * 40

    def test_the_actual_poisoned_transcript_is_rejected(self):
        self.assertTrue(whisper.degenerate(self.POISON, whisper.DEFAULT_HOTWORDS, 1.28))

    def test_ordinary_speech_passes(self):
        text = ("Okay so the next thing is the KV cache routing and how vLLM handles "
                "prefix caching across replicas in production")
        self.assertFalse(whisper.degenerate(text, whisper.DEFAULT_HOTWORDS, 6.0))

    def test_physically_impossible_output_is_rejected(self):
        self.assertTrue(whisper.degenerate("word " * 40, "", 1.0))

    def test_repetition_is_rejected_without_any_prompt(self):
        self.assertTrue(whisper.degenerate("the the the the the the the the the", "", 30.0))

    def test_prompt_words_dominating_the_output_are_rejected(self):
        echo = "Yahya Alhinai Oman Omi MLX vLLM SGLang Triton Helion Mojo CUDA"
        self.assertTrue(whisper.degenerate(echo, whisper.DEFAULT_HOTWORDS, 30.0))

    def test_short_utterances_are_never_flagged(self):
        self.assertFalse(whisper.degenerate("Okay.", "", 0.4))
        self.assertFalse(whisper.degenerate("", "", 0.4))

    def test_default_prompt_avoids_list_heading_phrasing(self):
        self.assertNotIn("Names and terms:", whisper.DEFAULT_HOTWORDS)


class DecodeGuardTests(unittest.TestCase):
    def recognizer(self, replies, languages=("en",), language="en"):
        recognizer = object.__new__(whisper.Recognizer)
        recognizer.hotwords = whisper.DEFAULT_HOTWORDS
        recognizer.language = language
        recognizer.languages = set(languages)
        calls = []

        def fake(pcm, prompt, temperature, language=whisper.DEFAULT):
            calls.append((prompt, temperature, language))
            return replies[len(calls) - 1]

        recognizer._transcribe = fake
        return recognizer, calls

    def test_prompt_echo_is_retried_without_the_prompt_then_dropped(self):
        poison = ("Names and terms, " * 40, "en", 0.1)
        recognizer, calls = self.recognizer([poison, poison])
        text, _, reason = recognizer.decode(b"\x00" * int(16000 * 2 * 1.28))
        self.assertEqual(text, "")
        self.assertEqual(reason, "degenerate")
        self.assertEqual(calls[0], (whisper.DEFAULT_HOTWORDS, 0.0, whisper.DEFAULT))
        self.assertEqual(calls[1], ("", 0.2, None))

    def test_a_clean_retry_is_published(self):
        poison = ("Names and terms, " * 40, "en", 0.1)
        recognizer, _ = self.recognizer([poison, ("the next topic is caching", "en", 0.1)])
        text, _, reason = recognizer.decode(b"\x00" * int(16000 * 2 * 1.28))
        self.assertEqual(text, "the next topic is caching")

    def test_clean_first_pass_is_not_retried(self):
        recognizer, calls = self.recognizer([("a normal sentence about caches", "en", 0.1)])
        text, _, reason = recognizer.decode(b"\x00" * int(16000 * 2 * 3))
        self.assertEqual(text, "a normal sentence about caches")
        self.assertEqual(len(calls), 1)

    def test_language_outside_the_allowlist_is_not_published(self):
        recognizer, _ = self.recognizer([("بسم الله الرحمن الرحيم", "ar", 0.1)])
        text, _, reason = recognizer.decode(b"\x00" * int(16000 * 2 * 2))
        self.assertEqual(text, "")
        self.assertEqual(reason, "language")

    def test_arabic_is_published_when_allowlisted(self):
        recognizer, _ = self.recognizer([("مرحبا بالجميع", "ar", 0.1)], languages=("en", "ar"))
        text, _, reason = recognizer.decode(b"\x00" * int(16000 * 2 * 2))
        self.assertEqual(text, "مرحبا بالجميع")


class LanguageTests(unittest.TestCase):
    def test_setting_parses_auto_and_codes(self):
        self.assertEqual(whisper.language_setting("EN"), "en")
        self.assertIsNone(whisper.language_setting("auto"))
        self.assertIsNone(whisper.language_setting(""))

    def test_allowlist_ignores_blanks(self):
        self.assertEqual(whisper.language_allowlist("en, ar ,"), {"en", "ar"})
        self.assertEqual(whisper.language_allowlist(""), set())


class SingleEncoderAndGuardTests(unittest.TestCase):
    def test_unspaced_repetition_is_caught_by_compression_guard(self):
        self.assertTrue(whisper.degenerate("ها" * 40, "", 4.0))

    def test_single_encoder_reuses_features_across_retry_and_detects_no_speech(self):
        class FakeNPBuf:
            def astype(self, _):
                return self

            def __truediv__(self, _):
                return "float_audio"

        rec = object.__new__(whisper.Recognizer)
        rec.np = types.SimpleNamespace(float32="float32", frombuffer=lambda pcm, dtype: FakeNPBuf())
        rec.mx = types.SimpleNamespace(
            gpu="gpu",
            float16="float16",
            set_default_device=lambda dev: None,
            synchronize=lambda: None,
        )
        rec._log_mel = lambda audio, n_mels=128: "raw_mel"

        class FakeArray:
            def astype(self, _):
                return self

            def __getitem__(self, _):
                return "mel_batch"

        rec._pad_or_trim = lambda mel, n, axis=-2: FakeArray()
        rec._n_frames = 3000
        rec.hotwords = whisper.DEFAULT_HOTWORDS
        rec.language = "en"
        rec.languages = {"en"}
        rec._cached_features = (None, None)
        rec._prompt_token_cache = {whisper.DEFAULT_HOTWORDS: [1, 2, 3]}
        rec._last_no_speech = False
        rec._tokenizer = types.SimpleNamespace(encode=lambda s: [4, 5])

        encoder_calls = []
        decode_options_seen = []
        replies = iter([
            types.SimpleNamespace(text="Names and terms, " * 20, language="en", no_speech_prob=0.1, avg_logprob=-0.2),
            types.SimpleNamespace(text="Clean retry result.", language="en", no_speech_prob=0.1, avg_logprob=-0.3),
            types.SimpleNamespace(text="Thank you.", language="en", no_speech_prob=0.85, avg_logprob=-1.4),
        ])

        def fake_encoder(mel_batch):
            encoder_calls.append(mel_batch)
            return ["encoded_features"]

        def fake_decode(features, opts):
            self.assertEqual(features, "encoded_features")
            decode_options_seen.append(opts)
            return next(replies)

        rec.model = types.SimpleNamespace(
            dims=types.SimpleNamespace(n_mels=128),
            encoder=fake_encoder,
            decode=fake_decode,
        )
        rec._decoding_options = lambda **kw: types.SimpleNamespace(**kw)

        pcm1 = bytes(RATE * 2)  # 1.0s audio -> degenerate first pass + clean retry on SAME pcm1 object
        text, _, reason = rec.decode(pcm1)
        self.assertEqual(text, "Clean retry result.")
        self.assertEqual(reason, "")
        self.assertEqual(len(encoder_calls), 1)  # Encoder ran ONCE across initial decode + retry!
        self.assertEqual(len(decode_options_seen), 2)
        self.assertEqual(decode_options_seen[0].sample_len, 56)

        pcm2 = bytes(RATE * 4)  # New PCM -> encodes once, hits no_speech guard
        text2, _, reason2 = rec.decode(pcm2)
        self.assertEqual((text2, reason2), ("", "no_speech"))
        self.assertEqual(len(encoder_calls), 2)


if __name__ == "__main__":
    unittest.main()
