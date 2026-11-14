"""Cloud settings are explicit, English-only and safe to inspect in diagnostics."""
import unittest

from backend_settings import DeepgramConfig


class DeepgramConfigTests(unittest.TestCase):
    def test_defaults_and_secret_representation(self):
        config = DeepgramConfig.from_env({"DEEPGRAM_API_KEY": "private-test-value"})
        self.assertEqual((config.model, config.language, config.endpointing_ms, config.diarize),
                         ("nova-3", "en", 300, False))
        self.assertNotIn("private-test-value", repr(config))

    def test_explicit_settings_and_english_dialect(self):
        config = DeepgramConfig.from_env({"DEEPGRAM_API_KEY": "key", "OMI_LANGUAGE": " EN-US ",
                    "OMI_LANGUAGES": "en, EN-us", "OMI_DEEPGRAM_ENDPOINTING_MS": "500",
                    "OMI_DEEPGRAM_DIARIZE": "TRUE"})
        self.assertEqual(config.language, "en-us")
        self.assertEqual(config.endpointing_ms, 500)
        self.assertTrue(config.diarize)

    def test_rejects_invalid_settings_without_echoing_values(self):
        bad = [{"DEEPGRAM_API_KEY": ""}, {"DEEPGRAM_API_KEY": "secret with whitespace"},
               {"OMI_LANGUAGE": "ar"}, {"OMI_LANGUAGE": "auto"}, {"OMI_LANGUAGES": "en,ar"},
               {"OMI_LANGUAGES": ""}, {"OMI_DEEPGRAM_MODEL": "flux-general-en"},
               {"OMI_DEEPGRAM_ENDPOINTING_MS": "0"}, {"OMI_DEEPGRAM_ENDPOINTING_MS": "5001"},
               {"OMI_DEEPGRAM_ENDPOINTING_MS": "secret-invalid-setting"}, {"OMI_DEEPGRAM_DIARIZE": "yes"}]
        for override in bad:
            with self.subTest(override=override), self.assertRaises(ValueError) as caught:
                DeepgramConfig.from_env({"DEEPGRAM_API_KEY": "private-test-value", **override})
            self.assertNotIn("private-test-value", str(caught.exception))
            self.assertNotIn("secret-invalid-setting", str(caught.exception))
