"""One parser must preserve literal secrets and process-over-file precedence."""
from pathlib import Path
import tempfile
import unittest
import config


class ConfigurationTests(unittest.TestCase):
    def test_parser_never_executes_shell_and_uses_last_assignment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, '.env')
            path.write_text('export VALUE=first\nVALUE="second # literal" # comment\n'
                            "SECRET='$(touch should-never-exist)'\nBROKEN-KEY=ignored\n")
            values = config.load_env(path, {"VALUE": "process"})
            self.assertEqual(values, {"VALUE": "process", "SECRET": "$(touch should-never-exist)"})
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_bad_quoting_reports_location_without_echoing_value(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, '.env')
            path.write_text('SECRET="private-unclosed\n')
            with self.assertRaises(ValueError) as error:
                config.load_env(path, {})
            self.assertIn(':1', str(error.exception))
            self.assertNotIn('private', str(error.exception))
