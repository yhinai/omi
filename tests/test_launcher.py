"""Check destructive reset behavior in a temporary directory only."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ClearTests(unittest.TestCase):
    def test_clear_removes_bridge_history_and_preserves_service_files(self):
        launcher = Path(__file__).resolve().parents[1] / 'run.sh'
        source = launcher.read_text()
        helper = source[source.index('clear_data() {'):source.index('if [ "$ACTION" = stop ]; then')]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logs = root / 'logs'
            logs.mkdir()
            removed = ['omi_transcript_events.jsonl', 'omi_transcript_events.jsonl.1',
                       'omi_live_transcript.log', 'omi_bridge.log.1',
                       'omi_bridge.console.log', '2026-10-24.md', 'capture.wav']
            kept = ['omi_bridge.launcher.lock', 'omi_research.pid']
            for name in removed + kept:
                (logs / name).write_text('fixture')
            (root / '.env').write_text('configuration')
            (logs / 'public-share').mkdir()
            (logs / 'public-share/server.py').write_text('service')
            (logs / 'editions/day').mkdir(parents=True)
            (logs / 'editions/day/edition.json').write_text('{}')
            outside = root / 'outside'
            outside.mkdir()
            (outside / 'safe').write_text('safe')
            (logs / 'linked.wav').symlink_to(outside, target_is_directory=True)
            result = subprocess.run(['bash', '-c', helper + '\nclear_data',
                                     'test', directory, sys.executable],
                                    env={'DIR': directory, 'CLI_PYTHON': sys.executable},
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(all(not (logs / name).exists() for name in removed))
            self.assertFalse((logs / 'editions').exists())
            self.assertFalse((logs / 'linked.wav').is_symlink())
            self.assertTrue(all((logs / name).exists() for name in kept))
            self.assertEqual((logs / 'public-share/server.py').read_text(), 'service')
            self.assertEqual((root / '.env').read_text(), 'configuration')
            self.assertEqual((outside / 'safe').read_text(), 'safe')

    def test_clear_dispatch_stops_clears_and_exits_without_startup(self):
        launcher = Path(__file__).resolve().parents[1] / 'run.sh'
        source = launcher.read_text()
        dispatch = source[source.index('if [ "$ACTION" = clear ]; then'):source.index('if [ "$ACTION" = status ]; then')]
        result = subprocess.run(['bash', '-c',
            'ACTION=clear\nstop_bridge() { echo stopped; }\n'
            'check_ports() { :; }\nclear_data() { echo cleared; }\n'
            + dispatch + '\necho unexpected-restart'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines()[:2], ['stopped', 'cleared'])
        self.assertNotIn('unexpected-restart', result.stdout)
