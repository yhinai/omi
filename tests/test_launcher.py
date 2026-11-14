"""Check reset behavior in a temporary directory only."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ClearTests(unittest.TestCase):
    def dispatch(self):
        source = (Path(__file__).resolve().parents[1] / 'run.sh').read_text()
        start = source.index('if [ "$ACTION" = restart ] || [ "$ACTION" = restart-clean ]; then')
        return source[start:source.index('PID="$("$PYTHON"', start)]

    def run_dispatch(self, action, stubs):
        return subprocess.run(['bash', '-c',
            'set -e\nACTION=' + action + '\nstop_bridge() { echo stopped; }\n'
            'check_ports() { echo checked; }\nclear_data() { echo cleared; }\n'
            + stubs + self.dispatch() + '\necho startup'], capture_output=True, text=True)

    def test_restart_preserves_history_and_continues_to_startup(self):
        result = self.run_dispatch('restart', '')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(),
                         ['stopped', 'checked', 'checked', 'startup'])

    def test_restart_clean_clears_and_continues_to_startup(self):
        result = self.run_dispatch('restart-clean', '')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(),
                         ['stopped', 'checked', 'cleared', 'checked', 'startup'])

    def test_a_failed_stop_or_port_check_never_reaches_clear_or_startup(self):
        for action in ('restart', 'restart-clean'):
            for failing in ('stop_bridge', 'check_ports'):
                with self.subTest(action=action, failing=failing):
                    result = self.run_dispatch(action, failing + '() { return 1; }\n')
                    self.assertNotEqual(result.returncode, 0)
                    self.assertNotIn('cleared', result.stdout)
                    self.assertNotIn('startup', result.stdout)

    def test_clear_archives_history_and_preserves_service_files(self):
        launcher = Path(__file__).resolve().parents[1] / 'run.sh'
        source = launcher.read_text()
        helper = source[source.index('clear_data() {'):source.index('if [ "$ACTION" = stop ]; then')]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logs = root / 'logs'
            logs.mkdir()
            archived = ['omi_transcript_events.jsonl', 'omi_transcript_events.jsonl.1',
                        'omi_live_transcript.log', 'omi_bridge.log.1',
                        'omi_bridge.console.log', '2026-10-24.md', 'capture.wav']
            kept = ['omi_bridge.launcher.lock', 'omi_research.pid']
            for name in archived + kept:
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
            stamps = list((logs / 'archive').iterdir())
            self.assertEqual(len(stamps), 1, 'one archive directory per clear')
            saved = stamps[0]
            self.assertTrue(all((saved / name).exists() for name in archived))
            self.assertTrue((saved / 'editions/day/edition.json').exists())
            self.assertTrue(all(not (logs / name).exists() for name in archived))
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

    def test_same_timestamp_archives_preserve_both_histories(self):
        source = (Path(__file__).resolve().parents[1] / 'run.sh').read_text()
        helper = source[source.index('clear_data() {'):source.index('if [ "$ACTION" = stop ]; then')]
        # Hold the UTC stamp fixed without relying on machine timing.
        helper = helper.replace('datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")', '"20261114T120000Z"')
        with tempfile.TemporaryDirectory() as directory:
            logs = Path(directory, 'logs')
            logs.mkdir()
            for text in ('first history', 'second history'):
                (logs / 'omi_transcript_events.jsonl').write_text(text)
                result = subprocess.run(['bash', '-c', helper + '\nclear_data'],
                                        env={'DIR': directory, 'CLI_PYTHON': sys.executable},
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            histories = sorted(path.read_text() for path in (logs / 'archive').glob('*/omi_transcript_events.jsonl'))
            self.assertEqual(histories, ['first history', 'second history'])

    def test_port_check_fails_if_lsof_is_missing(self):
        source = (Path(__file__).resolve().parents[1] / 'run.sh').read_text()
        helper = source[source.index('check_ports() {'):source.index('stop_bridge() {')]
        result = subprocess.run(['/bin/bash', '-c', helper + '\ncheck_ports'],
                                env={'PATH': '/nonexistent'}, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('lsof is required', result.stderr)


class DependencyTests(unittest.TestCase):
    def test_dependency_check_enforces_supported_aiohttp_range(self):
        source = (Path(__file__).resolve().parents[1]/'run.sh').read_text()
        line = next(line for line in source.splitlines() if "import aiohttp; assert" in line)
        check = line.split("-c '",1)[1].split("'",1)[0]
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)/'aiohttp.py'
            for version, supported in [('3.9.0',False),('3.14.4',False),('3.14.5',True),('3.15.0',True),('4.0.0',False)]:
                with self.subTest(version=version):
                    package.write_text('__version__ = '+repr(version)+'\n')
                    result = subprocess.run([sys.executable,'-B','-c',check],cwd=directory,capture_output=True,text=True)
                    self.assertEqual(result.returncode==0,supported)
