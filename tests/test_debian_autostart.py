"""验证 debhelper 生成的服务启用/清理脚本，并固定语音单元的启动语义。

不需要编译音频程序，但 dh_installdeb 需要 chown 权限，所以按仓库约定用
fakeroot 运行：``fakeroot python3 -m unittest discover -s tests``。
"""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

USER_UNITS = (
    'bibit-audio.service',
    'bibit-voice-tool.service',
    'bibit-voice-models.path',
)
SYSTEM_UNITS = ('bibit-voice-models.service',)
MODEL_MARKER = '/opt/bibit/models/.bibit-voice-models-downloaded'


def unit_text(name):
    return (ROOT / 'packaging' / name).read_text()


def directives(text, key):
    return [line.strip() for line in text.splitlines() if line.strip().startswith(f'{key}=')]


class DebianAutostartTest(unittest.TestCase):
    def test_generated_maintainer_scripts(self):
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)
            shutil.copytree(ROOT / 'debian', stage / 'debian')
            user_units = stage / 'debian/bibit-voice/usr/lib/systemd/user'
            system_units = stage / 'debian/bibit-voice/usr/lib/systemd/system'
            user_units.mkdir(parents=True, exist_ok=True)
            system_units.mkdir(parents=True, exist_ok=True)
            for unit in USER_UNITS:
                shutil.copy2(ROOT / 'packaging' / unit, user_units / unit)
            for unit in SYSTEM_UNITS:
                shutil.copy2(ROOT / 'packaging' / unit, system_units / unit)
            for command in ('dh_installsystemduser', 'dh_installsystemd', 'dh_installdeb'):
                subprocess.run([command, '-pbibit-voice'], cwd=stage, check=True)
            scripts = stage / 'debian/bibit-voice/DEBIAN'
            for name in ('postinst', 'postrm'):
                subprocess.run(['sh', '-n', str(scripts / name)], check=True)
            postinst = (scripts / 'postinst').read_text()
            postrm = (scripts / 'postrm').read_text()
            for unit in USER_UNITS:
                enable = f"deb-systemd-helper --user enable '{unit}'"
                self.assertIn(enable, postinst)
                self.assertLess(postinst.index(enable), postinst.index('exit 0'))
                self.assertIn(unit, postrm)
            for unit in SYSTEM_UNITS:
                enable = f"deb-systemd-helper enable '{unit}'"
                self.assertIn(enable, postinst)
                self.assertLess(postinst.index(enable), postinst.index('exit 0'))
                self.assertIn(unit, postrm)
            self.assertIn('deb-systemd-helper --user purge', postrm)
            self.assertLess(postrm.index('deb-systemd-helper --user purge'), postrm.index('exit 0'))

    def test_install_file_ships_every_unit(self):
        install = (ROOT / 'debian/bibit-voice.install').read_text()
        for unit in USER_UNITS:
            self.assertIn(f'packaging/{unit} usr/lib/systemd/user/', install)
        for unit in SYSTEM_UNITS:
            self.assertIn(f'packaging/{unit} usr/lib/systemd/system/', install)

    def test_no_unit_polls_forever_with_an_infinite_start_timeout(self):
        for path in sorted((ROOT / 'packaging').glob('*.service')):
            text = path.read_text()
            polls = 'while ' in text and 'sleep ' in text
            if polls and 'TimeoutStartSec=infinity' in text:
                self.fail(f'{path.name} combines an unbounded poll with an infinite start timeout')

    def test_audio_gates_on_the_asset_marker_instead_of_waiting(self):
        text = unit_text('bibit-audio.service')
        self.assertIn(f'ConditionPathExists={MODEL_MARKER}', text)
        self.assertNotIn('ExecStartPre', text)
        self.assertNotIn('TimeoutStartSec=infinity', text)

    def test_voice_tool_starts_independently_of_audio(self):
        text = unit_text('bibit-voice-tool.service')
        self.assertEqual(directives(text, 'After'), [])
        self.assertEqual(directives(text, 'Wants'), [])

    def test_models_path_unit_starts_audio_when_the_marker_appears(self):
        text = unit_text('bibit-voice-models.path')
        self.assertIn(f'PathExists={MODEL_MARKER}', text)
        self.assertIn('Unit=bibit-audio.service', text)

    def test_models_service_retries_with_a_bounded_timeout(self):
        text = unit_text('bibit-voice-models.service')
        self.assertIn('ExecStart=/usr/share/bibit-voice/download_models.sh --models-only', text)
        self.assertIn('Restart=on-failure', text)
        self.assertIn('RestartSec=', text)
        self.assertNotIn('TimeoutStartSec=infinity', text)


if __name__ == '__main__':
    unittest.main()
