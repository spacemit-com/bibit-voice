"""download_models.sh 的可重入性与就绪判据测试。

完全离线运行：用测试自带的 curl 替身按 URL 提供本地资产，验证真实脚本的
"已就绪跳过 / 中断续传 / 校验失败不落地 / marker 只反映完整校验"。
"""
import hashlib
import os
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'download_models.sh'

KWS_DIR = 'sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20'
ASR_DIR = 'qwen3-asr-0.6B-dynq-q40'
SENSEVOICE_DIR = 'sensevoice'
KWS_ARCHIVE = f'{KWS_DIR}.tar.bz2'
QWEN3_ARCHIVE = f'{ASR_DIR}.tar.gz'
SENSEVOICE_ARCHIVE = f'{SENSEVOICE_DIR}.tar.gz'
VAD_NAME = 'silero_vad.onnx'

KWS_FILES = {
    'encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx': b'kws-encoder',
    'decoder-epoch-13-avg-2-chunk-16-left-64.onnx': b'kws-decoder',
    'joiner-epoch-13-avg-2-chunk-16-left-64.int8.onnx': b'kws-joiner',
}
ASR_FILES = {
    'config.json': b'{}',
    'Qwen3-ASR-0.6B-text-q40.gguf': b'qwen3-text',
    'Qwen3-ASR-0.6B-encoder-frontend.dynq.onnx': b'qwen3-frontend',
    'Qwen3-ASR-0.6B-encoder-backend.dynq.onnx': b'qwen3-backend',
}
SENSEVOICE_FILES = {
    'model_quant_optimized.onnx': b'sensevoice-model',
    'tokenizer.vocab': b'a\t1\nb\t2\n',
}

# 复刻脚本的必需资产清单（--models-only 模式）。
REQUIRED_ASSETS = (
    [f'kws/{KWS_DIR}/{name}' for name in KWS_FILES]
    + ['vad/silero_vad.onnx']
    + [f'asr/{ASR_DIR}/{name}' for name in ASR_FILES]
    + [f'asr/{SENSEVOICE_DIR}/model_quant_optimized.onnx', f'asr/{SENSEVOICE_DIR}/tokens.sherpa.txt']
)

CURL_SHIM = r'''#!/bin/sh
set -eu
: "${CURL_SHIM_LOG:?}"
printf '%s\n' "$*" >>"$CURL_SHIM_LOG"
out=''
url=''
while [ $# -gt 0 ]; do
  case "$1" in
    -o) out=$2; shift 2 ;;
    -C) shift; [ "${1:-}" = "-" ] && shift ;;
    --retry|--retry-delay) shift 2 ;;
    -*) shift ;;
    *) url=$1; shift ;;
  esac
done
[ -n "$out" ] || exit 2
if [ "${CURL_SHIM_FAIL:-0}" = "1" ]; then
  printf 'shim: simulated transport failure\n' >&2
  exit 92
fi
: "${CURL_SHIM_FIXTURES:?}"
src="$CURL_SHIM_FIXTURES/$(basename "$url")"
[ -f "$src" ] || { printf 'shim: missing fixture for %s\n' "$url" >&2; exit 22; }
cp "$src" "$out"
'''


def _checksum(path, algorithm):
    digest = hashlib.new(algorithm)
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


class DownloadModelsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.stage = Path(self._tmp.name)
        self.model_root = self.stage / 'models'
        self.fixtures = self.stage / 'fixtures'
        self.fixtures.mkdir()
        self.shim_dir = self.stage / 'bin'
        self.shim_dir.mkdir()
        shim = self.shim_dir / 'curl'
        shim.write_text(CURL_SHIM)
        shim.chmod(0o755)
        self.curl_log = self.stage / 'curl.log'
        self._build_fixtures()

    def tearDown(self):
        self._tmp.cleanup()

    # ---- fixtures -------------------------------------------------------

    def _write_tree(self, root, files):
        for name, payload in files.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)

    def _build_fixtures(self):
        src = self.stage / 'src'
        for name, files in ((KWS_DIR, KWS_FILES), (ASR_DIR, ASR_FILES), (SENSEVOICE_DIR, SENSEVOICE_FILES)):
            self._write_tree(src / name, files)
        with tarfile.open(self.fixtures / KWS_ARCHIVE, 'w:bz2') as archive:
            archive.add(src / KWS_DIR, arcname=KWS_DIR)
        with tarfile.open(self.fixtures / QWEN3_ARCHIVE, 'w:gz') as archive:
            archive.add(src / ASR_DIR, arcname=ASR_DIR)
        with tarfile.open(self.fixtures / SENSEVOICE_ARCHIVE, 'w:gz') as archive:
            archive.add(src / SENSEVOICE_DIR, arcname=SENSEVOICE_DIR)
        (self.fixtures / VAD_NAME).write_bytes(b'vad-model')

    # ---- helpers --------------------------------------------------------

    @property
    def marker(self):
        return self.model_root / '.bibit-voice-models-downloaded'

    def _env(self, **overrides):
        env = os.environ.copy()
        env.update({
            'PATH': f"{self.shim_dir}:{env.get('PATH', '')}",
            'BIBIT_MODEL_ROOT': str(self.model_root),
            'CURL_SHIM_LOG': str(self.curl_log),
            'CURL_SHIM_FIXTURES': str(self.fixtures),
            'BIBIT_VAD_URL': f'https://example.invalid/{VAD_NAME}',
            'BIBIT_VAD_MD5': _checksum(self.fixtures / VAD_NAME, 'md5'),
            'BIBIT_KWS_URL': f'https://example.invalid/{KWS_ARCHIVE}',
            'BIBIT_KWS_SHA256': _checksum(self.fixtures / KWS_ARCHIVE, 'sha256'),
            'BIBIT_QWEN3_URL': f'https://example.invalid/{QWEN3_ARCHIVE}',
            'BIBIT_QWEN3_MD5': _checksum(self.fixtures / QWEN3_ARCHIVE, 'md5'),
            'BIBIT_SENSEVOICE_URL': f'https://example.invalid/{SENSEVOICE_ARCHIVE}',
            'BIBIT_SENSEVOICE_MD5': _checksum(self.fixtures / SENSEVOICE_ARCHIVE, 'md5'),
        })
        env.update(overrides)
        return env

    def run_script(self, *args, **overrides):
        return subprocess.run(
            ['bash', str(SCRIPT), '--models-only', *args],
            env=self._env(**overrides),
            capture_output=True,
            text=True,
        )

    def _curl_calls(self):
        return self.curl_log.read_text() if self.curl_log.exists() else ''

    # ---- tests ----------------------------------------------------------

    def test_first_run_downloads_with_resume_and_marks_ready(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.marker.exists())
        for relative in REQUIRED_ASSETS:
            self.assertTrue((self.model_root / relative).exists(), relative)
        self.assertIn('-C -', self._curl_calls())
        self.assertEqual(list((self.model_root / 'downloads').glob('*.part')), [])

    def test_verified_assets_are_reused_without_downloading(self):
        self.assertEqual(self.run_script().returncode, 0)
        self.marker.unlink()
        self.curl_log.unlink()

        result = self.run_script()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.marker.exists())
        self.assertEqual(self._curl_calls(), '', 'verified assets must not be downloaded again')

    def test_interrupted_download_resumes_from_part_file(self):
        downloads = self.model_root / 'downloads'
        downloads.mkdir(parents=True)
        part = downloads / f'{KWS_ARCHIVE}.part'
        part.write_bytes(b'interrupted')

        result = self.run_script()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(part.exists())
        self.assertEqual((downloads / KWS_ARCHIVE).read_bytes(), (self.fixtures / KWS_ARCHIVE).read_bytes())
        self.assertIn('-C -', self._curl_calls())

    def test_checksum_mismatch_leaves_no_asset_and_no_marker(self):
        result = self.run_script(BIBIT_KWS_SHA256='0' * 64)

        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.marker.exists())
        downloads = self.model_root / 'downloads'
        self.assertFalse((downloads / KWS_ARCHIVE).exists())
        self.assertFalse((downloads / f'{KWS_ARCHIVE}.part').exists())

    def test_check_mode_reports_missing_assets_without_network(self):
        result = self.run_script('--check')

        self.assertEqual(result.returncode, 1)
        self.assertIn('not ready', result.stderr)
        self.assertEqual(self._curl_calls(), '')
        self.assertFalse(self.marker.exists())

    def test_failed_repair_removes_a_stale_marker(self):
        self.model_root.mkdir(parents=True)
        self.marker.write_text('')

        result = self.run_script(CURL_SHIM_FAIL='1')

        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.marker.exists(), 'a failed repair must not leave a ready marker behind')


if __name__ == '__main__':
    unittest.main()
