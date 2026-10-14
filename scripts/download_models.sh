#!/usr/bin/env bash
set -euo pipefail

# Bibit model downloader for SpacemiT K3 Bianbu riscv64.
#
# By default it fetches the sherpa-onnx runtime + C API header (build-time
# assets) and the VAD/KWS/ASR models (runtime assets).
#
# With --models-only it skips the runtime/header and downloads only the
# runtime models. This is used by the bibit-voice-models.service, because the
# sherpa-onnx C API library is shipped by the libsherpa-onnx-c-api package
# and the onnxruntime engine by spacemit-onnxruntime.
#
# Reentrant by design: assets that already verify are skipped, an interrupted
# download resumes from its .part file, and the readiness marker is written
# only after every required asset passes verification. Any failure removes the
# marker, so a half-finished model directory can never be mistaken for ready.
#
# With --check it validates the assets and exits without downloading.

BIBIT_MODELS_ONLY=0
BIBIT_CHECK_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --models-only) BIBIT_MODELS_ONLY=1 ;;
    --check) BIBIT_CHECK_ONLY=1 ;;
    *) break ;;
  esac
  shift
done

BIBIT_MODEL_ROOT="${BIBIT_MODEL_ROOT:-/opt/bibit/models}"
BIBIT_DOWNLOAD_DIR="${BIBIT_MODEL_ROOT}/downloads"
BIBIT_RUNTIME_DIR="${BIBIT_MODEL_ROOT}/runtime"
BIBIT_VAD_DIR="${BIBIT_MODEL_ROOT}/vad"
BIBIT_KWS_DIR="${BIBIT_MODEL_ROOT}/kws"
BIBIT_ASR_DIR="${BIBIT_MODEL_ROOT}/asr"
BIBIT_MARKER="${BIBIT_MARKER:-${BIBIT_MODEL_ROOT}/.bibit-voice-models-downloaded}"

for BIBIT_REQUIRED_COMMAND in curl sha256sum md5sum tar bzip2 install flock; do
  if ! command -v "${BIBIT_REQUIRED_COMMAND}" >/dev/null 2>&1; then
    printf 'Missing required command: %s\n' "${BIBIT_REQUIRED_COMMAND}" >&2
    exit 1
  fi
done

mkdir -p \
  "${BIBIT_DOWNLOAD_DIR}" \
  "${BIBIT_RUNTIME_DIR}" \
  "${BIBIT_VAD_DIR}" \
  "${BIBIT_KWS_DIR}" \
  "${BIBIT_ASR_DIR}"

RUNTIME_NAME="sherpa-onnx-v1.12.36-linux-riscv64-spacemit-shared.tar.bz2"
KWS_NAME="sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2"
QWEN3_ASR_NAME="qwen3-asr-0.6B-dynq-q40.tar.gz"
SENSEVOICE_ASR_NAME="sensevoice.tar.gz"
SHERPA_C_API_HEADER_NAME="sherpa-onnx-v1.12.36-c-api.h"

# Sources and checksums are environment-overridable so the script can be
# exercised offline (tests, mirrors, air-gapped staging) without touching the
# shipped values.
BIBIT_VAD_URL="${BIBIT_VAD_URL:-https://archive.spacemit.com/spacemit-ai/model_zoo/vad/silero/silero_vad.onnx}"
BIBIT_VAD_MD5="${BIBIT_VAD_MD5:-ad78afa8b67a4f24c1201d7dd2f65bb1}"
BIBIT_KWS_URL="${BIBIT_KWS_URL:-https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/${KWS_NAME}}"
BIBIT_KWS_SHA256="${BIBIT_KWS_SHA256:-68447f4fbc67e70eee3a93961f36e81e98f47aef73ce7e7ca00885c6cd3616a6}"
BIBIT_QWEN3_URL="${BIBIT_QWEN3_URL:-https://archive.spacemit.com/spacemit-ai/model_zoo/asr/${QWEN3_ASR_NAME}}"
BIBIT_QWEN3_MD5="${BIBIT_QWEN3_MD5:-61dbc944077331670014cdfbe4de3a73}"
BIBIT_SENSEVOICE_URL="${BIBIT_SENSEVOICE_URL:-https://archive.spacemit.com/spacemit-ai/model_zoo/asr/${SENSEVOICE_ASR_NAME}}"
BIBIT_SENSEVOICE_MD5="${BIBIT_SENSEVOICE_MD5:-d2670fd0d6247c34b397ca4c48ee3d61}"
BIBIT_RUNTIME_URL="${BIBIT_RUNTIME_URL:-https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.12.36/${RUNTIME_NAME}}"
BIBIT_RUNTIME_SHA256="${BIBIT_RUNTIME_SHA256:-034a17f192fe74d4d153024c15a3a5445f5e4d6e89a2453482df58c519d06357}"
BIBIT_C_API_HEADER_URL="${BIBIT_C_API_HEADER_URL:-https://raw.githubusercontent.com/k2-fsa/sherpa-onnx/v1.12.36/sherpa-onnx/c-api/c-api.h}"
BIBIT_C_API_HEADER_SHA256="${BIBIT_C_API_HEADER_SHA256:-437b1279047877167d8fadc74a60d47f3df514d703fdac1c1b6851da9bc2fdb4}"

BIBIT_RUNTIME_ROOT="${BIBIT_RUNTIME_DIR}/sherpa-onnx-v1.12.36-linux-riscv64-spacemit-shared"
BIBIT_KWS_MODEL_ROOT="${BIBIT_KWS_DIR}/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
BIBIT_ASR_MODEL_ROOT="${BIBIT_ASR_DIR}/qwen3-asr-0.6B-dynq-q40"
BIBIT_SENSEVOICE_MODEL_ROOT="${BIBIT_ASR_DIR}/sensevoice"
# Build-time header lives outside the model root (the C API library package and
# CMake builds read it from /opt/bibit/include).
BIBIT_C_API_HEADER_ROOT="${BIBIT_C_API_HEADER_ROOT:-/opt/bibit/include/sherpa-onnx/c-api}"

BIBIT_REQUIRED_FILES=(
  "${BIBIT_KWS_MODEL_ROOT}/encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx"
  "${BIBIT_KWS_MODEL_ROOT}/decoder-epoch-13-avg-2-chunk-16-left-64.onnx"
  "${BIBIT_KWS_MODEL_ROOT}/joiner-epoch-13-avg-2-chunk-16-left-64.int8.onnx"
  "${BIBIT_VAD_DIR}/silero_vad.onnx"
  "${BIBIT_ASR_MODEL_ROOT}/config.json"
  "${BIBIT_ASR_MODEL_ROOT}/Qwen3-ASR-0.6B-text-q40.gguf"
  "${BIBIT_ASR_MODEL_ROOT}/Qwen3-ASR-0.6B-encoder-frontend.dynq.onnx"
  "${BIBIT_ASR_MODEL_ROOT}/Qwen3-ASR-0.6B-encoder-backend.dynq.onnx"
  "${BIBIT_SENSEVOICE_MODEL_ROOT}/model_quant_optimized.onnx"
  "${BIBIT_SENSEVOICE_MODEL_ROOT}/tokens.sherpa.txt"
)
if [[ ${BIBIT_MODELS_ONLY} -eq 0 ]]; then
  BIBIT_REQUIRED_FILES+=(
    "${BIBIT_RUNTIME_ROOT}/lib/libsherpa-onnx-c-api.so"
    "${BIBIT_C_API_HEADER_ROOT}/c-api.h"
  )
fi

# Asset presence is the readiness truth; the marker only mirrors it.
verify_file() {
  local path="$1" kind="$2" expected="$3"
  [[ -f "${path}" ]] || return 1
  case "${kind}" in
    sha256) printf '%s  %s\n' "${expected}" "${path}" | sha256sum -c --status - ;;
    md5) printf '%s  %s\n' "${expected}" "${path}" | md5sum -c --status - ;;
    *) printf 'Unknown checksum kind: %s\n' "${kind}" >&2; return 2 ;;
  esac
}

missing_assets() {
  local file
  for file in "${BIBIT_REQUIRED_FILES[@]}"; do
    [[ -f "${file}" ]] || printf '%s\n' "${file}"
  done
}

assets_ready() {
  [[ -z "$(missing_assets)" ]]
}

write_marker() {
  mkdir -p "$(dirname "${BIBIT_MARKER}")"
  : >"${BIBIT_MARKER}"
}

report_ready() {
  printf 'Bibit assets installed under %s\n' "${BIBIT_MODEL_ROOT}"
  printf 'KWS provider: cpu, num_threads: 1\n'
  printf 'VAD provider: cpu, num_threads: 1\n'
  printf 'ASR: sensevoice, provider: spacemit\n'
  printf 'Alternative ASR: qwen3_asr_0.6b, runtime: llama-server, media backend: smt\n'
}

# Concurrent runs (boot-time retry, manual re-run, package upgrade) must not
# share a .part file.
mkdir -p "${BIBIT_DOWNLOAD_DIR}"
exec 9>"${BIBIT_DOWNLOAD_DIR}/.bibit-voice-download.lock"
flock 9

if assets_ready; then
  write_marker
  printf 'All voice assets already verified under %s\n' "${BIBIT_MODEL_ROOT}"
  [[ ${BIBIT_CHECK_ONLY} -eq 1 ]] && exit 0
  report_ready
  exit 0
fi

if [[ ${BIBIT_CHECK_ONLY} -eq 1 ]]; then
  printf 'Voice assets are not ready; missing:\n%s\n' "$(missing_assets)" >&2
  exit 1
fi

# Assets are incomplete: drop the marker before touching anything so a crash
# mid-repair cannot leave a stale "ready" signal behind.
rm -f "${BIBIT_MARKER}"

fetch_asset() {
  local url="$1" dest="$2" kind="$3" expected="$4"
  if verify_file "${dest}" "${kind}" "${expected}"; then
    printf 'Verified, skipping download: %s\n' "${dest}"
    return 0
  fi
  local part="${dest}.part" attempt
  printf 'Downloading %s\n' "${url}"
  for attempt in 1 2; do
    local transport=()
    # curl's default retry set misses HTTP/2 stream errors (curl: (92)), the
    # exact failure that stranded this machine; --retry-all-errors covers it,
    # and the second attempt drops to HTTP/1.1 for servers that keep failing.
    if [[ ${attempt} -eq 2 ]]; then
      transport=(--http1.1)
      printf 'Retrying over HTTP/1.1: %s\n' "${url}" >&2
    fi
    if curl -fL --retry 5 --retry-delay 5 --retry-all-errors \
      "${transport[@]}" -C - -o "${part}" "${url}"; then
      break
    fi
    [[ ${attempt} -eq 2 ]] && return 1
  done
  if ! verify_file "${part}" "${kind}" "${expected}"; then
    printf 'Checksum verification failed: %s\n' "${part}" >&2
    rm -f "${part}"
    return 1
  fi
  mv -f "${part}" "${dest}"
}

fetch_asset "${BIBIT_VAD_URL}" "${BIBIT_DOWNLOAD_DIR}/silero_vad.onnx" md5 "${BIBIT_VAD_MD5}"
fetch_asset "${BIBIT_KWS_URL}" "${BIBIT_DOWNLOAD_DIR}/${KWS_NAME}" sha256 "${BIBIT_KWS_SHA256}"
fetch_asset "${BIBIT_QWEN3_URL}" "${BIBIT_DOWNLOAD_DIR}/${QWEN3_ASR_NAME}" md5 "${BIBIT_QWEN3_MD5}"
fetch_asset "${BIBIT_SENSEVOICE_URL}" "${BIBIT_DOWNLOAD_DIR}/${SENSEVOICE_ASR_NAME}" md5 "${BIBIT_SENSEVOICE_MD5}"

if [[ ${BIBIT_MODELS_ONLY} -eq 0 ]]; then
  fetch_asset "${BIBIT_RUNTIME_URL}" "${BIBIT_DOWNLOAD_DIR}/${RUNTIME_NAME}" sha256 "${BIBIT_RUNTIME_SHA256}"
  fetch_asset "${BIBIT_C_API_HEADER_URL}" "${BIBIT_DOWNLOAD_DIR}/${SHERPA_C_API_HEADER_NAME}" sha256 "${BIBIT_C_API_HEADER_SHA256}"
fi

install -m 0644 \
  "${BIBIT_DOWNLOAD_DIR}/silero_vad.onnx" \
  "${BIBIT_VAD_DIR}/silero_vad.onnx"

if [[ ${BIBIT_MODELS_ONLY} -eq 0 ]]; then
  tar -xjf "${BIBIT_DOWNLOAD_DIR}/${RUNTIME_NAME}" -C "${BIBIT_RUNTIME_DIR}"
fi
tar -xjf "${BIBIT_DOWNLOAD_DIR}/${KWS_NAME}" -C "${BIBIT_KWS_DIR}"
tar -xzf "${BIBIT_DOWNLOAD_DIR}/${QWEN3_ASR_NAME}" -C "${BIBIT_ASR_DIR}"
tar -xzf "${BIBIT_DOWNLOAD_DIR}/${SENSEVOICE_ASR_NAME}" -C "${BIBIT_ASR_DIR}"

if [[ ${BIBIT_MODELS_ONLY} -eq 0 ]]; then
  install -D -m 0644 \
    "${BIBIT_DOWNLOAD_DIR}/${SHERPA_C_API_HEADER_NAME}" \
    "${BIBIT_C_API_HEADER_ROOT}/c-api.h"
fi

awk -F '\t' '{ print $1 " " NR - 1 }' \
  "${BIBIT_SENSEVOICE_MODEL_ROOT}/tokenizer.vocab" \
  > "${BIBIT_SENSEVOICE_MODEL_ROOT}/tokens.sherpa.txt"

if ! assets_ready; then
  printf 'Required assets are still missing after extraction:\n%s\n' "$(missing_assets)" >&2
  exit 1
fi

write_marker

report_ready
