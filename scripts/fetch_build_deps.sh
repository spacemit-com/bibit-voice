#!/usr/bin/env bash
set -euo pipefail

# Fetch sherpa-onnx build-time dependencies (runtime lib + C API header) into
# /opt/bibit. Idempotent: skips download when both assets already exist.

BIBIT_SHERPA_INCLUDE_FILE="/opt/bibit/include/sherpa-onnx/c-api/c-api.h"
BIBIT_SHERPA_ROOT="/opt/bibit/models/runtime/sherpa-onnx-v1.12.36-linux-riscv64-spacemit-shared"
BIBIT_SHERPA_LIB="${BIBIT_SHERPA_ROOT}/lib/libsherpa-onnx-c-api.so"

for BIBIT_REQUIRED_COMMAND in curl sha256sum tar bzip2 install; do
  if ! command -v "${BIBIT_REQUIRED_COMMAND}" >/dev/null 2>&1; then
    printf 'Missing required command: %s\n' "${BIBIT_REQUIRED_COMMAND}" >&2
    exit 1
  fi
done

if [[ -f "${BIBIT_SHERPA_INCLUDE_FILE}" && -f "${BIBIT_SHERPA_LIB}" ]]; then
  printf 'sherpa-onnx build deps already present, skipping fetch\n'
  exit 0
fi

BIBIT_DOWNLOAD_DIR="/opt/bibit/models/downloads"
mkdir -p \
  "${BIBIT_DOWNLOAD_DIR}" \
  "/opt/bibit/models/runtime" \
  "/opt/bibit/include/sherpa-onnx/c-api"

RUNTIME_NAME="sherpa-onnx-v1.12.36-linux-riscv64-spacemit-shared.tar.bz2"
HEADER_NAME="sherpa-onnx-v1.12.36-c-api.h"

curl -fL --retry 3 \
  -o "${BIBIT_DOWNLOAD_DIR}/${RUNTIME_NAME}" \
  "https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.12.36/${RUNTIME_NAME}"

curl -fL --retry 3 \
  -o "${BIBIT_DOWNLOAD_DIR}/${HEADER_NAME}" \
  "https://raw.githubusercontent.com/k2-fsa/sherpa-onnx/v1.12.36/sherpa-onnx/c-api/c-api.h"

(
  cd "${BIBIT_DOWNLOAD_DIR}"
  printf '%s  %s\n' \
    '034a17f192fe74d4d153024c15a3a5445f5e4d6e89a2453482df58c519d06357' \
    "${RUNTIME_NAME}" | sha256sum -c -
  printf '%s  %s\n' \
    '437b1279047877167d8fadc74a60d47f3df514d703fdac1c1b6851da9bc2fdb4' \
    "${HEADER_NAME}" | sha256sum -c -
)

tar -xjf "${BIBIT_DOWNLOAD_DIR}/${RUNTIME_NAME}" -C "/opt/bibit/models/runtime"

install -D -m 0644 \
  "${BIBIT_DOWNLOAD_DIR}/${HEADER_NAME}" \
  "${BIBIT_SHERPA_INCLUDE_FILE}"

printf 'sherpa-onnx build deps installed under /opt/bibit\n'
