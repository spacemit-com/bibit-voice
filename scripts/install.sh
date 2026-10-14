#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "Run as root: sudo BIBIT_USER=<lxqt-user> ./scripts/install.sh" >&2
  exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
BIBIT_USER=${BIBIT_USER:-${SUDO_USER:-xhd}}
if [[ ${BIBIT_USER} == root ]]; then
  BIBIT_USER=xhd
fi
if ! id "${BIBIT_USER}" >/dev/null 2>&1; then
  echo "LXQt user does not exist: ${BIBIT_USER}" >&2
  exit 1
fi

BIBIT_UID=$(id -u "${BIBIT_USER}")
BIBIT_GROUP=$(id -gn "${BIBIT_USER}")
BIBIT_HOME=$(getent passwd "${BIBIT_USER}" | cut -d: -f6)
BIBIT_RUNTIME_ROOT=/opt/bibit/models/runtime/sherpa-onnx-v1.12.36-linux-riscv64-spacemit-shared
BIBIT_INCLUDE_ROOT=/opt/bibit/include

for command in cmake pkg-config c++ install python3 runuser systemctl getent sed pw-play \
  awk grep mktemp pgrep llama-server; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "Missing required command: ${command}" >&2
    exit 1
  fi
done

required_files=(
  "${BIBIT_RUNTIME_ROOT}/lib/libsherpa-onnx-c-api.so"
  "${BIBIT_INCLUDE_ROOT}/sherpa-onnx/c-api/c-api.h"
  "/opt/bibit/models/vad/silero_vad.onnx"
  "/opt/bibit/models/kws/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx"
  "/opt/bibit/models/asr/qwen3-asr-0.6B-dynq-q40/config.json"
  "/opt/bibit/models/asr/qwen3-asr-0.6B-dynq-q40/Qwen3-ASR-0.6B-text-q40.gguf"
  "/opt/bibit/models/asr/qwen3-asr-0.6B-dynq-q40/Qwen3-ASR-0.6B-encoder-frontend.dynq.onnx"
  "/opt/bibit/models/asr/qwen3-asr-0.6B-dynq-q40/Qwen3-ASR-0.6B-encoder-backend.dynq.onnx"
  "/opt/bibit/models/asr/sensevoice/model_quant_optimized.onnx"
  "/opt/bibit/models/asr/sensevoice/tokens.sherpa.txt"
)
for file in "${required_files[@]}"; do
  if [[ ! -f ${file} ]]; then
    echo "Missing model/runtime asset: ${file}" >&2
    echo "Run first: sudo ./scripts/download_models.sh" >&2
    exit 1
  fi
done

cmake -S "${PROJECT_ROOT}" -B "${PROJECT_ROOT}/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBIBIT_SHERPA_ROOT="${BIBIT_RUNTIME_ROOT}" \
  -DBIBIT_SHERPA_INCLUDE="${BIBIT_INCLUDE_ROOT}"
cmake --build "${PROJECT_ROOT}/build" -j"${BIBIT_BUILD_JOBS:-2}"
cmake --install "${PROJECT_ROOT}/build" --prefix /usr

install -d /etc/bibit /usr/share/bibit/sounds
install -m 0644 "${PROJECT_ROOT}/config/bibit.conf" /etc/bibit/bibit.conf
install -m 0644 "${PROJECT_ROOT}/config/keywords.txt" /etc/bibit/keywords.txt
install -m 0644 "${PROJECT_ROOT}/config/keywords-fallback.txt" /etc/bibit/keywords-fallback.txt
install -m 0644 "${PROJECT_ROOT}/config/bibit-asr-provider.config" \
  /etc/bibit/bibit-asr-provider.config
install -m 0644 "${PROJECT_ROOT}/assets/wo-zai.wav" /usr/share/bibit/sounds/wo-zai.wav

if [[ -n ${BIBIT_AUDIO_SOURCE:-} ]]; then
  escaped_source=${BIBIT_AUDIO_SOURCE//&/\\&}
  escaped_source=${escaped_source//|/\\|}
  sed -i "s|^audio_source=.*|audio_source=${escaped_source}|" /etc/bibit/bibit.conf
fi

USER_UNIT_DIR="${BIBIT_HOME}/.config/systemd/user"
install -d -o "${BIBIT_USER}" -g "${BIBIT_GROUP}" "${USER_UNIT_DIR}"
for unit in bibit-audio.service bibit-voice-tool.service bibit-voice-models.path; do
  install -m 0644 -o "${BIBIT_USER}" -g "${BIBIT_GROUP}" \
    "${PROJECT_ROOT}/packaging/${unit}" "${USER_UNIT_DIR}/${unit}"
done

# 资产获取走 systemd 一次性服务（与 deb 路径一致）：脚本可重入，失败留在
# journal 里并按退避重试，正常安装时只做一次校验。
install -m 0644 "${PROJECT_ROOT}/packaging/bibit-voice-models.service" \
  /usr/lib/systemd/system/bibit-voice-models.service
if [[ -d /run/systemd/system ]]; then
  systemctl daemon-reload
  systemctl enable --now bibit-voice-models.service >/dev/null 2>&1 || \
    echo "bibit-voice-models.service 未能立即启动，资产会在下次开机或手动 restart 时补齐" >&2
fi

XDG_RUNTIME_DIR="/run/user/${BIBIT_UID}"
DBUS_SESSION_BUS_ADDRESS="unix:path=${XDG_RUNTIME_DIR}/bus"
if [[ -S ${XDG_RUNTIME_DIR}/bus ]]; then
  user_systemctl() {
    runuser -u "${BIBIT_USER}" -- env \
      XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR}" \
      DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS}" \
      systemctl --user "$@"
  }
  user_systemctl daemon-reload
  user_systemctl enable bibit-audio.service bibit-voice-tool.service bibit-voice-models.path
  user_systemctl restart bibit-audio.service bibit-voice-tool.service bibit-voice-models.path
  user_systemctl --no-pager --full status \
    bibit-audio.service bibit-voice-tool.service bibit-voice-models.path || true
else
  echo "Installed successfully, but ${BIBIT_USER} has no active user bus." >&2
  echo "Log into LXQt as ${BIBIT_USER}, then run:" >&2
  echo "  systemctl --user daemon-reload" >&2
  echo "  systemctl --user enable --now bibit-audio.service bibit-voice-tool.service bibit-voice-models.path" >&2
fi

echo "Bibit Voice installed for user ${BIBIT_USER}."
