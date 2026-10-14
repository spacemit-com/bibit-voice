#!/usr/bin/env bash
set -euo pipefail

# End-to-end setup for a SpacemiT K3 running Bianbu Desktop/LXQt.

if [[ ${EUID} -ne 0 ]]; then
  echo "Run as root: sudo BIBIT_USER=<lxqt-user> ./scripts/setup.sh" >&2
  exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ $(uname -m) != riscv64 ]]; then
  echo "Unsupported architecture: $(uname -m). Bibit targets SpacemiT K3 riscv64." >&2
  exit 1
fi
if [[ ! -r /etc/os-release ]]; then
  echo "Cannot identify the operating system: /etc/os-release is missing." >&2
  exit 1
fi
# shellcheck disable=SC1091
source /etc/os-release
if [[ ${ID:-} != bianbu ]]; then
  echo "Unsupported operating system: ${PRETTY_NAME:-${ID:-unknown}}. Bianbu is required." >&2
  exit 1
fi

if [[ -z ${BIBIT_USER:-} ]]; then
  if [[ -n ${SUDO_USER:-} && ${SUDO_USER} != root ]]; then
    BIBIT_USER=${SUDO_USER}
  elif id xhd >/dev/null 2>&1; then
    BIBIT_USER=xhd
  else
    echo "Cannot determine the LXQt user. Set BIBIT_USER explicitly." >&2
    exit 1
  fi
fi
if [[ ${BIBIT_USER} == root ]] || ! id "${BIBIT_USER}" >/dev/null 2>&1; then
  echo "BIBIT_USER must name an existing non-root LXQt user: ${BIBIT_USER}" >&2
  exit 1
fi

BIBIT_UID=$(id -u "${BIBIT_USER}")
BIBIT_RUNTIME_DIR="/run/user/${BIBIT_UID}"
BIBIT_READY_TIMEOUT=${BIBIT_READY_TIMEOUT:-120}
if ! [[ ${BIBIT_READY_TIMEOUT} =~ ^[0-9]+$ ]] || (( BIBIT_READY_TIMEOUT < 1 )); then
  echo "BIBIT_READY_TIMEOUT must be a positive integer." >&2
  exit 1
fi

BIBIT_PACKAGES=(
  build-essential cmake pkg-config curl ca-certificates bzip2 python3 procps
  libpipewire-0.3-dev pipewire-audio pipewire-bin wireplumber pulseaudio-utils
  llama.cpp-tools-spacemit
)

echo "==> Bibit target: ${PRETTY_NAME:-Bianbu}, user=${BIBIT_USER}, arch=riscv64"
if [[ ${BIBIT_SKIP_APT:-0} == 1 ]]; then
  echo "==> Skipping APT installation (BIBIT_SKIP_APT=1)"
else
  echo "==> Installing build and runtime dependencies"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends "${BIBIT_PACKAGES[@]}"
fi

BIBIT_REQUIRED_ASSETS=(
  /opt/bibit/models/runtime/sherpa-onnx-v1.12.36-linux-riscv64-spacemit-shared/lib/libsherpa-onnx-c-api.so
  /opt/bibit/include/sherpa-onnx/c-api/c-api.h
  /opt/bibit/models/vad/silero_vad.onnx
  /opt/bibit/models/kws/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx
  /opt/bibit/models/asr/sensevoice/model_quant_optimized.onnx
  /opt/bibit/models/asr/sensevoice/tokens.sherpa.txt
  /opt/bibit/models/asr/qwen3-asr-0.6B-dynq-q40/config.json
  /opt/bibit/models/asr/qwen3-asr-0.6B-dynq-q40/Qwen3-ASR-0.6B-text-q40.gguf
  /opt/bibit/models/asr/qwen3-asr-0.6B-dynq-q40/Qwen3-ASR-0.6B-encoder-frontend.dynq.onnx
  /opt/bibit/models/asr/qwen3-asr-0.6B-dynq-q40/Qwen3-ASR-0.6B-encoder-backend.dynq.onnx
)

assets_ready=1
for asset in "${BIBIT_REQUIRED_ASSETS[@]}"; do
  if [[ ! -f ${asset} ]]; then
    assets_ready=0
    break
  fi
done

if [[ ${BIBIT_SKIP_MODELS:-0} == 1 ]]; then
  echo "==> Skipping model download (BIBIT_SKIP_MODELS=1)"
elif [[ ${assets_ready} -eq 1 && ${BIBIT_FORCE_MODELS:-0} != 1 ]]; then
  echo "==> Verified model/runtime layout already exists; skipping download"
else
  echo "==> Downloading and validating KWS, VAD, ASR, and sherpa-onnx assets"
  "${SCRIPT_DIR}/download_models.sh"
fi

for asset in "${BIBIT_REQUIRED_ASSETS[@]}"; do
  if [[ ! -f ${asset} ]]; then
    echo "Required model/runtime asset is missing: ${asset}" >&2
    exit 1
  fi
done

echo "==> Building, installing, and enabling Bibit"
export BIBIT_USER
"${SCRIPT_DIR}/install.sh"

if [[ ! -S ${BIBIT_RUNTIME_DIR}/bus ]]; then
  echo "Bibit is installed and enabled, but ${BIBIT_USER} has no active user session." >&2
  echo "Log into LXQt as ${BIBIT_USER}, then run this setup script again to start and verify Bibit." >&2
  exit 3
fi

user_systemctl() {
  runuser -u "${BIBIT_USER}" -- env \
    XDG_RUNTIME_DIR="${BIBIT_RUNTIME_DIR}" \
    DBUS_SESSION_BUS_ADDRESS="unix:path=${BIBIT_RUNTIME_DIR}/bus" \
    systemctl --user "$@"
}

echo "==> Waiting for the audio service and ASR model"
deadline=$((SECONDS + BIBIT_READY_TIMEOUT))
status_output=$(mktemp /tmp/bibit-setup-status.XXXXXX)
trap 'rm -f "${status_output}"' EXIT
ready=0
while (( SECONDS < deadline )); do
  if user_systemctl is-active --quiet bibit-audio.service && \
     user_systemctl is-active --quiet bibit-voice-tool.service && \
     env XDG_RUNTIME_DIR="${BIBIT_RUNTIME_DIR}" \
       python3 "${SCRIPT_DIR}/status.py" >"${status_output}" 2>/dev/null && \
     grep -q '"audio_ready": true' "${status_output}" && \
     grep -q '"kws_loaded": true' "${status_output}" && \
     grep -q '"asr_loaded": true' "${status_output}"; then
    ready=1
    break
  fi
  sleep 2
done

if [[ ${ready} -ne 1 ]]; then
  echo "Bibit audio/KWS/ASR did not become ready within ${BIBIT_READY_TIMEOUT}s." >&2
  user_systemctl --no-pager --full status \
    bibit-audio.service bibit-voice-tool.service >&2 || true
  user_systemctl --no-pager -n 80 status bibit-audio.service >&2 || true
  exit 1
fi
if ! user_systemctl is-active --quiet bibit-voice-tool.service; then
  echo "Bibit ASR is ready, but bibit-voice-tool.service is not active." >&2
  user_systemctl --no-pager --full status bibit-voice-tool.service >&2 || true
  exit 1
fi
cat "${status_output}"
echo "==> Bibit deployment completed: audio/tool=active, audio_ready=true, kws_loaded=true, asr_loaded=true"
echo "    Logs: journalctl --user -u bibit-audio.service -u bibit-voice-tool.service -f"
