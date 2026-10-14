# Bibit Voice

公开版本：**1.0Beta1**。

面向 SpacemiT K3 / Bianbu Desktop 的离线语音服务：唤醒词检测（KWS）、语音活动检测（VAD）与语音识别（ASR）。本仓库只提供**服务端、Socket 协议与文档**，不包含任何消费 Socket 的客户端实现；GUI、悬浮球等客户端在本仓库之外。

## 架构

```
bibit-audio (C++, bibit-audio.service)
  PipeWire 采集 → KWS 唤醒 → VAD → ASR (SenseVoice/Qwen3-ASR)
  监听 /run/bibit/audio.sock
          │
bibit-voice-tool (Python, bibit-voice-tool.service)
  监听 /run/bibit/tool.sock（多客户端透传代理）
  另提供 stdio MCP（bibit-voice-tool mcp）
          │
客户端（bibit-gui / CLI / MCP host，仓库外）
```

`bibit-voice-tool` 不是第二条音频流水线：它只透传 `tool.sock ↔ audio.sock`，不重复打开麦克风，也不重复加载 KWS/VAD/ASR 模型。

## 目录结构

```
src/audio/          bibit-audio 服务：PipeWire 采集、KWS/VAD/ASR、Socket 服务端
src/tool/           bibit-voice-tool：tool.sock 代理 + MCP 适配层
config/             bibit.conf、唤醒词、ASR provider 配置
assets/wo-zai.wav   唤醒应答「我在」提示音（服务端播放）
packaging/          systemd --user 服务单元
scripts/            依赖安装、模型下载、部署、状态查询
debian/             Debian/Bianbu 打包规范
docs/protocol.md    Socket 与 MCP 协议规范
docs/configuration.md  配置参数参考
```

## 协议

完整命令、事件、错误码与 MCP 工具定义见 [`docs/protocol.md`](docs/protocol.md)。

要点：

- 两层 Socket（`audio.sock`、`tool.sock`）均为 Unix 域 SOCK_STREAM，**一行一个 JSON**。
- `tool.sock` 是 `audio.sock` 的透传代理，命令/事件集合一致，额外提供 `tool_state` 与代理级 `error`。
- `bibit-voice-tool mcp` 暴露 8 个工具：状态、单次转写、等待交互、取消、开关、切模型、阈值、KWS debug。

## 运行环境

- Bianbu Desktop（PipeWire 音频）
- sherpa-onnx SpacemiT runtime（KWS/VAD/ASR 推理，位于 `/opt/bibit`）
- `llama.cpp-tools-spacemit`（仅 Qwen3-ASR 需要，DEB `Recommends`）
- 编译依赖：cmake、pkg-config、libpipewire-0.3-dev、g++

模型与 sherpa-onnx 运行时由 [`scripts/download_models.sh`](scripts/download_models.sh) 下载到 `/opt/bibit`，不随源码仓库分发。

## 配置

全部配置项、加载顺序与运行时可变范围见 [`docs/configuration.md`](docs/configuration.md)。要点：

- 主配置 `/etc/bibit/bibit.conf`；运行期阈值/开关写入 `$HOME/.config/bibit/thresholds.conf`，ASR 模型偏好写入 `$HOME/.config/bibit/asr-model.conf`。
- `kws_debug_dir` 不写死用户名，未配置时按 `$XDG_STATE_HOME`/`$HOME` 推导。
- 唤醒应答音统一安装在 `/usr/share/bibit/sounds/wo-zai.wav`（源码安装与 DEB 一致）。

## 快速部署

在已登录桌面会话的 K3 板上执行：

```bash
sudo BIBIT_USER=<lxqt-user> ./scripts/setup.sh
```

脚本安装系统依赖、下载并校验模型与运行时、编译安装，并启用 `bibit-voice-models.service`（资产获取）、`bibit-voice-models.path`（资产后到触发 audio）、`bibit-audio.service` 与 `bibit-voice-tool.service`，最后等待 `audio_ready/kws_loaded/asr_loaded` 就绪并打印状态。USB 麦克风需先在系统音频设置中设为默认输入，或通过 `BIBIT_AUDIO_SOURCE='<PipeWire source name>'` 指定。

## 服务验证

```bash
python3 scripts/status.py           # 输出 audio_ready/kws_loaded/asr_loaded 等状态
systemctl status bibit-voice-models.service        # 资产获取：成功 / 失败重试 / 日志
ls -l /opt/bibit/models/.bibit-voice-models-downloaded   # 就绪标记，只在全部资产校验通过后存在
systemctl --user status bibit-audio.service bibit-voice-tool.service
```

模型资产缺失时 `bibit-audio.service` 是 `inactive`（条件跳过）而不是排队等待；`bibit-voice-tool.service` 仍然 `active` 并监听 `/run/bibit/tool.sock`，客户端拿到的是"上游离线"而不是 socket 不存在。资产补齐后由 `bibit-voice-models.path` 自动拉起 audio，不需要重启 GUI 或整机。

涉及真模型的行为（切换语义、`llama-server` 生命周期）在开发板上跑（走 `bibit-dev` 独占租约）：

```bash
python3 scripts/verify_asr_switch_release.py --expect success   # 默认超时下正常切换
python3 scripts/verify_asr_switch_release.py --expect failure   # 调小 asr_release_timeout_ms：释放超时按失败收场
python3 scripts/verify_asr_switch_roundtrip.py --rounds 3       # 连续往返切换，且 SenseVoice 生效时 8063 无遗留监听
python3 scripts/verify_switch_intents.py                        # 构建期间开关的交错：收敛到最后一次意图、不自行翻转
python3 scripts/verify_switch_hammering.py --rounds 20          # 连点期间服务端始终可响应，收尾收敛到最后意图
python3 scripts/verify_single_instance.py --as-user bibit-test  # 第二个实例绑不上就退出，不许碰 8063
```

开关意图的收敛逻辑是纯逻辑单元（`src/audio/switch_intents.h`），对应的场景与交错扫描回归
不依赖开发板：

```bash
python3 -m unittest discover -s tests -p 'test_switch_intents.py' -t tests
```

## ASR 模型

默认 ASR 为 **Qwen3-ASR 0.6B**（生成式，由 `llama-server` 在 `127.0.0.1:8063` 提供，运行时依赖 `llama.cpp-tools-spacemit`），可热切换回 **SenseVoice**（整句识别）。

切换通过 `set_asr_model` 命令完成：它把 `asr_model_type` 和配套的 `asr_model`／`asr_tokens`／`asr_provider`／`asr_decoding_method` 一起改写并持久化到 `$HOME/.config/bibit/asr-model.conf`，服务进程不重启，PipeWire 采集与 Socket 连接保持不变。手改 `/etc/bibit/bibit.conf` 时必须成组改这五项，只改 `asr_model_type` 会让 `llama-server` 去加载另一个模型的文件。默认配置见 [`config/bibit.conf`](config/bibit.conf)。

## 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DBIBIT_SHERPA_ROOT=/opt/bibit/models/runtime/sherpa-onnx-v1.12.36-linux-riscv64-spacemit-shared \
  -DBIBIT_SHERPA_INCLUDE=/opt/bibit/include
cmake --build build -j"${JOBS:-2}"
```

## Debian 打包

仓库提供标准 `debian/` 目录，产物为单个 `bibit-voice` 二进制包：

```bash
dpkg-buildpackage -us -uc
```

二进制与 `bibit-voice-tool` 安装到 `/usr/bin`，配置到 `/etc/bibit`，提示音到 `/usr/share/bibit/sounds`，systemd 用户单元到 `/usr/lib/systemd/user`（资产获取服务到 `/usr/lib/systemd/system`），协议文档到 `/usr/share/doc/bibit-voice`。

## 常见问题

- 语音不可用：按"资产 → 服务 → socket"顺序看三层。`systemctl status bibit-voice-models.service` 给资产获取的结果（失败会写 journal 并按 60 秒退避重试，重试上限 10 次）；`ls -l /opt/bibit/models/.bibit-voice-models-downloaded` 给就绪标记（只在全部资产校验通过后存在，任何失败都会先删掉它）；`ls -l /run/bibit/audio.sock /run/bibit/tool.sock` 给运行态（tool 独立于 audio，audio 缺席时它照样在监听）。
- 手动补齐资产：`sudo systemctl restart bibit-voice-models.service`。脚本可重入，只补齐缺失资产并对每个文件做校验和验证；`--check` 只校验不下载，适合只看状态。
- 模型未加载：marker 存在但 `scripts/status.py` 报 `kws_loaded/asr_loaded` 为假时，`journalctl --user -u bibit-audio.service` 会给出具体缺失或加载失败的模型文件。
- KWS 误唤醒：在 `set_kws_debug` 打开后复现，分析 `/opt/bibit` 下的 KWS debug 采样。
