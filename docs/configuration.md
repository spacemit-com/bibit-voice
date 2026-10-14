# Bibit Voice 配置参考

本文档面向维护者和对接方，说明 `bibit-audio` 的全部配置项、加载顺序与运行时可变范围。默认值以 `src/audio/config.h` 为准。

## 配置文件与加载顺序

配置按以下顺序逐层覆盖，后者覆盖前者：

1. 内置默认值（`src/audio/config.h`）
2. `/etc/bibit/bibit.conf`（DEB 安装，系统级）
3. `$HOME/.config/bibit/thresholds.conf`（运行时由 `set_thresholds` / `set_kws_debug` 写入）
4. `$HOME/.config/bibit/asr-model.conf`（运行时由 `set_asr_model` 写入）

其中 3、4 是持久化状态，不属于 DEB 内容；用户删除后回落到系统配置。

## 运行时可变范围

| 命令 | 可变参数 | 是否持久化 |
|------|----------|-----------|
| `set_switches` | `wake_enabled`, `asr_enabled` | 否（仅内存） |
| `set_thresholds` | `kws_threshold`, `kws_score`, `vad_threshold`, `min_silence`, `wake_guard_ms` | 是（`thresholds.conf`） |
| `set_kws_debug` | `kws_debug_enabled` | 是（`thresholds.conf`） |
| `set_asr_model` | `asr_model_type` 及关联的模型路径/解码参数 | 是（`asr-model.conf`） |

其余参数只在服务启动时从配置文件读取，运行期只读。

## 参数分组

### 模型与运行时路径

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_root` | `/opt/bibit/models` | 模型根目录（外置，不随 DEB 分发） |
| `provider_config` | `/etc/bibit/bibit-asr-provider.config` | SpacemiT EP 线程亲和配置 |

### ASR — Qwen3-ASR（默认，生成式）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `asr_model_type` | `qwen3_asr_0.6b` | ASR 类型，`qwen3_asr_0.6b` 或 `sensevoice` |
| `asr_model` | `/opt/bibit/models/asr/qwen3-asr-0.6B-dynq-q40/Qwen3-ASR-0.6B-text-q40.gguf` | 文本解码模型（`llama-server` 的 `-m`） |
| `asr_tokens` | 空 | 生成式路径不使用 |
| `asr_provider` | `smt` | 推理 provider |
| `asr_num_threads` | `2` | 解码线程数 |
| `asr_decoding_method` | `generative` | 解码策略 |

以上五项**成组生效**：`set_asr_model` 会一起改写目标配置；只改 `asr_model_type` 会让 `llama-server` 加载另一个模型的文件，启动超时后 ASR 不可用。

切换是异步长任务，语义见 [`protocol.md` 的 4.3 节](protocol.md)：流水线先放下现役后端，
构建线程再构建目标后端，最后一次性接入并在此时写入 `$HOME/.config/bibit/asr-model.conf`。
采集、KWS、VAD 与 IPC 不受影响，识别本身在切换期间短暂不可用；构建失败时自动重建旧后端并报错，
不会写入偏好。`status.asr_model_type` 始终表示**生效模型**，切换中的目标见 `status.switching.to`。

### 模型切换

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `asr_release_timeout_ms` | `30000` | 等流水线线程放下现役后端的上限（ms） |

流水线线程被长解码占住时，放下现役后端的回执会迟到。超过上限不再继续构建——新旧后端
并存会争用 SpacemiT EP/TCM（实测 SEGV）——而是直接报切换失败，现役引擎保持生效、
识别不中断，客户端收到 `ASR_MODEL_SWITCH_FAILED`（`active_model` 为现役模型）。
调小这个值可以更快失败，代价是正常的慢切换更容易被判超时。

### ASR — SenseVoice（可选，整句识别）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `asr_model` | `/opt/bibit/models/asr/sensevoice/model_quant_optimized.onnx` | 模型文件（选用 SenseVoice 时） |
| `asr_tokens` | `/opt/bibit/models/asr/sensevoice/tokens.sherpa.txt` | 词表（选用 SenseVoice 时） |
| `asr_provider` | `spacemit` | 推理 provider |
| `asr_num_threads` | `2` | 解码线程数 |
| `asr_decoding_method` | `greedy_search` | 解码策略 |
| `asr_max_active_paths` | `4` | 束搜索最大活动路径 |
| `asr_pre_roll_ms` | `180` | 语音段前滚（ms） |
| `asr_post_roll_ms` | `120` | 语音段后滚（ms） |

### ASR — Qwen3-ASR 服务端（生成式路径专用）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `qwen3_model_dir` | `/opt/bibit/models/asr/qwen3-asr-0.6B-dynq-q40` | 模型目录 |
| `qwen3_server_bin` | `llama-server` | 推理服务二进制（由 `llama.cpp-tools-spacemit` 提供） |
| `qwen3_server_host` | `127.0.0.1` | 推理服务地址 |
| `qwen3_server_port` | `8063` | 推理服务端口 |
| `qwen3_thread_batch` | `2` | 批处理线程数 |
| `qwen3_ep_affinity` | `8;9` | SpacemiT EP 核亲和 |
| `qwen3_ctx_size` | `4096` | 上下文大小 |
| `qwen3_language` | `zh,en` | 支持语言 |
| `qwen3_max_tokens` | `100` | 单次最大输出 token |
| `qwen3_startup_timeout_ms` | `60000` | 服务启动超时（ms） |
| `qwen3_request_timeout_ms` | `60000` | 请求超时（ms） |

> 默认 ASR 是 Qwen3-ASR，`llama.cpp-tools-spacemit` 是本包的硬依赖（DEB `Depends`）。切到 SenseVoice 后就不再需要它，但包装仍会保留。

### 唤醒词

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `keywords_file` | `/etc/bibit/keywords.txt` | 主唤醒词（Hello Bibit） |
| `kws_fallback_keywords_file` | `/etc/bibit/keywords-fallback.txt` | 兼容旧唤醒词（Hello Bibi） |

### 音频采集

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `audio_source` | `default` | PipeWire 源；`default` 跟随系统声音设置 |
| `input_rate` | `16000` | 采集采样率（Hz） |
| `input_channels` | `2` | 采集声道数 |
| `audio_queue_max_ms` | `2000` | 采集队列上限（ms），钳到 200–30000；超出部分从最老的音频开始丢 |

### 采集队列与溢出

采集回调（生产者）把重采样到 16 kHz 的音频放入队列，处理线程（消费者，KWS+VAD+ASR）从队首取。队列**只用来吸收抖动**：超过 `audio_queue_max_ms` 时从队首丢弃最老的音频，并触发一次 `audio_discontinuity` 重置（重建 KWS 流、清空预滚与 VAD 待处理音频；若正在听写则按取消处理）。

队列上限限制最坏延迟，无法弥补处理速度持续低于实时的吞吐不足。调优时需同时观察采集速度、处理速度与队列深度。

溢出事件现在带判定信息，每秒最多一条（中间丢弃量聚合到下一条）：

| 字段 | 含义 |
|------|------|
| `dropped_samples` / `total_dropped_samples` | 本窗口丢弃量 / 累计丢弃量（采样数） |
| `queued_ms` / `queue_max_ms` | 丢弃后的队列深度 / 当前上限（ms） |
| `arrival_ms_per_s` | 上报窗口内采集侧送入的音频毫秒数（>1000 表示采集侧快于实时） |
| `process_ms_per_s` | 上报窗口内处理掉的音频毫秒数（<1000 表示处理链掉到实时以下） |
| `process_us_total` / `process_chunks` | 累计处理耗时（微秒）/ 累计块数；两次事件的差值即该窗口的每块平均耗时 |
| `process_us_max` | 单块处理耗时峰值（微秒）。一个块本身代表 16 kHz 下的 21.3 ms，超过它就追不回来 |

`status` 快照带 `queued_ms`（队列深度镜像，原子读，不取队列锁），可用于观察积压趋势。持续出现 `process_ms_per_s < arrival_ms_per_s` 时，应检查处理链耗时，而不是仅增加队列上限。

### 起音流前滚与队列吞吐

`kws_reset_pre_roll_ms=0` 为默认值：保留静音后起音重建，避免回灌历史音频的额外计算。起音窗口由 `kws_onset_window_ms` 控制，默认 3000 ms。

前滚归零后，起音流从当前块开始解码；长静音后紧接唤醒词时，词首召回由始终在线的主、备用连续流补充。如果短促唤醒的召回下降，可以将 `kws_reset_pre_roll_ms` 调回 800，并结合队列溢出指标评估吞吐影响；随后再考虑调整 `kws_silence_reset_ms` 或起音窗口。
