# Bibit Voice 协议规范

本文档定义 Bibit 语音服务的 IPC 协议。仓库只提供服务端、协议与文档，不包含任何
消费 Socket 的客户端实现（GUI/悬浮球等客户端在本仓库之外）。

## 1. 架构

```
┌─────────────────────────────────────────────────────────────┐
│ bibit-audio (C++, systemd user service bibit-audio.service) │
│   PipeWire 采集 → KWS 唤醒 → VAD → ASR (SenseVoice/Qwen3-ASR)│
│   监听 /run/bibit/audio.sock (多客户端 JSONL)     │
└───────────────────────────┬─────────────────────────────────┘
                            │ audio.sock 协议
┌───────────────────────────▼─────────────────────────────────┐
│ bibit-voice-tool (Python, bibit-voice-tool.service)          │
│   监听 /run/bibit/tool.sock                      │
│   透传 audio.sock，附加工具状态/错误事件；另提供 stdio MCP    │
└───────────────────────────┬─────────────────────────────────┘
                            │ tool.sock 协议（消费方在仓库外）
┌───────────────────────────▼─────────────────────────────────┐
│ 客户端（bibit-gui / CLI / MCP host，不在本仓库）             │
└─────────────────────────────────────────────────────────────┘
```

## 2. Socket 路径

| Socket | 默认路径 | 提供方 | 权限 |
|--------|----------|--------|------|
| `audio.sock` | `/run/bibit/audio.sock` | `bibit-audio` | 0600（创建者） |
| `tool.sock` | `/run/bibit/tool.sock` | `bibit-voice-tool serve` | 0600（创建者） |

两段 Socket 均为全局固定的 Unix 域路径（目录 `/run/bibit` 由 tmpfiles.d 创建，socket 文件 0600）。传输为 SOCK_STREAM、**换行符分隔的 JSON**（每行一条完整 JSON，无内嵌换行）。

## 3. 传输约定

- 客户端 → 服务：一行 JSON 命令，形如 `{"cmd":"<name>", ...}`。
- 服务 → 客户端：一行 JSON 事件，形如 `{"event":"<name>", ...}`。
- 命令和事件共用同一连接，无请求/响应配对，靠事件流驱动状态机。
- 单行上限：`tool.sock` 代理 16 KiB，`bibit-audio` 未设硬上限但命令均为短文本。

## 4. audio.sock 协议（bibit-audio）

### 4.1 命令（客户端 → 服务）

| cmd | 参数 | 说明 |
|-----|------|------|
| `get_status` | — | 返回 `status` + `parameters` 两个事件 |
| `set_switches` | `wake?:bool`, `asr?:bool`, `request_id?:string` | 按字段提交 KWS / ASR 开关意图（至少给一个字段，未给出沿用当前生效值），写持久化（`~/.config/bibit/switches.conf`）。未给任何字段、或给出的字段不是 JSON boolean 时回 `BAD_COMMAND` |
| `set_thresholds` | `kws_threshold`, `kws_score`, `vad_threshold`, `min_silence`, `wake_guard_ms` | 运行时阈值，写持久化 |
| `set_kws_debug` | `enabled:bool` | 是否保存 KWS 唤醒 WAV/JSON 采样 |
| `set_asr_model` | `model:"sensevoice"\|"qwen3_asr_0.6b"`, `request_id?:string` | 热切换 ASR 模型（异步长任务，见 §4.3） |
| `start_manual_asr` | — | 手动启动一次 ASR（不等唤醒词） |
| `cancel` | `session_id?:uint` | 取消当前会话；`session_id` 非 0 时仅当与当前活跃会话一致才执行，否则回 `SESSION_MISMATCH` |

### 4.2 事件（服务 → 客户端）

| event | 关键字段 | 说明 |
|-------|----------|------|
| `status` | `wake, audio_ready, queued_ms, kws_loaded, asr, asr_loaded, asr_model_type, switching, desired, partial_switches, queued_model_switch, switch_settlement, mode(idle/listening/recognizing), session_id, ..., reply?` | 完整状态快照；`switching` 为 `null` 或切换快照，`desired` 给出尚未达成的意图，`partial_switches`/`queued_model_switch`/`switch_settlement` 声明服务端能力（四者见 §4.4），`reply:true` 表示这是 `get_status` 的请求回执（见 §4.3） |
| `parameters` | `asr_model_type, asr_num_threads, qwen3_server_port, ...` | 完整参数快照 |
| `ready` | `asr_model_type, asr_loaded, ...` | 模型/流水线就绪 |
| `error` | `code, message?` | 错误，见错误码表 |
| `audio_state` | `message` | `audio_opened` / `audio_closed` |
| `switches_applied` | `request_id, wake, asr` | 开关命令的受理回执（广播）：开关态与采集已按这次命令调整。**不是**识别可用，也**不是**排队意图已达成；旧客户端不传 `request_id` 时该字段为空字符串 |
| `switches_settled` | `request_id, outcome(applied/superseded/failed), wake, asr` | 开关意图的终态（广播，见 §4.4）：`wake/asr` 是结算时的生效态。打开识别只有后端就绪才算 `applied` |
| `model_switching` | `model, to, from, request_id, queued?` | 开始切换模型；`queued:true` 表示该意图排在在途切换之后按最新意图收敛（广播） |
| `model_switched` | `model, to, from, request_id, duration_ms?, noop?, settled?` | 切换完成；`noop:true` 表示本来就是目标模型；`settled:true` 表示这是提交点之后为排队请求补发的终态（无 `duration_ms`，`from == to`） |
| `model_switch_superseded` | `request_id, to, active_model` | 该请求的目标被更新的意图取代（终态之一，不是失败） |
| `wake_triggered` | `session_id` | 唤醒词命中 |
| `listening` | `session_id` | 开始聆听 |
| `speech_started` | `session_id` | 检测到说话 |
| `recognizing` | `session_id, text` | 中间识别结果（可多次） |
| `asr_final` | `session_id, text, decode_ms?, audio_duration_ms?, rtf?` | 最终识别文本 |
| `session_cancelled` | `session_id` | 会话已取消 |
| `session_exit` | `session_id` | 会话退出 |
| `input_finished` | `session_id` | 输入结束 |
| `kws_reset` | — | KWS 状态复位 |
| `audio_queue_overflow` | `dropped_samples, total_dropped_samples, queued_ms, queue_max_ms, arrival_ms_per_s, process_ms_per_s, process_us_total, process_us_max, process_chunks` | 采集队列超过 `audio_queue_max_ms` 并丢弃最老音频（每秒最多一条，中间丢弃量聚合）；字段含义与判读见 `docs/configuration.md` |
| `audio_discontinuity` | `reason, queued_ms, session_id` | 丢弃造成音频不连续，KWS/VAD 状态已按此重置；正在听写时会话按取消处理（见 `docs/configuration.md`） |

### 4.3 ASR 模型切换语义（异步、可观测、不中断）

`set_asr_model` 是异步长任务：命令被接受后立即回执 `model_switching`（带 `request_id`），
目标后端的构建在独立线程进行，完成后广播 `model_switched`。

- **终态唯一**：每个带 `request_id` 的切换请求恰好收到一个终态——`model_switched`（成功）、
  `ASR_MODEL_SWITCH_FAILED` / `ASR_INIT_FAILED`（失败）或 `model_switch_superseded`（被更新的
  意图取代）。同一目标的多个请求各自收到终态，不会只回 `ASR_MODEL_SWITCH_BUSY` 后永久挂起。
- **BUSY 与排队**：目标不同于生效模型且已有切换在途时仍回 `ASR_MODEL_SWITCH_BUSY`（保持既有客户端的
  重试语义），同时该意图被记录，当前切换的提交点之后自动收敛到最新目标。客户端在
  `status.queued_model_switch` 为真时可以把 BUSY 当作非终态，继续等待终态事件而不重试。
- **失败不重试**：一次切换失败后对应的期望目标被清除，不会被收敛逻辑反复重试。
- **排队请求的失败分类**：目标被一次失败切换清掉时，跟随其后的排队请求收到
  `ASR_MODEL_SWITCH_FAILED` 而不是 `model_switch_superseded`。「被取代」只表示被更新的意图覆盖，
  不表示引擎切换失败；调用方不需要靠比对 `active_model` 才能分辨。
- **不中断采集与唤醒**：切换分三步——流水线线程先放下现役后端（短），构建线程再构建目标后端（长），
  最后由流水线线程接入（短）。PipeWire 采集、KWS、VAD 与 IPC 全程照常运行；
  识别本身在切换期间短暂不可用（`asr_loaded:false`），提交瞬间处于 `listening`/`recognizing`
  的会话会被取消（`session_cancelled`）。
- **为什么不能新旧并存**：SpacemiT EP 与 TCM 是独占资源，实测在同一进程里一边跑 Qwen3 的
  SMT 媒体后端、一边创建 SenseVoice 的 EP 会话会 SEGV，因此必须先放下现役后端再构建。
  历史实现把整段 destroy→init 压在流水线线程上，会让一次 Qwen3 冷启动把流水线按分钟级阻塞，已废弃。
- **生效语义**：`status.asr_model_type` 始终是**生效模型**。目标模型只出现在切换过程中，
  以 `status.switching.to` 或 `model_switching.to` 表达；切换失败时生效模型不变，且不需要回滚。
- **状态快照**：`switching` 为 `null`（空闲）或对象：
  `{"kind":"asr_model"|"asr_enable","from":"<生效模型>","to":"<目标模型>","request_id":"...","elapsed_ms":1234}`。
  任何客户端（含中途接入的 CLI、切换中途打开设置页的 GUI）都可以只凭快照重建界面。
  `kind:"asr_enable"` 表示打开识别开关时的后端构建（Qwen3 同样需要拉起 `llama-server`）。
- **幂等**：目标等于生效模型时是成功 no-op，回执 `model_switched`（`duration_ms:0`、`noop:true`），
  不返回 `ASR_MODEL_SWITCH_BUSY`。
- **忙碌**：目标不同于生效模型、且已有切换在进行时，回 `ASR_MODEL_SWITCH_BUSY`，
  带 `request_id`、`to` 与当时的 `switching` 快照。该意图同时被记录：当前切换的提交点
  之后会自动收敛到最新目标，客户端即使不重试也会切到它请求的引擎。
- **回执与广播**：`set_asr_model` 的所有结果（no-op、忙碌、文件缺失、切换完成/失败）都**广播**，
  用 `request_id` 对应发起方——客户端发命令用的是「写完即关」的连接，只回请求方的错误到不了订阅事件的界面。
  `get_status` 的回执带 `reply:true`（只发给请求方）；状态广播不带该字段，两者内容语义一致，
  客户端都可以当作当前快照消费。
- **失败回滚**：目标后端构建失败时，构建线程立即重建旧后端并在提交点接入，
  错误事件带 `rolled_back:true` 与 `active_model`；旧后端也重建失败时 `asr_enabled` 置为 false。
- **释放超时**：构建前必须先确认现役后端已放下。流水线线程被长解码占住、超过
  `asr_release_timeout_ms`（默认 30s）仍未回执时，切换按失败处理：不再构建目标后端
  （新旧并存会争用 SpacemiT EP/TCM），现役后端原样保留，错误事件带
  `rolled_back:false` 与 `active_model`（等于 `from`），识别保持可用。
- **持久化**：`~/.config/bibit/asr-model.conf` 在切换**成功后**写入；写入失败额外广播
  `ASR_MODEL_PERSIST_FAILED`，不影响本次已生效的切换。

### 4.4 命令执行模型

`bibit-audio` 的 socket 处理线程只做 accept、按行解析、投递与事件广播；任何会落盘、
起停 PipeWire 采集或构建 ASR 后端的命令都在独立的控制面线程执行。这条约束保证慢命令
不会推迟其它命令的接收、`get_status` 回执与事件广播。

- **字段级意图**：`wake`、`asr` 未给出的字段沿用当前生效值。客户端不必发送全量快照，
  也就不会因为本地快照过期而把另一个开关改回旧值。
- **最后意图胜出**：能力变更以「期望状态」收敛。切换在途时到达的新意图会被记录，并在
  当前切换提交后执行一次最新目标；更早的中间目标允许被跳过。
- **意图态与生效态并列**：`status.desired` 给出尚未达成的目标，`status.asr_model_type`、
  `status.wake`、`status.asr` 始终是当前生效态。任何客户端都能只凭快照重建界面。
- **回执**：`set_switches` 应用后广播 `switches_applied`；`set_asr_model` 的结果恒带
  `request_id`，用于把结果对应到发起它的那次意图。
- **能力协商**：`status.partial_switches` 表示服务端接受省略 `wake`/`asr` 的字段级提交；
  `status.queued_model_switch` 表示在途期间提交的目标会被记录并在提交点后收敛；
  `status.switch_settlement` 表示服务端用 `switches_settled` 给出开关意图的终态。客户端在省略
  字段或依赖排队语义之前必须先确认对应能力，否则退回全量提交与旧的重试规则：旧服务端会把缺失
  字段按默认 `true` 处理，只发一个字段会把另一个开关一起改掉。
- **开关意图恰好一个终态**（`switch_settlement`）：每条带 `request_id` 的开关命令，最终恰好
  收到一个 `switches_settled`（`outcome: applied|superseded|failed`），或一条归属它的
  `error`（如 `ASR_INIT_FAILED`）。`applied` 只表示这次命令请求的字段全部已生效：打开识别
  必须等后端就绪（`asr_loaded=true`）才算生效，构建中、排队中都不算；`superseded` 表示同一
  字段被更新的意图取代（事件里的 `wake/asr` 是结算时的生效态，客户端据此分辨哪一部分生效）。
  一条命令同时给出两个字段时按命令整体结算：任一字段被取代即 `superseded`。
- **构建期间关闭识别**：关闭不需要构建，立即生效并放下后端；在途构建的结果在提交点被丢弃
  （提交点只采纳「生效开关仍然是打开」的结果），排队中的打开意图随之判为 `superseded`，
  开关绝不会在构建结束后被翻回打开。构建失败只结算驱动它的那条意图（`ASR_INIT_FAILED`），
  期间到达的更新意图继续收敛（必要时重试一次构建）。
- **request_id 关联**：由请求触发的事件（`SWITCH_PERSIST_FAILED`、`AUDIO_OPEN_FAILED`、
  `ASR_INIT_FAILED`、`ASR_MODEL_PERSIST_FAILED`、`ASR_MODEL_SWITCH_FAILED`、
  `ASR_MODEL_SWITCH_BUSY`）都携带 `request_id`；`request_id` 是外部输入，进入 JSON 输出时统一转义。
- **`switches_applied` 的边界**：它表示开关与采集已应用。打开识别时 ASR 后端的构建仍是异步的，
  构建失败的终态是同一 `request_id` 的 `ASR_INIT_FAILED`；`switches_applied` 不是「识别已可用」，
  也不是排队意图的终态——终态看 `switches_settled`（`switch_settlement` 为真时）。
- **事件顺序**：`SWITCH_PERSIST_FAILED` 先于 `switches_applied` 发出（先落盘、后回执），
  回执本身不代表持久化成功。旧服务端没有 `switches_applied`，客户端不应把它当作唯一成功依据。
  开关偏好按「生效态」落盘：`switches_settled` 给出的 `wake/asr` 才写进
  `~/.config/bibit/switches.conf`，排队中或构建失败的打开不会留下一个启动时必然失败的预设。
- **能力位约定**：新增语义必须同时新增能力位；同一主版本内能力位只增不减。长连接消费者在服务
  重启后必须重新探测，不能永久沿用首次读取的能力快照。
- **`desired.asr_model_type` 为空串**表示当前没有未达成的引擎目标，不是「未知引擎」。
- **`request_id` 作用域**：只要求在单次请求范围内唯一。不要把同一个字符串同时用于
  `set_switches` 与 `set_asr_model`，否则引擎切换结算时可能摘掉另一个请求的等待项。

---

## 5. tool.sock 协议（bibit-voice-tool serve）

`tool.sock` 是 `audio.sock` 的**多客户端透传代理**：

- 客户端命令原样转发到上游 `audio.sock`，不重新包装。
- 上游事件原样广播给所有已连接客户端。
- 代理自身额外产生两类事件：

| event | 关键字段 | 说明 |
|-------|----------|------|
| `tool_state` | `state:"online"`, `upstream` | 代理已连上上游 |
| `error` | `code` | `SERVICE_DISCONNECTED` / `SERVICE_OFFLINE` / `BAD_COMMAND` |

因此，tool.sock 客户端消费的命令/事件集合 = 第 4 节集合 + 上述 `tool_state`。

## 6. MCP 协议（stdio）

`bibit-voice-tool mcp` 在标准输入/输出上运行 JSON-RPC 2.0，暴露 8 个工具：

| 工具名 | 说明 |
|--------|------|
| `bibit_voice_status` | 读取音频/KWS/VAD/ASR 状态 |
| `bibit_transcribe_once` | 启动一次手动 ASR 并等待 `asr_final` |
| `bibit_wait_for_interaction` | 等待下一次 KWS+ASR 交互 |
| `bibit_cancel_voice_session` | 取消当前会话 |
| `bibit_set_voice_switches` | 开关 KWS/ASR |
| `bibit_set_asr_model` | 热切换模型 |
| `bibit_set_voice_thresholds` | 修改 KWS/VAD 阈值 |
| `bibit_set_kws_debug` | 开关 KWS debug 采样 |

## 7. 典型事件流

唤醒交互：

```
→ {"cmd":"get_status"}
← {"event":"status","mode":"idle",...}
← {"event":"parameters",...}
   ...（等待唤醒词）...
← {"event":"wake_triggered","session_id":42}
← {"event":"listening","session_id":42}
← {"event":"speech_started","session_id":42}
← {"event":"recognizing","session_id":42,"text":"你好"}
← {"event":"recognizing","session_id":42,"text":"你好，我是语音"}
← {"event":"asr_final","session_id":42,"text":"你好，我是语音","decode_ms":123}
```

手动 ASR（`start_manual_asr`）与取消（`cancel`）复用同一套 `listening → recognizing → asr_final` 事件，`session_id` 由服务端分配。

## 8. 错误码

| code | 来源 | 含义 |
|------|------|------|
| `BAD_COMMAND` | audio / tool | 未知命令、非法 JSON，或 `set_switches` 未给出任何 `wake`/`asr` 字段、给出的字段不是 JSON boolean |
| `SERVICE_OFFLINE` | tool | 上游 audio 不可用 |
| `SERVICE_DISCONNECTED` | tool | 上游连接中断 |
| `AUDIO_OPEN_FAILED` | audio | PipeWire 采集打开失败 |
| `UNSUPPORTED_ASR_MODEL` | audio / tool | 未知模型名 |
| `ASR_MODEL_FILES_MISSING` | audio | 模型文件缺失 |
| `ASR_MODEL_SWITCH_BUSY` | audio | 目标不同于生效模型且正在切换；带 `request_id`、`to` 与 `switching` 快照。意图会被记录并在提交点后自动收敛（见 §4.3） |
| `ASR_MODEL_SWITCH_SUPERSEDED` | tool | 客户端侧：请求的目标被更新的意图取代且当前未生效（由 `model_switch_superseded` 映射） |
| `SWITCH_INTENT_SUPERSEDED` | tool | 客户端侧：开关意图被更新的命令取代且未生效（由 `switches_settled.outcome=superseded` 映射，见 §4.4） |
| `ASR_MODEL_SWITCH_FAILED` | audio | 目标后端构建失败，或释放现役后端超时（`asr_release_timeout_ms`）；带 `request_id`/`from`/`to`/`active_model`，生效模型未变 |
| `ASR_INIT_FAILED` | audio | 打开识别开关时后端构建失败 |
| `ASR_MODEL_PERSIST_FAILED` | audio | 切换已生效，但模型偏好写入失败 |
| `SWITCH_PERSIST_FAILED` | audio | 开关持久化失败 |
| `SESSION_MISMATCH` | audio | cancel 的 session_id 与当前活跃会话不符 |
| `THRESHOLD_PERSIST_FAILED` | audio | 阈值持久化失败 |
| `KWS_DEBUG_PERSIST_FAILED` | audio | KWS debug 持久化失败 |
| `QWEN3_ASR_FAILED` | audio | Qwen3-ASR 解码失败 |
| `ASR_STREAM_MISSING` | audio | ASR 流缺失 |
| `ASR_STREAM_CREATE_FAILED` | audio | ASR 流创建失败 |
