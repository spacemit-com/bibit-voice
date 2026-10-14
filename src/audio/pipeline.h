#pragma once

#include "config.h"
#include "audio_queue_guard.h"
#include "resampler.h"
#include "switch_intents.h"

#include <atomic>
#include <cstddef>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

struct SherpaOnnxKeywordSpotter;
struct SherpaOnnxOnlineStream;
struct SherpaOnnxVoiceActivityDetector;
struct SherpaOnnxSpeechSegment;
struct SherpaOnnxOfflineRecognizer;
struct SherpaOnnxOnlineRecognizer;
class Qwen3AsrClient;

class VoicePipeline {
 public:
  using EventCallback = std::function<void(const std::string &)>;

  VoicePipeline(BibitConfig config, EventCallback callback);
  ~VoicePipeline();
  bool initialize();
  bool start();
  void stop();
  void pushAudio(const float *samples, int count, int input_rate);
  void waitForAudioQueueBelow(size_t max_samples);
  void resetKwsForTest();
  // 开关命令：只记录命令里真正给出的字段，未给出的字段沿用当前生效值。
  // 调用方补全成「全量快照」是不安全的：补全用的镜像一旦过期，就会把另一个
  // 开关带回旧值，还会把没提过的字段记成一次新意图。
  struct SwitchRequest {
    bool wake_set = false;
    bool wake = false;
    bool asr_set = false;
    bool asr = false;
    std::string request_id;
  };
  void setSwitches(const SwitchRequest &request);
  void setThresholds(float kws_threshold, float kws_score,
                     float vad_threshold, float min_silence,
                     int wake_guard_ms);
  void setKwsDebug(bool enabled);
  bool setAsrModel(const BibitConfig &config, const std::string &request_id = "");
  bool modelSwitching() const;
  // 记录用户最新的引擎意图。切换在途时到达的请求不再被丢弃：提交点之后
  // 由 reconcileDesired() 只执行一次最新目标，中间目标被跳过。
  void noteDesiredModel(const std::string &model, const std::string &request_id);
  std::string effectiveAsrModel() const;
  std::string switchSnapshotJson() const;
  bool setAudioReady(bool ready) { return audio_ready_.exchange(ready) != ready; }
  void startManualAsr();
  void cancel();
  uint64_t activeSessionId() const { return session_id_.load(); }
  bool sessionActive() const { return mode_.load() != Mode::Idle; }
  void finishInput();
  bool recognizeDirect(const float *samples, int count, std::string *text,
                       int64_t *decode_us, std::string *error);
  // reply=true 标记这是 get_status 的请求回执（只回给请求方），
  // 与流水线主动广播的状态快照区分开（协议见 docs/protocol.md）。
  std::string statusJson(bool reply = false) const;
  std::string parametersJson() const;

 private:
  enum class Mode { Idle, Listening, Recognizing };
  enum class CommandType {
    Switches, Thresholds, KwsDebug, AsrRelease, AsrCommit, Manual, Cancel, Finish,
    KwsReset, Stop
  };
  // 后端构建的交接单：构建线程负责构建，流水线线程只在 AsrRelease /
  // AsrCommit 两个短命令里放下旧实例、接入新实例。
  struct AsrBackend {
    std::unique_ptr<Qwen3AsrClient> qwen3;
    const SherpaOnnxOfflineRecognizer *offline = nullptr;
    const SherpaOnnxOnlineRecognizer *online = nullptr;
    BibitConfig config;
    ~AsrBackend();
  };
  // 能力变更（换模型 / 打开识别）的可见状态，进入 status 快照的 switching 字段。
  struct SwitchState {
    bool active = false;
    std::string kind;
    std::string from;
    std::string to;
    std::string request_id;
    int64_t started_ms = 0;
  };
  // SpacemiT EP 与 TCM 是独占资源：新旧后端不能并存。构建线程必须先让
  // 流水线线程放下现役后端（AsrRelease），再开始构建目标后端。
  struct ReleaseTicket {
    std::mutex mutex;
    std::condition_variable cv;
    bool done = false;
    // 构建线程放弃这次切换时置位：流水线线程处理到这条命令必须跳过释放，
    // 否则会出现「回执说旧引擎仍生效、实际后端已被放下」的错位。
    bool abandoned = false;
  };
  struct Command {
    CommandType type;
    bool wake_set = false;
    bool wake = false;
    bool asr_set = false;
    bool asr = false;
    float kws_threshold = 0.10f;
    float kws_score = 3.0f;
    float vad_threshold = 0.40f;
    float min_silence = 0.50f;
    int wake_guard_ms = 180;
    bool kws_debug_enabled = false;
    std::shared_ptr<ReleaseTicket> release;
    // 发起这次能力变更的请求标识：结果事件据此关联回调用方。
    std::string request_id;
  };

  bool rebuildKws();
  bool rebuildVad();
  bool initializeAsr();
  bool buildAsr(const BibitConfig &cfg, AsrBackend *backend,
                std::string *error) const;
  void adoptAsrBackend(AsrBackend &&backend);
  bool startAsrSwitch(const BibitConfig &target, const std::string &kind,
                      const std::string &request_id);
  void commitAsrSwitch();
  void reconcileDesired();
  // 当前能力态快照：只能由流水线线程读取（后端句柄是非原子成员）。
  bibit::SwitchState currentSwitchState(bool switching) const;
  bool asrBackendReady() const {
    return qwen3_asr_ != nullptr || asr_offline_ != nullptr || asr_online_ != nullptr;
  }
  void emitSwitchSettlements(
      const std::vector<bibit::SwitchSettlement> &settlements) const;
  // 构建结束时把结果对应回发起它的那条意图；返回需要广播的终态（可能为空）。
  std::optional<bibit::SwitchSettlement> settleEnableBuild(bibit::SwitchOutcome outcome);
  void clearDesiredModel(const std::string &model);
  // 每个带 request_id 的切换请求恰好一个终态：成功、失败或被后续意图取代。
  struct WaitingRequest {
    std::string request_id;
    std::string target;
  };
  void settleWaitingRequests(const std::string &effective_model);
  void dropWaitingRequest(const std::string &request_id);
  // 调用方传入当前生效引擎，避免本函数再依赖 state_mutex_ 的持有前提。
  std::string desiredJson(const std::string &effective_model) const;
  bool beginSwitchState(const std::string &kind, const std::string &to,
                        const std::string &request_id);
  void endSwitchState();
  static std::string switchStateJson(const SwitchState &state);
  void destroyAsr();
  void destroyAsrStream();
  bool createAsrStream();
  void feedStreamingAsr(const float *samples, int count);
  void run();
  void processCommand(const Command &cmd);
  void processAudio(const std::vector<float> &audio);
  void processKws(const float *samples, int count, int64_t chunk_start);
  struct KwsMatch {
    std::string keyword;
    std::string path;
    int64_t start_ms = 0;
    int64_t end_ms = 0;
    int64_t duration_ms = 0;
    int64_t max_token_gap_ms = 0;
    int64_t candidate_start = 0;
    int64_t candidate_end = 0;
    double phrase_rms = 0.0;
    double pre_rms = 0.0;
    double rms_ratio = 0.0;
    std::string result_json;
    bool fallback_verify_applied = false;
    bool fallback_verified = false;
    int64_t fallback_verify_decode_us = 0;
    std::string fallback_verify_text;
    std::string fallback_verify_reason;
  };
  KwsMatch decodeKwsPath(const SherpaOnnxKeywordSpotter *spotter,
                            const SherpaOnnxOnlineStream *&stream,
                            const std::string &path, int max_duration_ms,
                            const float *samples, int count);
  bool verifyFallbackWithAsr(KwsMatch *match);
  void beginKwsDebugSession();
  void saveKwsDebugCapture(const KwsMatch &match, uint64_t session_id,
                           const std::string &outcome = "accepted");
  void beginListening(const std::string &keyword, bool feed_cached_audio = false,
                      int64_t cached_begin_sample = 0,
                      const std::string &reason = "wake");
  void feedVad(const float *samples, int count);
  std::vector<float> paddedSpeechSegment(const SherpaOnnxSpeechSegment *segment);
  void decodeSegment(const float *samples, int count);
  void emitAsrMetrics(int sample_count, int64_t decode_us,
                      const std::string &status,
                      const std::string &error_code = "") const;
  void resetToIdle(const std::string &reason = "reset");
  void resetKws();
  int playWakeAck();
  void discardCapturedAudio(const std::string &reason = "wake_ack_isolation");
  void emitEvent(const std::string &json) const;
  static std::string jsonEscape(const std::string &s);
  std::string stripWakePrefix(std::string text) const;

  BibitConfig config_;
  EventCallback callback_;
  mutable std::mutex state_mutex_;
  std::atomic<bool> wake_enabled_;
  std::atomic<bool> asr_enabled_;
  std::atomic<bool> audio_ready_{false};
  std::atomic<bool> kws_loaded_{false};
  std::atomic<bool> asr_loaded_{false};
  std::atomic<Mode> mode_{Mode::Idle};
  std::atomic<uint64_t> session_id_{0};

  // 切换状态由 state_mutex_ 保护：socket 线程读快照，流水线线程写状态。
  SwitchState switch_state_;
  std::mutex switch_mutex_;
  std::thread switch_thread_;
  std::atomic<bool> switch_cancel_{false};
  std::mutex pending_mutex_;
  std::shared_ptr<AsrBackend> pending_backend_;
  std::string pending_error_;
  bool pending_rolled_back_ = false;
  bool pending_active_kept_ = false;
  // 用户最新意图：切换在途时到达的请求记在这里，提交点之后收敛。
  mutable std::mutex desired_mutex_;
  std::string desired_model_type_;
  std::string desired_request_id_;
  // 开关意图账本（wake/asr）：与引擎目标并列，由 reconcileDesired() 统一收敛。
  bibit::SwitchIntentBook switch_book_;
  // 最近一次失败切换的目标：结算排队请求时据此按失败分类，而不是标成「被取代」。
  std::string failed_model_target_;
  std::vector<WaitingRequest> waiting_requests_;

  const SherpaOnnxKeywordSpotter *kws_ = nullptr;
  const SherpaOnnxOnlineStream *kws_stream_ = nullptr;
  const SherpaOnnxOnlineStream *kws_onset_stream_ = nullptr;
  const SherpaOnnxKeywordSpotter *kws_fallback_ = nullptr;
  const SherpaOnnxOnlineStream *kws_fallback_stream_ = nullptr;
  const SherpaOnnxOnlineStream *kws_fallback_onset_stream_ = nullptr;
  const SherpaOnnxVoiceActivityDetector *vad_ = nullptr;
  const SherpaOnnxOfflineRecognizer *asr_offline_ = nullptr;
  const SherpaOnnxOnlineRecognizer *asr_online_ = nullptr;
  const SherpaOnnxOnlineStream *asr_stream_ = nullptr;
  std::unique_ptr<Qwen3AsrClient> qwen3_asr_;

  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::deque<std::vector<float>> audio_queue_;
  size_t queued_audio_samples_ = 0;
  // Capture-queue ceiling; see audio_queue_guard.h for why it is not 30 s.
  const size_t queue_max_samples_;
  QueueOverflowReporter overflow_reporter_;
  // Per-chunk processing cost, and an atomic mirror of the queue depth so
  // statusJson (const, state_mutex_) can report it without taking
  // queue_mutex_ and inverting the lock order.
  ProcessTimeTracker process_time_;
  std::atomic<int64_t> queued_ms_snapshot_{0};
  // Snapshot for the arrival/processing rates reported with an overflow.
  int64_t overflow_window_ms_ = 0;
  int64_t overflow_arrived_ = 0;
  int64_t overflow_processed_ = 0;
  std::atomic<uint64_t> dropped_audio_samples_{0};
  // Audio totals used to tell a bursty capture side apart from a slow worker:
  // if arrival runs above realtime but processing stays at realtime, the
  // capture callback is delivering faster than the device produces.
  std::atomic<int64_t> arrived_samples_{0};
  std::atomic<int64_t> processed_samples_{0};
  // Set when the queue had to drop audio. The worker resets the KWS/VAD state
  // on the next chunk so a fragmented utterance is never decoded as one.
  std::atomic<bool> audio_discontinuity_{false};
  std::deque<Command> commands_;
  bool running_ = false;
  std::thread worker_;
  std::unique_ptr<StreamingResampler> resampler_;
  int resampler_rate_ = 0;

  std::deque<float> ring_;
  std::string kws_debug_session_dir_;
  int64_t ring_start_abs_ = 0;
  int64_t absolute_samples_ = 0;
  int64_t kws_stream_start_abs_ = 0;
  int64_t kws_silence_samples_ = 0;
  bool kws_sound_active_ = false;
  bool kws_onset_paths_active_ = false;
  int64_t kws_onset_paths_until_abs_ = 0;
  int64_t listening_samples_ = 0;
  int64_t streaming_asr_decode_us_ = 0;
  int64_t last_vad_log_sample_ = 0;
  double last_audio_rms_ = 0.0;
  std::vector<float> vad_pending_;
  std::vector<float> vad_audio_history_;
  int64_t vad_input_start_abs_ = 0;
  bool speech_started_ = false;
  bool asr_session_active_ = false;
  bool followup_listening_ = false;
  std::string active_keyword_;
};
