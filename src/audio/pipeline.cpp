#include "pipeline.h"
#include "qwen3_asr.h"
#include "wake_text.h"
#include "wav.h"

#include <sherpa-onnx/c-api/c-api.h>

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <spawn.h>
#include <sstream>
#include <stdexcept>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

extern char **environ;

namespace {
int64_t steadyNowMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

std::string joinPath(const std::string &a, const std::string &b) {
  return a + (a.empty() || a.back() == '/' ? "" : "/") + b;
}

std::string lowercase(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

bool isZipformerCtc(const std::string &model_type) {
  const std::string type = lowercase(model_type);
  return type == "zipformer" || type == "zipformer_ctc" ||
         type == "zipformer2_ctc";
}

bool isQwen3Asr(const std::string &model_type) {
  const std::string type = lowercase(model_type);
  return type == "qwen3" || type == "qwen3_asr" ||
         type == "qwen3_asr_0.6b" || type == "qwen3-asr-0.6b";
}

void trimText(std::string &s) {
  const std::vector<std::string> prefixes = {" ", "\t", "\r", "\n", ",", ".", "!", "?", "，", "。", "！", "？", "、", ":", "："};
  bool changed = true;
  while (changed && !s.empty()) {
    changed = false;
    for (const auto &p : prefixes) {
      if (s.rfind(p, 0) == 0) {
        s.erase(0, p.size());
        changed = true;
        break;
      }
    }
  }
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
}

bool isExitCommand(std::string text) {
  trimText(text);
  const std::vector<std::string> suffixes = {
      " ", "\t", "\r", "\n", ",", ".", "!", "?", "，", "。", "！", "？", "、", ":", "："};
  bool changed = true;
  while (changed && !text.empty()) {
    changed = false;
    for (const auto &suffix : suffixes) {
      if (text.size() >= suffix.size() &&
          text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0) {
        text.erase(text.size() - suffix.size());
        changed = true;
        break;
      }
    }
  }
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return text == "退出" || text == "exit" || text == "quit";
}

}  // namespace

VoicePipeline::VoicePipeline(BibitConfig config, EventCallback callback)
    : config_(std::move(config)),
      callback_(std::move(callback)),
      wake_enabled_(config_.wake_enabled),
      asr_enabled_(config_.asr_enabled),
      queue_max_samples_(queueMaxSamples(config_.audio_queue_max_ms)) {
  overflow_window_ms_ = steadyNowMs();
}

VoicePipeline::~VoicePipeline() {
  stop();
  if (kws_onset_stream_) SherpaOnnxDestroyOnlineStream(kws_onset_stream_);
  if (kws_stream_) SherpaOnnxDestroyOnlineStream(kws_stream_);
  if (kws_) SherpaOnnxDestroyKeywordSpotter(kws_);
  if (kws_fallback_onset_stream_)
    SherpaOnnxDestroyOnlineStream(kws_fallback_onset_stream_);
  if (kws_fallback_stream_) SherpaOnnxDestroyOnlineStream(kws_fallback_stream_);
  if (kws_fallback_) SherpaOnnxDestroyKeywordSpotter(kws_fallback_);
  if (vad_) SherpaOnnxDestroyVoiceActivityDetector(vad_);
  destroyAsr();
}

bool VoicePipeline::rebuildKws() {
  const std::string kws_root = joinPath(
      config_.model_root,
      "kws/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20");
  SherpaOnnxKeywordSpotterConfig kc{};
  kc.feat_config.sample_rate = 16000;
  kc.feat_config.feature_dim = 80;
  const std::string chunk = std::to_string(config_.kws_chunk_size);
  const std::string encoder = joinPath(kws_root, "encoder-epoch-13-avg-2-chunk-" + chunk + "-left-64.int8.onnx");
  const std::string decoder = joinPath(kws_root, "decoder-epoch-13-avg-2-chunk-" + chunk + "-left-64.onnx");
  const std::string joiner = joinPath(kws_root, "joiner-epoch-13-avg-2-chunk-" + chunk + "-left-64.int8.onnx");
  const std::string kws_tokens = joinPath(kws_root, "tokens.txt");
  kc.model_config.transducer.encoder = encoder.c_str();
  kc.model_config.transducer.decoder = decoder.c_str();
  kc.model_config.transducer.joiner = joiner.c_str();
  kc.model_config.tokens = kws_tokens.c_str();
  kc.model_config.provider = "cpu";
  kc.model_config.num_threads = 1;
  kc.max_active_paths = config_.kws_max_active_paths;
  kc.num_trailing_blanks = config_.kws_trailing_blanks;
  auto create_path = [&](const std::string &keywords_file, float score,
                         float threshold,
                         const SherpaOnnxKeywordSpotter **spotter,
                         const SherpaOnnxOnlineStream **stream) {
    kc.keywords_score = score;
    kc.keywords_threshold = threshold;
    kc.keywords_file = keywords_file.c_str();
    *spotter = SherpaOnnxCreateKeywordSpotter(&kc);
    if (!*spotter) return false;
    *stream = SherpaOnnxCreateKeywordStream(*spotter);
    if (!*stream) {
      SherpaOnnxDestroyKeywordSpotter(*spotter);
      *spotter = nullptr;
      return false;
    }
    return true;
  };

  const SherpaOnnxKeywordSpotter *new_kws = nullptr;
  const SherpaOnnxOnlineStream *new_stream = nullptr;
  if (!create_path(config_.keywords_file, config_.kws_score,
                   config_.kws_threshold, &new_kws, &new_stream)) return false;
  const SherpaOnnxOnlineStream *new_onset_stream =
      SherpaOnnxCreateKeywordStream(new_kws);
  if (!new_onset_stream) {
    SherpaOnnxDestroyOnlineStream(new_stream);
    SherpaOnnxDestroyKeywordSpotter(new_kws);
    return false;
  }

  const SherpaOnnxKeywordSpotter *new_fallback = nullptr;
  const SherpaOnnxOnlineStream *new_fallback_stream = nullptr;
  if (!config_.kws_fallback_keywords_file.empty() &&
      !create_path(config_.kws_fallback_keywords_file,
                   config_.kws_fallback_score,
                   config_.kws_fallback_threshold,
                   &new_fallback, &new_fallback_stream)) {
    SherpaOnnxDestroyOnlineStream(new_onset_stream);
    SherpaOnnxDestroyOnlineStream(new_stream);
    SherpaOnnxDestroyKeywordSpotter(new_kws);
    return false;
  }
  const SherpaOnnxOnlineStream *new_fallback_onset_stream = nullptr;
  if (new_fallback) {
    new_fallback_onset_stream = SherpaOnnxCreateKeywordStream(new_fallback);
    if (!new_fallback_onset_stream) {
      SherpaOnnxDestroyOnlineStream(new_fallback_stream);
      SherpaOnnxDestroyKeywordSpotter(new_fallback);
      SherpaOnnxDestroyOnlineStream(new_onset_stream);
      SherpaOnnxDestroyOnlineStream(new_stream);
      SherpaOnnxDestroyKeywordSpotter(new_kws);
      return false;
    }
  }
  if (kws_onset_stream_) SherpaOnnxDestroyOnlineStream(kws_onset_stream_);
  if (kws_stream_) SherpaOnnxDestroyOnlineStream(kws_stream_);
  if (kws_) SherpaOnnxDestroyKeywordSpotter(kws_);
  if (kws_fallback_onset_stream_)
    SherpaOnnxDestroyOnlineStream(kws_fallback_onset_stream_);
  if (kws_fallback_stream_) SherpaOnnxDestroyOnlineStream(kws_fallback_stream_);
  if (kws_fallback_) SherpaOnnxDestroyKeywordSpotter(kws_fallback_);
  kws_ = new_kws;
  kws_stream_ = new_stream;
  kws_onset_stream_ = new_onset_stream;
  kws_fallback_ = new_fallback;
  kws_fallback_stream_ = new_fallback_stream;
  kws_fallback_onset_stream_ = new_fallback_onset_stream;
  kws_onset_paths_active_ = false;
  kws_stream_start_abs_ = absolute_samples_;
  kws_loaded_.store(true);
  return true;
}

bool VoicePipeline::rebuildVad() {
  SherpaOnnxVadModelConfig vc{};
  const std::string vad_model = joinPath(config_.model_root, "vad/silero_vad.onnx");
  vc.silero_vad.model = vad_model.c_str();
  vc.silero_vad.threshold = config_.vad_threshold;
  vc.silero_vad.min_silence_duration = config_.min_silence;
  vc.silero_vad.min_speech_duration = config_.min_speech;
  vc.silero_vad.max_speech_duration = config_.max_speech;
  vc.silero_vad.window_size = 512;
  vc.sample_rate = 16000;
  vc.num_threads = 1;
  vc.provider = "cpu";
  const SherpaOnnxVoiceActivityDetector *new_vad =
      SherpaOnnxCreateVoiceActivityDetector(&vc, config_.max_speech + 5.0f);
  if (!new_vad) return false;
  if (vad_) SherpaOnnxDestroyVoiceActivityDetector(vad_);
  vad_ = new_vad;
  return true;
}

VoicePipeline::AsrBackend::~AsrBackend() {
  if (offline) SherpaOnnxDestroyOfflineRecognizer(offline);
  if (online) SherpaOnnxDestroyOnlineRecognizer(online);
  offline = nullptr;
  online = nullptr;
}

// 只构建，不改变流水线状态：失败时调用方持有的 AsrBackend 析构会回收半成品，
// 现役后端不会被碰。这样切换失败等于没发生，不需要回滚。
bool VoicePipeline::buildAsr(const BibitConfig &cfg, AsrBackend *backend,
                             std::string *error) const {
  if (!backend) return false;

  const std::string model_type = lowercase(cfg.asr_model_type);
  if (isQwen3Asr(model_type)) {
    auto client = std::make_unique<Qwen3AsrClient>(cfg);
    std::string reason;
    if (!client->initialize(&reason)) {
      std::cerr << "Qwen3-ASR initialization failed: " << reason << std::endl;
      if (error) {
        *error = reason.empty() ? "Qwen3-ASR initialization failed" : reason;
      }
      return false;
    }
    backend->qwen3 = std::move(client);
    backend->config = cfg;
    return true;
  }

  std::string provider = cfg.asr_provider.empty() ? "cpu" : cfg.asr_provider;
  if (provider == "spacemit" && !cfg.provider_config.empty()) {
    provider += ":" + cfg.provider_config;
  }
  const int num_threads = std::clamp(cfg.asr_num_threads, 1, 16);
  const std::string decoding_method = cfg.asr_decoding_method.empty()
      ? "greedy_search" : cfg.asr_decoding_method;
  std::vector<float> warmup(16000, 0.0f);

  if (isZipformerCtc(model_type)) {
    SherpaOnnxOnlineRecognizerConfig ac{};
    ac.feat_config.sample_rate = 16000;
    ac.feat_config.feature_dim = 80;
    ac.model_config.zipformer2_ctc.model = cfg.asr_model.c_str();
    ac.model_config.tokens = cfg.asr_tokens.c_str();
    ac.model_config.provider = provider.c_str();
    ac.model_config.num_threads = num_threads;
    ac.decoding_method = decoding_method.c_str();
    ac.max_active_paths = std::max(1, cfg.asr_max_active_paths);
    ac.enable_endpoint = 0;  // Silero VAD owns endpointing for Bibit.
    const SherpaOnnxOnlineRecognizer *online =
        SherpaOnnxCreateOnlineRecognizer(&ac);
    if (!online) {
      if (error) *error = "cannot create online recognizer: " + cfg.asr_model;
      return false;
    }
    backend->online = online;

    const SherpaOnnxOnlineStream *stream = SherpaOnnxCreateOnlineStream(online);
    if (!stream) {
      if (error) *error = "cannot create online stream: " + cfg.asr_model;
      return false;
    }
    SherpaOnnxOnlineStreamAcceptWaveform(stream, 16000, warmup.data(), warmup.size());
    SherpaOnnxOnlineStreamInputFinished(stream);
    while (SherpaOnnxIsOnlineStreamReady(online, stream)) {
      SherpaOnnxDecodeOnlineStream(online, stream);
    }
    const SherpaOnnxOnlineRecognizerResult *r =
        SherpaOnnxGetOnlineStreamResult(online, stream);
    if (r) SherpaOnnxDestroyOnlineRecognizerResult(r);
    SherpaOnnxDestroyOnlineStream(stream);
  } else if (model_type == "sensevoice" || model_type == "sense_voice") {
    SherpaOnnxOfflineRecognizerConfig ac{};
    ac.feat_config.sample_rate = 16000;
    ac.feat_config.feature_dim = 80;
    ac.model_config.sense_voice.model = cfg.asr_model.c_str();
    ac.model_config.sense_voice.language = "auto";
    ac.model_config.sense_voice.use_itn = 1;
    ac.model_config.tokens = cfg.asr_tokens.c_str();
    ac.model_config.provider = provider.c_str();
    ac.model_config.num_threads = num_threads;
    ac.decoding_method = decoding_method.c_str();
    const SherpaOnnxOfflineRecognizer *offline =
        SherpaOnnxCreateOfflineRecognizer(&ac);
    if (!offline) {
      if (error) *error = "cannot create offline recognizer: " + cfg.asr_model;
      return false;
    }
    backend->offline = offline;

    const SherpaOnnxOfflineStream *stream = SherpaOnnxCreateOfflineStream(offline);
    if (!stream) {
      if (error) *error = "cannot create offline stream: " + cfg.asr_model;
      return false;
    }
    SherpaOnnxAcceptWaveformOffline(stream, 16000, warmup.data(), warmup.size());
    SherpaOnnxDecodeOfflineStream(offline, stream);
    const SherpaOnnxOfflineRecognizerResult *r = SherpaOnnxGetOfflineStreamResult(stream);
    if (r) SherpaOnnxDestroyOfflineRecognizerResult(r);
    SherpaOnnxDestroyOfflineStream(stream);
  } else {
    std::cerr << "Unsupported ASR model type: " << cfg.asr_model_type << std::endl;
    if (error) *error = "unsupported ASR model type: " + cfg.asr_model_type;
    return false;
  }

  backend->config = cfg;
  return true;
}

void VoicePipeline::adoptAsrBackend(AsrBackend &&backend) {
  destroyAsr();
  qwen3_asr_ = std::move(backend.qwen3);
  asr_offline_ = backend.offline;
  asr_online_ = backend.online;
  backend.offline = nullptr;
  backend.online = nullptr;
  asr_loaded_.store(true);
}

bool VoicePipeline::initializeAsr() {
  if (asr_offline_ || asr_online_ || qwen3_asr_) return true;
  AsrBackend backend;
  if (!buildAsr(config_, &backend, nullptr)) return false;
  adoptAsrBackend(std::move(backend));
  return true;
}

void VoicePipeline::destroyAsr() {
  destroyAsrStream();
  qwen3_asr_.reset();
  if (asr_offline_) SherpaOnnxDestroyOfflineRecognizer(asr_offline_);
  if (asr_online_) SherpaOnnxDestroyOnlineRecognizer(asr_online_);
  asr_offline_ = nullptr;
  asr_online_ = nullptr;
  asr_loaded_.store(false);
}

void VoicePipeline::destroyAsrStream() {
  if (asr_stream_) SherpaOnnxDestroyOnlineStream(asr_stream_);
  asr_stream_ = nullptr;
}

bool VoicePipeline::createAsrStream() {
  destroyAsrStream();
  if (!asr_online_) return true;
  asr_stream_ = SherpaOnnxCreateOnlineStream(asr_online_);
  return asr_stream_ != nullptr;
}

void VoicePipeline::feedStreamingAsr(const float *samples, int count) {
  if (!asr_online_ || !asr_stream_ || !samples || count <= 0) return;
  const auto begin = std::chrono::steady_clock::now();
  SherpaOnnxOnlineStreamAcceptWaveform(asr_stream_, 16000, samples, count);
  while (SherpaOnnxIsOnlineStreamReady(asr_online_, asr_stream_)) {
    SherpaOnnxDecodeOnlineStream(asr_online_, asr_stream_);
  }
  streaming_asr_decode_us_ += std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - begin).count();
}

bool VoicePipeline::initialize() {
  if (!rebuildKws() || !rebuildVad()) return false;
  if (asr_enabled_.load() && !initializeAsr()) return false;

  if (config_.kws_debug_enabled) beginKwsDebugSession();
  emitEvent("{\"event\":\"ready\",\"wake\":" +
            std::string(wake_enabled_.load() ? "true" : "false") +
            ",\"audio_ready\":" + std::string(audio_ready_.load() ? "true" : "false") +
            ",\"kws_loaded\":" + std::string(kws_loaded_.load() ? "true" : "false") +
            ",\"asr\":" + std::string(asr_enabled_.load() ? "true" : "false") +
            ",\"asr_loaded\":" + std::string(asr_loaded_.load() ? "true" : "false") +
            ",\"asr_model_type\":\"" + jsonEscape(config_.asr_model_type) + "\"}");
  emitEvent(parametersJson());
  return true;
}

bool VoicePipeline::start() {
  std::lock_guard<std::mutex> lk(queue_mutex_);
  if (running_) return true;
  switch_cancel_.store(false);
  running_ = true;
  try {
    worker_ = std::thread(&VoicePipeline::run, this);
  } catch (...) {
    running_ = false;
    return false;
  }
  return true;
}

void VoicePipeline::stop() {
  switch_cancel_.store(true);
  {
    std::lock_guard<std::mutex> lk(queue_mutex_);
    if (running_) commands_.push_back({CommandType::Stop});
  }
  queue_cv_.notify_all();
  if (worker_.joinable()) worker_.join();
  // ponytail: 构建线程可能正阻塞在 llama-server 启动（最长 qwen3_startup_timeout_ms），
  // 这里同步等它退出；若要服务秒级停机，再给 buildAsr 传取消信号。
  std::thread builder;
  {
    std::lock_guard<std::mutex> lk(switch_mutex_);
    if (switch_thread_.joinable()) builder = std::move(switch_thread_);
  }
  if (builder.joinable()) builder.join();
}

void VoicePipeline::pushAudio(const float *samples, int count, int input_rate) {
  if (!samples || count <= 0 || input_rate <= 0) return;
  std::vector<float> out;
  size_t dropped = 0;
  size_t queued_after = 0;
  {
    std::lock_guard<std::mutex> lk(queue_mutex_);
    if (!resampler_ || resampler_rate_ != input_rate) {
      resampler_ = std::make_unique<StreamingResampler>(input_rate, 16000);
      resampler_rate_ = input_rate;
    }
    out = resampler_->process(samples, count);
    if (!out.empty()) {
      queued_audio_samples_ += out.size();
      arrived_samples_.fetch_add(static_cast<int64_t>(out.size()));
      audio_queue_.push_back(std::move(out));
      while (queued_audio_samples_ > queue_max_samples_ && !audio_queue_.empty()) {
        dropped += audio_queue_.front().size();
        queued_audio_samples_ -= audio_queue_.front().size();
        audio_queue_.pop_front();
      }
      queued_after = queued_audio_samples_;
      queued_ms_snapshot_.store(static_cast<int64_t>(queued_after / 16));
    }
  }
  if (dropped > 0) audio_discontinuity_.store(true);
  if (dropped > 0) {
    const uint64_t total = dropped_audio_samples_.fetch_add(dropped) + dropped;
    const int64_t now_ms = steadyNowMs();
    if (overflow_reporter_.add(now_ms, static_cast<int64_t>(dropped))) {
      // Rates over the reporting window say who is behind: arrival above
      // realtime means the capture side is delivering more audio than the
      // device produces; processing below realtime means the worker is the
      // one that fell behind.
      const int64_t window_ms = std::max<int64_t>(1, now_ms - overflow_window_ms_);
      const int64_t arrived = arrived_samples_.load();
      const int64_t processed = processed_samples_.load();
      const int64_t arrival_ms_per_s =
          (arrived - overflow_arrived_) * 1000 / window_ms / 16;
      const int64_t process_ms_per_s =
          (processed - overflow_processed_) * 1000 / window_ms / 16;
      overflow_arrived_ = arrived;
      overflow_processed_ = processed;
      overflow_window_ms_ = now_ms;
      emitEvent("{\"event\":\"audio_queue_overflow\",\"dropped_samples\":" +
                std::to_string(overflow_reporter_.takeDropped()) +
                ",\"total_dropped_samples\":" + std::to_string(total) +
                ",\"queued_ms\":" + std::to_string(queued_after / 16) +
                ",\"queue_max_ms\":" +
                std::to_string(static_cast<int64_t>(queue_max_samples_ / 16)) +
                ",\"arrival_ms_per_s\":" + std::to_string(arrival_ms_per_s) +
                ",\"process_ms_per_s\":" + std::to_string(process_ms_per_s) +
                ",\"process_us_total\":" + std::to_string(process_time_.totalUs()) +
                ",\"process_us_max\":" + std::to_string(process_time_.maxUs()) +
                ",\"process_chunks\":" + std::to_string(process_time_.chunks()) +
                "}");
    }
  }
  queue_cv_.notify_one();
}

void VoicePipeline::waitForAudioQueueBelow(size_t max_samples) {
  std::unique_lock<std::mutex> lk(queue_mutex_);
  queue_cv_.wait(lk, [this, max_samples]() {
    return queued_audio_samples_ <= max_samples || !running_;
  });
}

void VoicePipeline::resetKwsForTest() {
  {
    std::lock_guard<std::mutex> lk(queue_mutex_);
    commands_.push_back({CommandType::KwsReset});
  }
  queue_cv_.notify_one();
}

void VoicePipeline::setSwitches(const SwitchRequest &request) {
  Command cmd{CommandType::Switches};
  cmd.wake_set = request.wake_set;
  cmd.wake = request.wake;
  cmd.asr_set = request.asr_set;
  cmd.asr = request.asr;
  cmd.request_id = request.request_id;
  {
    std::lock_guard<std::mutex> lk(queue_mutex_);
    commands_.push_back(cmd);
  }
  queue_cv_.notify_one();
}

void VoicePipeline::setThresholds(float kws_threshold, float kws_score,
                                  float vad_threshold, float min_silence,
                                  int wake_guard_ms) {
  Command cmd{CommandType::Thresholds};
  cmd.kws_threshold = kws_threshold;
  cmd.kws_score = kws_score;
  cmd.vad_threshold = vad_threshold;
  cmd.min_silence = min_silence;
  cmd.wake_guard_ms = wake_guard_ms;
  {
    std::lock_guard<std::mutex> lk(queue_mutex_);
    commands_.push_back(cmd);
  }
  queue_cv_.notify_one();
}

void VoicePipeline::setKwsDebug(bool enabled) {
  Command cmd{CommandType::KwsDebug};
  cmd.kws_debug_enabled = enabled;
  {
    std::lock_guard<std::mutex> lk(queue_mutex_);
    commands_.push_back(cmd);
  }
  queue_cv_.notify_one();
}

bool VoicePipeline::modelSwitching() const {
  std::lock_guard<std::mutex> lk(state_mutex_);
  return switch_state_.active;
}

std::string VoicePipeline::effectiveAsrModel() const {
  std::lock_guard<std::mutex> lk(state_mutex_);
  return config_.asr_model_type;
}

std::string VoicePipeline::switchSnapshotJson() const {
  std::lock_guard<std::mutex> lk(state_mutex_);
  return switchStateJson(switch_state_);
}

std::string VoicePipeline::switchStateJson(const SwitchState &state) {
  if (!state.active) return "null";
  const int64_t elapsed = steadyNowMs() - state.started_ms;
  return "{\"kind\":\"" + jsonEscape(state.kind) + "\",\"from\":\"" +
         jsonEscape(state.from) + "\",\"to\":\"" + jsonEscape(state.to) +
         "\",\"request_id\":\"" + jsonEscape(state.request_id) +
         "\",\"elapsed_ms\":" + std::to_string(elapsed < 0 ? 0 : elapsed) + "}";
}

bool VoicePipeline::beginSwitchState(const std::string &kind, const std::string &to,
                                     const std::string &request_id) {
  std::lock_guard<std::mutex> lk(state_mutex_);
  if (switch_state_.active) return false;
  switch_state_.active = true;
  switch_state_.kind = kind;
  switch_state_.from = config_.asr_model_type;
  switch_state_.to = to;
  switch_state_.request_id = request_id;
  switch_state_.started_ms = steadyNowMs();
  return true;
}

void VoicePipeline::endSwitchState() {
  std::lock_guard<std::mutex> lk(state_mutex_);
  switch_state_ = SwitchState{};
}

bool VoicePipeline::setAsrModel(const BibitConfig &config,
                                const std::string &request_id) {
  noteDesiredModel(config.asr_model_type, request_id);
  return startAsrSwitch(config, "asr_model", request_id);
}

void VoicePipeline::noteDesiredModel(const std::string &model,
                                     const std::string &request_id) {
  std::lock_guard<std::mutex> lk(desired_mutex_);
  desired_model_type_ = model;
  desired_request_id_ = request_id;
  if (request_id.empty()) return;
  // 同一个 request_id 重复提交只更新目标；不同 request_id 各自等待终态。
  for (auto &waiting : waiting_requests_) {
    if (waiting.request_id == request_id) {
      waiting.target = model;
      return;
    }
  }
  waiting_requests_.push_back({request_id, model});
}

// 一次切换走完（成功或失败）后清掉对应的期望目标：成功即已达成，失败则不再自动
// 重试同一目标，否则 desired.pending 会永久为真并反复触发切换。
void VoicePipeline::clearDesiredModel(const std::string &model) {
  if (model.empty()) return;
  std::lock_guard<std::mutex> lk(desired_mutex_);
  if (desired_model_type_ == model) desired_model_type_.clear();
}

// 没有待追目标时的结算：目标已生效的请求报成功，其余报「被后续意图取代」。
// 这样一个带 request_id 的请求必然拿到恰好一个终态，不会永久挂起。
void VoicePipeline::settleWaitingRequests(const std::string &effective_model) {
  std::vector<WaitingRequest> pending;
  std::string failed_target;
  {
    std::lock_guard<std::mutex> lk(desired_mutex_);
    pending.swap(waiting_requests_);
    failed_target = failed_model_target_;
    failed_model_target_.clear();
  }
  for (const auto &request : pending) {
    if (request.request_id.empty()) continue;
    if (request.target == effective_model) {
      emitEvent("{\"event\":\"model_switched\",\"model\":\"" +
                jsonEscape(effective_model) + "\",\"request_id\":\"" +
                jsonEscape(request.request_id) + "\",\"from\":\"" +
                jsonEscape(effective_model) + "\",\"to\":\"" +
                jsonEscape(effective_model) + "\",\"settled\":true}");
    } else if (!failed_target.empty() && request.target == failed_target) {
      // 目标是被一次失败的切换清掉的：按失败收场，不能让调用方以为「被取代」。
      emitEvent("{\"event\":\"error\",\"code\":\"ASR_MODEL_SWITCH_FAILED\","
                "\"request_id\":\"" + jsonEscape(request.request_id) +
                "\",\"to\":\"" + jsonEscape(request.target) +
                "\",\"active_model\":\"" + jsonEscape(effective_model) +
                "\",\"rolled_back\":false,"
                "\"message\":\"the target switch failed before this request ran\"}");
    } else {
      emitEvent("{\"event\":\"model_switch_superseded\",\"request_id\":\"" +
                jsonEscape(request.request_id) + "\",\"to\":\"" +
                jsonEscape(request.target) + "\",\"active_model\":\"" +
                jsonEscape(effective_model) + "\"}");
    }
  }
}

// 失败路径已经为这个 request_id 发过终态事件，把它从等待表里摘掉，避免重复结算。
void VoicePipeline::dropWaitingRequest(const std::string &request_id) {
  if (request_id.empty()) return;
  std::lock_guard<std::mutex> lk(desired_mutex_);
  waiting_requests_.erase(
      std::remove_if(waiting_requests_.begin(), waiting_requests_.end(),
                     [&](const WaitingRequest &waiting) {
                       return waiting.request_id == request_id;
                     }),
      waiting_requests_.end());
}

// 提交点之后的收敛：正在执行的切换不打断，但紧随其后只执行一次最新意图，
// 更早的中间目标直接跳过（最后一次意图胜出）。
void VoicePipeline::reconcileDesired() {
  const bool switching = modelSwitching();
  // 开关意图的结算在途也要做：构建期间关掉识别，那条打开意图要立刻拿到「被取代」
  // 的终态，而不是等构建跑完才给答复。
  bibit::SwitchStep switch_step;
  {
    std::lock_guard<std::mutex> lk(desired_mutex_);
    switch_step = switch_book_.reconcile(currentSwitchState(switching));
  }
  emitSwitchSettlements(switch_step.settlements);
  if (switching) return;
  std::string want_model;
  std::string want_request_id;
  {
    std::lock_guard<std::mutex> lk(desired_mutex_);
    want_model = desired_model_type_;
    want_request_id = desired_request_id_;
  }
  BibitConfig current;
  {
    std::lock_guard<std::mutex> lk(state_mutex_);
    current = config_;
  }
  if (!want_model.empty() && want_model != current.asr_model_type) {
    const std::string from = current.asr_model_type;
    if (startAsrSwitch(asrModelConfig(current, want_model), "asr_model",
                       want_request_id)) {
      emitEvent("{\"event\":\"model_switching\",\"request_id\":\"" +
                jsonEscape(want_request_id) + "\",\"from\":\"" +
                jsonEscape(from) + "\",\"to\":\"" + jsonEscape(want_model) +
                "\",\"model\":\"" + jsonEscape(want_model) +
                "\",\"queued\":true}");
    }
    return;
  }
  // 没有还在追的引擎目标：结算所有等待请求，保证每个 request_id 都有终态。
  settleWaitingRequests(current.asr_model_type);

  // 开关动作：为最新的打开意图起一次构建。关闭意图不需要构建，命令处理时已经生效。
  if (switch_step.action.kind != bibit::SwitchActionKind::EnableAsr) return;
  // 这次构建服务的是「最新的打开意图」：开关态先落到打开（是否能识别由构建结果决定）。
  // 上一次尝试失败时会把开关退回关闭，这里必须再落回去，否则重试构建会以为不需要
  // 后端（build_needed=false），结果既没后端也不达成，意图被静默吃掉。
  asr_enabled_.store(true);
  if (startAsrSwitch(current, "asr_enable", switch_step.action.request_id)) {
    std::lock_guard<std::mutex> lk(desired_mutex_);
    switch_book_.noteEnableStarted(switch_step.action.request_id);
    return;
  }
  // 构建起不来（服务正在停止等）：未决的打开意图都按失败收场，界面不会停在
  // 「正在应用」。生效开关退回关闭，与结果一致。
  asr_enabled_.store(false);
  std::vector<bibit::SwitchSettlement> failed;
  {
    std::lock_guard<std::mutex> lk(desired_mutex_);
    failed = switch_book_.failAllPending(currentSwitchState(false));
  }
  emitSwitchSettlements(failed);
  for (const auto &settled : failed) {
    if (settled.request_id.empty()) continue;
    emitEvent("{\"event\":\"error\",\"code\":\"ASR_INIT_FAILED\",\"request_id\":\"" +
              jsonEscape(settled.request_id) + "\"}");
  }
}

bibit::SwitchState VoicePipeline::currentSwitchState(bool switching) const {
  bibit::SwitchState state;
  state.wake = wake_enabled_.load();
  state.asr = asr_enabled_.load();
  state.asr_ready = asrBackendReady();
  state.switching = switching;
  return state;
}

void VoicePipeline::emitSwitchSettlements(
    const std::vector<bibit::SwitchSettlement> &settlements) const {
  for (const auto &settled : settlements) {
    const char *outcome = settled.outcome == bibit::SwitchOutcome::Applied ? "applied"
                          : settled.outcome == bibit::SwitchOutcome::Superseded
                              ? "superseded"
                              : "failed";
    emitEvent("{\"event\":\"switches_settled\",\"request_id\":\"" +
              jsonEscape(settled.request_id) + "\",\"outcome\":\"" + outcome +
              "\",\"wake\":" + (settled.wake ? "true" : "false") +
              ",\"asr\":" + (settled.asr ? "true" : "false") + "}");
  }
}

std::optional<bibit::SwitchSettlement> VoicePipeline::settleEnableBuild(
    bibit::SwitchOutcome outcome) {
  std::optional<bibit::SwitchSettlement> settled;
  {
    std::lock_guard<std::mutex> lk(desired_mutex_);
    settled = switch_book_.noteEnableBuildEnded(outcome, currentSwitchState(false));
  }
  if (settled) emitSwitchSettlements({*settled});
  return settled;
}

// 切换分三步：流水线线程先放下现役后端（AsrRelease，短），构建线程再构建
// 目标后端（长），最后由流水线线程接入（AsrCommit，短）。
// 关键约束：SpacemiT EP / TCM 是独占资源，新旧后端不能并存——实测在同一
// 进程里一边跑 Qwen3 的 SMT 媒体后端、一边创建 SenseVoice 的 EP 会话会
// SEGV。这里换掉的是"整段构建都压在流水线线程上"的历史行为：KWS/VAD、
// PipeWire 采集与 IPC 在切换期间照常运行，只有识别本身短暂不可用。
bool VoicePipeline::startAsrSwitch(const BibitConfig &target,
                                   const std::string &kind,
                                   const std::string &request_id) {
  if (switch_cancel_.load()) return false;
  if (!beginSwitchState(kind, target.asr_model_type, request_id)) return false;

  std::thread previous;
  {
    std::lock_guard<std::mutex> lk(switch_mutex_);
    if (switch_thread_.joinable()) previous = std::move(switch_thread_);
  }
  if (previous.joinable()) previous.join();

  // 识别开关关闭时不存在"生效模型"：只更新配置，不加载后端（历史行为一致）。
  const bool build_needed = asr_enabled_.load();
  // 打开识别的构建失败时没有可回滚的「旧引擎」：目标就是当前生效引擎，重建同一个
  // 后端只会浪费一次冷启动，还会让「回了失败、开关却开着」同时成立。
  const bool allow_rollback = kind != "asr_enable";
  BibitConfig rollback_target;
  int release_timeout_ms = 0;
  {
    std::lock_guard<std::mutex> lk(state_mutex_);
    rollback_target = config_;
    release_timeout_ms = config_.asr_release_timeout_ms;
  }
  std::thread builder([this, target, rollback_target, build_needed, allow_rollback,
                       release_timeout_ms]() {
    auto backend = std::make_shared<AsrBackend>();
    std::string error;
    bool rolled_back = false;
    bool kept_active = false;
    bool ok = true;
    if (build_needed) {
      // 先让流水线放下现役后端，腾出 EP/TCM；等待有上限，避免服务已停时挂死。
      auto ticket = std::make_shared<ReleaseTicket>();
      {
        std::lock_guard<std::mutex> lk(queue_mutex_);
        if (!running_) return;
        Command release{CommandType::AsrRelease};
        release.release = ticket;
        commands_.push_back(release);
      }
      queue_cv_.notify_one();
      bool released = false;
      {
        std::unique_lock<std::mutex> lk(ticket->mutex);
        released = ticket->cv.wait_for(
            lk, std::chrono::milliseconds(release_timeout_ms),
            [&ticket]() { return ticket->done; });
        // 放弃这次切换：流水线线程稍后处理到这条命令时会跳过释放。
        if (!released) ticket->abandoned = true;
      }
      if (switch_cancel_.load()) return;

      if (!released) {
        // 现役后端还没放下就构建新后端，两者会争用 SpacemiT EP/TCM（实测 SEGV）。
        // 这里按切换失败处理：现役引擎保持生效，原因如实回给客户端。
        ok = false;
        kept_active = true;
        error = "ASR backend release timed out after " +
                std::to_string(release_timeout_ms) +
                "ms; the current engine stays active";
      } else {
        ok = buildAsr(target, backend.get(), &error);
        if (!ok && allow_rollback) {
          // 失败回滚也在这里做：重建旧后端仍然不占流水线线程。
          auto restore = std::make_shared<AsrBackend>();
          std::string restore_error;
          if (buildAsr(rollback_target, restore.get(), &restore_error)) {
            backend = restore;
            ok = true;
            rolled_back = true;
          }
        }
      }
    }
    if (!build_needed) backend->config = target;
    if (switch_cancel_.load()) return;
    {
      std::lock_guard<std::mutex> lk(pending_mutex_);
      pending_backend_ = ok ? backend : nullptr;
      // 回滚成功时也要保留原始失败原因：用户要知道为什么没切成。
      pending_error_ = (ok && !rolled_back) ? std::string() : error;
      pending_rolled_back_ = rolled_back;
      pending_active_kept_ = kept_active;
    }
    Command cmd{CommandType::AsrCommit};
    {
      std::lock_guard<std::mutex> lk(queue_mutex_);
      if (!running_) {
        std::lock_guard<std::mutex> pending_lk(pending_mutex_);
        pending_backend_.reset();
        pending_error_.clear();
        pending_rolled_back_ = false;
        return;
      }
      commands_.push_back(cmd);
    }
    queue_cv_.notify_one();
  });
  {
    std::lock_guard<std::mutex> lk(switch_mutex_);
    switch_thread_ = std::move(builder);
  }
  return true;
}

void VoicePipeline::commitAsrSwitch() {
  std::shared_ptr<AsrBackend> backend;
  std::string error;
  bool rolled_back = false;
  bool kept_active = false;
  {
    std::lock_guard<std::mutex> lk(pending_mutex_);
    backend = std::move(pending_backend_);
    error = pending_error_;
    pending_error_.clear();
    rolled_back = pending_rolled_back_;
    pending_rolled_back_ = false;
    kept_active = pending_active_kept_;
    pending_active_kept_ = false;
  }
  SwitchState state;
  {
    std::lock_guard<std::mutex> lk(state_mutex_);
    state = switch_state_;
  }
  // 只有「打开识别」的构建才有开关意图要结算；换引擎的结果归引擎链路。
  const bool settle_switch = state.kind == "asr_enable";
  // 无论本次切换以什么路径结束，都要收尾对应的期望目标。
  struct DesiredGuard {
    VoicePipeline *self;
    std::string model;
    ~DesiredGuard() { self->clearDesiredModel(model); }
  } desired_guard{this, state.to};

  if (rolled_back && backend) {
    // 目标后端构建失败，构建线程已经重建旧后端：接入它并如实报错。
    const BibitConfig restored = backend->config;
    // 打开识别不走回滚：它的「旧后端」就是没有后端，重建同一个引擎没有意义，
    // 接入它还会造出「回了失败、开关却是开」的错位。
    if (state.kind != "asr_enable" && asr_enabled_.load()) {
      adoptAsrBackend(std::move(*backend));
      {
        std::lock_guard<std::mutex> lk(state_mutex_);
        config_ = restored;
      }
    }
    if (state.kind == "asr_enable") asr_enabled_.store(false);
    endSwitchState();
    if (settle_switch) settleEnableBuild(bibit::SwitchOutcome::Failed);
    emitEvent("{\"event\":\"error\",\"code\":\"" +
              std::string(state.kind == "asr_enable" ? "ASR_INIT_FAILED"
                                                     : "ASR_MODEL_SWITCH_FAILED") +
              "\",\"request_id\":\"" + jsonEscape(state.request_id) +
              "\",\"from\":\"" + jsonEscape(state.from) + "\",\"to\":\"" +
              jsonEscape(state.to) + "\",\"active_model\":\"" +
              jsonEscape(restored.asr_model_type) +
              "\",\"rolled_back\":true,\"message\":\"" + jsonEscape(error) + "\"}");
    dropWaitingRequest(state.request_id);
    {
      std::lock_guard<std::mutex> lk(desired_mutex_);
      failed_model_target_ = state.to;
    }
    emitEvent(statusJson());
    emitEvent(parametersJson());
    return;
  }

  if (!backend) {
    if (kept_active) {
      // 切换在「放下现役后端」这一步就失败了：后端从未被放下，识别保持可用，
      // 只是这次切换没做成。打开识别失败时仍要退回关闭状态。
      if (state.kind == "asr_enable") asr_enabled_.store(false);
      endSwitchState();
      if (settle_switch) settleEnableBuild(bibit::SwitchOutcome::Failed);
      emitEvent("{\"event\":\"error\",\"code\":\"" +
                std::string(state.kind == "asr_enable" ? "ASR_INIT_FAILED"
                                                       : "ASR_MODEL_SWITCH_FAILED") +
                "\",\"request_id\":\"" + jsonEscape(state.request_id) +
                "\",\"from\":\"" + jsonEscape(state.from) + "\",\"to\":\"" +
                jsonEscape(state.to) + "\",\"active_model\":\"" +
                jsonEscape(state.from) + "\",\"rolled_back\":false,\"message\":\"" +
                jsonEscape(error) + "\"}");
      dropWaitingRequest(state.request_id);
      {
        std::lock_guard<std::mutex> lk(desired_mutex_);
        failed_model_target_ = state.to;
      }
      emitEvent(statusJson());
      emitEvent(parametersJson());
      return;
    }
    // 目标构建失败且旧后端也没能重建：识别确实不可用了，如实降级。
    asr_enabled_.store(false);
    endSwitchState();
    if (settle_switch) settleEnableBuild(bibit::SwitchOutcome::Failed);
    emitEvent("{\"event\":\"error\",\"code\":\"" +
              std::string(state.kind == "asr_enable" ? "ASR_INIT_FAILED"
                                                     : "ASR_MODEL_SWITCH_FAILED") +
              "\",\"request_id\":\"" + jsonEscape(state.request_id) +
              "\",\"from\":\"" + jsonEscape(state.from) + "\",\"to\":\"" +
              jsonEscape(state.to) + "\",\"active_model\":\"" +
              jsonEscape(state.from) + "\",\"rolled_back\":false,\"message\":\"" +
              jsonEscape(error) + "\"}");
    dropWaitingRequest(state.request_id);
    {
      std::lock_guard<std::mutex> lk(desired_mutex_);
      failed_model_target_ = state.to;
    }
    emitEvent(statusJson());
    emitEvent(parametersJson());
    return;
  }

  const BibitConfig target = backend->config;
  const bool has_backend =
      backend->qwen3 || backend->offline || backend->online;
  // 提交点：此刻才需要打断可能的会话，构建期间的会话仍由旧后端完成。
  // 开关已关闭时只落配置，不接管后端（构建结果随 shared_ptr 释放）。这条判据是
  // 「最后意图胜出」在构建上的落点：构建期间关掉识别，结果一律丢弃，绝不把开关
  // 翻回打开（历史实现在这里无条件置 true，用户关掉后照样被重新打开）。
  const bool adopt = has_backend && asr_enabled_.load();
  const int64_t elapsed = steadyNowMs() - state.started_ms;
  if (adopt) {
    if (mode_ != Mode::Idle) {
      emitEvent("{\"event\":\"session_cancelled\",\"session_id\":" +
                std::to_string(session_id_) + "}");
      resetToIdle("model_switched");
    }
    adoptAsrBackend(std::move(*backend));
    discardCapturedAudio("model_switch");
    resetKws();
  }
  {
    std::lock_guard<std::mutex> lk(state_mutex_);
    config_ = target;
  }
  endSwitchState();

  if (state.kind == "asr_enable") {
    // 打开识别的意图到此才有终态：采纳=达成；被丢弃=期间更新的关闭意图取胜。
    settleEnableBuild(adopt ? bibit::SwitchOutcome::Applied
                            : bibit::SwitchOutcome::Superseded);
    emitEvent(statusJson());
    emitEvent(parametersJson());
    return;
  }

  if (state.kind == "asr_model") {
    emitEvent("{\"event\":\"model_switched\",\"model\":\"" +
              jsonEscape(target.asr_model_type) + "\",\"request_id\":\"" +
              jsonEscape(state.request_id) + "\",\"from\":\"" +
              jsonEscape(state.from) + "\",\"to\":\"" +
              jsonEscape(target.asr_model_type) + "\",\"duration_ms\":" +
              std::to_string(elapsed < 0 ? 0 : elapsed) + "}");
    // 这个 request_id 已经拿到终态；同目标的其它等待请求交给结算补齐。
    dropWaitingRequest(state.request_id);
  }
  // 不在这里清 failed_model_target_：结算函数自己会消费它。否则「切 Y 失败 → 新意图切 Z
  // 成功」这种交错会把排队等 Y 的请求错标成 superseded，与协议文档不一致。
  emitEvent(statusJson());
  emitEvent(parametersJson());
}

void VoicePipeline::startManualAsr() {
  { std::lock_guard<std::mutex> lk(queue_mutex_); commands_.push_back({CommandType::Manual}); }
  queue_cv_.notify_one();
}

void VoicePipeline::cancel() {
  { std::lock_guard<std::mutex> lk(queue_mutex_); commands_.push_back({CommandType::Cancel}); }
  queue_cv_.notify_one();
}

void VoicePipeline::finishInput() {
  { std::lock_guard<std::mutex> lk(queue_mutex_); commands_.push_back({CommandType::Finish}); }
  queue_cv_.notify_one();
}

bool VoicePipeline::recognizeDirect(const float *samples, int count, std::string *text,
                                    int64_t *decode_us, std::string *error) {
  if (text) text->clear();
  if (decode_us) *decode_us = 0;
  if (!samples || count <= 0) {
    if (error) *error = "empty direct ASR input";
    return false;
  }
  if (!asr_offline_ && !qwen3_asr_) {
    if (error) *error = "direct ASR supports Qwen3 and offline ASR models";
    return false;
  }

  const auto begin = std::chrono::steady_clock::now();
  std::string recognized;
  if (qwen3_asr_) {
    if (!qwen3_asr_->recognize(samples, count, &recognized, error)) return false;
  } else {
    const SherpaOnnxOfflineStream *stream = SherpaOnnxCreateOfflineStream(asr_offline_);
    if (!stream) {
      if (error) *error = "failed to create offline ASR stream";
      return false;
    }
    SherpaOnnxAcceptWaveformOffline(stream, 16000, samples, count);
    SherpaOnnxDecodeOfflineStream(asr_offline_, stream);
    const SherpaOnnxOfflineRecognizerResult *result =
        SherpaOnnxGetOfflineStreamResult(stream);
    recognized = result && result->text ? result->text : "";
    if (result) SherpaOnnxDestroyOfflineRecognizerResult(result);
    SherpaOnnxDestroyOfflineStream(stream);
  }
  const int64_t elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - begin).count();
  if (text) *text = recognized;
  if (decode_us) *decode_us = elapsed;
  return true;
}

// 意图态快照：未达成的引擎目标与待处理的「打开识别」意图。客户端据此可以
// 只凭快照重建界面，而不依赖本地记忆或事件顺序。
std::string VoicePipeline::desiredJson(const std::string &effective_model) const {
  std::lock_guard<std::mutex> lk(desired_mutex_);
  const bool model_pending = !desired_model_type_.empty() &&
                             desired_model_type_ != effective_model;
  // 打开识别的意图只有后端就绪才算达成：排队中、构建中、构建失败后仍未重试成功，
  // 这里都要如实为真，客户端不必自己记忆「我刚点过开」。
  const bool pending = model_pending || switch_book_.enablePending();
  return "{\"asr_model_type\":\"" + jsonEscape(desired_model_type_) +
         "\",\"pending\":" + (pending ? "true" : "false") + "}";
}

std::string VoicePipeline::statusJson(bool reply) const {
  std::lock_guard<std::mutex> lk(state_mutex_);
  const Mode current_mode = mode_.load();
  const char *mode = current_mode == Mode::Idle ? "idle" : current_mode == Mode::Listening ? "listening" : "recognizing";
  return "{\"event\":\"status\",\"wake\":" + std::string(wake_enabled_.load() ? "true" : "false") +
         ",\"audio_ready\":" + std::string(audio_ready_.load() ? "true" : "false") +
         ",\"queued_ms\":" + std::to_string(queued_ms_snapshot_.load()) +
         ",\"kws_loaded\":" + std::string(kws_loaded_.load() ? "true" : "false") +
         ",\"asr\":" + std::string(asr_enabled_.load() ? "true" : "false") +
         ",\"asr_loaded\":" + std::string(asr_loaded_.load() ? "true" : "false") +
         ",\"switching\":" + switchStateJson(switch_state_) +
         ",\"desired\":" + desiredJson(config_.asr_model_type) +
         ",\"partial_switches\":true" +
         ",\"queued_model_switch\":true" +
         ",\"switch_settlement\":true" +
         ",\"asr_model_type\":\"" + jsonEscape(config_.asr_model_type) + "\"" +
         ",\"asr_provider\":\"" + jsonEscape(config_.asr_provider) + "\"" +
         ",\"mode\":\"" + mode + "\",\"session_id\":" + std::to_string(session_id_.load()) +
         ",\"kws_threshold\":" + std::to_string(config_.kws_threshold) +
         ",\"kws_score\":" + std::to_string(config_.kws_score) +
         ",\"vad_threshold\":" + std::to_string(config_.vad_threshold) +
         ",\"min_silence\":" + std::to_string(config_.min_silence) +
         ",\"wake_guard_ms\":" + std::to_string(config_.wake_guard_ms) +
         ",\"kws_max_active_paths\":" + std::to_string(config_.kws_max_active_paths) +
         ",\"kws_chunk_size\":" + std::to_string(config_.kws_chunk_size) +
         ",\"kws_debug_enabled\":" + std::string(config_.kws_debug_enabled ? "true" : "false") +
         ",\"kws_debug_dir\":\"" + jsonEscape(config_.kws_debug_dir) + "\"" +
         ",\"kws_debug_session_dir\":\"" + jsonEscape(kws_debug_session_dir_) + "\"" +
         ",\"wake_ack_enabled\":" + std::string(config_.wake_ack_enabled ? "true" : "false") +
         (reply ? ",\"reply\":true" : "") + "}";
}

std::string VoicePipeline::parametersJson() const {
  std::lock_guard<std::mutex> lk(state_mutex_);
  return "{\"event\":\"parameters\",\"asr_model_type\":\"" + jsonEscape(config_.asr_model_type) + "\"" +
      ",\"asr_model\":\"" + jsonEscape(config_.asr_model) + "\"" +
      ",\"asr_tokens\":\"" + jsonEscape(config_.asr_tokens) + "\"" +
      ",\"asr_provider\":\"" + jsonEscape(config_.asr_provider) + "\"" +
      ",\"asr_num_threads\":" + std::to_string(config_.asr_num_threads) +
      ",\"asr_decoding_method\":\"" + jsonEscape(config_.asr_decoding_method) + "\"" +
      ",\"asr_max_active_paths\":" + std::to_string(config_.asr_max_active_paths) +
      ",\"asr_pre_roll_ms\":" + std::to_string(config_.asr_pre_roll_ms) +
      ",\"asr_post_roll_ms\":" + std::to_string(config_.asr_post_roll_ms) +
      ",\"qwen3_model_dir\":\"" + jsonEscape(config_.qwen3_model_dir) + "\"" +
      ",\"qwen3_server_bin\":\"" + jsonEscape(config_.qwen3_server_bin) + "\"" +
      ",\"qwen3_server_host\":\"" + jsonEscape(config_.qwen3_server_host) + "\"" +
      ",\"qwen3_server_port\":" + std::to_string(config_.qwen3_server_port) +
      ",\"qwen3_thread_batch\":" + std::to_string(config_.qwen3_thread_batch) +
      ",\"qwen3_ep_affinity\":\"" + jsonEscape(config_.qwen3_ep_affinity) + "\"" +
      ",\"qwen3_ctx_size\":" + std::to_string(config_.qwen3_ctx_size) +
      ",\"qwen3_language\":\"" + jsonEscape(config_.qwen3_language) + "\"" +
      ",\"qwen3_max_tokens\":" + std::to_string(config_.qwen3_max_tokens) +
      ",\"qwen3_startup_timeout_ms\":" +
          std::to_string(config_.qwen3_startup_timeout_ms) +
      ",\"qwen3_request_timeout_ms\":" +
          std::to_string(config_.qwen3_request_timeout_ms) +
      ",\"kws_threshold\":" + std::to_string(config_.kws_threshold) +
      ",\"kws_score\":" + std::to_string(config_.kws_score) +
      ",\"kws_trailing_blanks\":" + std::to_string(config_.kws_trailing_blanks) +
      ",\"kws_max_token_gap_ms\":" + std::to_string(config_.kws_max_token_gap_ms) +
      ",\"kws_fallback_enabled\":" +
          std::string(config_.kws_fallback_keywords_file.empty() ? "false" : "true") +
      ",\"kws_fallback_threshold\":" + std::to_string(config_.kws_fallback_threshold) +
      ",\"kws_fallback_score\":" + std::to_string(config_.kws_fallback_score) +
      ",\"kws_fallback_max_duration_ms\":" +
          std::to_string(config_.kws_fallback_max_duration_ms) +
      ",\"kws_fallback_verify_enabled\":" +
          std::string(config_.kws_fallback_verify_enabled ? "true" : "false") +
      ",\"kws_fallback_verify_pre_roll_ms\":" +
          std::to_string(config_.kws_fallback_verify_pre_roll_ms) +
      ",\"kws_fallback_verify_max_audio_ms\":" +
          std::to_string(config_.kws_fallback_verify_max_audio_ms) +
      ",\"kws_long_candidate_ms\":" + std::to_string(config_.kws_long_candidate_ms) +
      ",\"kws_long_candidate_max_rms_ratio\":" +
          std::to_string(config_.kws_long_candidate_max_rms_ratio) +
      ",\"kws_parallel_continuous_max_rms_ratio\":" +
          std::to_string(config_.kws_parallel_continuous_max_rms_ratio) +
      ",\"kws_onset_window_ms\":" + std::to_string(config_.kws_onset_window_ms) +
      ",\"kws_reset_on_speech_after_silence\":" +
          std::string(config_.kws_reset_on_speech_after_silence ? "true" : "false") +
      ",\"kws_silence_reset_ms\":" + std::to_string(config_.kws_silence_reset_ms) +
      ",\"kws_speech_rms_threshold\":" +
          std::to_string(config_.kws_speech_rms_threshold) +
      ",\"kws_reset_pre_roll_ms\":" + std::to_string(config_.kws_reset_pre_roll_ms) +
      ",\"kws_max_active_paths\":" + std::to_string(config_.kws_max_active_paths) +
      ",\"kws_chunk_size\":" + std::to_string(config_.kws_chunk_size) +
      ",\"kws_debug_enabled\":" + std::string(config_.kws_debug_enabled ? "true" : "false") +
      ",\"kws_debug_dir\":\"" + jsonEscape(config_.kws_debug_dir) + "\"" +
      ",\"kws_debug_session_dir\":\"" + jsonEscape(kws_debug_session_dir_) + "\"" +
      ",\"kws_debug_pre_roll_ms\":" + std::to_string(config_.kws_debug_pre_roll_ms) +
      ",\"kws_debug_max_files\":" + std::to_string(config_.kws_debug_max_files) +
      ",\"vad_threshold\":" + std::to_string(config_.vad_threshold) +
      ",\"min_silence\":" + std::to_string(config_.min_silence) +
      ",\"min_speech\":" + std::to_string(config_.min_speech) +
      ",\"max_speech\":" + std::to_string(config_.max_speech) +
      ",\"vad_log_interval_ms\":" + std::to_string(config_.vad_log_interval_ms) +
      ",\"wake_guard_ms\":" + std::to_string(config_.wake_guard_ms) +
      ",\"wake_ack_enabled\":" + std::string(config_.wake_ack_enabled ? "true" : "false") +
      ",\"wake_ack_file\":\"" + jsonEscape(config_.wake_ack_file) + "\"" +
      ",\"followup_enabled\":" + std::string(config_.followup_enabled ? "true" : "false") +
      ",\"followup_timeout\":" + std::to_string(config_.followup_timeout) + "}";
}

void VoicePipeline::run() {
  while (true) {
    Command cmd{CommandType::Cancel};
    bool have_cmd = false;
    std::vector<float> audio;
    {
      std::unique_lock<std::mutex> lk(queue_mutex_);
      queue_cv_.wait(lk, [this]() { return !commands_.empty() || !audio_queue_.empty(); });
      const bool take_command = !commands_.empty() &&
          (commands_.front().type != CommandType::Finish || audio_queue_.empty());
      if (take_command) { cmd = commands_.front(); commands_.pop_front(); have_cmd = true; }
      else {
        queued_audio_samples_ -= audio_queue_.front().size();
        audio = std::move(audio_queue_.front());
        audio_queue_.pop_front();
        processed_samples_.fetch_add(static_cast<int64_t>(audio.size()));
        queued_ms_snapshot_.store(static_cast<int64_t>(queued_audio_samples_ / 16));
        queue_cv_.notify_all();
      }
    }
    if (have_cmd) {
      if (cmd.type == CommandType::Stop) break;
      processCommand(cmd);
    } else {
      const auto begin = std::chrono::steady_clock::now();
      processAudio(audio);
      process_time_.add(std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now() - begin)
                            .count());
    }
  }
  std::lock_guard<std::mutex> lk(queue_mutex_);
  running_ = false;
}

void VoicePipeline::processCommand(const Command &cmd) {
  if (cmd.type == CommandType::Switches) {
    // 命令只负责把「立即能生效的部分」落下并记录意图，随后由 reconcileDesired()
    // 统一收敛（结算终态、决定是否需要再起一次构建）。
    if (cmd.wake_set) wake_enabled_.store(cmd.wake);
    if (cmd.asr_set && cmd.asr) {
      // 打开识别也是一次能力变更：Qwen3 后端要拉起 llama-server，与换模型走同一条
      // build-then-swap 路径，别把流水线按分钟级卡住。这里的开关先置真，后端是否
      // 就绪由 asr_ready / switching 表达——「达成」要等提交点。
      asr_enabled_.store(true);
    } else if (cmd.asr_set) {
      // 关闭识别不需要构建：立即生效，并放下后端。在途构建的结果会在提交点被丢弃
      // （提交点的采纳判据只看生效开关），排队中的打开意图随之判为被取代。
      if (mode_.load() != Mode::Idle) resetToIdle("asr_disabled");
      asr_enabled_.store(false);
      destroyAsr();
    }
    {
      std::lock_guard<std::mutex> lk(desired_mutex_);
      switch_book_.record(cmd.wake_set, cmd.wake, cmd.asr_set, cmd.asr, cmd.request_id);
    }
    if (!wake_enabled_.load()) resetKws();
    // 先收敛再广播：客户端拿到的第一条快照就已经包含「正在构建」与待达成意图。
    reconcileDesired();
    emitEvent(statusJson());
  } else if (cmd.type == CommandType::Thresholds) {
    if (mode_ != Mode::Idle) {
      emitEvent("{\"event\":\"session_cancelled\",\"session_id\":" +
                std::to_string(session_id_) + "}");
      resetToIdle("threshold_changed");
    }
    const BibitConfig old = config_;
    {
      std::lock_guard<std::mutex> lk(state_mutex_);
      config_.kws_threshold = std::clamp(cmd.kws_threshold, 0.05f, 0.95f);
      config_.kws_score = std::clamp(cmd.kws_score, 0.1f, 5.0f);
      config_.vad_threshold = std::clamp(cmd.vad_threshold, 0.05f, 0.95f);
      config_.min_silence = std::clamp(cmd.min_silence, 0.15f, 3.0f);
      config_.wake_guard_ms = std::clamp(cmd.wake_guard_ms, 0, 600);
    }
    if (!rebuildKws() || !rebuildVad()) {
      {
        std::lock_guard<std::mutex> lk(state_mutex_);
        config_ = old;
      }
      rebuildKws();
      rebuildVad();
      emitEvent("{\"event\":\"error\",\"code\":\"THRESHOLD_APPLY_FAILED\"}");
    } else {
      emitEvent(statusJson());
      emitEvent(parametersJson());
    }
  } else if (cmd.type == CommandType::KwsDebug) {
    bool was_enabled = false;
    {
      std::lock_guard<std::mutex> lk(state_mutex_);
      was_enabled = config_.kws_debug_enabled;
      config_.kws_debug_enabled = cmd.kws_debug_enabled;
    }
    if (cmd.kws_debug_enabled && !was_enabled) beginKwsDebugSession();
    if (!cmd.kws_debug_enabled) {
      std::lock_guard<std::mutex> lk(state_mutex_);
      kws_debug_session_dir_.clear();
    }
    emitEvent("{\"event\":\"kws_debug\",\"enabled\":" +
              std::string(cmd.kws_debug_enabled ? "true" : "false") +
              ",\"dir\":\"" + jsonEscape(kws_debug_session_dir_) + "\"}");
    emitEvent(statusJson());
    emitEvent(parametersJson());
  } else if (cmd.type == CommandType::AsrCommit) {
    commitAsrSwitch();
    reconcileDesired();
  } else if (cmd.type == CommandType::AsrRelease) {
    // 构建线程的前置步骤：放下现役后端，腾出 SpacemiT EP / TCM。
    // 这一步很短（销毁句柄、结束 llama-server），KWS/VAD 与采集不受影响。
    // 「是否放弃」与这里在同一把锁下判定：要么真的放下后端，要么构建线程已经
    // 超时放弃并保留现役后端，两边不会同时成立。
    if (cmd.release) {
      std::lock_guard<std::mutex> lk(cmd.release->mutex);
      if (!cmd.release->abandoned) {
        if (mode_ != Mode::Idle) {
          // 识别马上不可用，进行中的会话在这里就干净地取消，而不是等到解码时再报错。
          emitEvent("{\"event\":\"session_cancelled\",\"session_id\":" +
                    std::to_string(session_id_) + "}");
          resetToIdle("model_switching");
        }
        destroyAsr();
        emitEvent(statusJson());
      }
      cmd.release->done = true;
      cmd.release->cv.notify_all();
    }
  } else if (cmd.type == CommandType::Manual) {
    if (!asr_enabled_) { emitEvent("{\"event\":\"error\",\"code\":\"ASR_DISABLED\"}"); return; }
    if (mode_ != Mode::Idle) {
      emitEvent("{\"event\":\"listening\",\"session_id\":" +
                std::to_string(session_id_) + "}");
      return;
    }
    beginListening("", false, 0, "manual");
  } else if (cmd.type == CommandType::Cancel) {
    if (mode_ != Mode::Idle) {
      emitEvent("{\"event\":\"session_cancelled\",\"session_id\":" + std::to_string(session_id_) + "}");
      resetToIdle("cancelled");
    }
  } else if (cmd.type == CommandType::Finish) {
    if (mode_ == Mode::Listening) {
      if (!vad_pending_.empty()) {
        std::vector<float> pending = std::move(vad_pending_);
        vad_pending_.clear();
        pending.resize(((pending.size() + 511) / 512) * 512, 0.0f);
        feedVad(pending.data(), static_cast<int>(pending.size()));
        if (mode_ != Mode::Listening) return;
      }
      SherpaOnnxVoiceActivityDetectorFlush(vad_);
      if (!SherpaOnnxVoiceActivityDetectorEmpty(vad_)) {
        const SherpaOnnxSpeechSegment *s = SherpaOnnxVoiceActivityDetectorFront(vad_);
        std::vector<float> segment;
        if (s) segment.assign(s->samples, s->samples + s->n);
        if (s) SherpaOnnxDestroySpeechSegment(s);
        SherpaOnnxVoiceActivityDetectorPop(vad_);
        if (!segment.empty()) decodeSegment(segment.data(), static_cast<int>(segment.size()));
      } else {
        emitEvent("{\"event\":\"error\",\"code\":\"NO_SPEECH\",\"session_id\":" +
                  std::to_string(session_id_.load()) + "}");
        resetToIdle("no_speech");
      }
    }
    emitEvent("{\"event\":\"input_finished\"}");
  } else if (cmd.type == CommandType::KwsReset) {
    resetKws();
    emitEvent("{\"event\":\"kws_reset\"}");
  }
}

void VoicePipeline::processAudio(const std::vector<float> &audio) {
  if (audio_discontinuity_.exchange(false)) {
    // Audio was dropped, so the stream is no longer continuous. Feeding the
    // pre-gap and post-gap halves to KWS/VAD as one utterance either loses the
    // wake word or invents one, and an in-flight utterance would be decoded
    // from audio the user never finished saying.
    emitEvent("{\"event\":\"audio_discontinuity\",\"reason\":\"queue_overflow\","
              "\"queued_ms\":" + std::to_string(queued_audio_samples_ / 16) +
              ",\"session_id\":" + std::to_string(session_id_.load()) + "}");
    resetKws();
    ring_.clear();
    ring_start_abs_ = absolute_samples_;
    vad_pending_.clear();
    if (mode_.load() != Mode::Idle) resetToIdle("audio_discontinuity");
    return;
  }
  if (audio.empty()) return;
  const int64_t chunk_start = absolute_samples_;
  absolute_samples_ += static_cast<int64_t>(audio.size());
  ring_.insert(ring_.end(), audio.begin(), audio.end());
  const size_t ring_limit = 16000 * 8;
  while (ring_.size() > ring_limit) { ring_.pop_front(); ++ring_start_abs_; }

  if (mode_ == Mode::Idle) {
    double sum = 0.0;
    for (float v : audio) sum += static_cast<double>(v) * v;
    const double rms = std::sqrt(sum / audio.size());
    const int64_t log_interval_samples =
        static_cast<int64_t>(std::clamp(config_.vad_log_interval_ms, 100, 5000)) * 16;
    if (absolute_samples_ - last_vad_log_sample_ >= log_interval_samples) {
      last_vad_log_sample_ = absolute_samples_;
      emitEvent("{\"event\":\"vad_monitor\",\"state\":\"inactive\",\"realtime\":false,"
                "\"audio_flow\":true,\"kws_active\":" +
                std::string(wake_enabled_.load() ? "true" : "false") +
                ",\"reason\":\"waiting_for_wake\",\"rms\":" +
                std::to_string(rms) + ",\"pending_samples\":0,\"listening_samples\":0,"
                "\"threshold\":" + std::to_string(config_.vad_threshold) +
                ",\"session_id\":" + std::to_string(session_id_.load()) + "}");
    }
    if (wake_enabled_) {
      const double speech_rms = std::clamp(
          static_cast<double>(config_.kws_speech_rms_threshold), 0.0001, 0.1);
      if (rms < speech_rms) {
        kws_silence_samples_ = std::min<int64_t>(
            kws_silence_samples_ + static_cast<int64_t>(audio.size()), 16000 * 60);
        kws_sound_active_ = false;
      } else {
        const int64_t required_silence =
            static_cast<int64_t>(std::clamp(config_.kws_silence_reset_ms, 200, 10000)) * 16;
        if (config_.kws_reset_on_speech_after_silence && !kws_sound_active_ &&
            kws_silence_samples_ >= required_silence) {
          const int64_t silence_ms = kws_silence_samples_ / 16;
          auto rearm_onset_path = [](const SherpaOnnxKeywordSpotter *spotter,
                                     const SherpaOnnxOnlineStream *&stream) {
            if (!spotter) return;
            const SherpaOnnxOnlineStream *fresh =
                SherpaOnnxCreateKeywordStream(spotter);
            if (fresh) {
              if (stream) SherpaOnnxDestroyOnlineStream(stream);
              stream = fresh;
            } else if (stream) {
              SherpaOnnxResetKeywordStream(spotter, stream);
            }
          };
          rearm_onset_path(kws_, kws_onset_stream_);
          rearm_onset_path(kws_fallback_, kws_fallback_onset_stream_);
          const int64_t pre_roll_samples =
              static_cast<int64_t>(std::clamp(config_.kws_reset_pre_roll_ms, 0, 2000)) * 16;
          const int64_t pre_begin = std::max(ring_start_abs_, chunk_start - pre_roll_samples);
          std::vector<float> pre_roll;
          pre_roll.reserve(static_cast<size_t>(chunk_start - pre_begin));
          for (int64_t at = pre_begin; at < chunk_start; ++at) {
            pre_roll.push_back(ring_[static_cast<size_t>(at - ring_start_abs_)]);
          }
          auto prime_path = [&pre_roll](const SherpaOnnxKeywordSpotter *spotter,
                                        const SherpaOnnxOnlineStream *stream) {
            if (!spotter || !stream || pre_roll.empty()) return;
            SherpaOnnxOnlineStreamAcceptWaveform(
                stream, 16000, pre_roll.data(), static_cast<int32_t>(pre_roll.size()));
            while (SherpaOnnxIsKeywordStreamReady(spotter, stream)) {
              SherpaOnnxDecodeKeywordStream(spotter, stream);
              const SherpaOnnxKeywordResult *result =
                  SherpaOnnxGetKeywordResult(spotter, stream);
              if (result) SherpaOnnxDestroyKeywordResult(result);
            }
          };
          prime_path(kws_, kws_onset_stream_);
          prime_path(kws_fallback_, kws_fallback_onset_stream_);
          kws_onset_paths_active_ = true;
          kws_onset_paths_until_abs_ = chunk_start +
              static_cast<int64_t>(std::clamp(config_.kws_onset_window_ms,
                                              1000, 10000)) * 16;
          kws_stream_start_abs_ = pre_begin;
          emitEvent("{\"event\":\"kws_stream_reset\","
                    "\"reason\":\"speech_after_silence\",\"paths\":\"onset\","
                    "\"silence_ms\":" +
                    std::to_string(silence_ms) + ",\"rms\":" +
                    std::to_string(rms) + ",\"pre_roll_ms\":" +
                    std::to_string((chunk_start - pre_begin) / 16) + "}");
        }
        kws_silence_samples_ = 0;
        kws_sound_active_ = true;
      }
      processKws(audio.data(), static_cast<int>(audio.size()), chunk_start);
    }
    return;
  }
  if (mode_ == Mode::Listening) {
    double sum = 0.0;
    for (float v : audio) sum += static_cast<double>(v) * v;
    const double rms = std::sqrt(sum / audio.size());
    last_audio_rms_ = rms;
    emitEvent("{\"event\":\"audio_level\",\"session_id\":" + std::to_string(session_id_) +
              ",\"level\":" + std::to_string(std::min(1.0, rms * 8.0)) + "}");
    feedStreamingAsr(audio.data(), static_cast<int>(audio.size()));
    feedVad(audio.data(), static_cast<int>(audio.size()));
    const int64_t log_interval_samples =
        static_cast<int64_t>(std::clamp(config_.vad_log_interval_ms, 100, 5000)) * 16;
    if (mode_ == Mode::Listening &&
        absolute_samples_ - last_vad_log_sample_ >= log_interval_samples) {
      last_vad_log_sample_ = absolute_samples_;
      emitEvent("{\"event\":\"vad_monitor\",\"state\":\"" +
                std::string(speech_started_ ? "triggered" : "waiting") +
                "\",\"realtime\":true,\"audio_flow\":true,\"rms\":" +
                std::to_string(last_audio_rms_) + ",\"pending_samples\":" +
                std::to_string(vad_pending_.size()) + ",\"listening_samples\":" +
                std::to_string(listening_samples_) + ",\"threshold\":" +
                std::to_string(config_.vad_threshold) + ",\"session_id\":" +
                std::to_string(session_id_.load()) + "}");
    }
  }
}

VoicePipeline::KwsMatch VoicePipeline::decodeKwsPath(
    const SherpaOnnxKeywordSpotter *spotter,
    const SherpaOnnxOnlineStream *&stream, const std::string &path,
    int max_duration_ms, const float *samples, int count) {
  if (!spotter || !stream) return {};
  SherpaOnnxOnlineStreamAcceptWaveform(stream, 16000, samples, count);
  while (SherpaOnnxIsKeywordStreamReady(spotter, stream)) {
    SherpaOnnxDecodeKeywordStream(spotter, stream);
    const SherpaOnnxKeywordResult *r = SherpaOnnxGetKeywordResult(spotter, stream);
    if (r && r->keyword && std::strlen(r->keyword) > 0) {
      double end_sec = r->start_time;
      if (r->count > 0 && r->timestamps) end_sec += r->timestamps[r->count - 1];
      double start_sec = r->start_time;
      if (r->count > 0 && r->timestamps) start_sec += r->timestamps[0];
      const int64_t duration_ms = static_cast<int64_t>(
          std::llround((end_sec - start_sec) * 1000.0));
      const int64_t duration_samples = std::max<int64_t>(1600, duration_ms * 16);
      const int64_t candidate_end = absolute_samples_;
      const int64_t candidate_start = candidate_end - duration_samples;
      int64_t max_token_gap_ms = 0;
      if (r->count > 1 && r->timestamps) {
        for (int32_t i = 1; i < r->count; ++i) {
          max_token_gap_ms = std::max<int64_t>(
              max_token_gap_ms,
              static_cast<int64_t>(std::llround(
                  (r->timestamps[i] - r->timestamps[i - 1]) * 1000.0)));
        }
      }
      auto range_rms = [this](int64_t begin, int64_t end) {
        begin = std::max(begin, ring_start_abs_);
        end = std::min(end, absolute_samples_);
        if (end <= begin) return 0.0;
        double energy = 0.0;
        size_t n = 0;
        for (int64_t at = begin; at < end; ++at) {
          const float value = ring_[static_cast<size_t>(at - ring_start_abs_)];
          energy += static_cast<double>(value) * value;
          ++n;
        }
        return n == 0 ? 0.0 : std::sqrt(energy / static_cast<double>(n));
      };
      const double phrase_rms = range_rms(candidate_start, candidate_end + 1600);
      const double pre_rms = range_rms(candidate_start - 6400, candidate_start - 800);
      const double rms_ratio = phrase_rms / std::max(0.0001, pre_rms);
      emitEvent("{\"event\":\"kws_candidate\",\"path\":\"" + path +
                "\",\"keyword\":\"" + jsonEscape(r->keyword) +
                "\",\"start_ms\":" +
                std::to_string(static_cast<int64_t>(std::llround(start_sec * 1000.0))) +
                ",\"end_ms\":" +
                std::to_string(static_cast<int64_t>(std::llround(end_sec * 1000.0))) +
                ",\"duration_ms\":" + std::to_string(duration_ms) +
                ",\"max_token_gap_ms\":" + std::to_string(max_token_gap_ms) +
                ",\"absolute_end_ms\":" + std::to_string(candidate_end / 16) +
                ",\"phrase_rms\":" + std::to_string(phrase_rms) +
                ",\"pre_rms\":" + std::to_string(pre_rms) +
                ",\"rms_ratio\":" + std::to_string(rms_ratio) +
                ",\"result_json\":\"" + jsonEscape(r->json ? r->json : "") + "\"}");

      // Absolute energy floor: a candidate assembled out of near-silence is
      // never a real wake word however clean its token timing looks. Every
      // other guard below compares the candidate against its own past, so a
      // quiet room could still produce a full keyword out of the noise floor.
      // limit for this reason is the floor in micro-RMS, not milliseconds.
      const double energy_floor = std::clamp(
          static_cast<double>(config_.kws_speech_rms_threshold), 0.0001, 0.1);
      const int max_allowed_gap_ms = std::clamp(config_.kws_max_token_gap_ms, 200, 1500);
      std::string reject_reason;
      int reject_limit = max_allowed_gap_ms;
      if (phrase_rms < energy_floor) {
        reject_reason = "energy_floor";
        reject_limit = static_cast<int>(std::lround(energy_floor * 1e6));
      } else if (max_token_gap_ms > max_allowed_gap_ms) {
        reject_reason = "token_gap";
      } else if (max_duration_ms > 0 && duration_ms > max_duration_ms) {
        reject_reason = "duration";
        reject_limit = max_duration_ms;
      } else if (kws_onset_paths_active_ &&
                 (path == "primary" || path == "fallback") &&
                 rms_ratio > std::clamp(
                     static_cast<double>(config_.kws_parallel_continuous_max_rms_ratio),
                     1.0, 20.0)) {
        reject_reason = "onset_contrast";
        reject_limit = static_cast<int>(std::lround(std::clamp(
            static_cast<double>(config_.kws_parallel_continuous_max_rms_ratio),
            1.0, 20.0)));
      } else if (duration_ms > std::clamp(config_.kws_long_candidate_ms, 600, 1600) &&
                 rms_ratio > std::clamp(
                     static_cast<double>(config_.kws_long_candidate_max_rms_ratio),
                     0.5, 5.0)) {
        reject_reason = "long_candidate";
        reject_limit = std::clamp(config_.kws_long_candidate_ms, 600, 1600);
      }
      KwsMatch match;
      match.keyword = r->keyword;
      match.path = path;
      match.start_ms = static_cast<int64_t>(std::llround(start_sec * 1000.0));
      match.end_ms = static_cast<int64_t>(std::llround(end_sec * 1000.0));
      match.duration_ms = duration_ms;
      match.max_token_gap_ms = max_token_gap_ms;
      match.candidate_start = candidate_start;
      match.candidate_end = candidate_end;
      match.phrase_rms = phrase_rms;
      match.pre_rms = pre_rms;
      match.rms_ratio = rms_ratio;
      match.result_json = r->json ? r->json : "";
      SherpaOnnxDestroyKeywordResult(r);
      if (!reject_reason.empty()) {
        emitEvent("{\"event\":\"kws_rejected\",\"path\":\"" + path +
                  "\",\"reason\":\"" + reject_reason +
                  "\",\"duration_ms\":" + std::to_string(duration_ms) +
                  ",\"max_token_gap_ms\":" + std::to_string(max_token_gap_ms) +
                  ",\"phrase_rms\":" + std::to_string(phrase_rms) +
                  ",\"limit\":" + std::to_string(reject_limit) + "}");
        const SherpaOnnxOnlineStream *fresh = SherpaOnnxCreateKeywordStream(spotter);
        if (fresh) {
          SherpaOnnxDestroyOnlineStream(stream);
          stream = fresh;
        } else {
          SherpaOnnxResetKeywordStream(spotter, stream);
        }
        return {};
      }
      return match;
    }
    if (r) SherpaOnnxDestroyKeywordResult(r);
  }
  return {};
}

bool VoicePipeline::verifyFallbackWithAsr(KwsMatch *match) {
  if (!match) return false;
  match->fallback_verify_applied = true;
  if (!config_.kws_fallback_verify_enabled) {
    match->fallback_verified = true;
    match->fallback_verify_reason = "disabled";
    return true;
  }
  // The verifier reuses whichever ASR is already loaded: the offline
  // recognizer or the generative client when it is the selected model.
  // ponytail: on the generative path this is a blocking request on the KWS
  // thread, so keep an eye on fallback_verify_decode_ms before making it the
  // default ASR; add a dedicated short timeout if it blocks audibly.
  if (!asr_offline_ && !qwen3_asr_) {
    match->fallback_verified = true;
    match->fallback_verify_reason = "asr_unavailable_fail_open";
    emitEvent("{\"event\":\"fallback_verify_skipped\",\"path\":\"" +
              jsonEscape(match->path) + "\",\"reason\":\"" +
              match->fallback_verify_reason + "\"}");
    return true;
  }
  const int64_t pre = static_cast<int64_t>(std::clamp(
      config_.kws_fallback_verify_pre_roll_ms, 0, 800)) * 16;
  const int64_t max_samples = static_cast<int64_t>(std::clamp(
      config_.kws_fallback_verify_max_audio_ms, 600, 3000)) * 16;
  const int64_t clip_end = absolute_samples_;
  const int64_t desired_begin = std::min(match->candidate_start - pre,
                                         clip_end - max_samples);
  const int64_t clip_begin = std::max(ring_start_abs_, desired_begin);
  if (clip_end - clip_begin < 1600) {
    match->fallback_verified = false;
    match->fallback_verify_reason = "insufficient_audio";
    return false;
  }
  std::vector<float> clip;
  clip.reserve(static_cast<size_t>(clip_end - clip_begin));
  for (int64_t at = clip_begin; at < clip_end; ++at) {
    clip.push_back(ring_[static_cast<size_t>(at - ring_start_abs_)]);
  }
  std::string error;
  if (!recognizeDirect(clip.data(), static_cast<int>(clip.size()),
                       &match->fallback_verify_text,
                       &match->fallback_verify_decode_us, &error)) {
    match->fallback_verified = false;
    match->fallback_verify_reason = "asr_error:" + error;
    return false;
  }
  match->fallback_verified = isFallbackWakeText(match->fallback_verify_text);
  match->fallback_verify_reason = match->fallback_verified ? "confirmed" : "text_mismatch";
  return match->fallback_verified;
}

void VoicePipeline::beginKwsDebugSession() {
  try {
    std::string root;
    {
      std::lock_guard<std::mutex> lk(state_mutex_);
      root = config_.kws_debug_dir;
    }
    const auto now = std::chrono::system_clock::now();
    const auto tt = std::chrono::system_clock::to_time_t(now);
    std::tm local{};
    localtime_r(&tt, &local);
    std::ostringstream name;
    name << "session-" << std::put_time(&local, "%Y%m%d-%H%M%S")
         << "-pid" << ::getpid();
    const std::filesystem::path dir = std::filesystem::path(root) / name.str();
    std::filesystem::create_directories(dir);
    {
      std::lock_guard<std::mutex> lk(state_mutex_);
      kws_debug_session_dir_ = dir.string();
    }
  } catch (const std::exception &e) {
    {
      std::lock_guard<std::mutex> lk(state_mutex_);
      kws_debug_session_dir_.clear();
    }
    emitEvent("{\"event\":\"kws_debug_error\",\"message\":\"" +
              jsonEscape(e.what()) + "\"}");
  }
}

void VoicePipeline::saveKwsDebugCapture(const KwsMatch &match, uint64_t session_id,
                                        const std::string &outcome) {
  BibitConfig debug_config;
  std::string debug_session_dir;
  {
    std::lock_guard<std::mutex> lk(state_mutex_);
    debug_config = config_;
    debug_session_dir = kws_debug_session_dir_;
  }
  if (!debug_config.kws_debug_enabled || match.keyword.empty()) return;
  try {
    namespace fs = std::filesystem;
    const int64_t pre_samples = static_cast<int64_t>(
        std::clamp(debug_config.kws_debug_pre_roll_ms, 1000, 8000)) * 16;
    const int64_t clip_end = absolute_samples_;
    const int64_t clip_begin = std::max(ring_start_abs_, clip_end - pre_samples);
    std::vector<float> clip;
    clip.reserve(static_cast<size_t>(clip_end - clip_begin));
    for (int64_t at = clip_begin; at < clip_end; ++at) {
      clip.push_back(ring_[static_cast<size_t>(at - ring_start_abs_)]);
    }
    const auto now = std::chrono::system_clock::now();
    const auto tt = std::chrono::system_clock::to_time_t(now);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count() % 1000;
    std::tm local{};
    localtime_r(&tt, &local);
    std::ostringstream name;
    name << std::put_time(&local, "%Y%m%d-%H%M%S") << '-'
         << std::setw(3) << std::setfill('0') << ms << "-s" << session_id
         << '-' << match.path;
    if (debug_session_dir.empty()) {
      beginKwsDebugSession();
      std::lock_guard<std::mutex> lk(state_mutex_);
      debug_session_dir = kws_debug_session_dir_;
    }
    const fs::path dir(debug_session_dir.empty()
        ? debug_config.kws_debug_dir : debug_session_dir);
    fs::create_directories(dir);
    const fs::path wav_path = dir / (name.str() + ".wav");
    std::string wav_error;
    if (!writeWavPcm16Mono(wav_path.string(), clip.data(), clip.size(), 16000,
                           &wav_error)) {
      emitEvent("{\"event\":\"kws_debug_error\",\"message\":\"" +
                jsonEscape(wav_error) + "\"}");
      return;
    }
    const fs::path json_path = dir / (name.str() + ".json");
    std::ofstream out(json_path.string() + ".tmp", std::ios::trunc);
    if (!out) throw std::runtime_error("cannot create KWS debug JSON");
    out << "{\"event\":\"kws_debug_saved\",\"outcome\":\"" << jsonEscape(outcome)
        << "\",\"session_id\":" << session_id
        << ",\"keyword\":\"" << jsonEscape(match.keyword)
        << "\",\"path\":\"" << jsonEscape(match.path)
        << "\",\"wav\":\"" << jsonEscape(wav_path.string())
        << "\",\"clip_start_ms\":" << clip_begin / 16
        << ",\"clip_end_ms\":" << clip_end / 16
        << ",\"start_ms\":" << match.start_ms
        << ",\"end_ms\":" << match.end_ms
        << ",\"duration_ms\":" << match.duration_ms
        << ",\"max_token_gap_ms\":" << match.max_token_gap_ms
        << ",\"phrase_rms\":" << match.phrase_rms
        << ",\"pre_rms\":" << match.pre_rms
        << ",\"rms_ratio\":" << match.rms_ratio
        << ",\"input_rate\":16000,\"input_channels\":" << debug_config.input_channels
        << ",\"audio_source\":\"" << jsonEscape(debug_config.audio_source)
        << "\",\"kws_threshold\":" << debug_config.kws_threshold
        << ",\"kws_score\":" << debug_config.kws_score
        << ",\"kws_fallback_threshold\":" << debug_config.kws_fallback_threshold
        << ",\"kws_fallback_score\":" << debug_config.kws_fallback_score
        << ",\"kws_fallback_max_duration_ms\":" << debug_config.kws_fallback_max_duration_ms
        << ",\"kws_fallback_verify_enabled\":" << (debug_config.kws_fallback_verify_enabled ? "true" : "false")
        << ",\"fallback_verify_applied\":" << (match.fallback_verify_applied ? "true" : "false")
        << ",\"fallback_verified\":" << (match.fallback_verified ? "true" : "false")
        << ",\"fallback_verify_decode_ms\":" << match.fallback_verify_decode_us / 1000.0
        << ",\"fallback_verify_text\":\"" << jsonEscape(match.fallback_verify_text)
        << "\",\"fallback_verify_reason\":\"" << jsonEscape(match.fallback_verify_reason)
        << ",\"kws_max_token_gap_ms\":" << debug_config.kws_max_token_gap_ms
        << ",\"kws_long_candidate_ms\":" << debug_config.kws_long_candidate_ms
        << ",\"kws_long_candidate_max_rms_ratio\":" << debug_config.kws_long_candidate_max_rms_ratio
        << ",\"kws_parallel_continuous_max_rms_ratio\":" << debug_config.kws_parallel_continuous_max_rms_ratio
        << ",\"kws_onset_window_ms\":" << debug_config.kws_onset_window_ms
        << ",\"kws_speech_rms_threshold\":" << debug_config.kws_speech_rms_threshold
        << ",\"kws_max_active_paths\":" << debug_config.kws_max_active_paths
        << ",\"kws_chunk_size\":" << debug_config.kws_chunk_size
        << ",\"result_json\":\"" << jsonEscape(match.result_json) << "\"}";
    out.close();
    if (!out) throw std::runtime_error("cannot finish KWS debug JSON");
    fs::rename(json_path.string() + ".tmp", json_path);

    const int max_files = std::clamp(debug_config.kws_debug_max_files, 20, 2000);
    std::vector<fs::path> wavs;
    for (const auto &entry : fs::directory_iterator(dir)) {
      if (entry.is_regular_file() && entry.path().extension() == ".wav") wavs.push_back(entry.path());
    }
    std::sort(wavs.begin(), wavs.end());
    while (static_cast<int>(wavs.size()) > max_files) {
      fs::path old = wavs.front(); wavs.erase(wavs.begin());
      std::error_code ignored;
      fs::remove(old, ignored); fs::remove(old.replace_extension(".json"), ignored);
    }
    emitEvent("{\"event\":\"kws_debug_saved\",\"session_id\":" +
              std::to_string(session_id) + ",\"keyword\":\"" +
              jsonEscape(match.keyword) + "\",\"path\":\"" +
              jsonEscape(match.path) + "\",\"outcome\":\"" +
              jsonEscape(outcome) + "\",\"wav\":\"" +
              jsonEscape(wav_path.string()) + "\",\"json\":\"" +
              jsonEscape(json_path.string()) + "\"}");
  } catch (const std::exception &e) {
    emitEvent("{\"event\":\"kws_debug_error\",\"message\":\"" +
              jsonEscape(e.what()) + "\"}");
  }
}

void VoicePipeline::processKws(const float *samples, int count, int64_t) {
  if (kws_onset_paths_active_ && absolute_samples_ > kws_onset_paths_until_abs_) {
    kws_onset_paths_active_ = false;
  }
  KwsMatch match = decodeKwsPath(kws_, kws_stream_, "primary", 0, samples, count);
  if (match.keyword.empty()) {
    const int fallback_max_duration = std::clamp(
        config_.kws_fallback_max_duration_ms, 500, 2500);
    match = decodeKwsPath(kws_fallback_, kws_fallback_stream_, "fallback",
                          fallback_max_duration, samples, count);
  }
  if (match.keyword.empty() && kws_onset_paths_active_) {
    match = decodeKwsPath(kws_, kws_onset_stream_, "primary_onset", 0, samples, count);
  }
  if (match.keyword.empty() && kws_onset_paths_active_) {
    const int fallback_max_duration = std::clamp(
        config_.kws_fallback_max_duration_ms, 500, 2500);
    match = decodeKwsPath(kws_fallback_, kws_fallback_onset_stream_,
                            "fallback_onset", fallback_max_duration,
                            samples, count);
  }
  if (match.keyword.empty()) return;

  // Every path goes through the ASR verifier. The primary path used to wake
  // straight from the KWS hit, which is why the same false wake that the
  // fallback path rejected still got through: the verifier was never the
  // problem, only the paths that bypassed it were.
  {
    const bool verified = verifyFallbackWithAsr(&match);
    emitEvent("{\"event\":\"fallback_verify\",\"path\":\"" +
              jsonEscape(match.path) + "\",\"accepted\":" +
              std::string(verified ? "true" : "false") +
              ",\"reason\":\"" + jsonEscape(match.fallback_verify_reason) +
              "\",\"text\":\"" + jsonEscape(match.fallback_verify_text) +
              "\",\"decode_ms\":" +
              std::to_string(match.fallback_verify_decode_us / 1000.0) + "}");
    if (!verified) {
      saveKwsDebugCapture(match, session_id_ + 1, "fallback_verify_rejected");
      emitEvent("{\"event\":\"kws_rejected\",\"path\":\"" +
                jsonEscape(match.path) + "\",\"reason\":\"fallback_verify\"}");
      resetKws();
      return;
    }
  }

  emitEvent("{\"event\":\"wake_triggered\",\"session_id\":" +
            std::to_string(session_id_ + 1) + ",\"keyword\":\"" +
            jsonEscape(match.keyword) + "\"}");
  saveKwsDebugCapture(match, session_id_ + 1);
  if (asr_enabled_) {
    int ack_result = 0;
    if (config_.wake_ack_enabled) {
      emitEvent("{\"event\":\"wake_ack\",\"state\":\"started\",\"file\":\"" +
                jsonEscape(config_.wake_ack_file) + "\"}");
      ack_result = playWakeAck();
      if (config_.wake_ack_post_delay_ms > 0) {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(config_.wake_ack_post_delay_ms));
      }
      emitEvent("{\"event\":\"wake_ack\",\"state\":\"finished\",\"result\":" +
                std::to_string(ack_result) + "}");
    }
    discardCapturedAudio();
    beginListening(match.keyword);
  }
  resetKws();
}

void VoicePipeline::beginListening(const std::string &keyword, bool feed_cached_audio,
                                   int64_t cached_begin_sample,
                                   const std::string &reason) {
  ++session_id_;
  mode_ = Mode::Listening;
  active_keyword_ = keyword;
  listening_samples_ = 0;
  streaming_asr_decode_us_ = 0;
  last_vad_log_sample_ = absolute_samples_;
  last_audio_rms_ = 0.0;
  speech_started_ = false;
  asr_session_active_ = true;
  followup_listening_ = reason == "followup";
  vad_pending_.clear();
  vad_audio_history_.clear();
  SherpaOnnxVoiceActivityDetectorReset(vad_);
  if (!createAsrStream()) {
    emitEvent("{\"event\":\"error\",\"code\":\"ASR_STREAM_CREATE_FAILED\",\"session_id\":" +
              std::to_string(session_id_.load()) + "}");
    resetToIdle("asr_stream_create_failed");
    return;
  }
  emitEvent("{\"event\":\"listening\",\"session_id\":" + std::to_string(session_id_) + "}");
  emitEvent("{\"event\":\"asr_state\",\"state\":\"started\",\"reason\":\"" +
            jsonEscape(reason) + "\",\"session_id\":" + std::to_string(session_id_) + "}");
  emitEvent("{\"event\":\"vad_state\",\"state\":\"waiting\",\"session_id\":" + std::to_string(session_id_) + "}");
  emitEvent("{\"event\":\"vad_monitor\",\"state\":\"waiting\",\"realtime\":true,"
            "\"audio_flow\":false,\"rms\":0,\"pending_samples\":0,\"listening_samples\":0,"
            "\"threshold\":" + std::to_string(config_.vad_threshold) +
            ",\"session_id\":" + std::to_string(session_id_.load()) + "}");
  const int64_t begin = std::max(cached_begin_sample, ring_start_abs_);
  vad_input_start_abs_ = feed_cached_audio ? begin : absolute_samples_;
  if (feed_cached_audio && begin < absolute_samples_) {
    const size_t offset = static_cast<size_t>(begin - ring_start_abs_);
    std::vector<float> cached;
    cached.reserve(static_cast<size_t>(absolute_samples_ - begin));
    for (size_t i = offset; i < ring_.size(); ++i) cached.push_back(ring_[i]);
    feedStreamingAsr(cached.data(), static_cast<int>(cached.size()));
    feedVad(cached.data(), static_cast<int>(cached.size()));
  }
}

void VoicePipeline::feedVad(const float *samples, int count) {
  if (count <= 0 || mode_ != Mode::Listening) return;
  vad_audio_history_.insert(vad_audio_history_.end(), samples, samples + count);
  vad_pending_.insert(vad_pending_.end(), samples, samples + count);
  listening_samples_ += count;
  while (vad_pending_.size() >= 512 && mode_ == Mode::Listening) {
    SherpaOnnxVoiceActivityDetectorAcceptWaveform(vad_, vad_pending_.data(), 512);
    vad_pending_.erase(vad_pending_.begin(), vad_pending_.begin() + 512);
    if (!speech_started_ && SherpaOnnxVoiceActivityDetectorDetected(vad_)) {
      speech_started_ = true;
      emitEvent("{\"event\":\"speech_started\",\"session_id\":" + std::to_string(session_id_) + "}");
      emitEvent("{\"event\":\"vad_state\",\"state\":\"triggered\",\"session_id\":" + std::to_string(session_id_) + "}");
    }
    if (!SherpaOnnxVoiceActivityDetectorEmpty(vad_)) {
      const SherpaOnnxSpeechSegment *s = SherpaOnnxVoiceActivityDetectorFront(vad_);
      std::vector<float> segment = paddedSpeechSegment(s);
      if (s) SherpaOnnxDestroySpeechSegment(s);
      SherpaOnnxVoiceActivityDetectorPop(vad_);
      std::vector<float> carry = std::move(vad_pending_);
      vad_pending_.clear();
      if (!segment.empty()) decodeSegment(segment.data(), static_cast<int>(segment.size()));
      if (mode_ == Mode::Listening && !carry.empty()) {
        feedStreamingAsr(carry.data(), static_cast<int>(carry.size()));
        feedVad(carry.data(), static_cast<int>(carry.size()));
      }
      return;
    }
  }
  const float followup_timeout = std::clamp(config_.followup_timeout, 2.0f, 30.0f);
  if (mode_ == Mode::Listening && followup_listening_ && !speech_started_ &&
      listening_samples_ > static_cast<int64_t>(followup_timeout * 16000)) {
    emitEvent("{\"event\":\"session_cancelled\",\"reason\":\"followup_timeout\",\"session_id\":" +
              std::to_string(session_id_) + "}");
    resetToIdle("followup_timeout");
    return;
  }
  if (mode_ == Mode::Listening && listening_samples_ > static_cast<int64_t>((config_.max_speech + 3.0f) * 16000)) {
    SherpaOnnxVoiceActivityDetectorFlush(vad_);
    if (!SherpaOnnxVoiceActivityDetectorEmpty(vad_)) {
      const SherpaOnnxSpeechSegment *s = SherpaOnnxVoiceActivityDetectorFront(vad_);
      std::vector<float> segment = paddedSpeechSegment(s);
      if (s) SherpaOnnxDestroySpeechSegment(s);
      SherpaOnnxVoiceActivityDetectorPop(vad_);
      if (!segment.empty()) decodeSegment(segment.data(), static_cast<int>(segment.size()));
    } else {
      emitEvent("{\"event\":\"error\",\"code\":\"NO_SPEECH\",\"session_id\":" + std::to_string(session_id_) + "}");
      resetToIdle("no_speech");
    }
  }
}

std::vector<float> VoicePipeline::paddedSpeechSegment(
    const SherpaOnnxSpeechSegment *segment) {
  if (!segment || !segment->samples || segment->n <= 0) return {};
  const int pre = std::clamp(config_.asr_pre_roll_ms, 0, 1000) * 16;
  const int post = std::clamp(config_.asr_post_roll_ms, 0, 1000) * 16;
  const int64_t raw_begin = std::max<int64_t>(0, segment->start);
  const int64_t raw_end = raw_begin + segment->n;
  const int64_t begin = std::max<int64_t>(0, raw_begin - pre);
  const int64_t end = std::min<int64_t>(vad_audio_history_.size(), raw_end + post);
  std::vector<float> result;
  if (begin < end && raw_end <= static_cast<int64_t>(vad_audio_history_.size())) {
    result.assign(vad_audio_history_.begin() + begin, vad_audio_history_.begin() + end);
  } else {
    result.assign(segment->samples, segment->samples + segment->n);
  }
  emitEvent("{\"event\":\"vad_segment\",\"start_sample\":" +
            std::to_string(segment->start) + ",\"raw_samples\":" +
            std::to_string(segment->n) + ",\"padded_samples\":" +
            std::to_string(result.size()) + ",\"pre_roll_ms\":" +
            std::to_string(config_.asr_pre_roll_ms) + ",\"post_roll_ms\":" +
            std::to_string(config_.asr_post_roll_ms) + "}");
  return result;
}

void VoicePipeline::decodeSegment(const float *samples, int count) {
  if (count <= 0) { resetToIdle("empty_segment"); return; }
  if (!asr_offline_ && !asr_online_ && !qwen3_asr_) {
    emitEvent("{\"event\":\"error\",\"code\":\"ASR_NOT_LOADED\",\"session_id\":" +
              std::to_string(session_id_.load()) + "}");
    resetToIdle("asr_not_loaded");
    return;
  }
  mode_ = Mode::Recognizing;
  emitEvent("{\"event\":\"recognizing\",\"session_id\":" + std::to_string(session_id_) + "}");
  const auto begin = std::chrono::steady_clock::now();
  const int64_t prior_streaming_decode_us = streaming_asr_decode_us_;
  std::string text;
  if (qwen3_asr_) {
    std::string error;
    if (!qwen3_asr_->recognize(samples, count, &text, &error)) {
      const int64_t decode_us = std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - begin).count();
      emitAsrMetrics(count, decode_us, "error", "QWEN3_ASR_FAILED");
      emitEvent("{\"event\":\"error\",\"code\":\"QWEN3_ASR_FAILED\",\"message\":\"" +
                jsonEscape(error) + "\",\"session_id\":" +
                std::to_string(session_id_.load()) + "}");
      resetToIdle("qwen3_asr_failed");
      return;
    }
  } else if (asr_online_) {
    if (!asr_stream_) {
      emitAsrMetrics(count, prior_streaming_decode_us, "error", "ASR_STREAM_MISSING");
      emitEvent("{\"event\":\"error\",\"code\":\"ASR_STREAM_MISSING\",\"session_id\":" +
                std::to_string(session_id_.load()) + "}");
      resetToIdle("asr_stream_missing");
      return;
    }
    SherpaOnnxOnlineStreamInputFinished(asr_stream_);
    while (SherpaOnnxIsOnlineStreamReady(asr_online_, asr_stream_)) {
      SherpaOnnxDecodeOnlineStream(asr_online_, asr_stream_);
    }
    const SherpaOnnxOnlineRecognizerResult *r =
        SherpaOnnxGetOnlineStreamResult(asr_online_, asr_stream_);
    text = r && r->text ? r->text : "";
    if (r) SherpaOnnxDestroyOnlineRecognizerResult(r);
    destroyAsrStream();
  } else {
    const SherpaOnnxOfflineStream *stream = SherpaOnnxCreateOfflineStream(asr_offline_);
    if (!stream) {
      emitAsrMetrics(count, 0, "error", "ASR_STREAM_CREATE_FAILED");
      emitEvent("{\"event\":\"error\",\"code\":\"ASR_STREAM_CREATE_FAILED\",\"session_id\":" +
                std::to_string(session_id_.load()) + "}");
      resetToIdle("asr_stream_create_failed");
      return;
    }
    SherpaOnnxAcceptWaveformOffline(stream, 16000, samples, count);
    SherpaOnnxDecodeOfflineStream(asr_offline_, stream);
    const SherpaOnnxOfflineRecognizerResult *r = SherpaOnnxGetOfflineStreamResult(stream);
    text = r && r->text ? r->text : "";
    if (r) SherpaOnnxDestroyOfflineRecognizerResult(r);
    SherpaOnnxDestroyOfflineStream(stream);
  }
  text = stripWakePrefix(text);
  const int64_t final_decode_us = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - begin).count();
  const int64_t decode_us = prior_streaming_decode_us + final_decode_us;
  const int64_t ms = (decode_us + 500) / 1000;
  const double audio_duration_ms = static_cast<double>(count) / 16.0;
  const double rtf = static_cast<double>(decode_us) * 16000.0 /
      (static_cast<double>(count) * 1000000.0);
  emitAsrMetrics(count, decode_us, "success");
  const bool exit_requested = isExitCommand(text);
  if (exit_requested) {
    emitEvent("{\"event\":\"session_exit\",\"session_id\":" + std::to_string(session_id_) +
              ",\"reason\":\"voice_command\",\"decode_ms\":" + std::to_string(ms) +
              ",\"audio_duration_ms\":" + std::to_string(audio_duration_ms) +
              ",\"rtf\":" + std::to_string(rtf) + "}");
  } else {
    emitEvent("{\"event\":\"asr_final\",\"session_id\":" + std::to_string(session_id_) +
              ",\"text\":\"" + jsonEscape(text) + "\",\"decode_ms\":" + std::to_string(ms) +
              ",\"audio_duration_ms\":" + std::to_string(audio_duration_ms) +
              ",\"rtf\":" + std::to_string(rtf) + "}");
  }
  size_t queued_samples = 0;
  {
    std::lock_guard<std::mutex> lk(queue_mutex_);
    queued_samples = queued_audio_samples_;
  }
  emitEvent("{\"event\":\"audio_backlog_preserved\",\"queued_samples\":" +
            std::to_string(queued_samples) + ",\"queued_ms\":" +
            std::to_string(queued_samples / 16) + "}");
  const bool continue_listening = !exit_requested && config_.followup_enabled && asr_enabled_.load();
  resetToIdle(exit_requested ? "voice_exit" : "final");
  if (continue_listening) beginListening("", false, 0, "followup");
}

void VoicePipeline::emitAsrMetrics(int sample_count, int64_t decode_us,
                                   const std::string &status,
                                   const std::string &error_code) const {
  const int safe_count = std::max(1, sample_count);
  const double audio_duration_ms = static_cast<double>(safe_count) / 16.0;
  const double decode_ms = static_cast<double>(std::max<int64_t>(0, decode_us)) / 1000.0;
  const double rtf = static_cast<double>(std::max<int64_t>(0, decode_us)) * 16000.0 /
      (static_cast<double>(safe_count) * 1000000.0);
  std::ostringstream json;
  json << std::fixed << std::setprecision(3)
       << "{\"event\":\"asr_metrics\",\"session_id\":" << session_id_.load()
       << ",\"model_type\":\"" << jsonEscape(config_.asr_model_type) << "\""
       << ",\"status\":\"" << jsonEscape(status) << "\""
       << ",\"audio_samples\":" << sample_count
       << ",\"audio_duration_ms\":" << audio_duration_ms
       << ",\"decode_ms\":" << decode_ms
       << ",\"rtf\":" << rtf;
  if (!error_code.empty()) {
    json << ",\"error_code\":\"" << jsonEscape(error_code) << "\"";
  }
  json << '}';
  emitEvent(json.str());
}

void VoicePipeline::resetToIdle(const std::string &reason) {
  if (asr_session_active_) {
    emitEvent("{\"event\":\"vad_state\",\"state\":\"idle\",\"session_id\":" + std::to_string(session_id_) + "}");
    emitEvent("{\"event\":\"asr_state\",\"state\":\"stopped\",\"reason\":\"" +
              jsonEscape(reason) + "\",\"session_id\":" + std::to_string(session_id_) + "}");
    asr_session_active_ = false;
  }
  emitEvent("{\"event\":\"vad_monitor\",\"state\":\"inactive\",\"realtime\":false,"
            "\"audio_flow\":false,\"reason\":\"" + jsonEscape(reason) +
            "\",\"session_id\":" + std::to_string(session_id_.load()) + "}");
  mode_ = Mode::Idle;
  last_vad_log_sample_ = absolute_samples_;
  listening_samples_ = 0;
  streaming_asr_decode_us_ = 0;
  speech_started_ = false;
  followup_listening_ = false;
  vad_pending_.clear();
  vad_audio_history_.clear();
  active_keyword_.clear();
  destroyAsrStream();
  SherpaOnnxVoiceActivityDetectorReset(vad_);
  resetKws();
}

int VoicePipeline::playWakeAck() {
  if (config_.wake_ack_file.empty() || ::access(config_.wake_ack_file.c_str(), R_OK) != 0) return 2;
  pid_t pid = -1;
  char program[] = "pw-play";
  char latency[] = "--latency=20ms";
  char *argv[] = {program, latency, const_cast<char *>(config_.wake_ack_file.c_str()), nullptr};
  const int spawn_result = ::posix_spawnp(&pid, program, nullptr, nullptr, argv, environ);
  if (spawn_result != 0) return 3;
  int status = 0;
  pid_t wait_result = -1;
  do {
    wait_result = ::waitpid(pid, &status, 0);
  } while (wait_result < 0 && errno == EINTR);
  if (wait_result < 0) return 4;
  return WIFEXITED(status) ? WEXITSTATUS(status) : 4;
}

void VoicePipeline::discardCapturedAudio(const std::string &reason) {
  {
    std::lock_guard<std::mutex> lk(queue_mutex_);
    audio_queue_.clear();
    queued_audio_samples_ = 0;
    if (resampler_) resampler_->reset();
  }
  queue_cv_.notify_all();
  ring_.clear();
  ring_start_abs_ = absolute_samples_;
  vad_pending_.clear();
  SherpaOnnxVoiceActivityDetectorReset(vad_);
  emitEvent("{\"event\":\"audio_discarded\",\"reason\":\"" +
            jsonEscape(reason) + "\"}");
}

void VoicePipeline::resetKws() {
  auto reset_path = [](const SherpaOnnxKeywordSpotter *spotter,
                       const SherpaOnnxOnlineStream *&stream) {
    if (spotter) {
      const SherpaOnnxOnlineStream *fresh = SherpaOnnxCreateKeywordStream(spotter);
      if (fresh) {
        if (stream) SherpaOnnxDestroyOnlineStream(stream);
        stream = fresh;
      } else if (stream) {
        SherpaOnnxResetKeywordStream(spotter, stream);
      }
    }
  };
  reset_path(kws_, kws_stream_);
  reset_path(kws_, kws_onset_stream_);
  reset_path(kws_fallback_, kws_fallback_stream_);
  reset_path(kws_fallback_, kws_fallback_onset_stream_);
  kws_stream_start_abs_ = absolute_samples_;
  kws_silence_samples_ = 0;
  kws_sound_active_ = false;
  kws_onset_paths_active_ = false;
  kws_onset_paths_until_abs_ = 0;
}

void VoicePipeline::emitEvent(const std::string &json) const {
  std::cout << json << std::endl;
  if (callback_) callback_(json);
}

std::string VoicePipeline::jsonEscape(const std::string &s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (unsigned char c : s) {
    if (c == '\\') out += "\\\\";
    else if (c == '"') out += "\\\"";
    else if (c == '\n') out += "\\n";
    else if (c == '\r') out += "\\r";
    else if (c == '\t') out += "\\t";
    else if (c >= 0x20) out.push_back(static_cast<char>(c));
  }
  return out;
}

std::string VoicePipeline::stripWakePrefix(std::string text) const {
  trimText(text);
  std::vector<std::string> candidates = {
      "hello bibit", "hello bbit", "hello bebit", "hello be bit",
      "hi bibit", "hi bbit", "hi bebit", "hi bee bit", "hi bit",
      "hey bibit", "hey bbit", "hey bebit", "hey bee bit", "hey bit",
      "你好bibit", "你好比比特", "你好比比", "你好必必特", "你好比必特",
      "bibit", "bbit", "bebit", "bee bit", "比比特", "比比"};
  if (!active_keyword_.empty()) candidates.insert(candidates.begin(), active_keyword_);

  const NormalizedText normalized = normalizeWakeText(text);
  size_t best_normalized_size = 0;
  size_t erase_end = 0;
  for (const std::string &candidate : candidates) {
    const std::string key = normalizeWakeText(candidate).text;
    if (key.size() <= best_normalized_size || key.empty()) continue;
    if (normalized.text.rfind(key, 0) == 0 && key.size() <= normalized.source_ends.size()) {
      best_normalized_size = key.size();
      erase_end = normalized.source_ends[key.size() - 1];
    }
  }
  if (erase_end > 0) {
    text.erase(0, erase_end);
    trimText(text);
  }
  return text;
}
