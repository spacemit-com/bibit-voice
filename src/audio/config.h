#pragma once

#include <string>

struct BibitConfig {
  std::string model_root = "/opt/bibit/models";
  std::string provider_config = "/etc/bibit/bibit-asr-provider.config";
  // Shipped default is Qwen3-ASR. asr_model_type, asr_model, asr_tokens,
  // asr_provider and asr_decoding_method move together - see asrModelConfig()
  // in main.cpp, which is what set_asr_model applies.
  std::string asr_model_type = "qwen3_asr_0.6b";
  std::string asr_model =
      "/opt/bibit/models/asr/qwen3-asr-0.6B-dynq-q40/"
      "Qwen3-ASR-0.6B-text-q40.gguf";
  std::string asr_tokens;
  std::string asr_provider = "smt";
  int asr_num_threads = 2;
  std::string asr_decoding_method = "generative";
  int asr_max_active_paths = 4;
  int asr_pre_roll_ms = 180;
  int asr_post_roll_ms = 120;
  // 切换时等待流水线线程放下现役后端的上限（ms）。超时按切换失败处理：
  // 现役后端原样保留，绝不在它还没放下时构建新后端（两者会争用 SpacemiT EP/TCM）。
  int asr_release_timeout_ms = 30000;
  std::string qwen3_model_dir =
      "/opt/bibit/models/asr/qwen3-asr-0.6B-dynq-q40";
  std::string qwen3_server_bin = "llama-server";
  std::string qwen3_server_host = "127.0.0.1";
  int qwen3_server_port = 8063;
  int qwen3_thread_batch = 2;
  std::string qwen3_ep_affinity = "8;9";
  int qwen3_ctx_size = 4096;
  std::string qwen3_language = "zh,en";
  int qwen3_max_tokens = 100;
  int qwen3_startup_timeout_ms = 60000;
  int qwen3_request_timeout_ms = 60000;
  std::string keywords_file = "/etc/bibit/keywords.txt";
  std::string kws_fallback_keywords_file;
  // "default" follows the input device selected in the system sound settings.
  std::string audio_source = "default";
  int input_rate = 16000;
  int input_channels = 2;
  // Capture-queue ceiling in milliseconds, clamped to 200..30000 by
  // queueMaxSamples(). Past the ceiling the oldest audio is dropped and the
  // KWS/VAD stream state is reset, so the pipeline falls at most this far
  // behind live audio instead of holding a 30 s backlog it cannot drain.
  int audio_queue_max_ms = 2000;
  bool wake_enabled = true;
  bool asr_enabled = true;
  float kws_score = 4.5f;
  float kws_threshold = 0.12f;
  int kws_trailing_blanks = 1;
  int kws_max_token_gap_ms = 600;
  float kws_fallback_score = 4.0f;
  float kws_fallback_threshold = 0.15f;
  int kws_fallback_max_duration_ms = 1100;
  // Only fallback (Hello Bibi compatibility) is verified by the independent
  // SenseVoice ASR model before it is allowed to wake the UI.
  bool kws_fallback_verify_enabled = true;
  int kws_fallback_verify_pre_roll_ms = 240;
  int kws_fallback_verify_max_audio_ms = 1600;
  int kws_long_candidate_ms = 900;
  float kws_long_candidate_max_rms_ratio = 1.0f;
  float kws_parallel_continuous_max_rms_ratio = 5.0f;
  int kws_onset_window_ms = 3000;
  bool kws_reset_on_speech_after_silence = true;
  int kws_silence_reset_ms = 1200;
  float kws_speech_rms_threshold = 0.0015f;
  // Re-feeding history to the onset streams costs more compute than the single
  // worker has left: 800 ms per stream per silence-then-speech transition is
  // what pushed the pipeline below realtime and overflowed the capture queue.
  // 0 keeps the reset and skips the re-decode. See docs/configuration.md for
  // the measured A/B/C comparison.
  int kws_reset_pre_roll_ms = 0;
  float vad_threshold = 0.30f;
  float min_silence = 0.50f;
  float min_speech = 0.20f;
  float max_speech = 12.0f;
  int vad_log_interval_ms = 500;
  int wake_guard_ms = 180;
  int kws_max_active_paths = 16;
  int kws_chunk_size = 16;
  // Disabled by default. When enabled, each accepted KWS wake writes a
  // short pre-trigger WAV plus a sidecar JSON record for false-wake analysis.
  bool kws_debug_enabled = false;
  // Empty means the directory is derived at load time from
  // $XDG_STATE_HOME/bibit/kws-debug, or $HOME/.local/share/bibit/kws-debug.
  std::string kws_debug_dir;
  int kws_debug_pre_roll_ms = 4000;
  int kws_debug_max_files = 200;
  bool wake_ack_enabled = true;
  std::string wake_ack_file = "/usr/share/bibit/sounds/wo-zai.wav";
  int wake_ack_post_delay_ms = 0;
  bool followup_enabled = true;
  float followup_timeout = 6.0f;

  static BibitConfig load(const std::string &path);
  void apply(const std::string &path);
};

// 识别引擎名到配置字段组的唯一映射：set_asr_model 与服务端能力收敛共用。
// 未知引擎名原样返回入参配置，由调用方决定如何拒绝。
BibitConfig asrModelConfig(BibitConfig config, const std::string &model);
