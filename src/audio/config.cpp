#include "config.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

namespace {
std::string trim(std::string s) {
  auto not_space = [](unsigned char c) { return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  return s;
}

bool to_bool(const std::string &s) {
  return s == "1" || s == "true" || s == "on" || s == "yes";
}
}  // namespace

BibitConfig BibitConfig::load(const std::string &path) {
  BibitConfig c;
  c.apply(path);
  if (c.kws_debug_dir.empty()) {
    const char *state = std::getenv("XDG_STATE_HOME");
    if (state && *state) {
      c.kws_debug_dir = std::string(state) + "/bibit/kws-debug";
    } else {
      const char *home = std::getenv("HOME");
      c.kws_debug_dir = std::string(home && *home ? home : "/tmp") +
                        "/.local/share/bibit/kws-debug";
    }
  }
  return c;
}

void BibitConfig::apply(const std::string &path) {
  std::ifstream in(path);
  if (!in) return;
  std::string line;
  while (std::getline(in, line)) {
    line = trim(line);
    if (line.empty() || line[0] == '#') continue;
    const auto pos = line.find('=');
    if (pos == std::string::npos) continue;
    const std::string key = trim(line.substr(0, pos));
    const std::string value = trim(line.substr(pos + 1));
    try {
      if (key == "model_root") model_root = value;
      else if (key == "provider_config") provider_config = value;
      else if (key == "asr_model_type") asr_model_type = value;
      else if (key == "asr_model") asr_model = value;
      else if (key == "asr_tokens") asr_tokens = value;
      else if (key == "asr_provider") asr_provider = value;
      else if (key == "asr_num_threads") asr_num_threads = std::stoi(value);
      else if (key == "asr_decoding_method") asr_decoding_method = value;
      else if (key == "asr_max_active_paths") asr_max_active_paths = std::stoi(value);
      else if (key == "asr_pre_roll_ms") asr_pre_roll_ms = std::stoi(value);
      else if (key == "asr_post_roll_ms") asr_post_roll_ms = std::stoi(value);
      else if (key == "asr_release_timeout_ms") asr_release_timeout_ms = std::stoi(value);
      else if (key == "qwen3_model_dir") qwen3_model_dir = value;
      else if (key == "qwen3_server_bin") qwen3_server_bin = value;
      else if (key == "qwen3_server_host") qwen3_server_host = value;
      else if (key == "qwen3_server_port") qwen3_server_port = std::stoi(value);
      else if (key == "qwen3_thread_batch") qwen3_thread_batch = std::stoi(value);
      else if (key == "qwen3_ep_affinity") qwen3_ep_affinity = value;
      else if (key == "qwen3_ctx_size") qwen3_ctx_size = std::stoi(value);
      else if (key == "qwen3_language") qwen3_language = value;
      else if (key == "qwen3_max_tokens") qwen3_max_tokens = std::stoi(value);
      else if (key == "qwen3_startup_timeout_ms") qwen3_startup_timeout_ms = std::stoi(value);
      else if (key == "qwen3_request_timeout_ms") qwen3_request_timeout_ms = std::stoi(value);
      else if (key == "keywords_file") keywords_file = value;
      else if (key == "kws_fallback_keywords_file") kws_fallback_keywords_file = value;
      else if (key == "audio_source") audio_source = value;
      else if (key == "input_rate") input_rate = std::stoi(value);
      else if (key == "input_channels") input_channels = std::stoi(value);
      else if (key == "audio_queue_max_ms") audio_queue_max_ms = std::stoi(value);
      else if (key == "wake_enabled") wake_enabled = to_bool(value);
      else if (key == "asr_enabled") asr_enabled = to_bool(value);
      else if (key == "kws_score") kws_score = std::stof(value);
      else if (key == "kws_threshold") kws_threshold = std::stof(value);
      else if (key == "kws_trailing_blanks") kws_trailing_blanks = std::stoi(value);
      else if (key == "kws_max_token_gap_ms") kws_max_token_gap_ms = std::stoi(value);
      else if (key == "kws_fallback_score") kws_fallback_score = std::stof(value);
      else if (key == "kws_fallback_threshold") kws_fallback_threshold = std::stof(value);
      else if (key == "kws_fallback_max_duration_ms") kws_fallback_max_duration_ms = std::stoi(value);
      else if (key == "kws_fallback_verify_enabled") kws_fallback_verify_enabled = to_bool(value);
      else if (key == "kws_fallback_verify_pre_roll_ms") kws_fallback_verify_pre_roll_ms = std::stoi(value);
      else if (key == "kws_fallback_verify_max_audio_ms") kws_fallback_verify_max_audio_ms = std::stoi(value);
      else if (key == "kws_long_candidate_ms") kws_long_candidate_ms = std::stoi(value);
      else if (key == "kws_long_candidate_max_rms_ratio") kws_long_candidate_max_rms_ratio = std::stof(value);
      else if (key == "kws_parallel_continuous_max_rms_ratio") kws_parallel_continuous_max_rms_ratio = std::stof(value);
      else if (key == "kws_onset_window_ms") kws_onset_window_ms = std::stoi(value);
      else if (key == "kws_reset_on_speech_after_silence") kws_reset_on_speech_after_silence = to_bool(value);
      else if (key == "kws_silence_reset_ms") kws_silence_reset_ms = std::stoi(value);
      else if (key == "kws_speech_rms_threshold") kws_speech_rms_threshold = std::stof(value);
      else if (key == "kws_reset_pre_roll_ms") kws_reset_pre_roll_ms = std::stoi(value);
      else if (key == "vad_threshold") vad_threshold = std::stof(value);
      else if (key == "min_silence") min_silence = std::stof(value);
      else if (key == "min_speech") min_speech = std::stof(value);
      else if (key == "max_speech") max_speech = std::stof(value);
      else if (key == "vad_log_interval_ms") vad_log_interval_ms = std::stoi(value);
      else if (key == "wake_guard_ms") wake_guard_ms = std::stoi(value);
      else if (key == "kws_max_active_paths") kws_max_active_paths = std::stoi(value);
      else if (key == "kws_chunk_size") kws_chunk_size = std::stoi(value);
      else if (key == "kws_debug_enabled") kws_debug_enabled = to_bool(value);
      else if (key == "kws_debug_dir") kws_debug_dir = value;
      else if (key == "kws_debug_pre_roll_ms") kws_debug_pre_roll_ms = std::stoi(value);
      else if (key == "kws_debug_max_files") kws_debug_max_files = std::stoi(value);
      else if (key == "wake_ack_enabled") wake_ack_enabled = to_bool(value);
      else if (key == "wake_ack_file") wake_ack_file = value;
      else if (key == "wake_ack_post_delay_ms") wake_ack_post_delay_ms = std::stoi(value);
      else if (key == "followup_enabled") followup_enabled = to_bool(value);
      else if (key == "followup_timeout") followup_timeout = std::stof(value);
    } catch (const std::invalid_argument &) {
      continue;
    } catch (const std::out_of_range &) {
      continue;
    }
  }
}

BibitConfig asrModelConfig(BibitConfig config, const std::string &model) {
  if (model == "qwen3_asr_0.6b") {
    config.asr_model_type = "qwen3_asr_0.6b";
    config.asr_model =
        "/opt/bibit/models/asr/qwen3-asr-0.6B-dynq-q40/"
        "Qwen3-ASR-0.6B-text-q40.gguf";
    config.asr_tokens.clear();
    config.asr_provider = "smt";
    config.asr_num_threads = 2;
    config.asr_decoding_method = "generative";
    config.qwen3_model_dir =
        "/opt/bibit/models/asr/qwen3-asr-0.6B-dynq-q40";
  } else if (model == "sensevoice") {
    config.asr_model_type = "sensevoice";
    config.asr_model =
        "/opt/bibit/models/asr/sensevoice/model_quant_optimized.onnx";
    config.asr_tokens =
        "/opt/bibit/models/asr/sensevoice/tokens.sherpa.txt";
    config.asr_provider = "spacemit";
    config.asr_num_threads = 2;
    config.asr_decoding_method = "greedy_search";
  }
  return config;
}
