#include "config.h"
#include "pipewire_capture.h"
#include "pipeline.h"
#include "socket_server.h"
#include "wav.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {
std::atomic<bool> g_stop{false};
void onSignal(int) { g_stop = true; }

// 只在「键位置」匹配字段名：必须紧跟在 { 或 , 之后（允许空白）。否则 request_id
// 这类自由字符串的值里出现同名子串时会被误当成字段，静默改写用户没点的开关。
size_t findKey(const std::string &line, const std::string &key) {
  const std::string needle = "\"" + key + "\"";
  size_t pos = line.find(needle);
  while (pos != std::string::npos) {
    size_t before = pos;
    while (before > 0 && (line[before - 1] == ' ' || line[before - 1] == '\t')) --before;
    if (before > 0 && (line[before - 1] == '{' || line[before - 1] == ',')) return pos;
    pos = line.find(needle, pos + needle.size());
  }
  return std::string::npos;
}

bool jsonBool(const std::string &line, const std::string &key, bool fallback) {
  const auto p = findKey(line, key);
  if (p == std::string::npos) return fallback;
  const auto c = line.find(':', p);
  if (c == std::string::npos) return fallback;
  const auto v = line.find_first_not_of(" \t", c + 1);
  if (v == std::string::npos) return fallback;
  return line.compare(v, 4, "true") == 0 || line.compare(v, 1, "1") == 0;
}

// 字段级意图判据：只有真正给出了该键才算一次意图，缺省字段沿用当前生效值。
bool jsonHas(const std::string &line, const std::string &key) {
  const auto p = findKey(line, key);
  if (p == std::string::npos) return false;
  return line.find(':', p) != std::string::npos;
}

// 严格布尔解析：只有字面 true/false 才算合法，数字或字符串一律拒绝。
// socket 是原始协议边界，不能依赖 CLI/MCP 的入参校验。
bool jsonBoolValue(const std::string &line, const std::string &key, bool *out) {
  const auto p = findKey(line, key);
  if (p == std::string::npos) return false;
  const auto c = line.find(':', p);
  if (c == std::string::npos) return false;
  const auto v = line.find_first_not_of(" \t", c + 1);
  if (v == std::string::npos) return false;
  // 字面量后面必须跟分隔符，否则 `truex` 这类前缀也会被当成合法布尔。
  const auto delimiter_ok = [&line](size_t end) {
    if (end >= line.size()) return true;
    const char ch = line[end];
    return ch == ',' || ch == '}' || ch == ' ' || ch == '\t' || ch == '\r';
  };
  if (line.compare(v, 4, "true") == 0 && delimiter_ok(v + 4)) {
    *out = true;
    return true;
  }
  if (line.compare(v, 5, "false") == 0 && delimiter_ok(v + 5)) {
    *out = false;
    return true;
  }
  return false;
}

float jsonFloat(const std::string &line, const std::string &key, float fallback) {
  const auto p = findKey(line, key);
  if (p == std::string::npos) return fallback;
  const auto c = line.find(':', p);
  if (c == std::string::npos) return fallback;
  try {
    return std::stof(line.substr(c + 1));
  } catch (...) {
    return fallback;
  }
}

uint64_t jsonUint(const std::string &line, const std::string &key, uint64_t fallback) {
  const auto p = findKey(line, key);
  if (p == std::string::npos) return fallback;
  const auto c = line.find(':', p);
  if (c == std::string::npos) return fallback;
  try {
    return std::stoull(line.substr(c + 1));
  } catch (...) {
    return fallback;
  }
}

std::string jsonString(const std::string &line, const std::string &key) {
  const auto p = findKey(line, key);
  if (p == std::string::npos) return {};
  const auto c = line.find(':', p);
  if (c == std::string::npos) return {};
  const auto quote = line.find('"', c + 1);
  if (quote == std::string::npos) return {};
  // 按 JSON 转义规则读取：request_id 等字段是外部输入，遇到 \" 不能提前截断，
  // 否则同一个值在回执里会与调用方发送的不同。
  std::string out;
  for (size_t i = quote + 1; i < line.size(); ++i) {
    const char ch = line[i];
    if (ch == '\\' && i + 1 < line.size()) {
      const char next = line[i + 1];
      switch (next) {
        case 'n': out.push_back('\n'); break;
        case 't': out.push_back('\t'); break;
        case 'r': out.push_back('\r'); break;
        default: out.push_back(next); break;
      }
      ++i;
      continue;
    }
    if (ch == '"') break;
    out.push_back(ch);
  }
  return out;
}

bool saveThresholds(const std::string &path, const BibitConfig &config) {
  try {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    const std::string temporary = path + ".tmp";
    std::ofstream out(temporary, std::ios::trunc);
    if (!out) return false;
    out << "kws_threshold=" << config.kws_threshold << '\n'
        << "kws_score=" << config.kws_score << '\n'
        << "vad_threshold=" << config.vad_threshold << '\n'
        << "min_silence=" << config.min_silence << '\n'
        << "wake_guard_ms=" << config.wake_guard_ms << '\n'
        << "kws_debug_enabled=" << (config.kws_debug_enabled ? "true" : "false") << '\n';
    out.close();
    if (!out) return false;
    std::filesystem::rename(temporary, path);
    return true;
  } catch (...) {
    return false;
  }
}

bool saveSwitches(const std::string &path, bool wake, bool asr) {
  try {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    const std::string temporary = path + ".tmp";
    std::ofstream out(temporary, std::ios::trunc);
    if (!out) return false;
    out << "wake_enabled=" << (wake ? "true" : "false") << '\n'
        << "asr_enabled=" << (asr ? "true" : "false") << '\n';
    out.close();
    if (!out) return false;
    std::filesystem::rename(temporary, path);
    return true;
  } catch (...) {
    return false;
  }
}

bool modelFilesAvailable(const std::string &model) {
  namespace fs = std::filesystem;
  if (model == "qwen3_asr_0.6b") {
    const fs::path root = "/opt/bibit/models/asr/qwen3-asr-0.6B-dynq-q40";
    return fs::is_regular_file(root / "config.json") &&
           fs::is_regular_file(root / "Qwen3-ASR-0.6B-text-q40.gguf") &&
           fs::is_regular_file(root / "Qwen3-ASR-0.6B-encoder-frontend.dynq.onnx") &&
           fs::is_regular_file(root / "Qwen3-ASR-0.6B-encoder-backend.dynq.onnx");
  }
  if (model == "sensevoice") {
    const fs::path root = "/opt/bibit/models/asr/sensevoice";
    return fs::is_regular_file(root / "model_quant_optimized.onnx") &&
           fs::is_regular_file(root / "tokens.sherpa.txt") &&
           fs::is_regular_file("/etc/bibit/bibit-asr-provider.config");
  }
  return false;
}

bool saveAsrModel(const std::string &path, const std::string &model) {
  try {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    const std::string temporary = path + ".tmp";
    std::ofstream out(temporary, std::ios::trunc);
    if (!out) return false;
    if (model == "qwen3_asr_0.6b") {
      out << "asr_model_type=qwen3_asr_0.6b\n"
             "asr_model=/opt/bibit/models/asr/qwen3-asr-0.6B-dynq-q40/Qwen3-ASR-0.6B-text-q40.gguf\n"
             "asr_tokens=\n"
             "asr_provider=smt\n"
             "asr_num_threads=2\n"
             "asr_decoding_method=generative\n"
             "qwen3_model_dir=/opt/bibit/models/asr/qwen3-asr-0.6B-dynq-q40\n";
    } else if (model == "sensevoice") {
      out << "asr_model_type=sensevoice\n"
             "asr_model=/opt/bibit/models/asr/sensevoice/model_quant_optimized.onnx\n"
             "asr_tokens=/opt/bibit/models/asr/sensevoice/tokens.sherpa.txt\n"
             "asr_provider=spacemit\n"
             "asr_num_threads=2\n"
             "asr_decoding_method=greedy_search\n";
    } else {
      return false;
    }
    out.close();
    if (!out) return false;
    std::filesystem::rename(temporary, path);
    return true;
  } catch (...) {
    return false;
  }
}

bool hasCommand(const std::string &line, const std::string &command) {
  // 只在键位置认 "cmd"：字符串值里出现的同名子串不能决定命令分派。
  const auto p = findKey(line, "cmd");
  if (p == std::string::npos) return false;
  const auto c = line.find(':', p);
  if (c == std::string::npos) return false;
  const auto v = line.find_first_not_of(" \t", c + 1);
  return v != std::string::npos && line.compare(v, command.size() + 2, "\"" + command + "\"") == 0;
}

std::string jsonEscapeForOutput(const std::string &value) {
  std::string output;
  output.reserve(value.size() + 8);
  for (const unsigned char c : value) {
    if (c == '\\') output += "\\\\";
    else if (c == '"') output += "\\\"";
    else if (c == '\n') output += "\\n";
    else if (c == '\r') output += "\\r";
    else if (c == '\t') output += "\\t";
    else if (c >= 0x20) output.push_back(static_cast<char>(c));
  }
  return output;
}
}  // namespace

int main(int argc, char **argv) {
  std::string config_path = "/etc/bibit/bibit.conf";
  std::string wav_path;
  std::string direct_asr_wav_path;
  std::vector<std::string> kws_stress_wavs;
  int kws_stress_rounds = 1;
  int direct_chunk_seconds = 20;
  bool wav_realtime = false;
  bool wav_no_throttle = false;
  bool manual = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--config" && i + 1 < argc) config_path = argv[++i];
    else if (arg == "--wav" && i + 1 < argc) wav_path = argv[++i];
    else if (arg == "--asr-direct-wav" && i + 1 < argc) direct_asr_wav_path = argv[++i];
    else if (arg == "--asr-direct-chunk-seconds" && i + 1 < argc) {
      direct_chunk_seconds = std::clamp(std::atoi(argv[++i]), 5, 60);
    }
    else if (arg == "--kws-stress-wav" && i + 1 < argc) kws_stress_wavs.emplace_back(argv[++i]);
    else if (arg == "--kws-stress-rounds" && i + 1 < argc) {
      kws_stress_rounds = std::max(1, std::atoi(argv[++i]));
    }
    else if (arg == "--wav-realtime") wav_realtime = true;
    else if (arg == "--wav-no-throttle") wav_no_throttle = true;
    else if (arg == "--manual") manual = true;
    else if (arg == "--help") {
      std::cout << "Usage: bibit-audio [--config file] [--wav file] [--manual]\n"
                   "                   [--wav-realtime]\n"
                   "                   [--wav-no-throttle]\n"
                   "                   [--asr-direct-wav file] [--asr-direct-chunk-seconds n]\n"
                   "                   [--kws-stress-wav file ...] [--kws-stress-rounds n]\n";
      return 0;
    }
  }

  BibitConfig config = BibitConfig::load(config_path);
  const char *home = std::getenv("HOME");
  const std::string threshold_path =
      std::string(home ? home : "/tmp") + "/.config/bibit/thresholds.conf";
  const std::string asr_model_path =
      std::string(home ? home : "/tmp") + "/.config/bibit/asr-model.conf";
  const std::string switches_path =
      std::string(home ? home : "/tmp") + "/.config/bibit/switches.conf";
  config.apply(threshold_path);
  config.apply(asr_model_path);
  config.apply(switches_path);
  std::mutex done_mutex;
  std::condition_variable done_cv;
  bool done = false;
  uint64_t wake_count = 0;
  uint64_t cancel_count = 0;
  uint64_t kws_reset_count = 0;
  std::mutex config_mutex;
  SocketServer *server_ptr = nullptr;
  // 单线程 socket 事件循环内自增，无需加锁。
  uint64_t switch_request_seq = 0;
  VoicePipeline pipeline(config, [&](const std::string &event) {
    if (server_ptr) server_ptr->broadcast(event);
    const std::string event_name = jsonString(event, "event");
    // 切换真正生效后才把偏好落盘：失败的服务不再留下一个启动时必然失败的预设。
    if (event_name == "model_switched") {
      const std::string applied = jsonString(event, "to");
      const std::string applied_request_id = jsonString(event, "request_id");
      if (applied == "qwen3_asr_0.6b" || applied == "sensevoice") {
        bool persisted = false;
        {
          std::lock_guard<std::mutex> config_lk(config_mutex);
          config = asrModelConfig(config, applied);
          persisted = saveAsrModel(asr_model_path, applied);
        }
        if (!persisted && server_ptr) {
          server_ptr->broadcast(
              "{\"event\":\"error\",\"code\":\"ASR_MODEL_PERSIST_FAILED\",\"model\":\"" +
              jsonEscapeForOutput(applied) + "\",\"request_id\":\"" +
              jsonEscapeForOutput(applied_request_id) + "\"}");
        }
      }
    }
    // 开关偏好按「生效态」落盘：switches_settled 里的 wake/asr 就是结算时的生效
    // 开关。排队中、被更新意图取代、构建失败的打开都不会写进偏好，重启后不会
    // 带着一个从未达成的「打开」自己启动。
    if (event_name == "switches_settled") {
      bool wake = false;
      bool asr = false;
      if (jsonBoolValue(event, "wake", &wake) && jsonBoolValue(event, "asr", &asr)) {
        bool persisted = false;
        {
          std::lock_guard<std::mutex> config_lk(config_mutex);
          config.wake_enabled = wake;
          config.asr_enabled = asr;
          persisted = saveSwitches(switches_path, wake, asr);
        }
        if (!persisted && server_ptr) {
          server_ptr->broadcast(
              "{\"event\":\"error\",\"code\":\"SWITCH_PERSIST_FAILED\",\"request_id\":\"" +
              jsonEscapeForOutput(jsonString(event, "request_id")) + "\"}");
        }
      }
    }
    if (event_name == "wake_triggered") {
      std::lock_guard<std::mutex> lk(done_mutex);
      ++wake_count;
      done_cv.notify_all();
    } else if (event_name == "session_cancelled") {
      std::lock_guard<std::mutex> lk(done_mutex);
      ++cancel_count;
      done_cv.notify_all();
    } else if (event_name == "kws_reset") {
      std::lock_guard<std::mutex> lk(done_mutex);
      ++kws_reset_count;
      done_cv.notify_all();
    }
    if (event_name == "asr_final" || event_name == "session_exit" ||
        event_name == "error" || event_name == "input_finished") {
      std::lock_guard<std::mutex> lk(done_mutex);
      done = true;
      done_cv.notify_all();
    }
  });
  // 机器级独占资源（Qwen3 的 8063 llama-server、SpacemiT EP/TCM）是全局的：
  // 先占住单实例 socket 再初始化 ASR。否则绑不上的第二个实例照样会把后端拉起来，
  // 失败退出时还可能留下占着 8063 的孤儿进程，把正在服务的实例卡在
  // "qwen3 port already has an HTTP server" 上。
  const bool standalone_mode = !wav_path.empty() || !direct_asr_wav_path.empty() ||
                               !kws_stress_wavs.empty();
  const std::string socket_path = "/run/bibit/audio.sock";
  SocketServer server(socket_path, SocketServer::LineHandler{});
  if (!standalone_mode && !server.acquire()) {
    std::cerr << "another bibit-audio instance is already serving " << socket_path << std::endl;
    return 7;
  }
  if (!pipeline.initialize()) {
    std::cerr << "Bibit pipeline initialization failed" << std::endl;
    if (std::filesystem::is_regular_file(asr_model_path)) {
      const std::string failed_path = asr_model_path + ".failed";
      std::error_code ec;
      std::filesystem::remove(failed_path, ec);
      ec.clear();
      std::filesystem::rename(asr_model_path, failed_path, ec);
      if (!ec) {
        std::cerr << "ASR model override rolled back to the system default; "
                     "failed preset saved at " << failed_path << std::endl;
        return 3;
      }
    }
    return 2;
  }

  if (!direct_asr_wav_path.empty()) {
    WavData wav;
    try {
      wav = readWav(direct_asr_wav_path);
    } catch (const std::exception &e) {
      std::cerr << "direct ASR WAV input failed: " << e.what() << std::endl;
      return 6;
    }
    std::vector<float> mono16k;
    if (wav.sample_rate == 16000) {
      mono16k = std::move(wav.mono);
    } else {
      StreamingResampler resampler(wav.sample_rate, 16000);
      mono16k = resampler.process(wav.mono.data(), static_cast<int>(wav.mono.size()));
    }
    const size_t chunk_samples = static_cast<size_t>(direct_chunk_seconds) * 16000;
    std::string combined;
    int chunk_index = 0;
    int64_t total_decode_us = 0;
    for (size_t offset = 0; offset < mono16k.size(); offset += chunk_samples) {
      const int count = static_cast<int>(
          std::min(chunk_samples, mono16k.size() - offset));
      std::string text;
      std::string error;
      int64_t decode_us = 0;
      if (!pipeline.recognizeDirect(mono16k.data() + offset, count, &text,
                                    &decode_us, &error)) {
        std::cerr << "direct ASR failed at chunk " << chunk_index << ": " << error
                  << std::endl;
        return 8;
      }
      total_decode_us += decode_us;
      combined += text;
      const double audio_ms = static_cast<double>(count) / 16.0;
      const double rtf = static_cast<double>(decode_us) * 16000.0 /
                         (static_cast<double>(count) * 1000000.0);
      std::cout << "{\"event\":\"asr_direct_final\",\"model_type\":\""
                << jsonEscapeForOutput(config.asr_model_type) << "\",\"chunk_index\":"
                << chunk_index << ",\"start_ms\":" << offset / 16
                << ",\"audio_duration_ms\":" << audio_ms
                << ",\"decode_ms\":" << decode_us / 1000.0
                << ",\"rtf\":" << rtf << ",\"text\":\""
                << jsonEscapeForOutput(text) << "\"}" << std::endl;
      ++chunk_index;
    }
    const double duration_ms = static_cast<double>(mono16k.size()) / 16.0;
    const double total_rtf = mono16k.empty() ? 0.0 :
        static_cast<double>(total_decode_us) * 16000.0 /
        (static_cast<double>(mono16k.size()) * 1000000.0);
    std::cout << "{\"event\":\"asr_direct_summary\",\"model_type\":\""
              << jsonEscapeForOutput(config.asr_model_type) << "\",\"chunks\":"
              << chunk_index << ",\"audio_duration_ms\":" << duration_ms
              << ",\"decode_ms\":" << total_decode_us / 1000.0
              << ",\"rtf\":" << total_rtf << ",\"text\":\""
              << jsonEscapeForOutput(combined) << "\"}" << std::endl;
    return 0;
  }
  if (!pipeline.start()) {
    std::cerr << "cannot start Bibit pipeline worker" << std::endl;
    return 2;
  }

  if (!kws_stress_wavs.empty()) {
    std::vector<WavData> tests;
    try {
      tests.reserve(kws_stress_wavs.size());
      for (const std::string &path : kws_stress_wavs) tests.push_back(readWav(path));
    } catch (const std::exception &e) {
      std::cerr << "KWS stress input failed: " << e.what() << std::endl;
      pipeline.stop();
      return 6;
    }

    int trials = 0;
    int hits = 0;
    int cancel_failures = 0;
    constexpr int kChunk = 800;  // 50 ms at 16 kHz; close to live capture cadence.
    const std::vector<float> silence(kChunk, 0.0f);
    auto pushSilence = [&](int chunks) {
      for (int i = 0; i < chunks; ++i) {
        pipeline.pushAudio(silence.data(), static_cast<int>(silence.size()), 16000);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    };

    for (int round = 0; round < kws_stress_rounds; ++round) {
      for (size_t test_index = 0; test_index < tests.size(); ++test_index) {
        ++trials;
        uint64_t before_wake = 0;
        uint64_t before_cancel = 0;
        {
          std::lock_guard<std::mutex> lk(done_mutex);
          before_wake = wake_count;
          before_cancel = cancel_count;
        }

        // Preserve a short inter-utterance blank period, as with a user closing
        // the panel and speaking the next wake word shortly afterwards.
        pushSilence(6);
        const WavData &wav = tests[test_index];
        for (size_t i = 0; i < wav.mono.size(); i += kChunk) {
          const int n = static_cast<int>(std::min<size_t>(kChunk, wav.mono.size() - i));
          pipeline.pushAudio(wav.mono.data() + i, n, wav.sample_rate);
          pipeline.waitForAudioQueueBelow(16000);
        }
        pushSilence(12);
        pipeline.waitForAudioQueueBelow(0);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        bool woke = false;
        {
          std::unique_lock<std::mutex> lk(done_mutex);
          woke = done_cv.wait_for(lk, std::chrono::seconds(3),
                                  [&]() { return wake_count > before_wake; });
        }
        bool cancelled = false;
        if (woke) {
          ++hits;
          if (config.asr_enabled) {
            pipeline.cancel();
            std::unique_lock<std::mutex> lk(done_mutex);
            cancelled = done_cv.wait_for(lk, std::chrono::seconds(3),
                                         [&]() { return cancel_count > before_cancel; });
          } else {
            // KWS-only regression never enters an ASR session; resetting the
            // keyword stream is already done at the trigger site.
            cancelled = true;
          }
          if (!cancelled) ++cancel_failures;
        }
        std::cout << "{\"event\":\"kws_stress_trial\",\"round\":" << (round + 1)
                  << ",\"test_index\":" << (test_index + 1)
                  << ",\"file\":\"" << kws_stress_wavs[test_index]
                  << "\",\"wake\":" << (woke ? "true" : "false")
                  << ",\"cancelled\":" << (cancelled ? "true" : "false") << "}"
                  << std::endl;
        uint64_t before_reset = 0;
        {
          std::lock_guard<std::mutex> lk(done_mutex);
          before_reset = kws_reset_count;
        }
        pipeline.resetKwsForTest();
        std::unique_lock<std::mutex> reset_lk(done_mutex);
        done_cv.wait_for(reset_lk, std::chrono::seconds(2),
                         [&]() { return kws_reset_count > before_reset; });
      }
    }
    pipeline.stop();
    std::cout << "{\"event\":\"kws_stress_summary\",\"trials\":" << trials
              << ",\"hits\":" << hits << ",\"misses\":" << (trials - hits)
              << ",\"cancel_failures\":" << cancel_failures << "}" << std::endl;
    return hits == trials && cancel_failures == 0 ? 0 : 7;
  }

  if (!wav_path.empty()) {
    if (manual) pipeline.startManualAsr();
    const WavData wav = readWav(wav_path);
    constexpr int kChunk = 1600;
    for (size_t i = 0; i < wav.mono.size(); i += kChunk) {
      const int n = static_cast<int>(std::min<size_t>(kChunk, wav.mono.size() - i));
      pipeline.pushAudio(wav.mono.data() + i, n, wav.sample_rate);
      if (wav_realtime) {
        const auto delay_us = static_cast<int64_t>(n) * 1000000 / wav.sample_rate;
        std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
      } else if (!wav_no_throttle) {
        // --wav-no-throttle pushes as fast as the loop runs, which is how the
        // capture-queue ceiling is exercised on SpacemiT K3: the producer outruns
        // the worker and the overflow path (drop + discontinuity reset) fires.
        pipeline.waitForAudioQueueBelow(16000 * 2);
      }
    }
    {
      std::lock_guard<std::mutex> done_lk(done_mutex);
      done = false;
    }
    pipeline.finishInput();
    std::unique_lock<std::mutex> lk(done_mutex);
    done_cv.wait_for(lk, std::chrono::seconds(90), [&]() { return done; });
    // The pipeline event callback also takes done_mutex.  Release it before
    // joining the worker so an in-flight final/error event cannot deadlock
    // the offline --wav path during shutdown.
    const bool completed = done;
    lk.unlock();
    pipeline.stop();
    return completed ? 0 : 3;
  }

  // 整机唯一实例：全局固定 socket 路径（/run/bibit 由 tmpfiles.d 创建），
  // 单实例由 bind 前的 connect 探测保证，不再 per-user。
  std::mutex capture_mutex;
  std::unique_ptr<PipeWireCapture> capture;
  // 控制面：会落盘、碰设备或碰后端的命令都在这里执行，Socket I/O 线程只投递。
  // 这样任何一条慢命令都不会推迟新命令的接收、get_status 回执与事件广播。
  std::deque<std::string> control_queue;
  std::mutex control_mutex;
  std::condition_variable control_cv;
  bool control_running = true;
  const auto is_control_command = [](const std::string &line) {
    return hasCommand(line, "set_switches") || hasCommand(line, "set_thresholds") ||
           hasCommand(line, "set_kws_debug") || hasCommand(line, "set_asr_model");
  };
  const auto run_control_command = [&](const std::string &line) {
    if (hasCommand(line, "set_switches")) {
      // 字段级意图：原样把给出的字段交给流水线，未给出的字段由服务端沿用当前
      // 生效值。本地镜像只用于「这次命令之后要不要开采集」，不再参与补全——
      // 镜像一旦与生效态分叉（例如上次打开失败），补全就会把开关再点一次。
      const bool has_wake = jsonHas(line, "wake");
      const bool has_asr = jsonHas(line, "asr");
      const std::string request_id = jsonString(line, "request_id");
      bool wake = false;
      bool asr = false;
      // 原始 socket 边界校验：至少给一个字段，且每个给出的字段必须是 JSON boolean。
      bool valid = has_wake || has_asr;
      if (valid && has_wake && !jsonBoolValue(line, "wake", &wake)) valid = false;
      if (valid && has_asr && !jsonBoolValue(line, "asr", &asr)) valid = false;
      if (!valid) {
        if (server_ptr) {
          server_ptr->broadcast(
              "{\"event\":\"error\",\"code\":\"BAD_COMMAND\",\"request_id\":\"" +
              jsonEscapeForOutput(request_id) +
              "\",\"message\":\"set_switches requires at least one boolean wake/asr field\"}");
        }
        return;
      }
      VoicePipeline::SwitchRequest request;
      request.wake_set = has_wake;
      request.wake = wake;
      request.asr_set = has_asr;
      request.asr = asr;
      request.request_id = request_id;
      pipeline.setSwitches(request);
      // 开关偏好不在受理时落盘：打开识别可能几十秒后才达成（甚至被更新的意图
      // 取代、或构建失败），落盘要等 switches_settled 给出的生效态，否则重启后
      // 会带着一个从未达成的「打开」自己启动。
      bool effective_wake = false;
      bool effective_asr = false;
      {
        std::lock_guard<std::mutex> config_lk(config_mutex);
        if (has_wake) config.wake_enabled = wake;
        if (has_asr) config.asr_enabled = asr;
        effective_wake = config.wake_enabled;
        effective_asr = config.asr_enabled;
      }
      {
        std::lock_guard<std::mutex> lk(capture_mutex);
        if (capture) {
          if (effective_wake || effective_asr) {
            if (!capture->start() && server_ptr) {
              pipeline.setAudioReady(false);
              server_ptr->broadcast(
                  "{\"event\":\"error\",\"code\":\"AUDIO_OPEN_FAILED\",\"request_id\":\"" +
                  jsonEscapeForOutput(request_id) + "\"}");
            }
          } else {
            capture->stop();
            pipeline.setAudioReady(false);
            if (server_ptr) {
              server_ptr->broadcast("{\"event\":\"audio_state\",\"message\":\"audio_closed\"}");
            }
          }
        }
      }
      if (server_ptr) {
        // 回执放在设备起停之后：switches_applied 表示开关态与采集已按这次命令调整，
        // 不代表识别后端已经就绪，也不代表这次意图已经达成——打开识别的终态在
        // switches_settled（或失败时那条 error）里给（协议见 docs/protocol.md §4.4）。
        server_ptr->broadcast(
            "{\"event\":\"switches_applied\",\"request_id\":\"" +
            jsonEscapeForOutput(request_id) +
            "\",\"wake\":" + (effective_wake ? "true" : "false") +
            ",\"asr\":" + (effective_asr ? "true" : "false") + "}");
      }
    } else if (hasCommand(line, "set_thresholds")) {
      BibitConfig updated;
      {
        std::lock_guard<std::mutex> config_lk(config_mutex);
        config.kws_threshold = std::clamp(
            jsonFloat(line, "kws_threshold", config.kws_threshold), 0.05f, 0.95f);
        config.kws_score = std::clamp(
            jsonFloat(line, "kws_score", config.kws_score), 0.1f, 5.0f);
        config.vad_threshold = std::clamp(
            jsonFloat(line, "vad_threshold", config.vad_threshold), 0.05f, 0.95f);
        config.min_silence = std::clamp(
            jsonFloat(line, "min_silence", config.min_silence), 0.15f, 3.0f);
        config.wake_guard_ms = std::clamp(static_cast<int>(std::lround(
            jsonFloat(line, "wake_guard_ms", config.wake_guard_ms))), 0, 600);
        updated = config;
      }
      pipeline.setThresholds(updated.kws_threshold, updated.kws_score,
                             updated.vad_threshold, updated.min_silence,
                             updated.wake_guard_ms);
      if (!saveThresholds(threshold_path, updated) && server_ptr) {
        server_ptr->broadcast("{\"event\":\"error\",\"code\":\"THRESHOLD_PERSIST_FAILED\"}");
      }
    } else if (hasCommand(line, "set_kws_debug")) {
      BibitConfig updated;
      {
        std::lock_guard<std::mutex> config_lk(config_mutex);
        config.kws_debug_enabled = jsonBool(
            line, "enabled", config.kws_debug_enabled);
        updated = config;
      }
      pipeline.setKwsDebug(updated.kws_debug_enabled);
      if (!saveThresholds(threshold_path, updated) && server_ptr) {
        server_ptr->broadcast("{\"event\":\"error\",\"code\":\"KWS_DEBUG_PERSIST_FAILED\"}");
      }
    } else if (hasCommand(line, "set_asr_model")) {
      const std::string requested = jsonString(line, "model");
      std::string request_id = jsonString(line, "request_id");
      if (request_id.empty()) {
        request_id = "sw-" + std::to_string(++switch_request_seq);
      }
      // 结果一律广播：客户端发命令用的是「写完即关」的连接（GUI 侧 Reply::Sent），
      // 只回请求方的错误到不了订阅事件的界面；request_id 负责把结果对应到发起方。
      if (requested != "qwen3_asr_0.6b" && requested != "sensevoice") {
        if (server_ptr) server_ptr->broadcast("{\"event\":\"error\",\"code\":\"UNSUPPORTED_ASR_MODEL\",\"request_id\":\"" + jsonEscapeForOutput(request_id) + "\",\"to\":\"" + jsonEscapeForOutput(requested) + "\"}");
      } else if (!modelFilesAvailable(requested)) {
        if (server_ptr) server_ptr->broadcast("{\"event\":\"error\",\"code\":\"ASR_MODEL_FILES_MISSING\",\"request_id\":\"" + jsonEscapeForOutput(request_id) + "\",\"to\":\"" + jsonEscapeForOutput(requested) + "\",\"model\":\"" + jsonEscapeForOutput(requested) + "\"}");
      } else if (pipeline.effectiveAsrModel() == requested &&
                 !pipeline.modelSwitching()) {
        // 幂等：已经是生效模型且没有在途切换，按成功 no-op 回执。
        // 有在途切换时不能走 no-op，否则用户「切回当前引擎」的意图会被丢掉。
        if (server_ptr) server_ptr->broadcast(
            "{\"event\":\"model_switched\",\"model\":\"" + requested +
            "\",\"request_id\":\"" + jsonEscapeForOutput(request_id) + "\",\"from\":\"" + requested +
            "\",\"to\":\"" + requested + "\",\"duration_ms\":0,\"noop\":true}");
      } else {
        BibitConfig target;
        {
          std::lock_guard<std::mutex> config_lk(config_mutex);
          target = asrModelConfig(config, requested);
        }
        // 切换开始时的生效模型：config_ 只在提交点变更，这里读到的就是 from。
        const std::string from = pipeline.effectiveAsrModel();
        if (pipeline.setAsrModel(target, request_id)) {
          if (server_ptr) server_ptr->broadcast(
              "{\"event\":\"model_switching\",\"request_id\":\"" + jsonEscapeForOutput(request_id) +
              "\",\"from\":\"" + jsonEscapeForOutput(from) +
              "\",\"to\":\"" + requested + "\",\"model\":\"" + requested + "\"}");
        } else {
          // 切换在途：仍然回 BUSY，保持既有消费者的重试语义；同时意图已记入
          // desired，提交点之后自动收敛到最新目标（客户端不重试也成立）。
          if (server_ptr) server_ptr->broadcast(
              "{\"event\":\"error\",\"code\":\"ASR_MODEL_SWITCH_BUSY\",\"request_id\":\"" +
              jsonEscapeForOutput(request_id) + "\",\"to\":\"" + requested + "\",\"switching\":" +
              pipeline.switchSnapshotJson() + "}");
        }
      }
    }
  };

  server.setHandler([&](int fd, const std::string &line) {
    if (is_control_command(line)) {
      {
        std::lock_guard<std::mutex> lk(control_mutex);
        control_queue.push_back(line);
      }
      control_cv.notify_one();
      return;
    }
    if (hasCommand(line, "start_manual_asr")) {
      pipeline.startManualAsr();
    } else if (hasCommand(line, "cancel")) {
      const uint64_t requested = jsonUint(line, "session_id", 0);
      if (requested != 0 &&
          (!pipeline.sessionActive() ||
           pipeline.activeSessionId() != requested)) {
        if (server_ptr) {
          server_ptr->sendTo(fd,
              "{\"event\":\"error\",\"code\":\"SESSION_MISMATCH\",\"session_id\":" +
              std::to_string(requested) + ",\"active_session_id\":" +
              std::to_string(pipeline.activeSessionId()) + "}");
        }
      } else {
        pipeline.cancel();
      }
    } else if (hasCommand(line, "get_status")) {
      if (server_ptr) {
        server_ptr->sendTo(fd, pipeline.statusJson(true));
        server_ptr->sendTo(fd, pipeline.parametersJson());
      }
    } else {
      if (server_ptr) server_ptr->sendTo(fd, "{\"event\":\"error\",\"code\":\"BAD_COMMAND\"}");
    }
  });
  server_ptr = &server;
  if (!server.start()) {
    std::cerr << "cannot start socket server at " << socket_path << std::endl;
    return 4;
  }

  capture = std::make_unique<PipeWireCapture>(config.audio_source, config.input_rate, config.input_channels,
      [&](const float *p, int n, int rate) { pipeline.pushAudio(p, n, rate); },
      [&](const std::string &state) {
        std::cerr << state << std::endl;
        bool audio_ready = false;
        bool has_audio_state = false;
        if (state.rfind("audio_opened ", 0) == 0) {
          audio_ready = state.find("format=s16") != std::string::npos;
          has_audio_state = true;
        } else if (state.rfind("audio_error", 0) == 0 ||
                   state.find("pipewire_state=error") != std::string::npos ||
                   state.find("pipewire_state=unconnected") != std::string::npos) {
          has_audio_state = true;
        }
        const bool readiness_changed =
            has_audio_state && pipeline.setAudioReady(audio_ready);
        server.broadcast("{\"event\":\"audio_state\",\"message\":\"" + state + "\"}");
        if (readiness_changed) server.broadcast(pipeline.statusJson());
      });
  if (config.wake_enabled || config.asr_enabled) {
    if (!capture->start()) {
      std::cerr << "cannot start PipeWire capture" << std::endl;
      return 5;
    }
  } else {
    server.broadcast("{\"event\":\"audio_state\",\"message\":\"audio_closed\"}");
  }

  std::thread control_thread([&]() {
    while (true) {
      std::string line;
      {
        std::unique_lock<std::mutex> lk(control_mutex);
        control_cv.wait(lk, [&]() { return !control_queue.empty() || !control_running; });
        if (!control_running) return;
        line = std::move(control_queue.front());
        control_queue.pop_front();
      }
      run_control_command(line);
    }
  });

  std::signal(SIGINT, onSignal);
  std::signal(SIGTERM, onSignal);
  std::cout << "Bibit service ready: " << socket_path << std::endl;
  while (!g_stop) std::this_thread::sleep_for(std::chrono::milliseconds(200));
  {
    std::lock_guard<std::mutex> lk(capture_mutex);
    capture->stop();
  }
  server.stop();
  {
    std::lock_guard<std::mutex> lk(control_mutex);
    control_running = false;
  }
  control_cv.notify_one();
  if (control_thread.joinable()) control_thread.join();
  pipeline.stop();
  return 0;
}
