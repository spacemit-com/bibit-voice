#include "qwen3_asr.h"

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <netdb.h>
#include <signal.h>
#include <sched.h>
#include <spawn.h>
#include <sstream>
#include <string_view>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

extern char **environ;

namespace {
struct HttpResponse {
  int status = 0;
  std::string body;
};

bool sendAll(int fd, const char *data, size_t size) {
  while (size > 0) {
#ifdef MSG_NOSIGNAL
    const ssize_t n = ::send(fd, data, size, MSG_NOSIGNAL);
#else
    const ssize_t n = ::send(fd, data, size, 0);
#endif
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return false;
    data += n;
    size -= static_cast<size_t>(n);
  }
  return true;
}

bool decodeChunked(std::string_view input, std::string *output) {
  output->clear();
  size_t pos = 0;
  while (pos < input.size()) {
    const size_t line_end = input.find("\r\n", pos);
    if (line_end == std::string_view::npos) return false;
    const std::string length_text(input.substr(pos, line_end - pos));
    char *end = nullptr;
    const unsigned long length = std::strtoul(length_text.c_str(), &end, 16);
    if (end == length_text.c_str()) return false;
    pos = line_end + 2;
    if (length == 0) return true;
    if (length > input.size() - pos) return false;
    output->append(input.data() + pos, static_cast<size_t>(length));
    pos += static_cast<size_t>(length);
    if (pos + 2 > input.size() || input.substr(pos, 2) != "\r\n") return false;
    pos += 2;
  }
  return false;
}

bool httpRequest(const std::string &host, int port, const std::string &method,
                 const std::string &path, const std::string &body,
                 int timeout_ms, HttpResponse *response, std::string *error) {
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo *addresses = nullptr;
  const std::string port_text = std::to_string(port);
  const int gai = ::getaddrinfo(host.c_str(), port_text.c_str(), &hints, &addresses);
  if (gai != 0) {
    if (error) *error = std::string("getaddrinfo: ") + gai_strerror(gai);
    return false;
  }

  int fd = -1;
  for (addrinfo *a = addresses; a; a = a->ai_next) {
    fd = ::socket(a->ai_family, a->ai_socktype, a->ai_protocol);
    if (fd < 0) continue;
    timeval tv{};
    const int bounded_timeout = std::clamp(timeout_ms, 100, 300000);
    tv.tv_sec = bounded_timeout / 1000;
    tv.tv_usec = (bounded_timeout % 1000) * 1000;
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    if (::connect(fd, a->ai_addr, a->ai_addrlen) == 0) break;
    ::close(fd);
    fd = -1;
  }
  ::freeaddrinfo(addresses);
  if (fd < 0) {
    if (error) *error = "connect failed: " + std::string(std::strerror(errno));
    return false;
  }

  std::ostringstream request;
  request << method << ' ' << path << " HTTP/1.1\r\n"
          << "Host: " << host << ':' << port << "\r\n"
          << "Connection: close\r\n";
  if (method == "POST") {
    request << "Content-Type: application/json\r\n"
            << "Content-Length: " << body.size() << "\r\n";
  }
  request << "\r\n" << body;
  const std::string wire = request.str();
  if (!sendAll(fd, wire.data(), wire.size())) {
    if (error) *error = "HTTP send failed: " + std::string(std::strerror(errno));
    ::close(fd);
    return false;
  }

  std::string raw;
  char buffer[16384];
  while (true) {
    const ssize_t n = ::recv(fd, buffer, sizeof(buffer), 0);
    if (n < 0 && errno == EINTR) continue;
    if (n < 0) {
      if (error) *error = "HTTP receive failed: " + std::string(std::strerror(errno));
      ::close(fd);
      return false;
    }
    if (n == 0) break;
    raw.append(buffer, static_cast<size_t>(n));
  }
  ::close(fd);

  const size_t header_end = raw.find("\r\n\r\n");
  if (header_end == std::string::npos) {
    if (error) *error = "invalid HTTP response";
    return false;
  }
  const size_t status_begin = raw.find(' ');
  if (status_begin == std::string::npos) return false;
  try {
    response->status = std::stoi(raw.substr(status_begin + 1, 3));
  } catch (...) {
    if (error) *error = "invalid HTTP status";
    return false;
  }
  const std::string headers = raw.substr(0, header_end);
  const std::string payload = raw.substr(header_end + 4);
  std::string lower_headers = headers;
  std::transform(lower_headers.begin(), lower_headers.end(), lower_headers.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (lower_headers.find("transfer-encoding: chunked") != std::string::npos) {
    if (!decodeChunked(payload, &response->body)) {
      if (error) *error = "invalid chunked HTTP body";
      return false;
    }
  } else {
    response->body = payload;
  }
  return true;
}

void appendLe16(std::string *out, uint16_t value) {
  out->push_back(static_cast<char>(value & 0xff));
  out->push_back(static_cast<char>((value >> 8) & 0xff));
}

void appendLe32(std::string *out, uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8) {
    out->push_back(static_cast<char>((value >> shift) & 0xff));
  }
}

std::string makeWav(const float *samples, int count) {
  const uint32_t data_size = static_cast<uint32_t>(std::max(0, count)) * 2u;
  std::string wav;
  wav.reserve(44u + data_size);
  wav.append("RIFF", 4);
  appendLe32(&wav, 36u + data_size);
  wav.append("WAVEfmt ", 8);
  appendLe32(&wav, 16);
  appendLe16(&wav, 1);
  appendLe16(&wav, 1);
  appendLe32(&wav, 16000);
  appendLe32(&wav, 32000);
  appendLe16(&wav, 2);
  appendLe16(&wav, 16);
  wav.append("data", 4);
  appendLe32(&wav, data_size);
  for (int i = 0; i < count; ++i) {
    const float value = std::clamp(samples[i], -1.0f, 1.0f);
    const int16_t pcm = static_cast<int16_t>(std::lround(value * 32767.0f));
    appendLe16(&wav, static_cast<uint16_t>(pcm));
  }
  return wav;
}

std::string base64Encode(std::string_view input) {
  static constexpr char kAlphabet[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string output;
  output.reserve(((input.size() + 2) / 3) * 4);
  for (size_t i = 0; i < input.size(); i += 3) {
    const uint32_t a = static_cast<unsigned char>(input[i]);
    const uint32_t b = i + 1 < input.size()
        ? static_cast<unsigned char>(input[i + 1]) : 0;
    const uint32_t c = i + 2 < input.size()
        ? static_cast<unsigned char>(input[i + 2]) : 0;
    const uint32_t value = (a << 16) | (b << 8) | c;
    output.push_back(kAlphabet[(value >> 18) & 63]);
    output.push_back(kAlphabet[(value >> 12) & 63]);
    output.push_back(i + 1 < input.size() ? kAlphabet[(value >> 6) & 63] : '=');
    output.push_back(i + 2 < input.size() ? kAlphabet[value & 63] : '=');
  }
  return output;
}

std::string jsonEscape(std::string_view value) {
  std::string output;
  output.reserve(value.size() + 16);
  for (unsigned char c : value) {
    switch (c) {
      case '\\': output += "\\\\"; break;
      case '"': output += "\\\""; break;
      case '\n': output += "\\n"; break;
      case '\r': output += "\\r"; break;
      case '\t': output += "\\t"; break;
      default:
        if (c < 0x20) {
          const char hex[] = "0123456789abcdef";
          output += "\\u00";
          output.push_back(hex[c >> 4]);
          output.push_back(hex[c & 15]);
        } else {
          output.push_back(static_cast<char>(c));
        }
    }
  }
  return output;
}

void appendUtf8(uint32_t codepoint, std::string *output) {
  if (codepoint <= 0x7f) {
    output->push_back(static_cast<char>(codepoint));
  } else if (codepoint <= 0x7ff) {
    output->push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
    output->push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
  } else if (codepoint <= 0xffff) {
    output->push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
    output->push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
    output->push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
  } else {
    output->push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
    output->push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
    output->push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
    output->push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
  }
}

int hexValue(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

bool readHex4(const std::string &input, size_t pos, uint32_t *value) {
  if (pos + 4 > input.size()) return false;
  *value = 0;
  for (size_t i = 0; i < 4; ++i) {
    const int h = hexValue(input[pos + i]);
    if (h < 0) return false;
    *value = (*value << 4) | static_cast<uint32_t>(h);
  }
  return true;
}

bool extractJsonString(const std::string &json, const std::string &key,
                       size_t begin, std::string *value) {
  const std::string needle = "\"" + key + "\"";
  size_t pos = json.find(needle, begin);
  if (pos == std::string::npos) return false;
  pos = json.find(':', pos + needle.size());
  if (pos == std::string::npos) return false;
  pos = json.find_first_not_of(" \t\r\n", pos + 1);
  if (pos == std::string::npos || json[pos] != '"') return false;
  ++pos;
  value->clear();
  while (pos < json.size()) {
    const char c = json[pos++];
    if (c == '"') return true;
    if (c != '\\') {
      value->push_back(c);
      continue;
    }
    if (pos >= json.size()) return false;
    const char escaped = json[pos++];
    switch (escaped) {
      case '"': value->push_back('"'); break;
      case '\\': value->push_back('\\'); break;
      case '/': value->push_back('/'); break;
      case 'b': value->push_back('\b'); break;
      case 'f': value->push_back('\f'); break;
      case 'n': value->push_back('\n'); break;
      case 'r': value->push_back('\r'); break;
      case 't': value->push_back('\t'); break;
      case 'u': {
        uint32_t first = 0;
        if (!readHex4(json, pos, &first)) return false;
        pos += 4;
        if (first >= 0xd800 && first <= 0xdbff && pos + 6 <= json.size() &&
            json[pos] == '\\' && json[pos + 1] == 'u') {
          uint32_t second = 0;
          if (readHex4(json, pos + 2, &second) && second >= 0xdc00 && second <= 0xdfff) {
            first = 0x10000 + ((first - 0xd800) << 10) + (second - 0xdc00);
            pos += 6;
          }
        }
        appendUtf8(first, value);
        break;
      }
      default: return false;
    }
  }
  return false;
}

std::string normalizeLanguage(const std::string &value) {
  std::istringstream stream(value.empty() ? "auto" : value);
  std::string item;
  std::string result;
  while (std::getline(stream, item, ',')) {
    item.erase(item.begin(), std::find_if(item.begin(), item.end(),
        [](unsigned char c) { return !std::isspace(c); }));
    item.erase(std::find_if(item.rbegin(), item.rend(),
        [](unsigned char c) { return !std::isspace(c); }).base(), item.end());
    std::string lower = item;
    std::transform(lower.begin(), lower.end(), lower.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lower == "auto") return "auto";
    if (lower == "zh" || lower == "cmn" || lower == "cn" || lower == "zh-cn") item = "Chinese";
    else if (lower == "en" || lower == "en-us" || lower == "en-gb") item = "English";
    else if (lower == "yue") item = "Cantonese";
    else if (lower == "ja" || lower == "jp") item = "Japanese";
    else if (lower == "ko") item = "Korean";
    if (!item.empty()) {
      if (!result.empty()) result += ',';
      result += item;
    }
  }
  return result.empty() ? "auto" : result;
}

void trim(std::string *value) {
  value->erase(value->begin(), std::find_if(value->begin(), value->end(),
      [](unsigned char c) { return !std::isspace(c); }));
  value->erase(std::find_if(value->rbegin(), value->rend(),
      [](unsigned char c) { return !std::isspace(c); }).base(), value->end());
}

bool findJsonObjectEnd(const std::string &json, size_t begin, size_t *end) {
  if (begin >= json.size() || json[begin] != '{') return false;
  int depth = 0;
  bool in_string = false;
  bool escaped = false;
  for (size_t i = begin; i < json.size(); ++i) {
    const char c = json[i];
    if (in_string) {
      if (escaped) escaped = false;
      else if (c == '\\') escaped = true;
      else if (c == '"') in_string = false;
      continue;
    }
    if (c == '"') {
      in_string = true;
    } else if (c == '{') {
      ++depth;
    } else if (c == '}' && --depth == 0) {
      *end = i;
      return true;
    }
  }
  return false;
}

bool findMemberObject(const std::string &json, const std::string &key,
                      size_t search_begin, size_t search_end,
                      size_t *object_begin, size_t *object_end) {
  const std::string needle = "\"" + key + "\"";
  const size_t key_pos = json.find(needle, search_begin);
  if (key_pos == std::string::npos || key_pos >= search_end) return false;
  const size_t colon = json.find(':', key_pos + needle.size());
  if (colon == std::string::npos || colon >= search_end) return false;
  const size_t begin = json.find_first_not_of(" \t\r\n", colon + 1);
  if (begin == std::string::npos || begin >= search_end || json[begin] != '{') return false;
  size_t end = 0;
  if (!findJsonObjectEnd(json, begin, &end) || end >= search_end) return false;
  *object_begin = begin;
  *object_end = end;
  return true;
}

std::string pathBasename(const std::string &path) {
  const size_t slash = path.find_last_of('/');
  return slash == std::string::npos ? path : path.substr(slash + 1);
}

bool createSymlink(const std::string &target, const std::string &link,
                   std::string *error) {
  if (::symlink(target.c_str(), link.c_str()) == 0) return true;
  if (error) *error = "cannot link Qwen3-ASR asset " + link + ": " +
      std::string(std::strerror(errno));
  return false;
}

std::string stripNativePrefix(std::string text) {
  constexpr std::string_view tag = "<asr_text>";
  const size_t tag_pos = text.find(tag);
  if (tag_pos != std::string::npos) text.erase(0, tag_pos + tag.size());
  trim(&text);
  return text;
}
}  // namespace

Qwen3AsrClient::Qwen3AsrClient(const BibitConfig &config) : config_(config) {}

Qwen3AsrClient::~Qwen3AsrClient() { shutdown(); }

void Qwen3AsrClient::startAffinityGuard() {
  stopAffinityGuard(nullptr);
  if (server_pid_ <= 0) return;

  cpu_set_t target{};
  CPU_ZERO(&target);
  std::string affinity = config_.qwen3_ep_affinity;
  std::replace(affinity.begin(), affinity.end(), ';', ',');
  std::istringstream values(affinity);
  std::string value;
  int target_count = 0;
  while (std::getline(values, value, ',')) {
    try {
      const int cpu = std::stoi(value);
      if (cpu >= 8 && cpu <= 15 && !CPU_ISSET(cpu, &target)) {
        CPU_SET(cpu, &target);
        ++target_count;
      }
    } catch (...) {
    }
  }
  if (target_count == 0) {
    std::cerr << "Qwen3-ASR affinity_guard disabled: invalid A100 affinity="
              << config_.qwen3_ep_affinity << std::endl;
    return;
  }

  affinity_guard_stop_ = false;
  affinity_guard_ready_ = false;
  affinity_guard_pinned_ = 0;
  affinity_guard_errors_ = 0;
  const pid_t process = static_cast<pid_t>(server_pid_);
  affinity_guard_thread_ = std::thread([this, process, target]() {
    const std::string task_dir = "/proc/" + std::to_string(process) + "/task";
    while (!affinity_guard_stop_) {
      DIR *directory = ::opendir(task_dir.c_str());
      if (!directory) {
        if (errno != ENOENT && errno != ESRCH) ++affinity_guard_errors_;
        affinity_guard_ready_ = true;
        break;
      }
      while (dirent *entry = ::readdir(directory)) {
        if (entry->d_name[0] < '0' || entry->d_name[0] > '9') continue;
        char *end = nullptr;
        const long parsed = std::strtol(entry->d_name, &end, 10);
        if (!end || *end != '\0' || parsed <= 0) continue;
        const pid_t tid = static_cast<pid_t>(parsed);
        cpu_set_t current{};
        if (::sched_getaffinity(tid, sizeof(current), &current) != 0) continue;
        bool has_a100 = false;
        bool has_other = false;
        bool outside_target = false;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
          if (!CPU_ISSET(cpu, &current)) continue;
          if (cpu >= 8 && cpu <= 15) {
            has_a100 = true;
            if (!CPU_ISSET(cpu, &target)) outside_target = true;
          } else {
            has_other = true;
          }
        }
        if (!has_a100 || has_other || !outside_target) continue;
        if (::sched_setaffinity(tid, sizeof(target), &target) == 0) {
          ++affinity_guard_pinned_;
        } else if (errno != ESRCH) {
          ++affinity_guard_errors_;
        }
      }
      ::closedir(directory);
      affinity_guard_ready_ = true;
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  });
  while (!affinity_guard_ready_ && affinity_guard_thread_.joinable()) {
    std::this_thread::yield();
  }
}

void Qwen3AsrClient::stopAffinityGuard(const char *phase) {
  affinity_guard_stop_ = true;
  const bool was_running = affinity_guard_thread_.joinable();
  if (was_running) affinity_guard_thread_.join();
  if (!was_running) return;
  if (phase && (affinity_guard_pinned_ > 0 || affinity_guard_errors_ > 0)) {
    std::cerr << "Qwen3-ASR affinity_guard phase=" << phase
              << " pinned_threads=" << affinity_guard_pinned_.load()
              << " errors=" << affinity_guard_errors_.load()
              << " target=" << config_.qwen3_ep_affinity << std::endl;
  }
}

bool Qwen3AsrClient::prepareSmtConfig(std::string *error) {
  cleanupSmtConfig();
  const std::string source_path = config_.qwen3_model_dir + "/config.json";
  std::ifstream source(source_path);
  if (!source) {
    if (error) *error = "cannot read Qwen3-ASR config: " + source_path;
    return false;
  }
  std::string json((std::istreambuf_iterator<char>(source)),
                   std::istreambuf_iterator<char>());

  size_t audio_begin = 0;
  size_t audio_end = 0;
  if (!findMemberObject(json, "audio_model", 0, json.size(),
                        &audio_begin, &audio_end)) {
    if (error) *error = "Qwen3-ASR config has no audio_model object";
    return false;
  }

  const int ep_threads = std::clamp(config_.asr_num_threads, 1, 2);
  std::string ep_affinity = config_.qwen3_ep_affinity;
  std::replace(ep_affinity.begin(), ep_affinity.end(), ',', ';');
  const std::string ep_config =
      "{\n      \"spacemit_ep_intra_thread_num\": " +
      std::to_string(ep_threads) +
      ",\n      \"spacemit_ep_inter_thread_num\": 1,\n"
      "      \"spacemit_ep_intra_thread_affinity\": \"" +
      jsonEscape(ep_affinity) + "\"\n    }";

  size_t ep_begin = 0;
  size_t ep_end = 0;
  if (findMemberObject(json, "ep_config", audio_begin + 1, audio_end,
                       &ep_begin, &ep_end)) {
    json.replace(ep_begin, ep_end - ep_begin + 1, ep_config);
  } else {
    json.insert(audio_end, ",\n    \"ep_config\": " + ep_config);
  }

  char directory_template[] = "/tmp/bibit-qwen3-XXXXXX";
  char *directory = ::mkdtemp(directory_template);
  if (!directory) {
    if (error) *error = "cannot create Qwen3-ASR runtime config directory: " +
        std::string(std::strerror(errno));
    return false;
  }
  runtime_config_dir_ = directory;
  const std::string runtime_config = runtime_config_dir_ + "/config.json";
  std::ofstream output(runtime_config, std::ios::binary | std::ios::trunc);
  if (!output || !(output << json)) {
    if (error) *error = "cannot write Qwen3-ASR runtime config: " + runtime_config;
    cleanupSmtConfig();
    return false;
  }
  output.close();
  if (!output) {
    if (error) *error = "cannot close Qwen3-ASR runtime config: " + runtime_config;
    cleanupSmtConfig();
    return false;
  }

  const std::vector<std::string> assets = {
      "Qwen3-ASR-0.6B-encoder-frontend.dynq.onnx",
      "Qwen3-ASR-0.6B-encoder-backend.dynq.onnx"};
  for (const std::string &asset : assets) {
    if (!createSymlink(config_.qwen3_model_dir + "/" + asset,
                       runtime_config_dir_ + "/" + asset, error)) {
      cleanupSmtConfig();
      return false;
    }
  }
  const std::string text_model = pathBasename(config_.asr_model);
  if (!createSymlink(config_.asr_model, runtime_config_dir_ + "/" + text_model,
                     error)) {
    cleanupSmtConfig();
    return false;
  }
  std::cerr << "Qwen3-ASR SMT runtime config=" << runtime_config
            << " ep_threads=" << ep_threads
            << " ep_affinity=" << ep_affinity << std::endl;
  return true;
}

void Qwen3AsrClient::cleanupSmtConfig() {
  if (runtime_config_dir_.empty()) return;
  const std::vector<std::string> files = {
      "config.json",
      "Qwen3-ASR-0.6B-encoder-frontend.dynq.onnx",
      "Qwen3-ASR-0.6B-encoder-backend.dynq.onnx",
      pathBasename(config_.asr_model)};
  for (const std::string &file : files) {
    ::unlink((runtime_config_dir_ + "/" + file).c_str());
  }
  ::rmdir(runtime_config_dir_.c_str());
  runtime_config_dir_.clear();
}

bool Qwen3AsrClient::startServer(std::string *error) {
  HttpResponse existing;
  std::string ignored;
  if (httpRequest(config_.qwen3_server_host, config_.qwen3_server_port, "GET",
                  "/health", "", 500, &existing, &ignored)) {
    if (error) *error = "qwen3 port already has an HTTP server; stop the old llama-server first";
    return false;
  }

  const int qwen_threads = std::clamp(config_.asr_num_threads, 1, 2);
  const int qwen_batch_threads = std::clamp(config_.qwen3_thread_batch, 1, 2);
  const std::vector<std::string> server_args = {
      config_.qwen3_server_bin,
      "-m", config_.asr_model,
      "--host", config_.qwen3_server_host,
      "--port", std::to_string(config_.qwen3_server_port),
      "-t", std::to_string(qwen_threads),
      "-tb", std::to_string(qwen_batch_threads),
      "-c", std::to_string(std::clamp(config_.qwen3_ctx_size, 1024, 65536)),
      "--media-backend", "smt",
      "--smt-config-dir", runtime_config_dir_,
      "--slot-save-path", "/tmp",
      "--warmup"};
  const std::vector<std::string> &args = server_args;
  std::vector<char *> argv;
  argv.reserve(args.size() + 1);
  for (const std::string &arg : args) argv.push_back(const_cast<char *>(arg.c_str()));
  argv.push_back(nullptr);

  std::vector<std::string> env_storage;
  for (char **entry = environ; entry && *entry; ++entry) {
    // bibit-audio carries the sherpa runtime path for KWS/VAD.  Do not pass
    // that private runtime path to llama-server; it must resolve its own SMT
    // media backend dependencies from the system installation.
    const std::string_view value(*entry);
    const auto is_variable = [&value](std::string_view name) {
      return value.size() > name.size() && value.substr(0, name.size()) == name &&
          value[name.size()] == '=';
    };
    if (!is_variable("SPACEMIT_DISABLE_TCM") &&
        !is_variable("SPACEMIT_EP_INTRA_THREAD_NUM") &&
        !is_variable("SPACEMIT_EP_INTRA_THREAD_AFFINITY") &&
        !is_variable("SPACEMIT_EP_INTER_THREAD_NUM") &&
        !is_variable("SPACEMIT_PERFER_CORE_ID") &&
        !is_variable("LD_LIBRARY_PATH")) {
      env_storage.emplace_back(*entry);
    }
  }
  env_storage.emplace_back("SPACEMIT_DISABLE_TCM=1");
  std::string a100_affinity = config_.qwen3_ep_affinity;
  std::replace(a100_affinity.begin(), a100_affinity.end(), ';', ',');
  env_storage.emplace_back("SPACEMIT_PERFER_CORE_ID=" + a100_affinity);
  std::vector<char *> envp;
  envp.reserve(env_storage.size() + 1);
  for (std::string &entry : env_storage) envp.push_back(entry.data());
  envp.push_back(nullptr);

  pid_t pid = -1;
  const int rc = ::posix_spawnp(&pid, config_.qwen3_server_bin.c_str(), nullptr,
                                nullptr, argv.data(), envp.data());
  if (rc != 0) {
    if (error) *error = "cannot start llama-server: " + std::string(std::strerror(rc));
    return false;
  }
  server_pid_ = static_cast<int>(pid);
  startAffinityGuard();
  std::cerr << "Qwen3-ASR llama-server pid=" << server_pid_
            << " port=" << config_.qwen3_server_port
            << " threads=" << qwen_threads
            << " ep_affinity=" << config_.qwen3_ep_affinity << std::endl;
  return true;
}

bool Qwen3AsrClient::waitUntilReady(std::string *error) {
  const int timeout_ms = std::clamp(config_.qwen3_startup_timeout_ms, 1000, 300000);
  const auto deadline = std::chrono::steady_clock::now() +
      std::chrono::milliseconds(timeout_ms);
  std::string last_error;
  while (std::chrono::steady_clock::now() < deadline) {
    int status = 0;
    const pid_t wait_result = ::waitpid(server_pid_, &status, WNOHANG);
    if (wait_result == server_pid_) {
      server_pid_ = -1;
      if (error) *error = "llama-server exited during startup, status=" + std::to_string(status);
      return false;
    }
    HttpResponse response;
    if (httpRequest(config_.qwen3_server_host, config_.qwen3_server_port, "GET",
                    "/health", "", 1000, &response, &last_error) &&
        response.status == 200) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
  }
  if (error) *error = "llama-server startup timeout: " + last_error;
  return false;
}

bool Qwen3AsrClient::initialize(std::string *error) {
  if (initialized_) return true;
  const std::vector<std::string> required = {
      config_.asr_model,
      config_.qwen3_model_dir + "/config.json",
      config_.qwen3_model_dir + "/Qwen3-ASR-0.6B-encoder-frontend.dynq.onnx",
      config_.qwen3_model_dir + "/Qwen3-ASR-0.6B-encoder-backend.dynq.onnx"};
  for (const std::string &path : required) {
    if (::access(path.c_str(), R_OK) != 0) {
      if (error) *error = "Qwen3-ASR asset missing: " + path;
      return false;
    }
  }
  if (!prepareSmtConfig(error)) return false;
  if (!startServer(error)) {
    cleanupSmtConfig();
    return false;
  }
  if (!waitUntilReady(error)) {
    shutdown();
    return false;
  }
  stopAffinityGuard("startup");
  initialized_ = true;
  return true;
}

bool Qwen3AsrClient::eraseSlot() {
  HttpResponse response;
  std::string ignored;
  return httpRequest(config_.qwen3_server_host, config_.qwen3_server_port,
                     "POST", "/slots/0?action=erase", "",
                     config_.qwen3_request_timeout_ms, &response, &ignored) &&
      response.status >= 200 && response.status < 300;
}

bool Qwen3AsrClient::recognize(const float *samples, int count, std::string *text,
                               std::string *error) {
  if (!initialized_ && !initialize(error)) return false;
  if (!samples || count <= 0) {
    if (error) *error = "Qwen3-ASR received empty audio";
    return false;
  }
  const std::string audio_b64 = base64Encode(makeWav(samples, count));
  const std::string language = normalizeLanguage(config_.qwen3_language);
  std::string content =
      "[{\"type\":\"input_audio\",\"input_audio\":{\"data\":\"" +
      audio_b64 + "\",\"format\":\"wav\"}}";
  if (language != "auto") {
    content += ",{\"type\":\"text\",\"text\":\"language " +
        jsonEscape(language) + "<asr_text>\"}";
  }
  content += "]";
  const std::string payload =
      "{\"model\":\"qwen3-asr\",\"id_slot\":0,\"messages\":[{\"role\":\"user\",\"content\":" +
      content + "}],\"max_tokens\":" +
      std::to_string(std::clamp(config_.qwen3_max_tokens, 16, 1024)) +
      ",\"temperature\":0}";

  startAffinityGuard();
  HttpResponse response;
  const bool ok = httpRequest(config_.qwen3_server_host, config_.qwen3_server_port,
      "POST", "/v1/chat/completions", payload,
      config_.qwen3_request_timeout_ms, &response, error);
  const bool slot_erased = ok && eraseSlot();
  stopAffinityGuard("inference");
  if (!ok) return false;
  if (!slot_erased) std::cerr << "Qwen3-ASR slot erase failed" << std::endl;
  if (response.status < 200 || response.status >= 300) {
    if (error) {
      *error = "llama-server HTTP " + std::to_string(response.status) + ": " +
          response.body.substr(0, 300);
    }
    return false;
  }
  const size_t message_pos = response.body.find("\"message\"");
  if (message_pos == std::string::npos ||
      !extractJsonString(response.body, "content", message_pos, text)) {
    if (error) *error = "llama-server response missing message.content";
    return false;
  }
  *text = stripNativePrefix(*text);
  return true;
}

void Qwen3AsrClient::shutdown() {
  initialized_ = false;
  stopAffinityGuard("shutdown");
  if (server_pid_ <= 0) {
    cleanupSmtConfig();
    return;
  }
  const pid_t pid = static_cast<pid_t>(server_pid_);
  ::kill(pid, SIGTERM);
  for (int i = 0; i < 80; ++i) {
    int status = 0;
    const pid_t result = ::waitpid(pid, &status, WNOHANG);
    if (result == pid || (result < 0 && errno == ECHILD)) {
      server_pid_ = -1;
      cleanupSmtConfig();
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  ::kill(pid, SIGKILL);
  int status = 0;
  while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
  server_pid_ = -1;
  cleanupSmtConfig();
}
