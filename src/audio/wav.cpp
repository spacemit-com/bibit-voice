#include "wav.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace {
uint16_t u16(const unsigned char *p) {
  return static_cast<uint16_t>(p[0] | (p[1] << 8));
}
uint32_t u32(const unsigned char *p) {
  return static_cast<uint32_t>(p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24));
}
}  // namespace

bool writeWavPcm16Mono(const std::string &path, const float *samples,
                       size_t count, int sample_rate, std::string *error) {
  if (!samples || count == 0 || sample_rate <= 0) {
    if (error) *error = "invalid WAV samples";
    return false;
  }
  try {
    namespace fs = std::filesystem;
    const fs::path target(path);
    fs::create_directories(target.parent_path());
    const std::string temporary = path + ".tmp";
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot create WAV");
    const uint32_t data_bytes = static_cast<uint32_t>(count * sizeof(int16_t));
    const uint32_t riff_bytes = 36 + data_bytes;
    const uint16_t format = 1, channels = 1, bits = 16;
    const uint32_t byte_rate = static_cast<uint32_t>(sample_rate * channels * bits / 8);
    const uint16_t block_align = channels * bits / 8;
    out.write("RIFF", 4); out.write(reinterpret_cast<const char *>(&riff_bytes), 4);
    out.write("WAVEfmt ", 8);
    const uint32_t fmt_bytes = 16;
    out.write(reinterpret_cast<const char *>(&fmt_bytes), 4);
    out.write(reinterpret_cast<const char *>(&format), 2);
    out.write(reinterpret_cast<const char *>(&channels), 2);
    out.write(reinterpret_cast<const char *>(&sample_rate), 4);
    out.write(reinterpret_cast<const char *>(&byte_rate), 4);
    out.write(reinterpret_cast<const char *>(&block_align), 2);
    out.write(reinterpret_cast<const char *>(&bits), 2);
    out.write("data", 4); out.write(reinterpret_cast<const char *>(&data_bytes), 4);
    for (size_t i = 0; i < count; ++i) {
      const float clipped = std::clamp(samples[i], -1.0f, 1.0f);
      const int16_t value = static_cast<int16_t>(std::lround(clipped * 32767.0f));
      out.write(reinterpret_cast<const char *>(&value), sizeof(value));
    }
    out.close();
    if (!out) throw std::runtime_error("cannot finish WAV");
    fs::rename(temporary, target);
    return true;
  } catch (const std::exception &e) {
    if (error) *error = e.what();
    return false;
  }
}

WavData readWav(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open wav: " + path);
  unsigned char header[12];
  in.read(reinterpret_cast<char *>(header), 12);
  if (in.gcount() != 12 || std::memcmp(header, "RIFF", 4) ||
      std::memcmp(header + 8, "WAVE", 4)) {
    throw std::runtime_error("unsupported wav header");
  }
  int format = 0, channels = 0, sample_rate = 0, bits = 0;
  std::vector<unsigned char> pcm;
  while (in) {
    unsigned char ch[8];
    in.read(reinterpret_cast<char *>(ch), 8);
    if (in.gcount() != 8) break;
    const uint32_t size = u32(ch + 4);
    std::vector<unsigned char> data(size);
    in.read(reinterpret_cast<char *>(data.data()), size);
    if (std::memcmp(ch, "fmt ", 4) == 0 && size >= 16) {
      format = u16(data.data());
      channels = u16(data.data() + 2);
      sample_rate = static_cast<int>(u32(data.data() + 4));
      bits = u16(data.data() + 14);
    } else if (std::memcmp(ch, "data", 4) == 0) {
      pcm = std::move(data);
    }
    if (size & 1) in.get();
  }
  if (channels <= 0 || sample_rate <= 0 || pcm.empty())
    throw std::runtime_error("wav missing fmt/data");
  if (!((format == 1 && bits == 16) || (format == 3 && bits == 32)))
    throw std::runtime_error("wav supports PCM16 or float32 only");
  const int bytes = bits / 8;
  const size_t frames = pcm.size() / (bytes * channels);
  WavData out{sample_rate, channels, {}};
  out.mono.reserve(frames);
  for (size_t f = 0; f < frames; ++f) {
    double sum = 0.0;
    for (int c = 0; c < channels; ++c) {
      const unsigned char *p = pcm.data() + (f * channels + c) * bytes;
      if (format == 1) {
        const int16_t v = static_cast<int16_t>(u16(p));
        sum += static_cast<double>(v) / 32768.0;
      } else {
        float v;
        std::memcpy(&v, p, sizeof(v));
        sum += v;
      }
    }
    out.mono.push_back(static_cast<float>(std::clamp(sum / channels, -1.0, 1.0)));
  }
  return out;
}
