#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct WavData {
  int sample_rate = 0;
  int channels = 0;
  std::vector<float> mono;
};

WavData readWav(const std::string &path);
bool writeWavPcm16Mono(const std::string &path, const float *samples,
                       size_t count, int sample_rate, std::string *error = nullptr);
