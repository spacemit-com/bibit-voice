#pragma once

#include <vector>

class StreamingResampler {
 public:
  StreamingResampler(int input_rate, int output_rate = 16000);
  std::vector<float> process(const float *samples, int count);
  void reset();
  int inputRate() const { return input_rate_; }

 private:
  double kernel(double x) const;

  int input_rate_;
  int output_rate_;
  int half_taps_ = 16;
  double step_;
  double cutoff_;
  double position_;
  std::vector<float> buffer_;
};
