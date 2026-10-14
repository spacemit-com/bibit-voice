#include "resampler.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {
constexpr double kPi = 3.14159265358979323846;
}

StreamingResampler::StreamingResampler(int input_rate, int output_rate)
    : input_rate_(input_rate),
      output_rate_(output_rate),
      step_(0.0),
      cutoff_(0.0),
      position_(16.0) {
  if (input_rate <= 0 || output_rate <= 0) throw std::invalid_argument("invalid sample rate");
  step_ = static_cast<double>(input_rate) / output_rate;
  cutoff_ = 0.45 * std::min(1.0, static_cast<double>(output_rate) / input_rate);
  reset();
}

void StreamingResampler::reset() {
  buffer_.assign(half_taps_ + 1, 0.0f);
  position_ = static_cast<double>(half_taps_);
}

double StreamingResampler::kernel(double x) const {
  if (std::abs(x) >= half_taps_) return 0.0;
  const double sinc_arg = 2.0 * cutoff_ * x;
  const double sinc = std::abs(sinc_arg) < 1e-12
                          ? 1.0
                          : std::sin(kPi * sinc_arg) / (kPi * sinc_arg);
  const double window = 0.5 + 0.5 * std::cos(kPi * x / half_taps_);
  return 2.0 * cutoff_ * sinc * window;
}

std::vector<float> StreamingResampler::process(const float *samples, int count) {
  if (!samples || count <= 0) return {};
  if (input_rate_ == output_rate_) return std::vector<float>(samples, samples + count);
  buffer_.insert(buffer_.end(), samples, samples + count);
  std::vector<float> out;
  out.reserve(static_cast<size_t>(count / step_ + 4));
  while (position_ + half_taps_ < static_cast<double>(buffer_.size())) {
    const int center = static_cast<int>(std::floor(position_));
    double sum = 0.0;
    double weight_sum = 0.0;
    for (int i = center - half_taps_ + 1; i <= center + half_taps_; ++i) {
      if (i < 0 || i >= static_cast<int>(buffer_.size())) continue;
      const double w = kernel(static_cast<double>(i) - position_);
      sum += static_cast<double>(buffer_[i]) * w;
      weight_sum += w;
    }
    out.push_back(static_cast<float>(weight_sum == 0.0 ? 0.0 : sum / weight_sum));
    position_ += step_;
  }
  const int drop = std::max(0, static_cast<int>(std::floor(position_)) - half_taps_ - 1);
  if (drop > 0) {
    buffer_.erase(buffer_.begin(), buffer_.begin() + drop);
    position_ -= drop;
  }
  return out;
}
