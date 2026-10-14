#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>

// The capture queue exists to absorb jitter, not to accumulate a history
// buffer.  It used to hold 30 s: one stall then pinned the pipeline half a
// minute behind live audio, and since the worker can only catch up by running
// faster than realtime it never came back - every arriving chunk was dropped
// oldest-first while the user's wake word sat in the discarded end of the
// queue (2026-09-17: 192 drops / 65537 samples over 26 s).
inline size_t queueMaxSamples(int configured_ms) {
  return static_cast<size_t>(std::clamp(configured_ms, 200, 30000)) * 16;
}

// A saturated queue used to log one line per capture period (~47/s), which
// buries the surrounding events.  Report at most once per second and fold the
// drops in between into that one line; the totals stay accurate.
class QueueOverflowReporter {
 public:
  // Returns true when this drop should be reported now.
  bool add(int64_t now_ms, int64_t dropped) {
    dropped_ += dropped;
    if (has_reported_ && now_ms - last_report_ms_ < kReportIntervalMs) {
      return false;
    }
    has_reported_ = true;
    last_report_ms_ = now_ms;
    return true;
  }

  // Drops accumulated since the last report; clears the accumulator.
  int64_t takeDropped() {
    const int64_t value = dropped_;
    dropped_ = 0;
    return value;
  }

  int64_t lastReportMs() const { return last_report_ms_; }

 private:
  static constexpr int64_t kReportIntervalMs = 1000;
  bool has_reported_ = false;
  int64_t last_report_ms_ = 0;
  int64_t dropped_ = 0;
};

// Per-chunk processing cost. The overflow event already says the worker is
// behind, but not where the time went: the 2026-09-28 field log showed
// arrival ~1000 ms/s against process ~999 ms/s - a ~0.1% deficit that pins the
// queue at its ceiling and drops a chunk every few seconds. Comparing a
// chunk's cost against the 21.3 ms that chunk itself represents is what turns
// "there is a deficit" into "this chunk cost 41 ms". Totals are process
// lifetime, like arrived/processed, so two events give the window average by
// subtraction.
class ProcessTimeTracker {
 public:
  // Single writer (the worker thread). The peak needs no CAS because readers
  // only ever see a previously published value.
  void add(int64_t us) {
    total_us_.fetch_add(us, std::memory_order_relaxed);
    chunks_.fetch_add(1, std::memory_order_relaxed);
    if (us > max_us_.load(std::memory_order_relaxed)) {
      max_us_.store(us, std::memory_order_relaxed);
    }
  }

  int64_t totalUs() const { return total_us_.load(std::memory_order_relaxed); }
  int64_t maxUs() const { return max_us_.load(std::memory_order_relaxed); }
  int64_t chunks() const { return chunks_.load(std::memory_order_relaxed); }

 private:
  std::atomic<int64_t> total_us_{0};
  std::atomic<int64_t> max_us_{0};
  std::atomic<int64_t> chunks_{0};
};
