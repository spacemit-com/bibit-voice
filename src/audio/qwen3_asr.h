#pragma once

#include "config.h"

#include <atomic>
#include <string>
#include <thread>

class Qwen3AsrClient {
 public:
  explicit Qwen3AsrClient(const BibitConfig &config);
  ~Qwen3AsrClient();

  bool initialize(std::string *error);
  bool recognize(const float *samples, int count, std::string *text,
                 std::string *error);
  void shutdown();

 private:
  bool prepareSmtConfig(std::string *error);
  void cleanupSmtConfig();
  bool startServer(std::string *error);
  bool waitUntilReady(std::string *error);
  bool eraseSlot();
  void startAffinityGuard();
  void stopAffinityGuard(const char *phase);

  BibitConfig config_;
  std::string runtime_config_dir_;
  int server_pid_ = -1;
  bool initialized_ = false;
  std::atomic<bool> affinity_guard_stop_{true};
  std::atomic<bool> affinity_guard_ready_{false};
  std::atomic<int> affinity_guard_pinned_{0};
  std::atomic<int> affinity_guard_errors_{0};
  std::thread affinity_guard_thread_;
};
