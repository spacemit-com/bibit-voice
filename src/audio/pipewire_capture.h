#pragma once

#include <pipewire/pipewire.h>
#include <pipewire/extensions/metadata.h>

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

class PipeWireCapture {
 public:
  using AudioCallback = std::function<void(const float *, int, int)>;
  using StateCallback = std::function<void(const std::string &)>;

  PipeWireCapture(std::string source, int rate, int channels,
                  AudioCallback audio_cb, StateCallback state_cb);
  ~PipeWireCapture();
  bool start();
  void stop();

 private:
  static void onStateChanged(void *data, pw_stream_state old_state,
                             pw_stream_state state, const char *error);
  static void onParamChanged(void *data, uint32_t id, const spa_pod *param);
  static void onProcess(void *data);
  static void onRegistryGlobal(void *data, uint32_t id, uint32_t permissions,
                               const char *type, uint32_t version,
                               const spa_dict *props);
  static void onRegistryGlobalRemove(void *data, uint32_t id);
  static int onMetadataProperty(void *data, uint32_t subject, const char *key,
                                const char *type, const char *value);
  void process();
  bool connectStream();
  void bindDefaultMetadata(uint32_t id);
  void followDefaultSource(const char *value);
  void cleanupLocked();
  void reportState(const std::string &message) noexcept;

  std::string source_;
  bool follow_system_default_ = false;
  std::string current_default_source_;
  int requested_rate_;
  int requested_channels_;
  int actual_rate_ = 0;
  int actual_channels_ = 0;
  bool actual_s16_ = true;
  AudioCallback audio_cb_;
  StateCallback state_cb_;
  pw_main_loop *loop_ = nullptr;
  pw_stream *stream_ = nullptr;
  pw_stream_events *events_ = nullptr;
  pw_registry *registry_ = nullptr;
  pw_metadata *metadata_ = nullptr;
  uint32_t metadata_id_ = PW_ID_ANY;
  spa_hook registry_listener_{};
  spa_hook metadata_listener_{};
  bool registry_listener_active_ = false;
  bool metadata_listener_active_ = false;
  std::thread thread_;
  std::atomic<bool> running_{false};
  bool pw_initialized_ = false;
  std::mutex lifecycle_mutex_;
};
