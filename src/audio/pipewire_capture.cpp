#include "pipewire_capture.h"

#include <pipewire/pipewire.h>
#include <spa/param/audio/format-utils.h>
#include <spa/param/audio/raw.h>
#include <spa/param/param.h>
#include <spa/pod/builder.h>
#include <spa/utils/json.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <new>
#include <sstream>

namespace {
bool isSystemDefault(const std::string &source) {
  return source.empty() || source == "default" || source == "system" ||
         source == "@DEFAULT_SOURCE@";
}

std::string metadataNodeName(const char *value) {
  if (!value) return {};
  spa_json it[2];
  spa_json_init(&it[0], value, std::strlen(value));
  if (spa_json_enter_object(&it[0], &it[1]) <= 0) return {};
  char key[128];
  char name[1024];
  while (spa_json_get_string(&it[1], key, sizeof(key)) > 0) {
    if (std::strcmp(key, "name") == 0) {
      return spa_json_get_string(&it[1], name, sizeof(name)) > 0
                 ? std::string(name)
                 : std::string();
    }
    if (spa_json_next(&it[1], nullptr) <= 0) break;
  }
  return {};
}
}  // namespace

PipeWireCapture::PipeWireCapture(std::string source, int rate, int channels,
                                 AudioCallback audio_cb, StateCallback state_cb)
    : source_(std::move(source)),
      follow_system_default_(isSystemDefault(source_)),
      requested_rate_(rate),
      requested_channels_(channels),
      audio_cb_(std::move(audio_cb)),
      state_cb_(std::move(state_cb)) {}

PipeWireCapture::~PipeWireCapture() { stop(); }

bool PipeWireCapture::start() {
  std::lock_guard<std::mutex> lk(lifecycle_mutex_);
  if (running_) return true;
  if (requested_rate_ <= 0 || requested_channels_ <= 0 || requested_channels_ > 32) {
    reportState("audio_error invalid_capture_format");
    return false;
  }
  if (loop_ || stream_ || events_ || pw_initialized_) cleanupLocked();
  pw_init(nullptr, nullptr);
  pw_initialized_ = true;
  loop_ = pw_main_loop_new(nullptr);
  if (!loop_) {
    cleanupLocked();
    return false;
  }
  events_ = new (std::nothrow) pw_stream_events{};
  if (!events_) {
    cleanupLocked();
    return false;
  }
  events_->version = PW_VERSION_STREAM_EVENTS;
  events_->state_changed = &PipeWireCapture::onStateChanged;
  events_->param_changed = &PipeWireCapture::onParamChanged;
  events_->process = &PipeWireCapture::onProcess;

  pw_properties *props = pw_properties_new(
      PW_KEY_MEDIA_TYPE, "Audio",
      PW_KEY_MEDIA_CATEGORY, "Capture",
      PW_KEY_MEDIA_ROLE, "Communication",
      PW_KEY_NODE_NAME, "bibit-audio-capture",
      nullptr);
  if (!props) {
    cleanupLocked();
    return false;
  }
  if (!follow_system_default_) {
    pw_properties_set(props, PW_KEY_TARGET_OBJECT, source_.c_str());
  }
  stream_ = pw_stream_new_simple(pw_main_loop_get_loop(loop_), "Bibit capture",
                                 props, events_, this);
  if (!stream_) {
    cleanupLocked();
    return false;
  }

  if (!connectStream()) {
    cleanupLocked();
    return false;
  }
  if (follow_system_default_) {
    pw_core *core = pw_stream_get_core(stream_);
    registry_ = core ? pw_core_get_registry(core, PW_VERSION_REGISTRY, 0) : nullptr;
    if (!registry_) {
      cleanupLocked();
      return false;
    }
    static const pw_registry_events registry_events = [] {
      pw_registry_events events{};
      events.version = PW_VERSION_REGISTRY_EVENTS;
      events.global = &PipeWireCapture::onRegistryGlobal;
      events.global_remove = &PipeWireCapture::onRegistryGlobalRemove;
      return events;
    }();
    pw_registry_add_listener(registry_, &registry_listener_, &registry_events, this);
    registry_listener_active_ = true;
  }
  running_ = true;
  try {
    thread_ = std::thread([this]() { pw_main_loop_run(loop_); });
  } catch (...) {
    cleanupLocked();
    return false;
  }
  return true;
}

bool PipeWireCapture::connectStream() {
  uint8_t storage[1024];
  spa_pod_builder builder = SPA_POD_BUILDER_INIT(storage, sizeof(storage));
  spa_audio_info_raw info{};
  info.format = SPA_AUDIO_FORMAT_S16;
  info.rate = static_cast<uint32_t>(requested_rate_);
  info.channels = static_cast<uint32_t>(requested_channels_);
  if (requested_channels_ >= 1) info.position[0] = SPA_AUDIO_CHANNEL_FL;
  if (requested_channels_ >= 2) info.position[1] = SPA_AUDIO_CHANNEL_FR;
  const spa_pod *params[] = {
      spa_format_audio_raw_build(&builder, SPA_PARAM_EnumFormat, &info)};
  return pw_stream_connect(
             stream_, PW_DIRECTION_INPUT, PW_ID_ANY,
             static_cast<pw_stream_flags>(PW_STREAM_FLAG_AUTOCONNECT |
                                          PW_STREAM_FLAG_MAP_BUFFERS),
             params, 1) >= 0;
}

void PipeWireCapture::stop() {
  std::lock_guard<std::mutex> lk(lifecycle_mutex_);
  cleanupLocked();
}

void PipeWireCapture::cleanupLocked() {
  running_ = false;
  if (loop_) pw_main_loop_quit(loop_);
  if (thread_.joinable()) thread_.join();
  if (metadata_listener_active_) {
    spa_hook_remove(&metadata_listener_);
    metadata_listener_active_ = false;
  }
  if (metadata_) pw_proxy_destroy(reinterpret_cast<pw_proxy *>(metadata_));
  metadata_ = nullptr;
  metadata_id_ = PW_ID_ANY;
  if (registry_listener_active_) {
    spa_hook_remove(&registry_listener_);
    registry_listener_active_ = false;
  }
  if (registry_) pw_proxy_destroy(reinterpret_cast<pw_proxy *>(registry_));
  registry_ = nullptr;
  if (stream_) pw_stream_destroy(stream_);
  stream_ = nullptr;
  if (loop_) pw_main_loop_destroy(loop_);
  loop_ = nullptr;
  delete events_;
  events_ = nullptr;
  if (pw_initialized_) {
    pw_deinit();
    pw_initialized_ = false;
  }
  actual_rate_ = 0;
  actual_channels_ = 0;
  actual_s16_ = true;
  current_default_source_.clear();
}

void PipeWireCapture::reportState(const std::string &message) noexcept {
  try {
    if (state_cb_) state_cb_(message);
  } catch (...) {
  }
}

void PipeWireCapture::onStateChanged(void *data, pw_stream_state,
                                     pw_stream_state state, const char *error) {
  auto *self = static_cast<PipeWireCapture *>(data);
  std::ostringstream os;
  os << "pipewire_state=" << pw_stream_state_as_string(state);
  if (error) os << " error=" << error;
  self->reportState(os.str());
}

void PipeWireCapture::onParamChanged(void *data, uint32_t id, const spa_pod *param) {
  auto *self = static_cast<PipeWireCapture *>(data);
  if (!param || id != SPA_PARAM_Format) return;
  spa_audio_info_raw info{};
  if (spa_format_audio_raw_parse(param, &info) < 0) return;
  self->actual_s16_ = info.format == SPA_AUDIO_FORMAT_S16;
  self->actual_rate_ = static_cast<int>(info.rate);
  self->actual_channels_ = static_cast<int>(info.channels);
  self->reportState("audio_opened rate=" + std::to_string(self->actual_rate_) +
                    " channels=" + std::to_string(self->actual_channels_) +
                    " format=" + (self->actual_s16_ ? "s16" : "unsupported"));
}

void PipeWireCapture::onProcess(void *data) {
  static_cast<PipeWireCapture *>(data)->process();
}

void PipeWireCapture::onRegistryGlobal(void *data, uint32_t id, uint32_t,
                                       const char *type, uint32_t,
                                       const spa_dict *props) {
  auto *self = static_cast<PipeWireCapture *>(data);
  if (!self->follow_system_default_ || self->metadata_ || !type ||
      std::strcmp(type, PW_TYPE_INTERFACE_Metadata) != 0) {
    return;
  }
  const char *name = props ? spa_dict_lookup(props, PW_KEY_METADATA_NAME) : nullptr;
  if (name && std::strcmp(name, "default") == 0) self->bindDefaultMetadata(id);
}

void PipeWireCapture::onRegistryGlobalRemove(void *data, uint32_t id) {
  auto *self = static_cast<PipeWireCapture *>(data);
  if (id != self->metadata_id_) return;
  if (self->metadata_listener_active_) {
    spa_hook_remove(&self->metadata_listener_);
    self->metadata_listener_active_ = false;
  }
  if (self->metadata_) pw_proxy_destroy(reinterpret_cast<pw_proxy *>(self->metadata_));
  self->metadata_ = nullptr;
  self->metadata_id_ = PW_ID_ANY;
  self->current_default_source_.clear();
}

void PipeWireCapture::bindDefaultMetadata(uint32_t id) {
  metadata_ = static_cast<pw_metadata *>(pw_registry_bind(
      registry_, id, PW_TYPE_INTERFACE_Metadata, PW_VERSION_METADATA, 0));
  if (!metadata_) return;
  metadata_id_ = id;
  static const pw_metadata_events metadata_events = [] {
    pw_metadata_events events{};
    events.version = PW_VERSION_METADATA_EVENTS;
    events.property = &PipeWireCapture::onMetadataProperty;
    return events;
  }();
  pw_metadata_add_listener(metadata_, &metadata_listener_, &metadata_events, this);
  metadata_listener_active_ = true;
}

int PipeWireCapture::onMetadataProperty(void *data, uint32_t, const char *key,
                                        const char *, const char *value) {
  auto *self = static_cast<PipeWireCapture *>(data);
  if (key && std::strcmp(key, "default.audio.source") == 0) {
    self->followDefaultSource(value);
  }
  return 0;
}

void PipeWireCapture::followDefaultSource(const char *value) {
  const std::string next = metadataNodeName(value);
  if (next.empty() || next == current_default_source_) return;
  const bool initial = current_default_source_.empty();
  current_default_source_ = next;
  reportState("audio_source=" + next + " mode=system_default");
  if (initial || !stream_ || !running_) return;

  const spa_dict_item item = {PW_KEY_TARGET_OBJECT, current_default_source_.c_str()};
  const spa_dict props = SPA_DICT_INIT(&item, 1);
  if (pw_stream_update_properties(stream_, &props) < 0) {
    reportState("audio_error default_source_update_failed source=" +
                current_default_source_);
  } else {
    reportState("audio_source_changed source=" + current_default_source_);
  }
}

void PipeWireCapture::process() {
  pw_buffer *pb = pw_stream_dequeue_buffer(stream_);
  if (!pb) return;
  try {
    spa_buffer *b = pb->buffer;
    if (b && b->n_datas > 0) {
      spa_data &d = b->datas[0];
      const int channels = actual_channels_ > 0 ? actual_channels_ : requested_channels_;
      const int rate = actual_rate_ > 0 ? actual_rate_ : requested_rate_;
      const size_t offset = d.chunk ? static_cast<size_t>(d.chunk->offset) : 0;
      const size_t size = d.chunk ? static_cast<size_t>(d.chunk->size) : 0;
      const size_t max_size = static_cast<size_t>(d.maxsize);
      const bool valid_bounds = offset <= max_size && size <= max_size - offset;
      const bool valid_alignment = d.data &&
          (reinterpret_cast<uintptr_t>(d.data) + offset) % alignof(int16_t) == 0;
      if (d.data && d.chunk && size > 0 && actual_s16_ && channels > 0 &&
          channels <= 32 && rate > 0 && valid_bounds && valid_alignment) {
        const size_t frame_size = sizeof(int16_t) * static_cast<size_t>(channels);
        const size_t frame_count = size / frame_size;
        if (frame_count <= static_cast<size_t>(std::numeric_limits<int>::max())) {
          const auto *p = reinterpret_cast<const int16_t *>(
              static_cast<const uint8_t *>(d.data) + offset);
          std::vector<float> mono(frame_count);
          for (size_t i = 0; i < frame_count; ++i) {
            int64_t sum = 0;
            for (int c = 0; c < channels; ++c) sum += p[i * channels + c];
            mono[i] = static_cast<float>(sum) / (32768.0f * channels);
          }
          if (audio_cb_ && !mono.empty()) {
            audio_cb_(mono.data(), static_cast<int>(mono.size()), rate);
          }
        }
      }
    }
  } catch (const std::exception &e) {
    reportState(std::string("audio_error process_exception=") + e.what());
  } catch (...) {
    reportState("audio_error process_exception=unknown");
  }
  pw_stream_queue_buffer(stream_, pb);
}
