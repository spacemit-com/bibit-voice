#pragma once

// 语音开关意图的收敛模型（纯逻辑、无第三方依赖，主机上可直接编译测试）。
//
// 背景：打开识别是一次异步的后端构建（Qwen3 还要拉起 llama-server），关闭识别却是
// 立即生效。如果构建期间到达的命令只写进一个布尔「待处理」标记，就会出现三类错位：
//
//   1. 被取代的意图复活：关掉之后构建才完成，又把开关置回打开；
//   2. 更新的意图被吞掉：构建失败时顺手清掉了用户后来的打开意图；
//   3. 意图没有终态：被取代的请求既不成功也不失败，客户端只能靠超时猜。
//
// 这里把意图显式记成一条条待结算记录，规则只有三条：
//
//   * 最后意图胜出：同一字段上只有最新意图会被执行，更早的中间意图判为被取代；
//   * 达成才结算：打开识别的意图要等后端真正就绪（asr && asr_ready）才算达成，
//     排队或构建中都不算「已经可用」；
//   * 失败只吃自己那一份：一次构建失败只结算驱动它的那条意图，更新的意图继续收敛。
//
// 生效态始终由流水线持有（它才是真正碰后端的人），本模型只回答两个问题：
// 「现在要不要再起一次构建」和「哪些意图可以给出终态了」。

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace bibit {

enum class SwitchOutcome { Applied, Superseded, Failed };

// 一条意图的终态：request_id 用于对应回发起它的那次命令，wake/asr 是结算时的生效态。
struct SwitchSettlement {
  SwitchOutcome outcome = SwitchOutcome::Applied;
  std::string request_id;
  bool wake = false;
  bool asr = false;
};

// 生效态快照（由流水线在收敛点读出）。
struct SwitchState {
  bool wake = false;
  // 开关态：打开识别的意图到达时立即为真。后端就绪与否由 asr_ready 表达。
  bool asr = false;
  // 后端是否已经就绪（qwen3 / offline / online 三者之一存在）。
  bool asr_ready = false;
  // 是否有构建在途：在途时不启动新构建，等提交点再收敛。
  bool switching = false;
};

enum class SwitchActionKind { None, EnableAsr };

struct SwitchAction {
  SwitchActionKind kind = SwitchActionKind::None;
  // 驱动这次构建的请求标识（最新那条未决的打开意图）。
  std::string request_id;
};

struct SwitchStep {
  SwitchAction action;
  std::vector<SwitchSettlement> settlements;
};

class SwitchIntentBook {
 public:
  struct Entry {
    std::string request_id;
    // 只记命令里真正给出的字段：未给出的字段沿用当前生效值，不产生新意图。
    bool wake_set = false;
    bool wake = false;
    bool asr_set = false;
    bool asr = false;
    int64_t seq = 0;
  };

  // 记录一次 set_switches 命令里给出的字段。
  void record(bool wake_set, bool wake, bool asr_set, bool asr,
              const std::string &request_id) {
    Entry entry;
    entry.request_id = request_id;
    entry.wake_set = wake_set;
    entry.wake = wake;
    entry.asr_set = asr_set;
    entry.asr = asr;
    entry.seq = ++seq_;
    entries_.push_back(std::move(entry));
  }

  // 空闲点（命令处理完、构建提交完）调用：结算已达成/被取代的意图，并给出下一步动作。
  // 在途构建期间调用只做结算，不返回新动作（构建结果会在提交点重算）。
  SwitchStep reconcile(const SwitchState &state) {
    SwitchStep step;
    std::vector<Entry> kept;
    kept.reserve(entries_.size());
    for (const auto &entry : entries_) {
      const bool wake_ok = !entry.wake_set || entry.wake == state.wake;
      const bool asr_ok = !entry.asr_set ||
                          (entry.asr == state.asr && (!entry.asr || state.asr_ready));
      // 字段级取代：更新的意图在同一个字段上要求了别的值，这个字段已经不可能达成。
      // 被取代的字段不再等待，但同一命令里其它字段仍然有效——一条命令同时请求
      // 唤醒和识别时，不能因为其中一个被取代就把另一个也丢掉。
      const bool wake_done = wake_ok || (entry.wake_set && supersededField(entry, true));
      const bool asr_done = asr_ok || (entry.asr_set && supersededField(entry, false));
      if (!wake_done || !asr_done) {
        kept.push_back(entry);
        continue;
      }
      // 请求的字段全部生效才算「达成」；有字段被更新意图取代就是「被取代」。
      // 事件同时带生效态，客户端据此分辨哪一部分生效了。
      const SwitchOutcome outcome =
          (wake_ok && asr_ok) ? SwitchOutcome::Applied : SwitchOutcome::Superseded;
      step.settlements.push_back(settlement(entry, outcome, state));
    }
    entries_ = std::move(kept);
    if (state.switching) return step;
    const Entry *want = newestPendingEnable();
    if (want != nullptr && !(state.asr && state.asr_ready)) {
      step.action.kind = SwitchActionKind::EnableAsr;
      step.action.request_id = want->request_id;
    }
    return step;
  }

  // 构建确实起来了才记账：只返回动作而不执行（例如同一个收敛点上先去做引擎切换）
  // 不能让后续的构建结果错记到这条意图头上。
  void noteEnableStarted(const std::string &request_id) {
    building_request_id_ = request_id;
  }

  // 打开识别的构建结束。outcome 由流水线按提交结果给出：
  // Applied=已采纳（后端接入）、Superseded=构建结果被丢弃、Failed=构建失败。
  // 若这条意图已经在更早的收敛点结算过（例如期间被关闭取代），返回空。
  std::optional<SwitchSettlement> noteEnableBuildEnded(SwitchOutcome outcome,
                                                       const SwitchState &state) {
    if (building_request_id_.empty()) return std::nullopt;
    const std::string request_id = building_request_id_;
    building_request_id_.clear();
    auto it = find(request_id);
    if (it == entries_.end()) return std::nullopt;
    // 构建成功但结果没被采纳，说明期间出现了更新的关闭意图：这条意图是被取代，
    // 不是「打开失败」，避免客户端收到一个假的失败。
    if (outcome == SwitchOutcome::Applied && !(state.asr && state.asr_ready)) {
      outcome = SwitchOutcome::Superseded;
    }
    SwitchSettlement settled = settlement(*it, outcome, state);
    entries_.erase(it);
    return settled;
  }

  // 构建没能启动（服务正在停止等）：所有未决意图都按失败收场，界面不会停在「正在应用」。
  std::vector<SwitchSettlement> failAllPending(const SwitchState &state) {
    std::vector<SwitchSettlement> settled;
    settled.reserve(entries_.size());
    for (const auto &entry : entries_) {
      settled.push_back(settlement(entry, SwitchOutcome::Failed, state));
    }
    entries_.clear();
    building_request_id_.clear();
    return settled;
  }

  // 还有未达成的打开意图（排队中或构建中）：status.desired.pending 用它。
  bool enablePending() const {
    for (const auto &entry : entries_) {
      if (entry.asr_set && entry.asr && !supersededField(entry, false)) return true;
    }
    return false;
  }

  bool empty() const { return entries_.empty() && building_request_id_.empty(); }

  const std::vector<Entry> &entries() const { return entries_; }

 private:
  // 有没有更新的意图在同一个字段上要求了别的值（wake_field=true 查唤醒字段）。
  bool supersededField(const Entry &entry, bool wake_field) const {
    for (const auto &other : entries_) {
      if (other.seq <= entry.seq) continue;
      if (wake_field) {
        if (entry.wake_set && other.wake_set && other.wake != entry.wake) return true;
      } else if (entry.asr_set && other.asr_set && other.asr != entry.asr) {
        return true;
      }
    }
    return false;
  }

  const Entry *newestPendingEnable() const {
    const Entry *newest = nullptr;
    for (const auto &entry : entries_) {
      if (!entry.asr_set || !entry.asr) continue;
      // 识别字段已经被更新的意图取代：这条命令不再驱动构建。
      if (supersededField(entry, false)) continue;
      if (newest == nullptr || entry.seq > newest->seq) newest = &entry;
    }
    return newest;
  }

  std::vector<Entry>::iterator find(const std::string &request_id) {
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
      if (it->request_id == request_id) return it;
    }
    return entries_.end();
  }

  static SwitchSettlement settlement(const Entry &entry, SwitchOutcome outcome,
                                     const SwitchState &state) {
    SwitchSettlement settled;
    settled.outcome = outcome;
    settled.request_id = entry.request_id;
    settled.wake = state.wake;
    settled.asr = state.asr;
    return settled;
  }

  std::vector<Entry> entries_;
  // 正在驱动构建的请求（尚未结算）。构建结束后用它把结果对应回发起方。
  std::string building_request_id_;
  int64_t seq_ = 0;
};

}  // namespace bibit
