// 语音开关意图收敛的行为回归（纯逻辑，无第三方依赖）。
//
// 由 tests/test_switch_intents.py 编译运行；也可以直接：
//   g++ -std=c++17 -I src/audio tests/switch_intent_scenarios.cpp -o /tmp/t && /tmp/t
//
// 每个场景都按真实时序驱动：命令 → 收敛点 → 构建结束 → 收敛点。模拟器只做流水线
// 那几条固定规则（命令立即作用在生效开关上；提交点只采纳「开关还开着」的构建结果），
// 其余全部交给被测的 SwitchIntentBook。断言集中在这一类问题的三条不变量：
//
//   1. 最后意图胜出：构建期间关掉，构建结束后开关不得回到打开；
//   2. 更新的意图不被吞：构建失败只结算驱动它的那条意图，更新的打开意图继续收敛；
//   3. 意图恰好一个终态：每条 request_id 要么达成、要么被取代、要么失败，不重不漏。

#include "switch_intents.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using bibit::SwitchActionKind;
using bibit::SwitchIntentBook;
using bibit::SwitchOutcome;
using bibit::SwitchSettlement;
using bibit::SwitchState;

int g_checks = 0;
int g_failures = 0;

void expect(bool ok, const std::string &what) {
  ++g_checks;
  if (ok) return;
  ++g_failures;
  std::printf("FAIL: %s\n", what.c_str());
}

std::string outcomeName(SwitchOutcome outcome) {
  switch (outcome) {
    case SwitchOutcome::Applied: return "applied";
    case SwitchOutcome::Superseded: return "superseded";
    case SwitchOutcome::Failed: return "failed";
  }
  return "?";
}

// 事件轨迹：settle <rid> <outcome> 与 action enable <rid>。
class Log {
 public:
  void add(const std::string &line) { lines_.push_back(line); }
  bool has(const std::string &line) const {
    for (const auto &entry : lines_) {
      if (entry == line) return true;
    }
    return false;
  }
  // 事件行带诊断后缀（当时的生效态），断言按前缀匹配。
  bool hasPrefix(const std::string &prefix) const {
    for (const auto &entry : lines_) {
      if (entry.rfind(prefix, 0) == 0) return true;
    }
    return false;
  }
  int count(const std::string &prefix) const {
    int total = 0;
    for (const auto &entry : lines_) {
      if (entry.rfind(prefix, 0) == 0) ++total;
    }
    return total;
  }
  const std::vector<std::string> &lines() const { return lines_; }

 private:
  std::vector<std::string> lines_;
};

// 模拟流水线的固定规则，只把意图账本当作被测对象。
class Sim {
 public:
  // 命令到达：给出的字段立即作用在生效开关上（关闭还会放下后端）。
  void command(bool wake_set, bool wake, bool asr_set, bool asr,
               const std::string &request_id) {
    if (wake_set) state.wake = wake;
    if (asr_set) {
      state.asr = asr;
      if (!asr) ready = false;  // destroyAsr()
    }
    log.add("command " + request_id + " wake=" + (wake_set ? (wake ? "1" : "0") : "-") +
            " asr=" + (asr_set ? (asr ? "1" : "0") : "-"));
    book.record(wake_set, wake, asr_set, asr, request_id);
    settle();
  }

  // 收敛点：结算终态，必要时起一次构建。
  void settle() {
    SwitchState snapshot = state;
    snapshot.asr_ready = ready;
    snapshot.switching = switching;
    const bibit::SwitchStep step = book.reconcile(snapshot);
    for (const auto &settled : step.settlements) {
      log.add("settle " + settled.request_id + " " + outcomeName(settled.outcome) +
              " (asr=" + (state.asr ? "1" : "0") + " ready=" + (ready ? "1" : "0") + ")");
      terminals.push_back(settled.request_id);
    }
    if (step.action.kind == SwitchActionKind::EnableAsr) {
      expect(!switching, "构建在途时不应再返回新动作");
      switching = true;
      // 这次构建服务的是最新打开意图，开关态跟着落到打开。
      state.asr = true;
      book.noteEnableStarted(step.action.request_id);
      log.add("action enable " + step.action.request_id);
    }
  }

  // 构建结束：提交点只采纳「开关还开着」的结果，失败则退回关闭。
  void buildEnded(bool failed) {
    expect(switching, "没有在途构建时不应有构建结束事件");
    switching = false;
    SwitchOutcome outcome = SwitchOutcome::Superseded;
    if (failed) {
      state.asr = false;
      ready = false;
      outcome = SwitchOutcome::Failed;
    } else if (state.asr) {
      ready = true;
      outcome = SwitchOutcome::Applied;
    }
    log.add(std::string("build-ended ") + (failed ? "failed" : "ok") +
            " outcome=" + outcomeName(outcome));
    {
      SwitchState snapshot = state;
      snapshot.asr_ready = ready;
      const auto settled = book.noteEnableBuildEnded(outcome, snapshot);
      if (settled) {
        log.add("settle " + settled->request_id + " " + outcomeName(settled->outcome) +
                " (asr=" + (state.asr ? "1" : "0") + " ready=" + (ready ? "1" : "0") + ")");
        terminals.push_back(settled->request_id);
      }
    }
    settle();
  }

  // 构建没能起来：全部未决意图按失败收场。
  void buildStartFailed() {
    const SwitchState snapshot = state;
    for (const auto &settled : book.failAllPending(snapshot)) {
      log.add("settle " + settled.request_id + " " + outcomeName(settled.outcome) +
              " (asr=" + (state.asr ? "1" : "0") + " ready=" + (ready ? "1" : "0") + ")");
      terminals.push_back(settled.request_id);
    }
    state.asr = false;
    ready = false;
    switching = false;
  }

  // 引擎切换在途：占用构建时隙但不属于开关账本（模拟 asr_model 切换）。
  void foreignSwitchStarted() { switching = true; }

  void foreignSwitchEnded() {
    switching = false;
    settle();
  }

  SwitchIntentBook book;
  SwitchState state;
  bool ready = false;
  bool switching = false;
  Log log;
  std::vector<std::string> terminals;
};

void scenario_off_during_enable_build() {
  Sim sim;
  sim.command(false, false, true, true, "r1");
  expect(sim.log.hasPrefix("action enable r1"), "打开识别要起一次后端构建");
  expect(sim.switching, "构建应当处于在途状态");

  sim.command(false, false, true, false, "r2");
  expect(!sim.state.asr, "关闭立即可见");
  expect(sim.log.hasPrefix("settle r1 superseded"), "构建中关闭：打开意图判为被取代");
  expect(sim.log.hasPrefix("settle r2 applied"), "关闭意图当次达成");

  sim.buildEnded(false);
  expect(!sim.state.asr, "构建结束后开关必须保持关闭（历史缺陷：会被翻回打开）");
  expect(!sim.ready, "被丢弃的构建结果不得接入后端");
  expect(!sim.log.hasPrefix("settle r1 applied"), "被取代的意图不得报成达成");
  expect(!sim.book.enablePending(), "收敛后不应残留未决意图");
}

void scenario_off_then_on_during_build() {
  Sim sim;
  sim.command(false, false, true, true, "r1");   // 起构建
  sim.command(false, false, true, false, "r2");  // 构建中关闭
  sim.command(false, false, true, true, "r3");   // 又打开：意图排在在途构建之后
  expect(sim.log.hasPrefix("settle r1 superseded"), "r1 立即被取代");
  expect(sim.log.hasPrefix("settle r2 applied"), "关闭立即达成");
  expect(!sim.log.hasPrefix("action enable r3"), "在途构建期间不并发第二个构建");

  sim.buildEnded(false);
  expect(sim.state.asr, "最后一次意图是打开，构建结果应当被采纳");
  expect(sim.ready, "采纳后后端就绪");
  expect(sim.log.hasPrefix("settle r3 applied"), "最新意图拿到达成终态");
  expect(!sim.book.enablePending(), "收敛后无未决意图");
}

void scenario_failure_does_not_eat_newer_intent() {
  Sim sim;
  sim.command(false, false, true, true, "r1");   // 起构建
  sim.command(false, false, true, true, "r2");   // 构建中再次打开：更新意图
  sim.buildEnded(true);                          // 构建失败
  expect(sim.log.hasPrefix("settle r1 failed"), "驱动构建的意图按失败收场");
  expect(sim.log.hasPrefix("action enable r2"), "更新的打开意图要继续收敛，不能被失败带走");
  sim.buildEnded(false);
  expect(sim.state.asr && sim.ready, "重试成功后就绪");
  expect(sim.log.hasPrefix("settle r2 applied"), "r2 达成");
}

void scenario_failure_after_off_never_reopens() {
  Sim sim;
  sim.command(false, false, true, true, "r1");
  sim.command(false, false, true, false, "r2");  // 关闭取代
  sim.buildEnded(true);                          // 构建随后失败
  expect(!sim.state.asr, "失败的构建不得把开关打开");
  expect(!sim.log.hasPrefix("settle r1 applied"), "r1 早已被取代，不得再给达成");
  expect(!sim.log.hasPrefix("settle r1 failed"), "被取代的意图不得改判为失败");
  expect(sim.log.count("settle r1 ") == 1, "同一条意图只能有一个终态");
}

void scenario_off_on_off_then_failure() {
  Sim sim;
  sim.command(false, false, true, true, "r1");
  sim.command(false, false, true, false, "r2");
  sim.command(false, false, true, true, "r3");
  sim.command(false, false, true, false, "r4");  // 最后一次仍是关闭
  sim.buildEnded(false);                         // 构建成功，但开关已关
  expect(!sim.state.asr, "最后意图是关闭，构建结果必须丢弃");
  expect(!sim.ready, "丢弃的构建不得接入后端");
  expect(sim.log.hasPrefix("settle r3 superseded"), "r3 被 r4 取代");
  expect(!sim.log.hasPrefix("settle r3 applied"), "r3 不得报成达成");
  expect(!sim.book.enablePending(), "收敛后无未决意图");
}

void scenario_off_on_off_then_failure_then_reopen() {
  Sim sim;
  sim.command(false, false, true, true, "r1");
  sim.command(false, false, true, false, "r2");
  sim.command(false, false, true, true, "r3");   // 在途
  sim.command(false, false, true, false, "r4");  // 又关
  sim.buildEnded(false);                         // 构建成功但被丢弃
  expect(!sim.state.asr, "丢弃后仍是关闭");
  sim.command(false, false, true, true, "r5");   // 再次打开
  expect(sim.log.hasPrefix("action enable r5"), "关闭后的新打开意图要重新起一次构建");
  sim.buildEnded(false);
  expect(sim.state.asr && sim.ready, "最终按最后一次意图打开");
  expect(sim.log.hasPrefix("settle r5 applied"), "r5 达成");
}

void scenario_wake_only_leaves_asr_alone() {
  Sim sim;
  sim.command(false, false, true, true, "r1");     // 打开识别（构建中）
  sim.command(true, false, false, false, "w1");    // 只动唤醒
  expect(!sim.state.wake, "唤醒立即生效");
  expect(sim.log.hasPrefix("settle w1 applied"), "只给唤醒字段的命令当次达成");
  expect(!sim.log.hasPrefix("settle r1 superseded"), "没提识别字段的命令不得取代识别意图");
  sim.buildEnded(false);
  expect(sim.state.asr && sim.ready, "识别意图照常收敛");
  expect(sim.log.hasPrefix("settle r1 applied"), "r1 达成");
}

void scenario_mixed_fields_are_independent() {
  Sim sim;
  // 一条命令同时请求「唤醒关 + 识别开」，随后只改唤醒。识别字段没有被取代，
  // 仍然要驱动构建；唤醒字段被取代不能把识别一起丢掉。
  sim.foreignSwitchStarted();
  sim.command(true, false, true, true, "both");
  sim.command(true, true, false, false, "wake");
  expect(!sim.log.hasPrefix("action enable"), "在途构建期间不并发第二个构建");
  sim.foreignSwitchEnded();
  expect(sim.log.hasPrefix("action enable both"), "识别字段未被取代时必须驱动构建");
  sim.buildEnded(false);
  expect(sim.state.asr && sim.ready, "识别字段照常达成");
  expect(!sim.book.enablePending(), "收敛后无残留意图");

  // 反向：识别字段被更新的关闭取代，唤醒字段仍然按自己的意图生效。
  Sim other;
  other.command(true, true, true, true, "r1");
  other.command(true, false, true, false, "r2");
  expect(!other.state.wake, "r1 的唤醒开被 r2 的关取代");
  expect(!other.state.asr, "r1 的识别开被 r2 的关取代");
  expect(other.log.hasPrefix("settle r1 superseded"), "r1 两个字段都被取代：按被取代结算");
  expect(other.log.hasPrefix("settle r2 applied"), "r2 达成");
  other.buildEnded(false);
  expect(!other.state.asr, "被取代的构建结果不得接入");
}

void scenario_idempotent_targets() {
  Sim sim;
  sim.ready = true;
  sim.state.asr = true;
  sim.command(false, false, true, true, "r1");   // 已经开着且就绪
  expect(sim.log.hasPrefix("settle r1 applied"), "已是目标态的命令立即达成");
  expect(!sim.log.hasPrefix("action enable r1"), "已是目标态不应再起构建");
  sim.command(false, false, true, false, "r2");
  sim.command(false, false, true, false, "r3");
  expect(sim.log.hasPrefix("settle r2 applied") && sim.log.hasPrefix("settle r3 applied"),
         "重复关闭各自达成，不互相取代");
}

void scenario_build_start_failure_settles_everything() {
  Sim sim;
  sim.command(false, false, true, true, "r1");
  sim.command(false, false, true, true, "r2");
  sim.buildStartFailed();
  expect(sim.log.hasPrefix("settle r1 failed") && sim.log.hasPrefix("settle r2 failed"),
         "构建起不来时每条未决意图都要有失败终态");
  expect(!sim.book.enablePending(), "不留未决意图，界面不会停在「正在应用」");
  expect(!sim.state.asr, "开关退回关闭，与失败结果一致");
}

// 交错扫描：随机混排命令与构建结果，检查三条不变量在任何顺序下都成立。
void stress_interleavings() {
  uint32_t rng = 0x9e3779b9u;
  const auto next = [&rng]() {
    rng = rng * 1664525u + 1013904223u;
    return static_cast<uint32_t>(rng >> 8);
  };
  for (int seed = 0; seed < 400; ++seed) {
    Sim sim;
    const int failures_before = g_failures;
    std::string last_asr_rid;
    std::string last_asr_outcome;
    bool last_asr_on = false;
    std::vector<std::string> settled_rids;
    size_t consumed = 0;
    int sequence = 0;
    for (int step = 0; step < 24; ++step) {
      if (sim.switching && next() % 2 == 0) {
        sim.buildEnded(next() % 3 == 0);
      } else {
        const bool wake_set = next() % 3 == 0;
        const bool asr_set = next() % 4 != 0;
        if (!wake_set && !asr_set) continue;
        const bool wake = next() % 2 == 0;
        const bool asr = next() % 2 == 0;
        const std::string rid = "s" + std::to_string(sequence++);
        if (asr_set) {
          last_asr_rid = rid;
          last_asr_outcome.clear();
          last_asr_on = asr;
        }
        sim.command(wake_set, wake, asr_set, asr, rid);
      }
      // 终态不重不缺：每条 request_id 至多一个终态。
      for (; consumed < sim.terminals.size(); ++consumed) {
        const std::string &rid = sim.terminals[consumed];
        for (const auto &other : settled_rids) {
          expect(other != rid, "同一条意图出现了多个终态：" + rid);
        }
        settled_rids.push_back(rid);
      }
    }
    // 把所有构建跑完，检查收敛结果。
    int guard = 0;
    while ((sim.switching || sim.book.enablePending()) && guard++ < 100) {
      if (sim.switching) {
        sim.buildEnded(false);
      } else {
        sim.settle();
      }
    }
    expect(guard < 100, "收敛过程不应当无法终止");
    expect(!sim.book.enablePending(), "收敛后不应残留未决意图");
    expect(sim.state.asr == sim.ready, "收敛后生效开关与后端就绪程度必须一致");
    if (!last_asr_rid.empty()) {
      for (const auto &line : sim.log.lines()) {
        if (line.rfind("settle " + last_asr_rid + " ", 0) != 0) continue;
        // 行格式：settle <rid> <outcome> (asr=.. ready=..)——取第三个字段。
        const size_t first = line.find(' ');
        const size_t second = line.find(' ', first + 1);
        const size_t third = line.find(' ', second + 1);
        last_asr_outcome = line.substr(second + 1, third - second - 1);

      }
      expect(!last_asr_outcome.empty(), "最后一条识别意图必须有终态：" + last_asr_rid);
      if (!last_asr_on) {
        // 最后一次意图是关闭：不需要构建，必须在关闭状态下收尾。
        expect(!sim.state.asr, "最后一次关闭意图之后开关必须保持关闭");
      } else if (sim.state.asr) {
        expect(sim.ready, "最后一次打开意图达成就必须真的就绪");
      } else {
        // 打开意图没达成：只能是这次尝试失败，不能是被静默吃掉。
        expect(last_asr_outcome == "failed",
               "打开意图落空时必须如实报失败：" + last_asr_outcome);
      }
    }
    if (g_failures != failures_before) {
      std::printf("seed %d 的轨迹：\n", seed);
      for (const auto &line : sim.log.lines()) {
        std::printf("  %s\n", line.c_str());
      }
      std::printf("  最终：asr=%d ready=%d\n", sim.state.asr ? 1 : 0, sim.ready ? 1 : 0);
    }
  }
}

}  // namespace

int main() {
  scenario_off_during_enable_build();
  scenario_off_then_on_during_build();
  scenario_failure_does_not_eat_newer_intent();
  scenario_failure_after_off_never_reopens();
  scenario_off_on_off_then_failure();
  scenario_off_on_off_then_failure_then_reopen();
  scenario_wake_only_leaves_asr_alone();
  scenario_mixed_fields_are_independent();
  scenario_idempotent_targets();
  scenario_build_start_failure_settles_everything();
  stress_interleavings();
  if (g_failures == 0) {
    std::printf("ok: %d checks\n", g_checks);
    return 0;
  }
  std::printf("failed: %d/%d checks\n", g_failures, g_checks);
  return 1;
}
