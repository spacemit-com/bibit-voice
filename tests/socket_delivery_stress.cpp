// 回归：客户端「写完即关」的命令不允许被服务端静默丢掉。
//
// 背景：SocketServer::broadcast() 曾经在发送失败时顺手删掉该连接的接收缓冲。
// 客户端连上、写完就关（GUI、CLI、脚本都这么做）时，广播先到、命令还在接收队列里
// 等着被处理，于是那条命令连同缓冲一起消失：客户端没报错，服务端也没有任何记录，
// 表现就是「点了一下开关没反应」。有广播负载时实测丢 1%–3%。
//
// 由 tests/test_socket_delivery.py 编译运行；也可以直接：
//   g++ -std=c++17 -pthread -I src/audio tests/socket_delivery_stress.cpp \
//       src/audio/socket_server.cpp -o /tmp/t && /tmp/t

#include "socket_server.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

const char *kPath = "/tmp/bibit-socket-delivery-stress.sock";

int connectTo() {
  const int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (fd < 0) return -1;
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::strncpy(addr.sun_path, kPath, sizeof(addr.sun_path) - 1);
  if (::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
    ::close(fd);
    return -1;
  }
  return fd;
}

// 写完即关：与 GUI 的命令连接形态一致。
int sendAndClose(const std::string &line) {
  const int fd = connectTo();
  if (fd < 0) return -1;
  const ssize_t written = ::send(fd, line.data(), line.size(), 0);
  ::close(fd);
  return written == static_cast<ssize_t>(line.size()) ? 0 : -2;
}

struct RoundResult {
  int sent = 0;
  int send_failures = 0;
  int received = 0;
  int missing = 0;
};

// 一轮：commands 条「写完即关」的命令，可选后台广播负载 + 常驻订阅者。
RoundResult runRound(SocketServer *server, int commands, bool with_broadcast,
                     const std::string &prefix) {
  std::atomic<int> received{0};
  std::mutex seen_mutex;
  std::vector<std::string> seen;
  server->setHandler([&](int, const std::string &line) {
    received.fetch_add(1);
    std::lock_guard<std::mutex> lk(seen_mutex);
    seen.push_back(line);
  });

  int subscriber = -1;
  std::atomic<bool> stop{false};
  std::thread subscriber_thread;
  std::thread broadcaster;
  if (with_broadcast) {
    subscriber = connectTo();
    if (subscriber < 0) return RoundResult{commands, -1, 0, commands};
    subscriber_thread = std::thread([&] {
      std::vector<char> buffer(65536);
      while (!stop.load()) {
        if (::recv(subscriber, buffer.data(), buffer.size(), 0) <= 0) return;
      }
    });
    broadcaster = std::thread([&] {
      // 与真实服务一致：命令处理完就会广播状态，广播先到、命令还没被处理。
      while (!stop.load()) {
        server->broadcast("{\"event\":\"status\",\"pad\":\"" + std::string(300, 'x') + "\"}");
        std::this_thread::sleep_for(std::chrono::microseconds(200));
      }
    });
  }

  RoundResult result;
  result.sent = commands;
  for (int index = 0; index < commands; ++index) {
    const std::string line = "{\"cmd\":\"set_switches\",\"wake\":true,\"request_id\":\"" +
                             prefix + std::to_string(index) + "\"}\n";
    if (sendAndClose(line) != 0) ++result.send_failures;
  }
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (received.load() < result.sent - result.send_failures &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  stop.store(true);
  if (broadcaster.joinable()) broadcaster.join();
  if (subscriber >= 0) ::shutdown(subscriber, SHUT_RDWR);
  if (subscriber_thread.joinable()) subscriber_thread.join();
  if (subscriber >= 0) ::close(subscriber);
  result.received = received.load();

  std::lock_guard<std::mutex> lk(seen_mutex);
  for (int index = 0; index < commands; ++index) {
    const std::string rid = prefix + std::to_string(index);
    bool found = false;
    for (const auto &line : seen) {
      if (line.find(rid) != std::string::npos) {
        found = true;
        break;
      }
    }
    if (!found) ++result.missing;
  }
  return result;
}

}  // namespace

int main(int argc, char **argv) {
  const int commands = argc > 1 ? std::atoi(argv[1]) : 100;
  ::unlink(kPath);
  SocketServer server(kPath, SocketServer::LineHandler{});
  if (!server.start()) {
    std::printf("{\"error\":\"server start failed\"}\n");
    return 2;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  // 有广播负载那一轮才是历史缺陷的现场：命令与广播交错到达。
  const RoundResult loaded = runRound(&server, commands, true, "loaded-");
  const RoundResult quiet = runRound(&server, commands, false, "quiet-");
  server.stop();

  std::printf("{\"loaded\":{\"sent\":%d,\"send_failures\":%d,\"received\":%d,\"missing\":%d},"
              "\"quiet\":{\"sent\":%d,\"send_failures\":%d,\"received\":%d,\"missing\":%d}}\n",
              loaded.sent, loaded.send_failures, loaded.received, loaded.missing,
              quiet.sent, quiet.send_failures, quiet.received, quiet.missing);
  return (loaded.missing == 0 && loaded.send_failures == 0 && quiet.missing == 0)
             ? 0
             : 1;
}
