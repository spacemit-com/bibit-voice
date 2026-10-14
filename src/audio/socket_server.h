#pragma once

#include <atomic>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

class SocketServer {
 public:
  using LineHandler = std::function<void(int, const std::string &)>;
  SocketServer(std::string path, LineHandler handler);
  ~SocketServer();
  // 先占住 socket（bind+listen，不启线程）：调用方要在碰机器级独占资源
  // （Qwen3 的 8063、SpacemiT EP/TCM）之前就确认自己是唯一实例。
  bool acquire();
  bool start();
  void stop();
  void setHandler(LineHandler handler) { handler_ = std::move(handler); }
  void broadcast(const std::string &line);
  void sendTo(int fd, const std::string &line);
  const std::string &path() const { return path_; }

 private:
  void run();
  void removeClient(int fd);

  std::string path_;
  LineHandler handler_;
  int listen_fd_ = -1;
  std::atomic<bool> running_{false};
  std::thread thread_;
  std::mutex clients_mutex_;
  std::vector<int> clients_;
  // 广播发不出去的连接：只停发，不关连接、不清缓冲。客户端「写完即关」时命令
  // 可能还在接收队列里，广播失败就把缓冲删掉会把那条命令无声吞掉。
  std::set<int> muted_clients_;
  std::map<int, std::string> buffers_;
};
