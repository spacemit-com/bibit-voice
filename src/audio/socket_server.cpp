#include "socket_server.h"

#include <sys/poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <system_error>

SocketServer::SocketServer(std::string path, LineHandler handler)
    : path_(std::move(path)), handler_(std::move(handler)) {}

SocketServer::~SocketServer() { stop(); }

namespace {
bool socketInUse(const std::string &path) {
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  if (path.size() >= sizeof(addr.sun_path)) return false;
  std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
  const int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (fd < 0) return false;
  const int rc = ::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr));
  const int connect_errno = rc == 0 ? 0 : errno;
  ::close(fd);
  // EACCES 同样是「路径上已经有别人的 socket」：按在用处理，否则我们会 unlink
  // 掉活跃实例的 socket 再 bind 成功，两个实例各占一个 inode，谁也管不了谁。
  return rc == 0 || connect_errno == EACCES;
}
}  // namespace

bool SocketServer::start() {
  if (running_) return true;
  if (listen_fd_ < 0 && !acquire()) return false;
  running_ = true;
  try {
    thread_ = std::thread(&SocketServer::run, this);
  } catch (...) {
    running_ = false;
    stop();
    return false;
  }
  return true;
}

bool SocketServer::acquire() {
  if (listen_fd_ >= 0) return true;
  std::error_code fs_error;
  const std::filesystem::path dir = std::filesystem::path(path_).parent_path();
  std::filesystem::create_directories(dir, fs_error);
  if (fs_error) return false;
  // 已有活跃实例在监听同一路径：单实例冲突，不做任何 unlink（unlink 会把
  // 活跃实例的 socket 路径摘掉，导致两个实例都能 bind 成功）。
  if (socketInUse(path_)) {
    return false;
  }
  // 路径上只剩残留的 socket 文件（无进程监听）时再清掉，否则 bind 会 EADDRINUSE。
  ::unlink(path_.c_str());
  listen_fd_ = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (listen_fd_ < 0) return false;
  const auto fail = [this]() {
    if (listen_fd_ >= 0) ::close(listen_fd_);
    listen_fd_ = -1;
    ::unlink(path_.c_str());
    return false;
  };
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  if (path_.size() >= sizeof(addr.sun_path)) return fail();
  std::strncpy(addr.sun_path, path_.c_str(), sizeof(addr.sun_path) - 1);
  if (::bind(listen_fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
    std::cerr << "bind " << path_ << ": " << std::strerror(errno) << std::endl;
    return fail();
  }
  ::chmod(path_.c_str(), 0600);
  if (::listen(listen_fd_, 64) < 0) return fail();
  // 监听 fd 设成非阻塞：接受连接时可以把队列一次排空。阻塞式 accept 每次 poll
  // 只取一个连接，突发命令（GUI 连点、脚本背靠背）会堆到 backlog 上限，
  // 之后的 connect 直接 EAGAIN 失败——命令还没发出去就被顶回来了。
  const int flags = ::fcntl(listen_fd_, F_GETFL, 0);
  if (flags >= 0) ::fcntl(listen_fd_, F_SETFL, flags | O_NONBLOCK);
  return true;
}

void SocketServer::stop() {
  const bool was_running = running_.exchange(false);
  const bool was_bound = listen_fd_ >= 0;
  if (was_bound) {
    ::shutdown(listen_fd_, SHUT_RDWR);
    ::close(listen_fd_);
    listen_fd_ = -1;
  }
  if (was_running && thread_.joinable()) thread_.join();
  std::lock_guard<std::mutex> lk(clients_mutex_);
  for (int fd : clients_) ::close(fd);
  clients_.clear();
  muted_clients_.clear();
  buffers_.clear();
  // 只有真的占过这个路径才删：没 acquire 成功过时 unlink 会摘掉别人正在用的 socket。
  if (was_bound) ::unlink(path_.c_str());
}

void SocketServer::broadcast(const std::string &line) {
  std::lock_guard<std::mutex> lk(clients_mutex_);
  const std::string payload = line + "\n";
  for (int fd : clients_) {
    // 已经发不出去的连接只标记停发，不关连接也不清缓冲：客户端「写完即关」时
    // 命令可能还在接收队列里等着被处理，删缓冲等于无声吞掉一条命令。
    if (muted_clients_.count(fd)) continue;
    const ssize_t n = ::send(fd, payload.data(), payload.size(), MSG_NOSIGNAL | MSG_DONTWAIT);
    const bool failed = n < 0 && errno != EAGAIN && errno != EWOULDBLOCK;
    const bool partial = n >= 0 && static_cast<size_t>(n) != payload.size();
    if (failed || partial) muted_clients_.insert(fd);
  }
}

void SocketServer::sendTo(int fd, const std::string &line) {
  const std::string payload = line + "\n";
  ::send(fd, payload.data(), payload.size(), MSG_NOSIGNAL | MSG_DONTWAIT);
}

void SocketServer::removeClient(int fd) {
  std::lock_guard<std::mutex> lk(clients_mutex_);
  clients_.erase(std::remove(clients_.begin(), clients_.end(), fd), clients_.end());
  muted_clients_.erase(fd);
  buffers_.erase(fd);
  ::close(fd);
}

void SocketServer::run() {
  while (running_) {
    std::vector<pollfd> fds;
    fds.push_back({listen_fd_, POLLIN, 0});
    {
      std::lock_guard<std::mutex> lk(clients_mutex_);
      for (int fd : clients_) fds.push_back({fd, POLLIN | POLLHUP | POLLERR, 0});
    }
    const int rc = ::poll(fds.data(), fds.size(), 250);
    if (rc <= 0) continue;
    if (fds[0].revents & POLLIN) {
      // 一次把 accept 队列排空：突发连接不会堆在 backlog 上把后来的 connect 顶回去。
      for (int accepted = 0; accepted < 64; ++accepted) {
        const int fd = ::accept4(listen_fd_, nullptr, nullptr, SOCK_CLOEXEC);
        if (fd < 0) break;
        std::lock_guard<std::mutex> lk(clients_mutex_);
        clients_.push_back(fd);
        buffers_[fd] = {};
      }
    }
    for (size_t i = 1; i < fds.size(); ++i) {
      const int fd = fds[i].fd;
      {
        std::lock_guard<std::mutex> lk(clients_mutex_);
        if (std::find(clients_.begin(), clients_.end(), fd) == clients_.end()) continue;
      }
      // 先收数据、再处理挂断：客户端「写完即关」时 poll 会同时报 POLLIN|POLLHUP，
      // 若先按挂断清理连接，刚刚到达的那条命令就被丢掉了（GUI 的即发即忘命令、
      // CLI 的一次性连接都走这条路径）。
      bool closed = false;
      if (fds[i].revents & POLLIN) {
        char data[4096];
        const ssize_t n = ::recv(fd, data, sizeof(data), 0);
        if (n <= 0) {
          closed = true;
        } else {
          std::lock_guard<std::mutex> lk(clients_mutex_);
          buffers_[fd].append(data, static_cast<size_t>(n));
        }
      }
      while (true) {
        std::string line;
        {
          std::lock_guard<std::mutex> lk(clients_mutex_);
          auto &b = buffers_[fd];
          const size_t pos = b.find('\n');
          if (pos == std::string::npos) {
            if (b.size() > 16384) b.clear();
            break;
          }
          line = b.substr(0, pos);
          b.erase(0, pos + 1);
        }
        if (handler_ && !line.empty()) handler_(fd, line);
      }
      if (closed || (fds[i].revents & (POLLHUP | POLLERR | POLLNVAL))) {
        removeClient(fd);
        continue;
      }
    }
  }
}
