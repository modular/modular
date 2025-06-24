//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "HostInfo.h"

#include "motr/Log.h"
#include "motr/Time.h"
#include "motr/motr.h"
#include <cstring>
#include <ctime>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>

namespace M::motr {

struct IPV4 {
  uint8_t a, b, c, d;
};

std::vector<IPV4> getIPAddresses() {
  std::vector<IPV4> addresses;
  struct ifaddrs *ifaddr;
  if (getifaddrs(&ifaddr) == 0) {
    for (struct ifaddrs *ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
      if (ifa->ifa_addr && ifa->ifa_addr->sa_family == AF_INET) {
        struct sockaddr_in *sa = (struct sockaddr_in *)ifa->ifa_addr;
        IPV4 ip;
        ip.a = static_cast<uint8_t>(sa->sin_addr.s_addr & 0xFF);
        ip.b = static_cast<uint8_t>((sa->sin_addr.s_addr >> 8) & 0xFF);
        ip.c = static_cast<uint8_t>((sa->sin_addr.s_addr >> 16) & 0xFF);
        ip.d = static_cast<uint8_t>((sa->sin_addr.s_addr >> 24) & 0xFF);
        if (ip.a == 127 && ip.b == 0 && ip.c == 0 && ip.d == 1)
          continue;
        addresses.push_back(ip);
      }
    }
    freeifaddrs(ifaddr);
  }
  return addresses;
}

std::string getHostname() {
  char hostname[256];
  if (gethostname(hostname, sizeof(hostname)) == 0)
    return std::string(hostname);
  return "<UNKNOWN>";
}

uint64_t getBootTime() {
#ifdef __APPLE__
  // On macOS, calculate boot time = current time - uptime
  struct timespec current_ts, uptime_ts;

  // Get current wall clock time
  if (clock_gettime(CLOCK_REALTIME, &current_ts) != 0)
    return 0;

  // Get system uptime using CLOCK_MONOTONIC_RAW (truly monotonic)
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &uptime_ts) != 0)
    return 0;

  // Calculate boot time = current time - uptime
  uint64_t current_ns = static_cast<uint64_t>(
      current_ts.tv_sec * 1000000000ULL + current_ts.tv_nsec);
  uint64_t uptime_ns = static_cast<uint64_t>(uptime_ts.tv_sec * 1000000000ULL +
                                             uptime_ts.tv_nsec);

  return current_ns - uptime_ns;
#else
  // Linux implementation
  struct timespec ts;
  if (clock_gettime(CLOCK_BOOTTIME, &ts) == 0)
    return static_cast<uint64_t>(ts.tv_sec * 1000000000ULL + ts.tv_nsec);
  return 0;
#endif
}

std::string getUsername() {
  char uname[256];
  if (getlogin_r(uname, sizeof(uname)) == 0)
    return std::string(uname);
  return "<UNKNOWN>";
}

std::string getOS() {
#ifdef MOTR_PLATFORM_MACOS
  return "MacOS";
#else
  return "Linux";
#endif
}

HostInfo getHostInfo() {
  std::vector<IPV4> ipAddresses = getIPAddresses();
  for (const auto &ip : ipAddresses) {
    MOTR_LOG("IP address: {}.{}.{}.{}", ip.a, ip.b, ip.c, ip.d);
  }
  return {
      .hostname = getHostname(),
      .uname = getUsername(),
      .os = getOS(),
      .t0 = getBootTime(),
      .numCPUs = std::thread::hardware_concurrency(),
      .numGPUs = 0,
  };
}

MotrServerInfo getMotrServerInfo() {
  return {
      .version = MOTR_VERSION_STRING,
      .versionMajor = MOTR_VERSION_MAJOR,
      .versionMinor = MOTR_VERSION_MINOR,
      .versionPatch = MOTR_VERSION_PATCH,
      .buildTimestamp = static_cast<uint64_t>(Time::getBuildTimestamp().v),
      .startTimestamp = static_cast<uint64_t>(Time::getStartTimestamp().v),
      .processId = getProcessID(),
  };
}

} // namespace M::motr
