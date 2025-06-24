//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Types that are reflectable to structured messages for transport over
// websocket

#ifndef MOTR_TYPES_H
#define MOTR_TYPES_H

#include "motr/Common.h"
#include "motr/Hash.h"
#include "motr/RPC.h"
#include <string>

namespace M::motr {

struct HostInfo {
  std::string hostname;
  std::string uname;
  std::string os;
  uint64_t t0 = 0;
  uint64_t numCPUs = 0;
  uint64_t numGPUs = 0;
  RPC_REFLECTABLE(hostname, uname, os, t0, numCPUs, numGPUs);
};

struct MotrServerInfo {
  std::string version = MOTR_VERSION_STRING;
  uint64_t versionMajor = MOTR_VERSION_MAJOR;
  uint64_t versionMinor;
  uint64_t versionPatch;
  uint64_t buildTimestamp = 0;
  uint64_t startTimestamp = 0;
  uint64_t processId = 0;
  RPC_REFLECTABLE(version, versionMajor, versionMinor, versionPatch,
                  buildTimestamp, startTimestamp, processId);
};

} // namespace M::motr

#endif // MOTR_TYPES_H
