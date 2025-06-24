//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOTR_HOSTINFO_H
#define MOTR_HOSTINFO_H

#include "motr/Types/Types.h" // actual HostInfo struct definition w/ RPC reflection

namespace M::motr {

HostInfo getHostInfo();

MotrServerInfo getMotrServerInfo();

std::string getHostname();
std::string getOS();
std::string getUsername();
uint64_t getBootTime();

} // namespace M::motr

#endif // MOTR_HOSTINFO_H
