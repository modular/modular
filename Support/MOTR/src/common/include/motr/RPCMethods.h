//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOTR_RPC_METHODS_H
#define MOTR_RPC_METHODS_H

#include "motr/RPCMailbox.h"
#include "motr/Types/Types.h"
#include <functional>
#include <optional>

namespace M::motr::RPC::Methods {

struct getHostInfo {
  static constexpr const char rpcCallName[] = "getHostInfo";
  static constexpr Hash::Value rpcCallNameHash{rpcCallName};
  using ReturnType = HostInfo;
  using Call = RPC::RPCCall<rpcCallName, ReturnType>;
  using FuncType = ReturnType (*)();
  static FuncType func;
};

struct getMotrServerInfo {
  static constexpr const char rpcCallName[] = "getMotrServerInfo";
  static constexpr Hash::Value rpcCallNameHash{rpcCallName};
  using ReturnType = MotrServerInfo;
  using Call = RPC::RPCCall<rpcCallName, ReturnType>;
  using FuncType = ReturnType (*)();
  static FuncType func;
};

template <typename T, typename... Args>
static std::optional<typename T::ReturnType> execute(Args &&...args) {
  if (!T::func) {
    MOTR_LOG("{}", "RPC Method not found");
    return std::nullopt;
  }
  return T::func(std::forward<Args>(args)...);
}

} // namespace M::motr::RPC::Methods

#endif // MOTR_RPC_METHODS_H
