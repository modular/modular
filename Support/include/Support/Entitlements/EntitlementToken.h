//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_ENTITLEMENTS_ENTITLEMENTTOKEN_H
#define SUPPORT_ENTITLEMENTS_ENTITLEMENTTOKEN_H

#include <memory>
#include <string>
#include <vector>

#include "Support/ErrorOr.h"
#include "llvm/ADT/StringRef.h"

namespace M {

struct EntitlementToken {
  std::string key;
  std::vector<std::string> certChain;
};

ErrorOr<std::unique_ptr<EntitlementToken>> unpackToken(StringRef b64token);

std::string packToken(const EntitlementToken &token);

} // namespace M

#endif // SUPPORT_ENTITLEMENTS_ENTITLEMENT_H
