//===- KGENVerifyHelper.h -------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_VERIFY_HELPER_H
#define KGEN_VERIFY_HELPER_H

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"

namespace M::KGEN {

/// Compare a range of values from an "originator" to a corresponding range of
/// values from a "target".  If the two mismatch, emit an error that tries to
/// explain the issue in a nice way.
template <typename TargetRange, typename OriginatorRange>
inline ParseResult verifyMatchingLists(
    const OriginatorRange &originatorRange, const TargetRange &targetRange,
    const char *originatorName, Location originatorLoc, const char *targetName,
    Location targetLoc, const char *itemName, const char *propertyName) {
  // Check that the ranges have the same size.  If not, diagnose this.
  size_t numOriginator =
      std::distance(originatorRange.begin(), originatorRange.end());
  size_t numTarget = std::distance(targetRange.begin(), targetRange.end());
  if (numOriginator != numTarget) {
    auto diag = emitError(originatorLoc, originatorName)
                << " has " << numOriginator << " " << itemName
                << (numOriginator != 1 ? "s" : "") << " but " << targetName
                << " expects " << numTarget;
    if (originatorLoc != targetLoc)
      diag.attachNote(targetLoc) << targetName << " declared here";
    return failure();
  }

  // If they have the same sizes, diagnose any mismatches between their
  // elements.

  // NOTE: llvm::zip doesn't work with LLVM mapped iterators.
  auto targetIt = targetRange.begin();
  auto originatorIt = originatorRange.begin();
  for (size_t itemNum = 0; itemNum != numTarget; ++itemNum) {
    auto targetVal = *targetIt++;
    auto originatorVal = *originatorIt++;
    if (originatorVal == targetVal)
      continue;

    auto diag = emitError(originatorLoc, originatorName)
                << ' ' << itemName << " #" << itemNum << " has " << propertyName
                << ' ' << originatorVal << " but " << targetName << " expected "
                << propertyName << ' ' << targetVal;
    if (originatorLoc != targetLoc)
      diag.attachNote(targetLoc) << targetName << " declared here";
    return failure();
  }

  return success();
}

} // namespace M::KGEN

#endif // KGEN_VERIFY_HELPER_H