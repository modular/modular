//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_SUPPORT_FORCELINKMLIRC_H
#define KGEN_SUPPORT_FORCELINKMLIRC_H

namespace M::KGEN {

//===----------------------------------------------------------------------===//
// Force Link MLIR C API
//===----------------------------------------------------------------------===//

/// Calling this function forces the linking of MLIR C API symbols. This allows
/// JIT'ed Mojo code to use the same MLIR C API symbols as the current process,
/// which is necessary to avoid conflicting TypeIDs.
void forceLinkMLIRC();

} // namespace M::KGEN

#endif // KGEN_SUPPORT_FORCELINKMLIRC_H
