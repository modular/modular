//===- LLCL/Support/Chain.h -----------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef LLCL_SUPPORT_CHAIN_H
#define LLCL_SUPPORT_CHAIN_H

namespace M::LLCL {

/// This type is used to model dependences between side-effecting operations,
/// by turning these side effects into explicitly modeled values.  Its runtime
/// representation is a zero sized value.
///
class Chain {};

} // namespace M::LLCL

#endif // LLCL_SUPPORT_CHAIN_H
