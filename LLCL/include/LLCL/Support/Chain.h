//===- LLCL/Support/Chain.h -------------------------------------*- C++ -*-===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef LLCL_SUPPORT_CHAIN_H
#define LLCL_SUPPORT_CHAIN_H

namespace LLCL {

/// This type is used to model dependences between side-effecting operations,
/// by turning these side effects into explicitly modeled values.  Its runtime
/// representation is a zero sized value.
///
class Chain {};

} // end namespace LLCL

#endif // LLCL_SUPPORT_CHAIN_H