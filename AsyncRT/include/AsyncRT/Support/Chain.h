//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef ASYNCRT_SUPPORT_CHAIN_H
#define ASYNCRT_SUPPORT_CHAIN_H

namespace M::AsyncRT {

/// This type is used to model dependencies between side-effecting operations,
/// by turning these side effects into explicitly modeled values.  Its runtime
/// representation is a zero sized value.
///
class Chain {
public:
  static void swap(Chain &lhs, Chain &rhs) {}
};

} // namespace M::AsyncRT

#endif // ASYNCRT_SUPPORT_CHAIN_H
